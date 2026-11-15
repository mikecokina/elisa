from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

import numpy as np
import pandas as pd
from scipy import interpolate

from elisa import const, settings, utils
from elisa import umpy as up
from elisa.base.error import (
    ElisaLookupError,
    raise_elisa_initial_parameters_error,
    raise_elisa_limb_darkening_error,
    raise_elisa_unexpected_error,
)
from elisa.buffer import buffer
from elisa.logger import getLogger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float

logger = getLogger(__name__)

_LD_LAWS_LINEAR: Final[set[str]] = {"linear", "cosine"}
_LD_ERROR_SAMPLE_SIZE: Final[int] = 10


class CustomLimbDarkeningCoefficients(dict):
    """Custom passband coefficients with the law used to validate them."""

    def __init__(self, coefficients: dict, law: str) -> None:
        super().__init__(coefficients)
        self.law = law

    def copy(self) -> CustomLimbDarkeningCoefficients:
        """Copy the mapping without losing its defining law."""
        return type(self)(self, self.law)

    def validate_law(self) -> None:
        """Reject incompatible fixed configuration with ElisaInitialParametersError."""
        current = settings.LIMB_DARKENING_LAW
        if self.law == current or {self.law, current} <= _LD_LAWS_LINEAR:
            return
        msg = (
            f"Custom limb-darkening coefficients were defined for '{self.law}', "
            f"but the current limb-darkening law is '{current}'. "
            "Supply new coefficients for the current law or restore the original law."
        )
        # The defining law and custom coefficients are fixed configuration, not
        # physical values that drift between proposals. Reusing them under another
        # law must abort fitting rather than penalize every candidate as an LD failure.
        # TODO: consider to use new critical LD error  # noqa: FIX002, TD002, TD003
        raise_elisa_initial_parameters_error(msg)


class LDFileRevision(NamedTuple):
    """Filesystem revision used to invalidate cached LD source data."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


class LDTableIdentity(NamedTuple):
    """An ordered source table filename and its filesystem revision."""

    filename: str
    revision: LDFileRevision


class LDSourceIdentity(NamedTuple):
    """Immutable, hashable identity of one complete LD interpolation source.

    Table and column order are significant. Both cache layers use this identity
    so changes to files, law, directory or column selection cannot reuse stale
    structures or evaluated coefficients.
    """

    directory: str
    law: str
    tables: tuple[LDTableIdentity, ...]
    csv_columns: tuple[str, ...]
    domain_columns: tuple[str, ...]
    coefficient_columns: tuple[str, ...]


def get_metallicity_from_ld_table_filename(filename: str) -> float:
    """Extract the metallicity encoded in a Van Hamme LD table filename.

    The expected filename format is typically::

        <prefix>.<passband>.<metallicity>.csv

    where the metallicity field is the second-to-last dot-separated token.

    :param filename: Path to an LD table file.
    :returns: Metallicity as a numeric value.
    """
    basename = Path(filename).name
    metallicity_token = str(basename).split(".")[-2]
    return utils.numeric_metallicity_from_string(metallicity_token)


def get_ld_table_filename(passband: str, metallicity: float, law: str | None = None) -> str:
    """Build a limb darkening table filename for a passband and metallicity.

    If *law* is not provided, the configured default law is used.

    :param passband: Passband identifier (for example ``"V"`` or ``"bolometric"``).
    :param metallicity: Metallicity value.
    :param law: Limb darkening law name (``"linear"``, ``"cosine"``, ``"logarithmic"``, ``"square_root"``).
    :returns: Filename (without directory) of the corresponding CSV table.
    """
    resolved_law = law if not utils.is_empty(law) else settings.LIMB_DARKENING_LAW
    prefix = settings.LD_LAW_TO_FILE_PREFIX[resolved_law]
    m_str = utils.numeric_metallicity_to_string(metallicity)
    return f"{prefix}.{passband}.{m_str}.csv"


def get_relevant_ld_tables(passband: str, metallicity: float, law: str | None = None) -> list[str]:
    """Get filenames of tables surrounding the requested metallicity.

    The surrounding metallicities are determined from :data:`elisa.const.METALLICITY_LIST_LD`.

    :param passband: Passband identifier.
    :param metallicity: Metallicity value.
    :param law: Limb darkening law name.
    :returns: List of CSV filenames to use for interpolation.
    """
    resolved_law = law if not utils.is_empty(law) else settings.LIMB_DARKENING_LAW
    # Nonfinite candidate metallicity cannot select a physical LD grid. Reject it
    # explicitly: NaN bypasses the ordinary lower/upper-bound comparisons.
    if not np.isfinite(metallicity):
        msg = f"Cannot select limb-darkening tables for nonfinite metallicity {metallicity!r}."
        raise_elisa_limb_darkening_error(msg)
    try:
        surrounded = utils.find_surrounded(const.METALLICITY_LIST_LD, metallicity)
    except (ElisaLookupError, ValueError) as exc:
        msg = f"Cannot select limb-darkening tables for metallicity {metallicity!r}: {exc}"
        raise_elisa_limb_darkening_error(msg, exc)
    return [get_ld_table_filename(passband, m, resolved_law) for m in surrounded]


def clear_ld_caches() -> None:
    """Release process-local LD caches after an asset update or on demand.

    Ordinary file edits/replacements are detected by stat revision keys. Call
    this after updates on filesystems that do not provide reliable nanosecond
    mtime/ctime/inode metadata, and before resuming calculations after such an
    update. Do not modify tables concurrently with an active calculation.
    """
    from elisa.base.curves.utils import interpolate_on_ld_grid_cached  # noqa: PLC0415

    buffer.LD_CFS_TABLES.clear()
    get_ld_source_interpolator.cache_clear()
    interpolate_on_ld_grid_cached.cache_clear()


def get_ld_source_identities(
    metallicity: float,
    passbands: Iterable[str],
) -> tuple[LDSourceIdentity, ...]:
    """Snapshot every effective source input without reading/hashing CSV contents.

    Ordered filenames (including duplicates), column order, resolved directory,
    law and file revisions are shared by both cache layers. ctime catches
    same-size rewrites with restored mtime; inode/device detect replacement.
    """
    directory = str(Path(settings.LD_TABLES).resolve())
    law = settings.LIMB_DARKENING_LAW
    sources = []
    for band in passbands:
        tables = []
        for name in get_relevant_ld_tables("bolometric" if band == "rv_band" else band, metallicity, law):
            path = Path(directory) / name
            try:
                revision = path.stat()
            except FileNotFoundError as exc:
                raise_elisa_limb_darkening_error(f"There is no file like {path}.", exc)
            except OSError as exc:
                raise_elisa_unexpected_error(f"Cannot access limb-darkening table {path}: {exc}", exc)
            tables.append(LDTableIdentity(
                filename=name,
                revision=LDFileRevision(
                    device=revision.st_dev,
                    inode=revision.st_ino,
                    size=revision.st_size,
                    mtime_ns=revision.st_mtime_ns,
                    ctime_ns=revision.st_ctime_ns,
                ),
            ))
        sources.append(LDSourceIdentity(
            directory=directory,
            law=law,
            tables=tuple(tables),
            csv_columns=tuple(settings.LD_LAW_COLS_ORDER[law]),
            domain_columns=tuple(settings.LD_DOMAIN_COLS),
            coefficient_columns=tuple(settings.LD_LAW_CFS_COLUMNS[law]),
        ))
    return tuple(sources)


def _read_ld_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Read selected source columns using the same failure policy as atmospheres."""
    try:
        # noinspection PyArgumentList
        return pd.read_csv(path)[list(columns)]
    except FileNotFoundError as exc:
        raise_elisa_limb_darkening_error(f"There is no file like {path}.", exc)
    except OSError as exc:
        # Permissions, device failures and resource limits are not candidate physics.
        raise_elisa_unexpected_error(f"Cannot read limb-darkening table {path}: {exc}", exc)
    except Exception as exc:  # noqa: BLE001 - fixed table content and columns are source contracts
        # Bad CSV, missing columns and reader failures must abort fitting, just as
        # they do for atmosphere tables. Preserve the failure for diagnosis.
        raise_elisa_unexpected_error(f"Cannot decode limb-darkening table {path}: {exc}", exc)


@lru_cache(maxsize=32)
def get_ld_source_interpolator(source: LDSourceIdentity) -> interpolate.LinearNDInterpolator:
    """Retain complete source interpolators until LRU eviction or explicit clear.

    The 32-entry bound is per process, independent of the 128 evaluated-result
    entries. No surface coordinates or evaluated coefficients are stored here.
    """
    frames = []
    for table in source.tables:
        frame_key = (source.directory, table.filename, table.revision, source.csv_columns)
        if frame_key not in buffer.LD_CFS_TABLES:
            path = Path(source.directory) / table.filename
            # noinspection PyArgumentList
            buffer.LD_CFS_TABLES[frame_key] = _read_ld_csv(path, source.csv_columns)
        frames.append(buffer.LD_CFS_TABLES[frame_key])
    buffer.reduce_buffer(buffer.LD_CFS_TABLES)
    # Preserve ordered concatenation and full-row (not coordinate) deduplication.
    frame = pd.concat(frames, ignore_index=True).drop_duplicates()
    domain = frame[list(source.domain_columns)].to_numpy()
    coefficients = frame[list(source.coefficient_columns)].to_numpy()
    try:
        return interpolate.LinearNDInterpolator(domain, coefficients, fill_value=np.nan, rescale=False)
    except Exception as exc:  # noqa: BLE001 - interpolator construction failures reject candidates by current policy
        # Treat every interpolation failure, including Qhull triangulation errors,
        # as an unusable LD model for now. Future review may separate pathological
        # source dimensions/geometry into unexpected errors; no such split is made here.
        msg = f"Cannot construct limb-darkening interpolator from {source.directory}, tables={source.tables!r}: {exc}"
        return raise_elisa_limb_darkening_error(msg, exc)


def _prepare_ld_interpolation_inputs(
    temperature: NDArray, log_g: NDArray, metallicity: float,
) -> tuple[NDArray, float]:
    """Prepare numerical candidate inputs under the current rejection policy."""
    try:
        t_eff = np.asarray(temperature, dtype=float)
        log_g_arr = np.asarray(log_g, dtype=float)
        log_g_cgs = utils.convert_gravity_acceleration_array(log_g_arr, units="log_cgs")
        return np.column_stack((t_eff, log_g_cgs)), float(metallicity)
    except Exception as exc:  # noqa: BLE001 - candidate input failures are deliberately penalized for now
        # Future review should decide whether type/shape failures need separating
        # into ElisaUnexpectedError. For now, they reject proposals like bad values.
        raise_elisa_limb_darkening_error(f"Cannot prepare limb-darkening interpolation inputs: {exc}", exc)


def _evaluate_ld_query(interpolator: interpolate.LinearNDInterpolator, domain: NDArray) -> NDArray:
    """Evaluate candidate coordinates separately from building the source interpolator."""
    try:
        return np.asarray(interpolator(domain))
    except Exception as exc:  # noqa: BLE001 - includes unsupported candidate coordinate shapes by explicit policy
        # Source loading/construction happens outside this boundary, so its fatal
        # failures retain their meaning. Revisit coordinate shape errors separately.
        raise_elisa_limb_darkening_error(f"Cannot evaluate limb-darkening interpolation inputs: {exc}", exc)


# noinspection PyUnusedLocal
def interpolate_on_ld_grid(
    temperature: NDArray,
    log_g: NDArray,
    metallicity: float,
    passband: Iterable[str] | Mapping[str, str],
    author: str | None = None,
) -> dict[str, NDArray[Float]]:
    """Interpolate limb darkening coefficients on the Van Hamme grid.

    The interpolation is piecewise linear in (T_eff, log_g) using cached
    :class:`scipy.interpolate.LinearNDInterpolator` instances. Metallicity selects
    the same ordered source tables as before. Input *log_g* is expected in log(SI) and is
    converted to log(cgs) prior to interpolation.

    :param temperature: Effective temperature values (triangle-wise).
    :param log_g: Surface gravity values in log(SI) units (triangle-wise).
    :param metallicity: Metallicity value.
    :param passband: Passband names. If a mapping is provided, its keys are used.
    :param author: Table author selector (currently unused).
    :returns: Mapping of passband name to interpolated coefficients array.
    :raises ElisaLimbDarkeningError: If numerical inputs cannot be processed or interpolation yields invalid values.
    """
    del author  # Not implemented yet.

    bands = list(passband.keys()) if isinstance(passband, Mapping) else list(passband)

    uvw_domain, metallicity = _prepare_ld_interpolation_inputs(temperature, log_g, metallicity)

    results: dict[str, NDArray[Float]] = {}
    logger.debug("interpolating limb darkening coefficients")

    for band in bands:
        source = get_ld_source_identities(metallicity, (band,))[0]
        interpolator = get_ld_source_interpolator(source)
        uvw_values_arr = _evaluate_ld_query(interpolator, uvw_domain)

        if not np.all(np.isfinite(uvw_values_arr)):
            invalid_indices = np.flatnonzero(np.any(~np.isfinite(uvw_values_arr), axis=1))
            sample_indices = invalid_indices[:_LD_ERROR_SAMPLE_SIZE]
            table_paths = [str(Path(source.directory) / table.filename) for table in source.tables]
            msg = (
                "Limb darkening interpolation produced numpy.nan/None or infinity.\n"
                f"Passband: {band}; law: {source.law}; metallicity: {metallicity!r}.\n"
                f"Source tables: {table_paths!r}.\n"
                f"Failed interpolation points: {invalid_indices.size} of {len(uvw_domain)}. "
                f"First {_LD_ERROR_SAMPLE_SIZE} failing input rows (zero-based): {sample_indices.tolist()}.\n"
                "Requested values at these rows (t_eff [K], log_g [log10(cm/s^2)]): "
                f"{uvw_domain[sample_indices].tolist()}.\n"
                "Some of the surface parameters (t_eff, log_g, metallicity) are probably out of range.\n"
                "Adjust star parameters or provide custom LD coefficients via Star.limb_darkening_coefficients."
            )
            raise_elisa_limb_darkening_error(msg)

        results[band] = uvw_values_arr

    logger.debug("limb darkening coefficients interpolation finished")
    return results


def limb_darkening_factor(
    normal_vector: NDArray | None = None,
    line_of_sight: NDArray | None = None,
    coefficients: NDArray | None = None,
    limb_darkening_law: str | None = None,
    cos_theta: NDArray | None = None,
) -> NDArray[Float]:
    """Compute limb darkening factor for given surface elements.

    If *cos_theta* is provided, the function will not compute it from *normal_vector*
    and *line_of_sight*.

    Coefficient shapes:

    - Linear and cosine laws: ``(N, 1)`` or ``(N, )``.
    - Logarithmic and square-root laws: ``(N, 2)`` where columns correspond to the
      law parameters in the configured order.

    :param normal_vector: Normal vectors, normalized to length 1.
    :param line_of_sight: Line-of-sight vectors, normalized to length 1.
    :param coefficients: Limb darkening coefficients.
    :param limb_darkening_law: Limb darkening law name.
    :param cos_theta: Precomputed cosine of the angle to the line of sight.
    :returns: Limb darkening factors with shape matching the input elements.
    :raises ElisaUnexpectedError: If required geometry, prepared coefficients or a supported law are missing.
    """
    # Two supported input routes share the same LD formula implementation.
    # Precomputed cosines take precedence; unused vectors may legitimately be None.
    if cos_theta is not None:
        return _limb_darkening_factor_from_cosines(cos_theta, coefficients, limb_darkening_law)

    computed_cosines = _cosines_from_vectors(normal_vector, line_of_sight)
    return _limb_darkening_factor_from_cosines(computed_cosines, coefficients, limb_darkening_law)


def _cosines_from_vectors(normal_vector: NDArray | None, line_of_sight: NDArray | None) -> NDArray:
    """Resolve the vector route, which requires both normals and viewing direction."""
    # Geometry preparation is the caller's responsibility. Missing vectors without
    # precomputed cosines are an internal flow error, not candidate parameter drift.
    if normal_vector is None:
        msg = "Normal vector(s) was not supplied."
        raise_elisa_unexpected_error(msg)
    if line_of_sight is None:
        msg = "Line of sight vector(s) was not supplied."
        raise_elisa_unexpected_error(msg)
    normals = np.asarray(normal_vector, dtype=float)
    direction = np.asarray(line_of_sight, dtype=float)
    return np.sum(normals * direction, axis=-1)


def _limb_darkening_factor_from_cosines(
    cos_theta: NDArray,
    coefficients: NDArray | None,
    limb_darkening_law: str | None,
) -> NDArray[Float]:
    """Evaluate LD factors identically for precomputed and vector-derived cosines."""
    if coefficients is None:
        # The caller must supply prepared coefficients. Interpolation returns an
        # array or raises on unusable candidates; None is an internal flow error.
        msg = "Limb darkening coefficients were not supplied."
        raise_elisa_unexpected_error(msg)
    if limb_darkening_law is None:
        # The caller must provide the fixed, validated law before synthesis.
        # Its absence is an internal contract failure, independent of parameter drift.
        msg = "Limb darkening rule was not supplied choose from: `linear` or `cosine`, `logarithmic`, `square_root`."
        raise_elisa_unexpected_error(msg)

    coeffs = np.asarray(coefficients, dtype=float)

    mu = np.asarray(cos_theta, dtype=float)
    mu = mu[:, np.newaxis] if mu.ndim == 1 else mu
    mu = mu.copy()
    neg = mu <= 0

    if limb_darkening_law in _LD_LAWS_LINEAR:
        mu[neg] = 0.0
        retval = 1.0 - coeffs + coeffs * mu
        retval[neg] = 0.0
    elif limb_darkening_law == "logarithmic":
        mu_for_log = mu.copy()
        mu[neg] = 0.0
        mu_for_log[neg] = 1.0
        retval = 1.0 - coeffs[:, :1] * (1 - mu) - coeffs[:, 1:] * mu * up.log(mu_for_log)
        retval[neg] = 0.0
    elif limb_darkening_law == "square_root":
        mu[neg] = 0.0
        retval = 1.0 - coeffs[:, :1] * (1 - mu) - coeffs[:, 1:] * (1 - up.sqrt(mu))
        retval[neg] = 0.0
    else:
        # Settings validate this fixed choice before synthesis. An unsupported law
        # here means an internal error or reconfiguration during calculation, not
        # candidate parameter drift; it must never become a fitting penalty.
        msg = f"Unsupported limb-darkening law {limb_darkening_law!r} during factor calculation."
        raise_elisa_unexpected_error(msg)

    retval_arr = np.asarray(retval, dtype=float)
    return retval_arr[:, 0] if retval_arr.shape[1] == 1 else retval_arr


def calculate_integrated_limb_darkening_factor(
    limb_darkening_law: str | None = None,
    coefficients: NDArray | None = None,
) -> NDArray[Float]:
    """Compute the integrated limb darkening factor for hemisphere integration.

    This factor is used to convert interpolated radiosity to normal radiance.

    :param limb_darkening_law: Limb darkening law name.
    :param coefficients: Limb darkening coefficients.
    :returns: Integrated limb darkening factor per surface element.
    :raises ElisaUnexpectedError: If prepared coefficients or a supported law are missing.
    """
    if coefficients is None:
        # Bolometric preparation returns an array or fails before integration.
        # Missing coefficients violate the caller's contract, not candidate physics.
        msg = "Limb darkening coefficients were not supplied."
        raise_elisa_unexpected_error(msg)
    if limb_darkening_law is None:
        # Like an unsupported law, a missing law violates the synthesis contract.
        msg = "Limb darkening rule was not supplied choose from: `linear` or `cosine`, `logarithmic`, `square_root`."
        raise_elisa_unexpected_error(msg)

    coeffs = np.asarray(coefficients, dtype=float)

    if limb_darkening_law in _LD_LAWS_LINEAR:
        return const.PI * (1 - coeffs[0, :] / 3)
    if limb_darkening_law == "logarithmic":
        return const.PI * (1 - coeffs[0, :] / 3 + 2 * coeffs[1, :] / 9)
    if limb_darkening_law == "square_root":
        return const.PI * (1 - coeffs[0, :] / 3 - coeffs[1, :] / 5)

    # The configured law is fixed and validated before synthesis, as in the
    # per-element factor calculation. Reaching this branch violates that contract.
    msg = f"Unsupported limb-darkening law {limb_darkening_law!r} during integrated factor calculation."
    return raise_elisa_unexpected_error(msg)


def get_bolometric_ld_coefficients(
    temperature: NDArray,
    log_g: NDArray,
    metallicity: float,
    custom_ld_coefs: Mapping[str, NDArray] | None = None,
) -> NDArray[Float]:
    """Obtain bolometric limb darkening coefficients for each face.

    If *custom_ld_coefs* is provided, it must contain a ``"bolometric"`` entry.
    Otherwise, coefficients are interpolated from the configured tables.

    :param temperature: Effective temperature values (triangle-wise).
    :param log_g: Surface gravity values in log(SI) units (triangle-wise).
    :param metallicity: Metallicity value.
    :param custom_ld_coefs: Optional custom coefficients mapping.
    :returns: Coefficients with shape ``(n_coeffs, n_faces)``.
    :raises ElisaInitialParametersError: If custom coefficients lack the required ``"bolometric"`` entry.
    """
    t_eff = np.asarray(temperature, dtype=float)

    if custom_ld_coefs is not None:
        if isinstance(custom_ld_coefs, CustomLimbDarkeningCoefficients):
            custom_ld_coefs.validate_law()
        if "bolometric" not in custom_ld_coefs:
            # Bolometric coefficients are required by the synthesis/reflection
            # flow. An incomplete fixed custom map is not a penalizable candidate.
            msg = (
                "Please add `bolometric` limb-darkening coefficients to your custom set of limb-darkening coefficients."
            )
            raise_elisa_initial_parameters_error(msg)
        bol = np.asarray(custom_ld_coefs["bolometric"], dtype=float)
        coeffs = np.tile(bol, (t_eff.shape[0], 1))
    else:
        coeffs = interpolate_on_ld_grid(
            temperature=t_eff,
            log_g=log_g,
            metallicity=metallicity,
            passband=["bolometric"],
        )["bolometric"]

    coeffs_arr = np.asarray(coeffs, dtype=float)
    return coeffs_arr.T
