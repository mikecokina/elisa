from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from scipy import interpolate

from elisa import const, settings, utils
from elisa import umpy as up
from elisa.base.error import LimbDarkeningError
from elisa.buffer import buffer
from elisa.logger import getLogger

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from elisa.types import Float

logger = getLogger(__name__)

_LD_LAWS_LINEAR: Final[set[str]] = {"linear", "cosine"}


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


def get_ld_table(passband: str, metallicity: float, law: str | None = None) -> pd.DataFrame:
    """Load a Van Hamme limb darkening table from the configured tables' directory.

    :param passband: Passband identifier.
    :param metallicity: Metallicity value.
    :param law: Limb darkening law name. If not provided, the configured default is used.
    :returns: Table content as a :class:`pandas.DataFrame`.
    :raises FileNotFoundError: If the expected CSV file is not present.
    """
    resolved_law = law if not utils.is_empty(law) else settings.LIMB_DARKENING_LAW
    filename = get_ld_table_filename(passband, metallicity, law=resolved_law)
    path = Path(settings.LD_TABLES) / filename
    if not path.is_file():
        msg = f"There is no file like {path}."
        raise FileNotFoundError(msg)
    return pd.read_csv(path)


def get_ld_table_by_name(fname: str) -> pd.DataFrame:
    """Load a Van Hamme limb darkening table by filename from the configured tables' directory.

    :param fname: Filename of the CSV table (no directory).
    :returns: Table content as a :class:`pandas.DataFrame`.
    :raises FileNotFoundError: If the file is not present in the configured directory.
    """
    logger.debug("accessing limb darkening file %s", fname)
    path = Path(settings.LD_TABLES) / fname
    if not path.is_file():
        msg = f"There is no file like {path}."
        raise FileNotFoundError(msg)
    return pd.read_csv(path)


def get_relevant_ld_tables(passband: str, metallicity: float, law: str | None = None) -> list[str]:
    """Get filenames of tables surrounding the requested metallicity.

    The surrounding metallicities are determined from :data:`elisa.const.METALLICITY_LIST_LD`.

    :param passband: Passband identifier.
    :param metallicity: Metallicity value.
    :param law: Limb darkening law name.
    :returns: List of CSV filenames to use for interpolation.
    """
    resolved_law = law if not utils.is_empty(law) else settings.LIMB_DARKENING_LAW
    surrounded = utils.find_surrounded(const.METALLICITY_LIST_LD, metallicity)
    return [get_ld_table_filename(passband, m, resolved_law) for m in surrounded]


# noinspection PyUnusedLocal
def interpolate_on_ld_grid(
        temperature: ArrayLike,
        log_g: ArrayLike,
        metallicity: float,
        passband: Iterable[str] | Mapping[str, str],
        author: str | None = None,
) -> dict[str, NDArray[Float]]:
    """Interpolate limb darkening coefficients on the Van Hamme grid.

    The interpolation is performed in the (T_eff, log_g, metallicity) domain using
    :func:`scipy.interpolate.griddata`. Input *log_g* is expected in log(SI) and is
    converted to log(cgs) prior to interpolation.

    :param temperature: Effective temperature values (triangle-wise).
    :param log_g: Surface gravity values in log(SI) units (triangle-wise).
    :param metallicity: Metallicity value.
    :param passband: Passband names. If a mapping is provided, its keys are used.
    :param author: Table author selector (currently unused).
    :returns: Mapping of passband name to interpolated coefficients array.
    :raises LimbDarkeningError: If interpolation yields invalid values.
    """
    del author  # Not implemented yet.

    bands = list(passband.keys()) if isinstance(passband, Mapping) else list(passband)

    t_eff = np.asarray(temperature, dtype=float)
    log_g_arr = np.asarray(log_g, dtype=float)

    # Convert logg from log(SI) to log(cgs).
    log_g_cgs = utils.convert_gravity_acceleration_array(log_g_arr, units="log_cgs")

    results: dict[str, NDArray[Float]] = {}
    logger.debug("interpolating limb darkening coefficients")

    for band in bands:
        interp_band = "bolometric" if band == "rv_band" else band

        relevant_tables = get_relevant_ld_tables(
            passband=interp_band,
            metallicity=metallicity,
            law=settings.LIMB_DARKENING_LAW,
        )

        csv_columns = settings.LD_LAW_COLS_ORDER[settings.LIMB_DARKENING_LAW]
        frames: list[pd.DataFrame] = []

        for table in relevant_tables:
            if table in buffer.LD_CFS_TABLES:
                df_tbl = buffer.LD_CFS_TABLES[table]
            else:
                df_tbl = get_ld_table_by_name(table)[csv_columns]
                buffer.LD_CFS_TABLES[table] = df_tbl
            frames.append(df_tbl)

        buffer.reduce_buffer(buffer.LD_CFS_TABLES)

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=csv_columns)
        df = df.drop_duplicates()

        xyz_domain = df[settings.LD_DOMAIN_COLS].to_numpy()
        xyz_values = df[settings.LD_LAW_CFS_COLUMNS[settings.LIMB_DARKENING_LAW]].to_numpy()

        uvw_domain = np.column_stack((t_eff, log_g_cgs))
        uvw_values = interpolate.griddata(xyz_domain, xyz_values, uvw_domain, method="linear")

        uvw_values_arr = np.asarray(uvw_values)

        if np.any(up.isnan(uvw_values_arr)):
            msg = (
                "Limb darkening interpolation produced numpy.nan/None.\n"
                "Some of the surface parameters (t_eff, log_g, metallicity) are probably out of range.\n"
                "Adjust star parameters or provide custom LD coefficients via Star.limb_darkening_coefficients."
            )
            raise LimbDarkeningError(msg)

        results[band] = uvw_values_arr

    logger.debug("limb darkening coefficients interpolation finished")
    return results


def limb_darkening_factor(
        normal_vector: ArrayLike | None = None,
        line_of_sight: ArrayLike | None = None,
        coefficients: ArrayLike | None = None,
        limb_darkening_law: str | None = None,
        cos_theta: ArrayLike | None = None,
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
    :raises ValueError: If required vectors are missing and *cos_theta* is not provided.
    :raises LimbDarkeningError: If coefficients or law are missing or invalid.
    """
    if normal_vector is None and cos_theta is None:
        msg = "Normal vector(s) was not supplied."
        raise ValueError(msg)
    if line_of_sight is None and cos_theta is None:
        msg = "Line of sight vector(s) was not supplied."
        raise ValueError(msg)
    if coefficients is None:
        msg = "Limb darkening coefficients were not supplied."
        raise LimbDarkeningError(msg)
    if limb_darkening_law is None:
        msg = (
            "Limb darkening rule was not supplied choose from: "
            "`linear` or `cosine`, `logarithmic`, `square_root`."
        )
        raise LimbDarkeningError(msg)

    coeffs = np.asarray(coefficients, dtype=float)

    if cos_theta is None:
        n = np.asarray(normal_vector, dtype=float)
        los = np.asarray(line_of_sight, dtype=float)
        mu = np.sum(n * los, axis=-1)
    else:
        mu = np.asarray(cos_theta, dtype=float)

    mu = mu[:, np.newaxis] if mu.ndim == 1 else mu.copy()

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
        msg = "Invalid limb darkening."
        raise LimbDarkeningError(msg)

    retval_arr = np.asarray(retval, dtype=float)
    return retval_arr[:, 0] if retval_arr.shape[1] == 1 else retval_arr


def calculate_integrated_limb_darkening_factor(
        limb_darkening_law: str | None = None,
        coefficients: ArrayLike | None = None,
) -> NDArray[Float]:
    """Compute the integrated limb darkening factor for hemisphere integration.

    This factor is used to convert interpolated radiosity to normal radiance.

    :param limb_darkening_law: Limb darkening law name.
    :param coefficients: Limb darkening coefficients.
    :returns: Integrated limb darkening factor per surface element.
    :raises LimbDarkeningError: If coefficients or law are missing.
    """
    if coefficients is None:
        msg = "Limb darkening coefficients were not supplied."
        raise LimbDarkeningError(msg)
    if limb_darkening_law is None:
        msg = (
            "Limb darkening rule was not supplied choose from: "
            "`linear` or `cosine`, `logarithmic`, `square_root`."
        )
        raise LimbDarkeningError(msg)

    coeffs = np.asarray(coefficients, dtype=float)

    if limb_darkening_law in _LD_LAWS_LINEAR:
        return const.PI * (1 - coeffs[0, :] / 3)
    if limb_darkening_law == "logarithmic":
        return const.PI * (1 - coeffs[0, :] / 3 + 2 * coeffs[1, :] / 9)
    if limb_darkening_law == "square_root":
        return const.PI * (1 - coeffs[0, :] / 3 - coeffs[1, :] / 5)

    msg = "Invalid limb darkening."
    raise LimbDarkeningError(msg)


def get_bolometric_ld_coefficients(
        temperature: ArrayLike,
        log_g: ArrayLike,
        metallicity: float,
        custom_ld_coefs: Mapping[str, ArrayLike] | None = None,
) -> NDArray[Float]:
    """Obtain bolometric limb darkening coefficients for each face.

    If *custom_ld_coefs* is provided, it must contain a ``"bolometric"`` entry.
    Otherwise, coefficients are interpolated from the configured tables.

    :param temperature: Effective temperature values (triangle-wise).
    :param log_g: Surface gravity values in log(SI) units (triangle-wise).
    :param metallicity: Metallicity value.
    :param custom_ld_coefs: Optional custom coefficients mapping.
    :returns: Coefficients with shape ``(n_coeffs, n_faces)``.
    :raises ValueError: If custom coefficients are provided without a ``"bolometric"`` entry.
    """
    t_eff = np.asarray(temperature, dtype=float)

    if custom_ld_coefs is not None:
        if "bolometric" not in custom_ld_coefs:
            msg = (
                "Please add `bolometric` limb-darkening coefficients to your custom set "
                "of limb-darkening coefficients."
            )
            raise ValueError(msg)
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
