"""Core computation logic for LC fitting - no Gradio dependency.

Translates flat Gradio value dicts and passband row data into ELISa
analytics objects, runs LSQRT or MCMC optimisation, and returns plain
Python / matplotlib objects for the UI layer.

The ``semi_major_axis`` parameter supports three modes via the
``system_semi_major_axis_mode`` key:

- ``"free"`` - fitted freely with ``min`` / ``max`` bounds.
- ``"fixed"`` - held at its initial value.
- ``"constrained"`` - value expression from ``system_semi_major_axis_constraint``.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt
import pandas as pd

from elisa import units as u
from elisa.analytics import LCBinaryAnalyticsTask, LCData
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.ui.shared.logging_config import fit_logging

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.figure import Figure

    from elisa.types import Float

# ---------------------------------------------------------------------------
# Unit maps
# ---------------------------------------------------------------------------

_X_UNIT_MAP: dict[str, object] = {
    "Julian days (JD)": u.d,
    "Phases (dimensionless)": u.dimensionless_unscaled,
}

_Y_UNIT_MAP: dict[str, object] = {
    "Flux (dimensionless)": u.dimensionless_unscaled,
    "Magnitude (mag)": u.mag,
}

# Unit strings injected into the x0 dict for each parameter.
# None means dimensionless (unit key omitted).
_SYSTEM_PARAM_UNITS: dict[str, str | None] = {
    "inclination": "deg",
    "eccentricity": None,
    "argument_of_periastron": "deg",
    "mass_ratio": None,
    "period": "d",
    "primary_minimum_time": "d",
    "additional_light": None,
    "phase_shift": None,
    "semi_major_axis": "solRad",
}

_COMPONENT_PARAM_UNITS: dict[str, str | None] = {
    "mass": "solMass",
    "t_eff": "K",
    "surface_potential": None,
    "gravity_darkening": None,
    "albedo": None,
}


# ---------------------------------------------------------------------------
# Passband row typing
# ---------------------------------------------------------------------------


class LCRowData(TypedDict):
    """Flat data for one passband row collected from the UI.

    :cvar passband: Photometric passband name.
    :cvar file_path: Absolute path to the uploaded data file, or ``None``.
    :cvar y_unit: Y-axis unit choice string from the UI dropdown.
    :cvar reference_magnitude: Reference magnitude for mag-to-flux conversion,
        or ``None`` when y-unit is flux.
    """

    passband: str
    file_path: str | None
    y_unit: str
    reference_magnitude: Float | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _capture_figure() -> Iterator[list[Figure]]:
    """Replace ``plt.show`` temporarily to capture the figure it would display.

    Yields a list that will contain the captured ``Figure`` after the
    ``with`` block exits.  Only the figure alive at the moment
    ``plt.show()`` is called is captured; subsequent calls overwrite it.

    :yields: A list that will hold the captured ``matplotlib.figure.Figure``
        after the context exits.
    :rtype: list[matplotlib.figure.Figure]
    """
    captured: list[Figure] = []
    original_show = plt.show

    def _mock_show(*_args: object, **_kwargs: object) -> None:
        fig = plt.gcf()
        if fig is not None:
            captured.append(fig)

    plt.show = _mock_show  # type: ignore[assignment]
    try:
        yield captured
    finally:
        plt.show = original_show  # type: ignore[assignment]


def _opt_float(value: object) -> Float | None:
    """Convert *value* to float or return ``None`` for empty/None inputs.

    :param value: Raw value from a Gradio Number component.
    :type value: object
    :returns: Parsed float or ``None``.
    :rtype: Float | None
    """
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _result_temp_path(prefix: str) -> Path:
    """Return a timestamped temp-file path for a fit result JSON.

    :param prefix: Short label embedded in the filename (e.g. ``"lsqrt"`` or ``"mcmc"``).
    :type prefix: str
    :returns: Path inside the system temp directory.
    :rtype: Path
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    return Path(tempfile.gettempdir()) / f"elisa_lc_{prefix}_{ts}.json"


# ---------------------------------------------------------------------------
# Public API - data loading
# ---------------------------------------------------------------------------


def load_lc_data(
    file_path: str,
    passband: str,
    x_unit_str: str,
    y_unit_str: str,
    reference_magnitude: Float | None,
) -> LCData:
    """Load a single LC data file into an :class:`~elisa.analytics.LCData` object.

    :param file_path: Absolute path to the whitespace-delimited data file.
        Columns: ``x`` (time/phase) | ``flux or mag`` | ``err`` (optional).
    :type file_path: str
    :param passband: Photometric passband name (e.g. ``"Generic.Bessell.V"``).
    :type passband: str
    :param x_unit_str: X-axis unit label from the UI dropdown.
    :type x_unit_str: str
    :param y_unit_str: Y-axis unit label from the UI dropdown.
    :type y_unit_str: str
    :param reference_magnitude: Reference magnitude for magnitude-to-flux
        conversion.  Required when *y_unit_str* is ``"Magnitude (mag)"``;
        ignored for flux data.
    :type reference_magnitude: Float | None
    :returns: Loaded :class:`~elisa.analytics.LCData`.
    :rtype: LCData
    """
    x_unit = _X_UNIT_MAP.get(x_unit_str, u.d)
    y_unit = _Y_UNIT_MAP.get(y_unit_str, u.dimensionless_unscaled)

    kwargs: dict[str, object] = {"passband": passband}
    if y_unit is u.mag and reference_magnitude is not None:
        kwargs["reference_magnitude"] = float(reference_magnitude)

    return LCData.load_from_file(file_path, x_unit=x_unit, y_unit=y_unit, **kwargs)


# ---------------------------------------------------------------------------
# Public API - x0 construction
# ---------------------------------------------------------------------------


def _build_regular_param_entry(
    param_values: dict[str, object],
    prefix: str,
    unit_str: str | None,
) -> dict[str, object]:
    """Build one regular parameter entry for the x0 dict.

    Supports three parameter modes via the ``{prefix}_mode`` key:

    - ``"free"`` - fitted freely with ``min`` / ``max`` bounds.
    - ``"fixed"`` - held at its initial value.
    - ``"constrained"`` - value determined by expression from ``{prefix}_constraint``.

    :param param_values: Flat UI parameter dict.
    :type param_values: dict[str, object]
    :param prefix: Full key prefix, e.g. ``"system_inclination"`` or
        ``"primary_t_eff"``.
    :type prefix: str
    :param unit_str: Unit string to inject, or ``None`` for dimensionless.
    :type unit_str: str | None
    :returns: Per-parameter entry dict ready for ``BinaryInitialParameters``.
    :rtype: dict[str, object]
    """
    mode = str(param_values.get(f"{prefix}_mode", "free"))
    value = _opt_float(param_values.get(f"{prefix}_value"))
    entry: dict[str, object] = {"value": value}

    if mode == "constrained":
        entry["constraint"] = str(param_values.get(f"{prefix}_constraint", ""))
    elif mode == "fixed":
        entry["fixed"] = True
    else:  # free
        entry["fixed"] = False
        entry["min"] = _opt_float(param_values.get(f"{prefix}_min"))
        entry["max"] = _opt_float(param_values.get(f"{prefix}_max"))

    if unit_str is not None:
        entry["unit"] = unit_str
    return entry


def _build_sma_entry(param_values: dict[str, object]) -> dict[str, object]:
    """Build the ``semi_major_axis`` parameter entry for the x0 dict.

    Reads ``system_semi_major_axis_mode`` and constructs either a free,
    fixed, or constrained entry accordingly.

    :param param_values: Flat UI parameter dict.
    :type param_values: dict[str, object]
    :returns: ``semi_major_axis`` entry dict.
    :rtype: dict[str, object]
    """
    mode = str(param_values.get("system_semi_major_axis_mode", "constrained"))
    entry: dict[str, object] = {
        "value": _opt_float(param_values.get("system_semi_major_axis_value")),
        "unit": "solRad",
    }
    if mode == "constrained":
        entry["constraint"] = str(
            param_values.get("system_semi_major_axis_constraint", ""),
        )
    elif mode == "fixed":
        entry["fixed"] = True
    else:  # free
        entry["fixed"] = False
        entry["min"] = _opt_float(param_values.get("system_semi_major_axis_min"))
        entry["max"] = _opt_float(param_values.get("system_semi_major_axis_max"))
    return entry


def _build_component_params(
    param_values: dict[str, object],
    section: str,
    base_params: tuple[str, ...],
) -> dict[str, dict]:
    """Build one component section (primary or secondary) for the x0 dict.

    Iterates over *base_params* (community component params without mass),
    then appends the optional ``mass`` entry when present in *param_values*
    (standard approach).

    :param param_values: Flat UI parameter dict.
    :type param_values: dict[str, object]
    :param section: Component section prefix (``"primary"`` or ``"secondary"``).
    :type section: str
    :param base_params: Ordered names of the common component parameters.
    :type base_params: tuple[str, ...]
    :returns: Per-component parameter dict ready for ``BinaryInitialParameters``.
    :rtype: dict[str, dict]
    """
    result: dict[str, dict] = {
        name: _build_regular_param_entry(
            param_values, f"{section}_{name}", _COMPONENT_PARAM_UNITS.get(name),
        )
        for name in base_params
    }
    # Optional: mass present only in the standard approach.
    if f"{section}_mass_value" in param_values:
        result["mass"] = _build_regular_param_entry(
            param_values, f"{section}_mass", _COMPONENT_PARAM_UNITS.get("mass"),
        )
    return result


def _build_nuisance_entry(param_values: dict[str, object]) -> dict[str, object]:
    """Build the ``ln_f`` nuisance parameter entry for the x0 dict.

    :param param_values: Flat UI parameter dict.
    :type param_values: dict[str, object]
    :returns: ``ln_f`` entry dict.
    :rtype: dict[str, object]
    """
    ln_f_mode = str(param_values.get("nuisance_ln_f_mode", "fixed"))
    entry: dict[str, object] = {
        "value": _opt_float(param_values.get("nuisance_ln_f_value")),
    }
    if ln_f_mode == "constrained":
        entry["constraint"] = str(param_values.get("nuisance_ln_f_constraint", ""))
    elif ln_f_mode == "fixed":
        entry["fixed"] = True
    else:  # free
        entry["fixed"] = False
        ln_f_min = _opt_float(param_values.get("nuisance_ln_f_min"))
        ln_f_max = _opt_float(param_values.get("nuisance_ln_f_max"))
        if ln_f_min is not None:
            entry["min"] = ln_f_min
        if ln_f_max is not None:
            entry["max"] = ln_f_max
    return entry


def build_x0(param_values: dict[str, object], *, include_nuisance: bool = False) -> dict:
    """Build the ``BinaryInitialParameters`` input dict from flat UI values.

    Constructs the nested
    ``{"system": {...}, "primary": {...}, "secondary": {...}}`` structure
    expected by :class:`~elisa.analytics.LCBinaryAnalyticsTask`. Optionally
    includes nuisance parameters required for MCMC fitting.

    Parameters are built from the flat UI dict using the ``{section}_{name}_mode``
    key to determine the mode (free, fixed, or constrained):

    - ``"free"`` - ``{"value": ..., "fixed": False, "min": ..., "max": ..., ...}``
    - ``"fixed"`` - ``{"value": ..., "fixed": True, ...}``
    - ``"constrained"`` - ``{"value": ..., "constraint": "...", ...}``

    :param param_values: Flat dict keyed by
        :data:`~elisa.ui.tabs.lc_fitting.components.param_inputs.FIELD_ORDER`
        entries.
    :type param_values: dict[str, object]
    :param include_nuisance: Whether to include nuisance parameters (used only for MCMC).
        Defaults to False (LSQRT fitting).
    :type include_nuisance: bool
    :returns: Nested initial-parameter dict suitable for
        ``LCBinaryAnalyticsTask.fit(x0=...)``.
    :rtype: dict
    """
    from elisa.ui.tabs.lc_fitting.components.param_inputs import (  # noqa: PLC0415
        COMPONENT_PARAMS,
        SYSTEM_REGULAR_PARAMS,
    )

    # Build all regular system parameters (common to both standard and community)
    system: dict[str, dict] = {
        name: _build_regular_param_entry(
            param_values, f"system_{name}", _SYSTEM_PARAM_UNITS.get(name),
        )
        for name in SYSTEM_REGULAR_PARAMS
    }

    # Add optional system parameters if present
    # (semi_major_axis + mass_ratio: community; semi_major_axis: both)
    for optional_param in ("semi_major_axis", "mass_ratio"):
        if f"system_{optional_param}_value" in param_values:
            if optional_param == "semi_major_axis":
                system[optional_param] = _build_sma_entry(param_values)
            else:
                system[optional_param] = _build_regular_param_entry(
                    param_values, f"system_{optional_param}", _SYSTEM_PARAM_UNITS.get(optional_param),
                )

    result: dict[str, dict] = {
        "system": system,
        "primary": _build_component_params(param_values, "primary", COMPONENT_PARAMS),
        "secondary": _build_component_params(param_values, "secondary", COMPONENT_PARAMS),
    }

    if include_nuisance:
        result["nuisance"] = {"ln_f": _build_nuisance_entry(param_values)}

    return result


# ---------------------------------------------------------------------------
# Public API - result helpers
# ---------------------------------------------------------------------------


def result_to_dataframe(flat_result: dict) -> pd.DataFrame:
    """Convert a flat fit result dict into a display DataFrame.

    :param flat_result: Flat result dict as returned by
        ``task.fit_cls.flat_result`` (``"section@param"`` keys).
    :type flat_result: dict
    :returns: DataFrame with columns *Parameter*, *Value*, *-1s*, *+1s*,
        *Unit*, *Status*.
    :rtype: pandas.DataFrame
    """
    rows = []
    for key, meta in flat_result.items():
        value = meta.get("value")
        ci = meta.get("confidence_interval") or {}
        lo = ci.get("min")
        hi = ci.get("max")
        unit = meta.get("unit") or "-"
        fixed = meta.get("fixed", False)
        constraint = meta.get("constraint")
        if constraint:
            status = f"constrained: {constraint}"
        elif fixed:
            status = "fixed"
        else:
            status = "free"

        rows.append(
            {
                "Parameter": key,
                "Value": f"{value:.6g}" if value is not None else "-",
                "-1s": f"{lo:.6g}" if lo is not None else "-",
                "+1s": f"{hi:.6g}" if hi is not None else "-",
                "Unit": unit,
                "Status": status,
            },
        )
    return pd.DataFrame(rows)


def extract_values_for_transfer(result: dict) -> dict[str, object]:
    """Extract fitted parameter values from a result dict for form population.

    Maps the nested result structure (``system``, ``primary``, ``secondary``,
    ``nuisance`` sections) to the flat
    ``"{section}_{name}_value"`` keys used by the parameter form.

    :param result: Nested fit result dict as returned by
        :class:`~elisa.analytics.LCBinaryAnalyticsTask`.
    :type result: dict
    :returns: Flat dict with ``"{section}_{name}_value"`` keys.
    :rtype: dict[str, object]
    """
    out: dict[str, object] = {}
    for section in ("system", "primary", "secondary"):
        for name, meta in (result.get(section) or {}).items():
            if isinstance(meta, dict) and "value" in meta:
                out[f"{section}_{name}_value"] = meta["value"]
    for name, meta in (result.get("nuisance") or {}).items():
        if isinstance(meta, dict) and "value" in meta:
            out[f"nuisance_{name}_value"] = meta["value"]
    return out


def _param_meta_to_flat(
    section: str,
    name: str,
    meta: dict,
) -> dict[str, object]:
    """Convert one parameter's result metadata to flat form entries.

    Handles the special ``semi_major_axis`` case where a ``"constraint"``
    key determines the mode rather than ``"fixed"``.

    :param section: Section name (``"system"``, ``"primary"``, etc.).
    :type section: str
    :param name: Parameter name.
    :type name: str
    :param meta: Per-parameter metadata dict from the JSON result.
    :type meta: dict
    :returns: Flat dict entries for this parameter.
    :rtype: dict[str, object]
    """
    value = meta.get("value")
    fixed = bool(meta.get("fixed", False))
    constraint = meta.get("constraint")
    min_val = meta.get("min")
    max_val = meta.get("max")

    # Always use the full "{section}_{name}" prefix so keys match the UI
    # component dict (e.g. "nuisance_ln_f_value", not "ln_f_value").
    prefix = f"{section}_{name}"

    out: dict[str, object] = {}
    if value is not None:
        out[f"{prefix}_value"] = value

    # Derive mode uniformly for all params.
    if constraint:
        out[f"{prefix}_mode"] = "constrained"
        out[f"{prefix}_constraint"] = constraint
    elif fixed:
        out[f"{prefix}_mode"] = "fixed"
    else:
        out[f"{prefix}_mode"] = "free"

    if min_val is not None:
        out[f"{prefix}_min"] = min_val
    if max_val is not None:
        out[f"{prefix}_max"] = max_val

    return out


def detect_approach_from_json(path: str) -> str:
    """Detect whether a saved LC result JSON uses Community or Standard approach.

    Uses :func:`~elisa.binary_system.utils.resolve_json_kind` to inspect
    the ``system``, ``primary``, and ``secondary`` sections and determine
    whether the file was produced by a Community fit (``mass_ratio`` +
    ``semi_major_axis``) or a Standard fit (individual component masses).

    :param path: Absolute path to the JSON file produced by ``save_result()``.
    :type path: str
    :returns: ``"Community"`` or ``"Standard"``.
    :rtype: str
    :raises ValueError: If the file cannot be read or the approach cannot be determined.
    """
    from elisa.binary_system.utils import resolve_json_kind  # noqa: PLC0415

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        msg = f"Cannot read result file: {exc}"
        raise ValueError(msg) from exc

    # resolve_json_kind expects plain values, not the fitting meta-dicts.
    # Extract just the first-level keys per section so it can check presence.
    def _extract_keys(section_data: object) -> dict:
        if not isinstance(section_data, dict):
            return {}
        return {name: meta.get("value") for name, meta in section_data.items() if isinstance(meta, dict)}

    plain = {
        "system": _extract_keys(data.get("system")),
        "primary": _extract_keys(data.get("primary")),
        "secondary": _extract_keys(data.get("secondary")),
    }

    try:
        kind = resolve_json_kind(plain)
    except LookupError as exc:
        msg = f"Cannot determine fitting approach from JSON: {exc}"
        raise ValueError(msg) from exc

    return "Community" if kind == "community" else "Standard"


def load_params_from_json(path: str) -> dict[str, object]:
    """Load parameter values, bounds, and modes from a saved LC result JSON.

    Parses the ``system``, ``primary``, ``secondary``, and ``nuisance``
    sections written by
    :meth:`~elisa.analytics.LCBinaryAnalyticsTask.save_result` and returns
    a flat dict compatible with the LC parameter form.

    The ``semi_major_axis`` entry is analysed: if it contains a
    ``"constraint"`` key the mode is set to ``"constrained"``; if
    ``"fixed": True`` then ``"fixed"``; otherwise ``"free"``.

    :param path: Absolute path to the JSON file produced by ``save_result()``.
    :type path: str
    :returns: Flat dict with ``"{section}_{name}_{sub}"`` keys.
    :rtype: dict[str, object]
    :raises ValueError: If the file cannot be read or lacks a ``"system"`` key.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        msg = f"Cannot read result file: {exc}"
        raise ValueError(msg) from exc

    if "system" not in data:
        msg = "File does not look like an ELISa LC result - missing 'system' key."
        raise ValueError(msg)

    out: dict[str, object] = {}
    for section in ("system", "primary", "secondary", "nuisance"):
        for name, meta in (data.get(section) or {}).items():
            if isinstance(meta, dict):
                out.update(_param_meta_to_flat(section, name, meta))
    return out


# ---------------------------------------------------------------------------
# Public API - fit runners
# ---------------------------------------------------------------------------


def run_lsqrt(
    lc_rows: list[LCRowData],
    x_unit_str: str,
    param_values: dict[str, object],
    morphology: str,
) -> tuple[dict, Figure, pd.DataFrame, str]:
    """Run a least-squares LC fit and return all displayable artefacts.

    Loads :class:`~elisa.analytics.LCData` objects from the supplied row
    descriptors, builds the initial-parameter structure, runs the LSQRT
    optimisation via :class:`~elisa.analytics.LCBinaryAnalyticsTask`, and
    returns the fitted result together with a model plot, a results table,
    and a path to the saved JSON result.

    :param lc_rows: List of passband row dicts (only rows with a non-``None``
        ``file_path`` and non-empty ``passband`` are loaded).
    :type lc_rows: list[LCRowData]
    :param x_unit_str: X-axis unit label from the UI dropdown.
    :type x_unit_str: str
    :param param_values: Flat parameter dict from the Gradio form.
    :type param_values: dict[str, object]
    :param morphology: Expected binary morphology (``"detached"`` or
        ``"over-contact"``).
    :type morphology: str
    :returns: Tuple of ``(result_dict, model_figure, results_dataframe, json_path)``.
    :rtype: tuple[dict, matplotlib.figure.Figure, pandas.DataFrame, str]
    :raises ValueError: If no valid LC data rows are provided.
    """
    data: dict[str, LCData] = {}
    for row in lc_rows:
        passband = row.get("passband") or ""
        file_path = row.get("file_path")
        if not passband or not file_path:
            continue
        lc = load_lc_data(
            file_path,
            passband,
            x_unit_str,
            row.get("y_unit", "Flux (dimensionless)"),
            row.get("reference_magnitude"),
        )
        data[passband] = lc

    if not data:
        msg = "At least one light curve with a valid data file must be provided."
        raise ValueError(msg)

    x0_dict = build_x0(param_values, include_nuisance=False)
    x0 = BinaryInitialParameters(**x0_dict)

    plt.close("all")
    # noinspection PyTypeChecker
    with fit_logging():
        task = LCBinaryAnalyticsTask(
            data=data, method="least_squares", expected_morphology=morphology,
        )
        result = task.fit(x0=x0)

    model_fig: Figure = task.plot.model(return_figure_instance=True)
    df = result_to_dataframe(task.fit_cls.flat_result)

    json_path = _result_temp_path("lsqrt")
    task.save_result(str(json_path))

    return result, model_fig, df, str(json_path)


def run_mcmc(
    lc_rows: list[LCRowData],
    x_unit_str: str,
    param_values: dict[str, object],
    morphology: str,
    nwalkers: int = 50,
    nsteps: int = 500,
    burn_in: int = 50,
    fit_id: str = "mcmc_lc_fit",
    *,
    save: bool = True,
    progress: bool = True,
) -> tuple[dict, Figure, Figure | None, Figure | None, pd.DataFrame, str]:
    """Run an MCMC LC fit and return all displayable artifacts.

    Loads LC data, constructs parameters, runs the MCMC sampler, and captures
    the model plot, corner plot, and traces plot as ``Figure`` instances by
    temporarily intercepting ``plt.show()``.

    :param lc_rows: List of passband row dicts.
    :type lc_rows: list[LCRowData]
    :param x_unit_str: X-axis unit label from the UI dropdown.
    :type x_unit_str: str
    :param param_values: Flat parameter dict from the Gradio form.
    :type param_values: dict[str, object]
    :param morphology: Expected binary morphology (``"detached"`` or
        ``"over-contact"``).
    :type morphology: str
    :param nwalkers: Number of MCMC walkers.
    :type nwalkers: int
    :param nsteps: Number of MCMC steps per walker.
    :type nsteps: int
    :param burn_in: Number of burn-in steps to discard.
    :type burn_in: int
    :param fit_id: Chain file identifier used when *save* is ``True``.
    :type fit_id: str
    :param save: Whether to save the chain to disk.
    :type save: bool
    :param progress: Whether to show progress bar during MCMC sampling.
    :type progress: bool
    :returns: Tuple of
        ``(result_dict, model_figure, corner_figure, traces_figure,
        results_dataframe, json_path)``.
    :rtype: tuple[dict, Figure, Figure | None, Figure | None, pandas.DataFrame, str]
    :raises ValueError: If no valid LC data rows are provided.
    """
    data: dict[str, LCData] = {}
    for row in lc_rows:
        passband = row.get("passband") or ""
        file_path = row.get("file_path")
        if not passband or not file_path:
            continue
        lc = load_lc_data(
            file_path,
            passband,
            x_unit_str,
            row.get("y_unit", "Flux (dimensionless)"),
            row.get("reference_magnitude"),
        )
        data[passband] = lc

    if not data:
        msg = "At least one light curve with a valid data file must be provided."
        raise ValueError(msg)

    x0_dict = build_x0(param_values, include_nuisance=True)
    x0 = BinaryInitialParameters(**x0_dict)

    plt.close("all")
    # noinspection PyTypeChecker
    with fit_logging():
        task = LCBinaryAnalyticsTask(
            data=data, method="mcmc", expected_morphology=morphology,
        )
        result = task.fit(
            x0=x0,
            nwalkers=nwalkers,
            nsteps=nsteps,
            burn_in=burn_in,
            save=save,
            fit_id=fit_id,
            progress=progress,
        )

    model_fig: Figure = task.plot.model(return_figure_instance=True)

    with _capture_figure() as corner_captured:
        task.plot.corner(truths=True)
    corner_fig: Figure | None = corner_captured[0] if corner_captured else None

    with _capture_figure() as traces_captured:
        task.plot.traces(truths=True)
    traces_fig: Figure | None = traces_captured[0] if traces_captured else None

    df = result_to_dataframe(task.fit_cls.flat_result)

    json_path = _result_temp_path("mcmc")
    task.save_result(str(json_path))

    return result, model_fig, corner_fig, traces_fig, df, str(json_path)
