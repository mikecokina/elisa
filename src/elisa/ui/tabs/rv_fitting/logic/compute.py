"""Core computation logic for RV fitting - no Gradio dependency.

Translates flat Gradio value dictionaries into ELISa analytics objects,
runs the LSQRT or MCMC optimisation, and returns plain Python / matplotlib
objects that the UI layer can display.

The module-level :func:`_capture_figure` context manager intercepts
``plt.show()`` so that MCMC diagnostic plots (corner, traces) which call
``plt.show()`` internally can be captured as ``Figure`` objects suitable
for ``gr.Plot``.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from elisa import units as u
from elisa.analytics import RVBinaryAnalyticsTask, RVData

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.figure import Figure

    from elisa.types import Float

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Maps UI unit strings to astropy units used when loading RVData.
_X_UNIT_MAP: dict[str, object] = {
    "Julian days (JD)": u.d,
    "Phases (dimensionless)": u.dimensionless_unscaled,
}

# Maps param name to the unit string accepted by BinaryInitialParameters.
# None means dimensionless (no unit key in the dict).
_PARAM_UNITS: dict[str, str | None] = {
    "eccentricity": None,
    "asini": "solRad",
    "mass_ratio": None,
    "argument_of_periastron": "deg",
    "gamma": "km / s",
    "period": "d",
    "primary_minimum_time": "d",
    "ln_f": None,
}

# Parameters that belong to the "nuisance" section rather than "system".
_NUISANCE_PARAMS: frozenset[str] = frozenset({"ln_f"})


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


def load_rv_data(file_path: str | None, x_unit_str: str) -> RVData | None:
    """Load an RV data file into an :class:`~elisa.analytics.RVData` object.

    Returns ``None`` when *file_path* is ``None`` (optional secondary component
    not uploaded).

    :param file_path: Path to the whitespace-delimited data file.  Columns
        must be ordered as ``x | RV [km/s] | sigma (optional)``.
    :type file_path: str | None
    :param x_unit_str: UI string for the x-axis unit - one of the values in
        :data:`data_inputs.X_UNIT_CHOICES`.
    :type x_unit_str: str
    :returns: Loaded :class:`~elisa.analytics.RVData` or ``None``.
    :rtype: RVData | None
    """
    if file_path is None:
        return None
    x_unit = _X_UNIT_MAP.get(x_unit_str, u.d)
    return RVData.load_from_file(file_path, x_unit=x_unit, y_unit=u.km / u.s)


def build_x0(param_values: dict[str, object]) -> dict:
    """Build the ``BinaryInitialParameters`` input dict from flat UI values.

    Constructs the nested ``{"system": {...}, "nuisance": {...}}`` JSON
    structure that :class:`~elisa.analytics.RVBinaryAnalyticsTask` expects
    as ``x0``.

    :param param_values: Flat dict keyed by ``"{name}_value"``,
        ``"{name}_fixed"``, ``"{name}_min"``, ``"{name}_max"`` for every
        parameter in :data:`~elisa.ui.tabs.rv_fitting.components.param_inputs.PARAMS`.
    :type param_values: dict[str, object]
    :returns: Nested initial-parameter dict suitable for
        ``RVBinaryAnalyticsTask.fit(x0=...)``.
    :rtype: dict
    """
    system: dict[str, dict] = {}
    nuisance: dict[str, dict] = {}

    for name, unit_str in _PARAM_UNITS.items():
        value = _opt_float(param_values.get(f"{name}_value"))
        fixed = bool(param_values.get(f"{name}_fixed", False))
        lo = _opt_float(param_values.get(f"{name}_min"))
        hi = _opt_float(param_values.get(f"{name}_max"))

        entry: dict[str, object] = {"value": value, "fixed": fixed}
        if not fixed:
            entry["min"] = lo
            entry["max"] = hi
        if unit_str is not None:
            entry["unit"] = unit_str

        target = nuisance if name in _NUISANCE_PARAMS else system
        target[name] = entry

    result: dict = {"system": system}
    if nuisance:
        result["nuisance"] = nuisance
    return result


def result_to_dataframe(result: dict) -> pd.DataFrame:
    """Convert a flat fit result into a display DataFrame.

    Iterates over the flat result (``"section@param"`` keys) and builds
    a table with columns *Parameter*, *Value*, *-1s*, *+1s*, *Unit*,
    *Status*.

    :param result: Flat result dict as returned by
        :meth:`~elisa.analytics.binary_fit.rv_fit.RVFit.flat_result`.
    :type result: dict
    :returns: DataFrame suitable for ``gr.DataFrame``.
    :rtype: pandas.DataFrame
    """
    rows = []
    for key, meta in result.items():
        value = meta.get("value")
        ci = meta.get("confidence_interval") or {}
        lo = ci.get("min")
        hi = ci.get("max")
        unit = meta.get("unit") or "-"
        fixed = meta.get("fixed", False)
        status = "fixed" if fixed else "free"

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


def _save_result_json(result: dict) -> str:
    """Write *result* to a timestamped JSON file in the system temp directory.

    :param result: Fit result dict to serialise.
    :type result: dict
    :returns: Absolute path of the written file.
    :rtype: str
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(tempfile.gettempdir()) / f"elisa_rv_result_{ts}.json"
    path.write_text(json.dumps(result, indent=4, default=str))
    return str(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_lsqrt(
    primary_path: str | None,
    secondary_path: str | None,
    x_unit_str: str,
    param_values: dict[str, object],
) -> tuple[dict, Figure, pd.DataFrame, str]:
    """Run a least-squares RV fit and return all displayable artefacts.

    Builds :class:`~elisa.analytics.RVData` objects from the uploaded files,
    constructs the initial-parameter dict, runs the LSQRT optimisation via
    :class:`~elisa.analytics.RVBinaryAnalyticsTask`, and returns the fitted
    result together with a model plot, a results table, and a path to the
    saved JSON result.

    :param primary_path: Path to the primary component RV data file.
    :type primary_path: str | None
    :param secondary_path: Path to the secondary component RV data file
        (``None`` if not uploaded).
    :type secondary_path: str | None
    :param x_unit_str: X-axis unit label from the UI dropdown.
    :type x_unit_str: str
    :param param_values: Flat parameter dict from the Gradio form
        (keys follow the :data:`~components.param_inputs.FIELD_ORDER` convention).
    :type param_values: dict[str, object]
    :returns: Tuple of ``(result_dict, model_figure, results_dataframe, json_path)``.
    :rtype: tuple[dict, Figure, pandas.DataFrame, str]
    :raises ValueError: If no primary data file is provided.
    """
    if primary_path is None:
        msg = "Primary RV data file is required."
        raise ValueError(msg)

    rv_primary = load_rv_data(primary_path, x_unit_str)
    rv_secondary = load_rv_data(secondary_path, x_unit_str)

    data: dict[str, RVData] = {"primary": rv_primary}
    if rv_secondary is not None:
        data["secondary"] = rv_secondary

    x0 = build_x0(param_values)

    plt.close("all")
    task = RVBinaryAnalyticsTask(data=data, method="least_squares")
    result = task.fit(x0=x0)

    model_fig: Figure = task.plot.model(return_figure_instance=True)
    df = result_to_dataframe(task.fit_cls.flat_result)
    json_path = _save_result_json(result)

    return result, model_fig, df, json_path


def run_mcmc(
    primary_path: str | None,
    secondary_path: str | None,
    x_unit_str: str,
    param_values: dict[str, object],
    nwalkers: int,
    nsteps: int,
    burn_in: int,
    fit_id: str,
    *,
    save: bool = True,
) -> tuple[dict, Figure, Figure, Figure, pd.DataFrame, str]:
    """Run an MCMC RV fit and return all displayable artefacts.

    Builds the ELISa data and parameter objects, runs the MCMC sampling,
    and captures the model plot, corner plot, and traces plot as
    ``Figure`` instances by temporarily intercepting ``plt.show()``.

    :param primary_path: Path to the primary component RV data file.
    :type primary_path: str | None
    :param secondary_path: Path to the secondary component RV data file.
    :type secondary_path: str | None
    :param x_unit_str: X-axis unit label from the UI dropdown.
    :type x_unit_str: str
    :param param_values: Flat parameter dict from the Gradio form.
    :type param_values: dict[str, object]
    :param nwalkers: Number of MCMC walkers.
    :type nwalkers: int
    :param nsteps: Number of MCMC steps.
    :type nsteps: int
    :param burn_in: Number of burn-in steps to discard.
    :type burn_in: int
    :param fit_id: Chain file identifier used when *save* is ``True``.
    :type fit_id: str
    :param save: Whether to save the chain to disk.
    :type save: bool
    :returns: Tuple of
        ``(result_dict, model_figure, corner_figure, traces_figure,
        results_dataframe, json_path)``.
    :rtype: tuple[dict, Figure, Figure, Figure, pandas.DataFrame, str]
    :raises ValueError: If no primary data file is provided.
    """
    if primary_path is None:
        msg = "Primary RV data file is required."
        raise ValueError(msg)

    rv_primary = load_rv_data(primary_path, x_unit_str)
    rv_secondary = load_rv_data(secondary_path, x_unit_str)

    data: dict[str, RVData] = {"primary": rv_primary}
    if rv_secondary is not None:
        data["secondary"] = rv_secondary

    x0 = build_x0(param_values)

    plt.close("all")
    task = RVBinaryAnalyticsTask(data=data, method="mcmc")
    result = task.fit(
        x0=x0,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burn_in=burn_in,
        save=save,
        fit_id=fit_id,
        progress=False,
    )

    model_fig: Figure = task.plot.model(return_figure_instance=True)

    with _capture_figure() as corner_captured:
        task.plot.corner(truths=True)
    corner_fig: Figure | None = corner_captured[0] if corner_captured else None

    with _capture_figure() as traces_captured:
        task.plot.traces(truths=True)
    traces_fig: Figure | None = traces_captured[0] if traces_captured else None

    df = result_to_dataframe(task.fit_cls.flat_result)
    json_path = _save_result_json(result)

    return result, model_fig, corner_fig, traces_fig, df, json_path


def extract_values_for_transfer(result: dict) -> dict[str, object]:
    """Extract fitted parameter values from a result dict for form population.

    Maps the nested ``result["system"][param]["value"]`` and
    ``result["nuisance"][param]["value"]`` entries to the flat
    ``"{name}_value"`` keys used by the parameter form.

    Performs unit conversion where the result's default unit differs from
    the UI's expected unit:
    - ``gamma``: m/s (ELISa default) → km/s (UI form)

    :param result: Nested fit result dict as returned by
        :class:`~elisa.analytics.RVBinaryAnalyticsTask`.
    :type result: dict
    :returns: Flat dict with ``"{name}_value"`` keys suitable for passing as
        *defaults* to
        :func:`~elisa.ui.tabs.rv_fitting.components.param_inputs.build`.
    :rtype: dict[str, object]
    """
    out: dict[str, object] = {}
    for section in ("system", "nuisance"):
        section_data = result.get(section) or {}
        for name, meta in section_data.items():
            if isinstance(meta, dict) and "value" in meta:
                value = meta["value"]
                unit_str = meta.get("unit")

                # Convert gamma from m/s (ELISa default) to km/s (UI form)
                if name == "gamma" and unit_str == "m / s":
                    value_with_unit = value * u.m / u.s
                    value = value_with_unit.to(u.km / u.s).value

                out[f"{name}_value"] = value
    return out

