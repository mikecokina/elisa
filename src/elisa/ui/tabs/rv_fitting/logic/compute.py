"""Core computation logic for RV fitting - no Gradio dependency.

Translates flat Gradio value dictionaries into ELISa analytics objects,
runs the LSQRT or MCMC optimization, and returns plain Python / PIL image
objects that the UI layer can display.

The module-level :func:`_capture_figure` context manager intercepts
``plt.show()`` so that MCMC diagnostic plots (corner, traces) which call
``plt.show()`` internally can be captured and converted to PIL images
suitable for ``gr.Image``.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from elisa import units as u
from elisa.analytics import RVBinaryAnalyticsTask, RVData
from elisa.ui.shared.logging_config import fit_logging
from elisa.ui.shared.plotting import figure_to_pil
from elisa.utc import UTC

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from PIL import Image

    from elisa.types import Float, Int

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

    The context manager intercepts ``plt.show()`` and collects the
    current :class:`matplotlib.figure.Figure` into a list which is
    yielded to the caller. Only the figure present at the moment of
    the ``plt.show()`` call is captured; subsequent calls overwrite the
    captured value.

    :yields: A list that will contain the captured
        :class:`matplotlib.figure.Figure` after the context exits.
    :rtype: list[matplotlib.figure.Figure]
    """
    captured: list[Figure] = []
    original_show = plt.show

    def _mock_show(*_args: object, **_kwargs: Any) -> None:
        fig = plt.gcf()
        if fig is not None:
            captured.append(fig)

    plt.show = _mock_show  # type: ignore[assignment]
    try:
        yield captured
    finally:
        plt.show = original_show  # type: ignore[assignment]


def _opt_float(value: object) -> Float | None:
    """Convert *value* to a floating value or return ``None``.

    Accepts user-supplied values from Gradio number inputs and attempts to
    coerce them to a numeric type used internally by ELISa.

    :param value: Raw value from a Gradio Number component.
    :type value: object
    :returns: Parsed floating value or ``None`` when the input is empty or
        cannot be converted.
    :rtype: elisa.types.Float | None
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
        ``"{name}_mode"``, ``"{name}_constraint"``, ``"{name}_min"``,
        ``"{name}_max"`` for every parameter in
        :data:`~elisa.ui.tabs.rv_fitting.components.param_inputs.PARAMS`.
    :type param_values: dict[str, object]
    :returns: Nested initial-parameter dict suitable for
        ``RVBinaryAnalyticsTask.fit(x0=...)``.
    :rtype: dict
    """
    system: dict[str, dict] = {}
    nuisance: dict[str, dict] = {}

    for name, unit_str in _PARAM_UNITS.items():
        value = _opt_float(param_values.get(f"{name}_value"))
        mode = str(param_values.get(f"{name}_mode", "free"))
        fixed = mode == "fixed"
        constraint = str(param_values.get(f"{name}_constraint", "") or "").strip()
        lo = _opt_float(param_values.get(f"{name}_min"))
        hi = _opt_float(param_values.get(f"{name}_max"))

        entry: dict[str, object] = {"value": value, "fixed": fixed}
        if mode == "constrained" and constraint:
            entry["constraint"] = constraint
        elif not fixed:
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


def _result_temp_path(prefix: str) -> Path:
    """Return a timestamped temp-file path for a fit result JSON.

    :param prefix: Short label embedded in the filename, e.g. ``"lsqrt"`` or ``"mcmc"``.
    :type prefix: str
    :returns: Path inside the system temp directory.
    :rtype: Path
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    return Path(tempfile.gettempdir()) / f"elisa_rv_{prefix}_{ts}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_lsqrt(
    primary_path: str | None,
    secondary_path: str | None,
    x_unit_str: str,
    param_values: dict[str, object],
    *,
    task: RVBinaryAnalyticsTask | None = None,
) -> tuple[dict, Image.Image, pd.DataFrame, str]:
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
    :param task: Optional pre-constructed :class:`~elisa.analytics.RVBinaryAnalyticsTask`.
        When supplied the fit will be run on this task instance, and it will be
        populated with the resulting fit state (useful when the caller wants
        to retain the task object across UI actions).
    :type task: RVBinaryAnalyticsTask | None
    :returns: Tuple of ``(result_dict, model_image, results_dataframe, json_path)``.
    :rtype: tuple[dict, PIL.Image.Image, pandas.DataFrame, str]
    :raises ValueError: If no primary data file is provided.
    """
    if primary_path is None:
        msg = "Primary RV data file is required."
        raise ValueError(msg)

    # If a pre-constructed task instance is provided, use it; otherwise
    # construct a new RVBinaryAnalyticsTask from uploaded files.
    if task is None:
        rv_primary = load_rv_data(primary_path, x_unit_str)
        rv_secondary = load_rv_data(secondary_path, x_unit_str)

        data: dict[str, RVData] = {"primary": rv_primary}
        if rv_secondary is not None:
            data["secondary"] = rv_secondary

        x0 = build_x0(param_values)

        plt.close("all")
        # noinspection PyTypeChecker
        with fit_logging():
            task = RVBinaryAnalyticsTask(data=data, method="least_squares")
            result = task.fit(x0=x0)
    else:
        # Use the provided task - caller is responsible for creating it and
        # ensuring its .data attribute is set if necessary.
        x0 = build_x0(param_values)
        plt.close("all")
        # noinspection PyTypeChecker
        with fit_logging():
            result = task.fit(x0=x0)

    model_fig: Figure = task.plot.model(return_figure_instance=True)
    df = result_to_dataframe(task.fit_cls.flat_result)

    json_path = _result_temp_path("lsqrt")
    task.save_result(str(json_path))

    return result, figure_to_pil(model_fig), df, str(json_path)


def run_mcmc(
    primary_path: str | None,
    secondary_path: str | None,
    x_unit_str: str,
    param_values: dict[str, object],
    nwalkers: Int,
    nsteps: Int,
    burn_in: Int,
    fit_id: str,
    *,
    save: bool = True,
    progress: bool = True,
    task: RVBinaryAnalyticsTask | None = None,
    initial_state: object = None,
) -> tuple[dict, Image.Image, Image.Image | None, Image.Image | None, pd.DataFrame, str]:
    """Run an MCMC RV fit and return all displayable artefacts.

    Builds the ELISa data and parameter objects, runs the MCMC sampling,
    and captures the model plot, corner plot, and traces plot as PIL images
    by temporarily intercepting ``plt.show()``.

    :param primary_path: Path to the primary component RV data file.
    :type primary_path: str | None
    :param secondary_path: Path to the secondary component RV data file.
    :type secondary_path: str | None
    :param x_unit_str: X-axis unit label from the UI dropdown.
    :type x_unit_str: str
    :param param_values: Flat parameter dict from the Gradio form.
    :type param_values: dict[str, object]
    :param nwalkers: Number of MCMC walkers.
    :type nwalkers: elisa.types.Int
    :param nsteps: Number of MCMC steps.
    :type nsteps: elisa.types.Int
    :param burn_in: Number of burn-in steps to discard.
    :type burn_in: elisa.types.Int
    :param fit_id: Chain file identifier used when *save* is ``True``.
    :type fit_id: str
    :param save: Whether to save the chain to disk.
    :type save: bool
    :param progress: Whether to show a progress indicator during MCMC sampling.
    :type progress: bool
    :param task: Optional pre-constructed :class:`~elisa.analytics.RVBinaryAnalyticsTask`.
        When supplied the MCMC fit will be run on this task instance, and it
        will be populated with the resulting chain and fit state. The caller
        is responsible for providing appropriate ``data`` on the task.
    :type task: RVBinaryAnalyticsTask | None
    :param initial_state: Optional initial state array for MCMC walkers.
        Should be an array of shape (nwalkers, n_params) containing the
        starting positions for each walker. When ``None``, the sampler
        initializes walkers from the prior.
    :type initial_state: object
    :returns: Tuple of
        ``(result_dict, model_image, corner_image, traces_image,
        results_dataframe, json_path)``.
    :rtype: tuple[dict, PIL.Image.Image, PIL.Image.Image | None, PIL.Image.Image | None, pandas.DataFrame, str]
    :raises ValueError: If no primary data file is provided.
    """
    # Accept an optional pre-built task so callers (UI) can create the
    # analytics instance and retain it in session state. If no task is
    # provided, create a fresh one from uploaded files.
    #
    # NOTE: The `task` parameter is accepted as a keyword-only argument.
    # Callers should pass it like: `run_mcmc(..., task=task_obj)`.
    if primary_path is None:
        msg = "Primary RV data file is required."
        raise ValueError(msg)

    # Build x0 from the form values
    x0 = build_x0(param_values)

    # If caller provided a task instance, use it; otherwise build one from files
    if task is None:
        rv_primary = load_rv_data(primary_path, x_unit_str)
        rv_secondary = load_rv_data(secondary_path, x_unit_str)

        data: dict[str, RVData] = {"primary": rv_primary}
        if rv_secondary is not None:
            data["secondary"] = rv_secondary

        task = RVBinaryAnalyticsTask(data=data, method="mcmc")

    else:
        # Ensure task has observational data when possible: prefer existing
        # data on the task, otherwise load from provided paths.
        try:
            has_primary = bool(getattr(task, "data", None) and task.data.get("primary"))
        except (AttributeError, TypeError):
            has_primary = False
        if not has_primary:
            rv_primary = load_rv_data(primary_path, x_unit_str)
            rv_secondary = load_rv_data(secondary_path, x_unit_str)
            task.data = {"primary": rv_primary}
            if rv_secondary is not None:
                task.data["secondary"] = rv_secondary

    plt.close("all")
    # noinspection PyTypeChecker
    with fit_logging():
        fit_kwargs = {
            "x0": x0,
            "nwalkers": nwalkers,
            "nsteps": nsteps,
            "burn_in": burn_in,
            "save": save,
            "fit_id": fit_id,
            "progress": progress,
        }
        if initial_state is not None:
            fit_kwargs["initial_state"] = initial_state  # type: ignore[typeddict-unknown-key]
        result = task.fit(**fit_kwargs)

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

    return (
        result,
        figure_to_pil(model_fig),
        figure_to_pil(corner_fig),
        figure_to_pil(traces_fig),
        df,
        str(json_path),
    )


# Chain-loading helpers removed per request. Use the analytics layer
# (RVBinaryAnalyticsTask.load_result/load_chain) directly in scripts if
# you need to reconstruct tasks from saved files.


def _gamma_ms_to_kms(value: Float | None) -> Float | None:
    """Convert a gamma velocity value from m/s to km/s.

    Safely handles ``None`` inputs and returns ``None`` in that case so
    callers can use the helper without additional guards.

    :param value: Velocity in m/s, or ``None``.
    :type value: elisa.types.Float | None
    :returns: Velocity in km/s, or ``None`` if *value* is ``None``.
    :rtype: elisa.types.Float | None
    """
    if value is None:
        return None
    return (value * u.m / u.s).to(u.km / u.s).value


def _param_meta_to_flat(name: str, meta: dict) -> dict[str, object]:
    """Convert a single parameter metadata dict to flat form entries.

    Reads ``value``, ``fixed``, ``constraint``, ``min``, and ``max`` from
    *meta* and maps them to ``"{name}_value"``, ``"{name}_mode"``,
    ``"{name}_constraint"``, ``"{name}_min"``, and ``"{name}_max"`` keys.
    Applies m/s to km/s conversion for ``gamma``.

    :param name: Parameter name (e.g. ``"gamma"``, ``"asini"``).
    :type name: str
    :param meta: Metadata dict for one parameter as stored in the JSON result.
    :type meta: dict
    :returns: Flat dict entries for this parameter.
    :rtype: dict[str, object]
    """
    value = meta.get("value")
    fixed = bool(meta.get("fixed", False))
    constraint = meta.get("constraint")
    unit_str = meta.get("unit") or ""
    min_val = meta.get("min")
    max_val = meta.get("max")

    if name == "gamma" and unit_str == "m / s":
        value = _gamma_ms_to_kms(value)
        min_val = _gamma_ms_to_kms(min_val)
        max_val = _gamma_ms_to_kms(max_val)

    entry: dict[str, object] = {}
    if value is not None:
        entry[f"{name}_value"] = value

    # Derive mode string for the UI dropdown (matching LC fitting convention).
    if constraint:
        entry[f"{name}_mode"] = "constrained"
        entry[f"{name}_constraint"] = constraint
    elif fixed:
        entry[f"{name}_mode"] = "fixed"
    else:
        entry[f"{name}_mode"] = "free"

    if min_val is not None:
        entry[f"{name}_min"] = min_val
    if max_val is not None:
        entry[f"{name}_max"] = max_val
    return entry


def extract_values_for_transfer(result: dict) -> dict[str, object]:
    """Extract fitted parameter values from a result dict for form population.

    Maps the nested ``result["system"][param]["value"]`` and
    ``result["nuisance"][param]["value"]`` entries to the flat
    ``"{name}_value"`` keys used by the parameter form.
    Performs unit conversion where the result's default unit differs from
    the UI's expected unit:

    - ``gamma``: m/s (ELISa default) -> km/s (UI form)

    :param result: Nested fit result dict as returned by
        :class:`~elisa.analytics.RVBinaryAnalyticsTask`.
    :type result: dict
    :returns: Flat dict with ``"{name}_value"`` keys suitable for passing as
        defaults to
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
                if name == "gamma" and unit_str == "m / s":
                    value = _gamma_ms_to_kms(value)
                out[f"{name}_value"] = value
    return out


def load_params_from_json(path: str) -> dict[str, object]:
    """Load parameter values, bounds and fixed flags from a saved result JSON.

    Parses the ``system`` and ``nuisance`` sections written by
    :meth:`~elisa.analytics.RVBinaryAnalyticsTask.save_result` and returns a
    flat dict compatible with the parameter form.
    Unit conversion applied where ELISa internal unit differs from the UI:

    - ``gamma``: stored as ``m / s`` -> converted to ``km/s`` (value and bounds).

    Fixed parameters have no ``min``/``max`` in the file - those keys are
    omitted from the return dict so the form keeps its existing bound values.

    :param path: Absolute path to the JSON file produced by ``save_result()``.
    :type path: str
    :returns: Flat dict with ``"{name}_value"``, ``"{name}_fixed"`` and,
        when present, ``"{name}_min"`` and ``"{name}_max"`` keys.
    :rtype: dict[str, object]
    :raises ValueError: If the file cannot be read or does not contain a
        recognised ``system`` section.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"Cannot read result file: {exc}"
        raise ValueError(msg) from exc
    if "system" not in data:
        msg = "File does not look like an ELISa RV result - missing 'system' key."
        raise ValueError(msg)
    out: dict[str, object] = {}
    for section in ("system", "nuisance"):
        for name, meta in (data.get(section) or {}).items():
            if not isinstance(meta, dict):
                continue
            out.update(_param_meta_to_flat(name, meta))
    return out


def load_chain(
    chain_file_path: str,
    result_file_path: str,
    discard: Int = 0,
) -> tuple[RVBinaryAnalyticsTask, Image.Image | None, Image.Image | None, pd.DataFrame | None]:
    """Load a previously saved MCMC chain and results for plotting and reuse.

    Loads both the result JSON (containing fitted parameters) and the chain
    JSON file, then creates an :class:`~elisa.analytics.RVBinaryAnalyticsTask`
    with the loaded state. Automatically generates corner and traces plots
    if both files load successfully.

    :param chain_file_path: Path to the saved flat-chain JSON file.
    :type chain_file_path: str
    :param result_file_path: Path to the saved result JSON file.
    :type result_file_path: str
    :param discard: Number of initial chain samples to discard per walker.
        Used to trim burn-in before generating plots.
    :type discard: elisa.types.Int
    :returns: Tuple of (task, corner_image, traces_image, results_dataframe).
        Images are ``None`` if plotting fails. DataFrame is ``None`` if
        result extraction fails.
    :rtype: tuple[RVBinaryAnalyticsTask, PIL.Image.Image | None, PIL.Image.Image | None, pandas.DataFrame | None]
    :raises gr.Error: If result file cannot be loaded or if chain file
        is missing or invalid.
    """
    # Create a task to hold the loaded state.
    task = RVBinaryAnalyticsTask(data={}, method="mcmc")

    try:
        # Load result first - this populates the task with fit parameters.
        task.load_result(filename=result_file_path)
    except Exception as exc:
        msg = f"Failed to load result file '{result_file_path}': {exc}"
        raise gr.Error(msg) from exc

    try:
        # Load chain - this populates task.fit_cls.flat_chain
        task.load_chain(chain_file_path, discard=discard)
    except Exception as exc:
        msg = f"Failed to load chain file '{chain_file_path}': {exc}"
        raise gr.Error(msg) from exc

    # Generate plots using the task's plotting interface.
    corner_fig: Figure | None = None
    traces_fig: Figure | None = None
    results_df: pd.DataFrame | None = None

    plt.close("all")
    try:
        with _capture_figure() as corner_captured:
            task.plot.corner(truths=True)
        corner_fig = corner_captured[0] if corner_captured else None
    except Exception as exc:  # noqa: BLE001
        msg = f"Corner plot generation failed: {exc}"
        gr.Warning(msg)

    plt.close("all")
    try:
        with _capture_figure() as traces_captured:
            task.plot.traces(truths=True)
        traces_fig = traces_captured[0] if traces_captured else None
    except Exception as exc:  # noqa: BLE001
        msg = f"Traces plot generation failed: {exc}"
        gr.Warning(msg)

    try:
        results_df = result_to_dataframe(task.fit_cls.flat_result)
    except Exception as exc:  # noqa: BLE001
        msg = f"Results dataframe generation failed: {exc}"
        gr.Warning(msg)

    return task, figure_to_pil(corner_fig), figure_to_pil(traces_fig), results_df


def extract_initial_state_from_chain(
    task: RVBinaryAnalyticsTask,
    nwalkers: Int,
) -> NDArray:
    """Extract the last nwalkers samples from a loaded chain as initial state.

    Uses the last nwalkers rows from the chain to create an initial state
    array suitable for passing to :meth:`~elisa.analytics.RVBinaryAnalyticsTask.fit`
    as the ``initial_state`` parameter.

    :param task: Task with a loaded chain (task.fit_cls.flat_chain populated).
    :type task: RVBinaryAnalyticsTask
    :param nwalkers: Number of walkers for the next MCMC run.
    :type nwalkers: elisa.types.Int
    :returns: Array of shape (nwalkers, n_params) containing initial positions.
    :rtype: object
    :raises gr.Error: If the chain is too short or if extraction fails.
    """
    try:
        chain = task.fit_cls.flat_chain
    except Exception as exc:
        msg = f"Failed to access chain from loaded task: {exc}"
        raise gr.Error(msg) from exc

    if chain is None or chain.size == 0:
        msg = "Chain is empty or not loaded."
        raise gr.Error(msg)

    n_samples = chain.shape[0]
    if n_samples < nwalkers:
        msg = (
            f"Chain has {n_samples} samples but {nwalkers} walkers requested. "
            f"Load a longer chain or reduce the number of walkers."
        )
        raise gr.Error(msg)

    # Extract the last nwalkers rows
    return chain[-nwalkers:, :]
