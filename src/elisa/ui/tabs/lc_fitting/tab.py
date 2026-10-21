"""LC Fitting tab for the ELISa Gradio UI.

Provides a full workflow for fitting light curves using either the
least-squares (LSQRT) or MCMC method:

1. Upload one or more light-curve data files, each assigned a passband.
2. Set initial parameter values, bounds, and fixed/constrained flags.
3. Choose the binary morphology (detached or over-contact).
4. Run LSQRT - fast deterministic fit, result stored in session state.
5. Optionally transfer the LSQRT result as MCMC starting point.
6. Run MCMC - full posterior sampling, produces corner and traces plots.
7. Inspect the summary table, model plot, and MCMC diagnostics.
8. Download the JSON result file.

The ``semi_major_axis`` parameter supports a three-way mode selector:
- **free** - fitted with explicit min/max bounds.
- **fixed** - held at its initial value.
- **constrained** - derived via a Python/math expression referencing other
  parameters (e.g. ``"16.515 / sin(radians(system@inclination))"`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, SupportsIndex, cast

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from elisa.analytics import LCBinaryAnalyticsTask
from elisa.analytics.binary_fit.shared import extend_observations_to_desired_interval
from elisa.ui.shared.plotting import figure_to_pil
from elisa.ui.shared.utils import collect_param_values
from elisa.ui.tabs.lc_fitting.components import data_inputs, param_inputs
from elisa.ui.tabs.lc_fitting.components.data_inputs import MAX_PASSBAND_ROWS
from elisa.ui.tabs.lc_fitting.components.param_inputs import (
    COMPONENT_PARAMS,
    FIELD_ORDER_UNIFIED,
    SYSTEM_REGULAR_PARAMS,
)
from elisa.ui.tabs.lc_fitting.logic import compute

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Int

# ---------------------------------------------------------------------------
# Tab ID constants
# ---------------------------------------------------------------------------

_TAB_LSQRT = "lc_lsqrt_results"
_TAB_MCMC = "lc_mcmc_results"

_X_UNIT_JD = "Julian days (JD)"
_PRIMARY_MIN_TIME_PREFIX = "system_primary_minimum_time"
_PRIMARY_MIN_TIME_KEYS: tuple[str, ...] = (
    f"{_PRIMARY_MIN_TIME_PREFIX}_value",
    f"{_PRIMARY_MIN_TIME_PREFIX}_mode",
    f"{_PRIMARY_MIN_TIME_PREFIX}_constraint",
    f"{_PRIMARY_MIN_TIME_PREFIX}_min",
    f"{_PRIMARY_MIN_TIME_PREFIX}_max",
)


def _is_jd_x_unit(x_unit: str) -> bool:
    """Return whether the selected x-axis unit is Julian days.

    :param x_unit: X-axis unit label from the dropdown.
    :type x_unit: str
    :returns: ``True`` when x-unit is Julian days.
    :rtype: bool
    """
    return x_unit == _X_UNIT_JD


def _primary_min_time_interactivity(mode: str, *, enabled: bool) -> tuple[bool, bool, bool, bool, bool]:
    """Compute interactivity tuple for primary-minimum-time controls.

    :param mode: Parameter mode value (``"free"``, ``"fixed"``, ``"constrained"``).
    :type mode: str
    :param enabled: Whether the whole parameter row is enabled by x-unit context.
    :type enabled: bool
    :returns: ``(value, mode, constraint, min, max)`` interactivity flags.
    :rtype: tuple[bool, bool, bool, bool, bool]
    """
    if not enabled:
        return False, False, False, False, False
    return (
        True,
        True,
        mode == "constrained",
        mode == "free",
        mode == "free",
    )


def _primary_min_time_label_update(*, disabled: bool) -> object:
    """Build label update for ``system_primary_minimum_time``.

    :param disabled: Whether label should use disabled styling.
    :type disabled: bool
    :returns: ``gr.update`` payload for the label Markdown component.
    :rtype: object
    """
    base_label = param_inputs.get_label_for_prefix(_PRIMARY_MIN_TIME_PREFIX)
    if base_label is None:
        return gr.update()
    return gr.update(value=param_inputs.style_param_label(base_label, disabled=disabled))


def _is_x_unit_restricted_key(key: str) -> bool:
    """Return whether *key* belongs to the x-unit restricted T0 parameter.

    :param key: Flat form key.
    :type key: str
    :returns: ``True`` for ``system_primary_minimum_time_*`` keys.
    :rtype: bool
    """
    return any(key == restricted_key for restricted_key in _PRIMARY_MIN_TIME_KEYS)


# ---------------------------------------------------------------------------
# Stateless event handlers (module-level)
# ---------------------------------------------------------------------------


def _add_passband(count: object) -> list[object]:
    """Increment passband row count and show the newly active row.

    :param count: Current active row count from ``gr.State``.
    :type count: object
    :returns: ``[new_count, *row_visibility_updates]``.
    :rtype: list[object]
    """
    new_count = min(int(count), MAX_PASSBAND_ROWS - 1) + 1  # type: ignore[arg-type]
    return [new_count, *[gr.update(visible=i < new_count) for i in range(MAX_PASSBAND_ROWS)]]


def _remove_passband(count: object) -> list[object]:
    """Decrement passband row count and hide the last active row.

    :param count: Current active row count from ``gr.State``.
    :type count: object
    :returns: ``[new_count, *row_visibility_updates]``.
    :rtype: list[object]
    """
    new_count = max(int(count) - 1, 1)  # type: ignore[arg-type]
    return [new_count, *[gr.update(visible=i < new_count) for i in range(MAX_PASSBAND_ROWS)]]


def _sma_mode_changed(mode: str) -> tuple[object, object, object]:
    """Update constraint/min/max visibility and interactivity based on mode.

    :param mode: Selected mode - ``"free"``, ``"fixed"``, or ``"constrained"``.
    :type mode: str
    :returns: ``gr.update`` for constraint (visible), min (interactive), max (interactive).
    :rtype: tuple[object, object, object]
    """
    is_free = mode == "free"
    return (
        gr.update(visible=mode == "constrained"),
        gr.update(interactive=is_free),
        gr.update(interactive=is_free),
    )


def _parse_lc_inputs_simple(
    values: tuple[object, ...],
    n_rows: int,
) -> tuple[str, list[compute.LCRowData], str, int]:
    """Parse the flat input tuple from a fit handler (without approach).

    Expects inputs in the order laid out by :func:`build`:
    ``[x_unit, passband_count, row_0_pb, row_0_file, row_0_yu, row_0_rm, ..., morphology, ...]``

    :param values: Flat Gradio values tuple.
    :type values: tuple[object, ...]
    :param n_rows: Pre-rendered row count (``MAX_PASSBAND_ROWS``).
    :type n_rows: int
    :returns: Tuple of ``(x_unit_str, active_lc_rows, morphology, fit_offset)``.
    :rtype: tuple[str, list[compute.LCRowData], str, int]
    """
    x_unit_str = str(values[0])
    passband_count = int(values[1])  # type: ignore[arg-type]
    offset = 2
    lc_rows: list[compute.LCRowData] = []

    for i in range(n_rows):
        row_offset = offset + i * 4
        passband = str(values[row_offset]) if values[row_offset] else ""
        file_obj = values[row_offset + 1]
        y_unit = str(values[row_offset + 2]) if values[row_offset + 2] else "Flux (dimensionless)"
        ref_mag_raw = values[row_offset + 3]
        ref_mag: compute.Float | None = (
            float(ref_mag_raw) if ref_mag_raw is not None else None  # type: ignore[arg-type]
        )
        if i < passband_count:
            lc_rows.append(
                compute.LCRowData(
                    passband=passband,
                    file_path=getattr(file_obj, "name", None),
                    y_unit=y_unit,
                    reference_magnitude=ref_mag,
                ),
            )

    morphology_offset = offset + n_rows * 4
    morphology = str(values[morphology_offset])
    fit_offset = morphology_offset + 1
    return x_unit_str, lc_rows, morphology, fit_offset


# ---------------------------------------------------------------------------
# Results sub-tab builders
# ---------------------------------------------------------------------------


def _build_lsqrt_results() -> tuple[gr.Image, gr.DataFrame, gr.DownloadButton]:
    """Render the LSQRT results sub-tab and return output components.

    :returns: Tuple of ``(model_plot, table, download_button)``.
    :rtype: tuple[gr.Image, gr.DataFrame, gr.DownloadButton]
    """
    with gr.Row():
        model_plot = gr.Image(label="Model fit", type="pil")
    with gr.Row():
        table = gr.DataFrame(label="Fitted parameters", wrap=True)
    with gr.Row():
        download = gr.DownloadButton(label="💾 Download result JSON", visible=False)
    return model_plot, table, download


def _build_mcmc_results() -> tuple[gr.Image, gr.DataFrame, gr.Image, gr.Image, gr.DownloadButton]:
    """Render the MCMC results sub-tab and return output components.

    :returns: Tuple of
        ``(model_plot, table, corner_plot, traces_plot, download_button)``.
    :rtype: tuple[gr.Image, gr.DataFrame, gr.Image, gr.Image, gr.DownloadButton]
    """
    with gr.Row():
        model_plot = gr.Image(label="Model fit (MCMC median)", type="pil")
    with gr.Row():
        table = gr.DataFrame(label="Fitted parameters", wrap=True)
    with gr.Row():
        corner_plot = gr.Image(label="Corner plot", type="pil")
    with gr.Row():
        traces_plot = gr.Image(label="Parameter traces", type="pil")
    with gr.Row():
        download = gr.DownloadButton(label="💾 Download result JSON", visible=False)
    return model_plot, table, corner_plot, traces_plot, download


# ---------------------------------------------------------------------------
# Event wiring helpers
# ---------------------------------------------------------------------------


def _build_active_param_update(k: str, params: dict[str, object], approach: str, x_unit: str) -> object:
    """Build one ``gr.update`` for field *k* using the loaded *params* dict.

    For ``_constraint`` / ``_min`` / ``_max`` keys the correct ``interactive``
    state is included so the form reflects the loaded mode without requiring a
    manual dropdown click.  Svelte only recreates a DOM node when
    ``interactive`` actually changes, so only params whose mode differs from
    the form default incur a lifecycle event.

    :param k: Field key from the loader key tuple.
    :type k: str
    :param params: Flat parameter dict from
        :func:`~logic.compute.load_params_from_json`.
    :type params: dict[str, object]
    :param approach: Lowercase approach (``"community"`` or ``"standard"``).
    :type approach: str
    :param x_unit: Current x-axis unit label used for x-unit-dependent gating.
    :type x_unit: str
    :returns: A ``gr.update`` with value and, where applicable, interactive.
    :rtype: object
    """
    normalized: Literal["community", "standard"] = "community" if approach == "community" else "standard"
    interactive_allowed = param_inputs.is_key_interactive_for_approach(k, normalized)
    if _is_x_unit_restricted_key(k):
        interactive_allowed = interactive_allowed and _is_jd_x_unit(x_unit)

    update_kwargs: dict[str, object] = {}
    if k.endswith("_constraint"):
        prefix = k[:-11]  # strip "_constraint" (11 chars)
        mode = str(params.get(f"{prefix}_mode", "free"))
        val = params.get(k)
        update_kwargs["value"] = str(val) if val is not None else ""
        update_kwargs["interactive"] = (mode == "constrained") and interactive_allowed
    elif k.endswith(("_min", "_max")):
        prefix = k[:-4]  # strip "_min" or "_max" (4 chars each)
        mode = str(params.get(f"{prefix}_mode", "free"))
        if k in params:
            update_kwargs["value"] = params[k]
        update_kwargs["interactive"] = (mode == "free") and interactive_allowed
    else:
        if k in params:
            update_kwargs["value"] = params[k]
        if not interactive_allowed:
            update_kwargs["interactive"] = False

    return gr.update(**update_kwargs)


def _normalize_ui_approach(approach: str) -> Literal["community", "standard"]:
    """Normalize radio value to internal approach literal.

    :param approach: Approach label from the radio component.
    :type approach: str
    :returns: Lowercase approach literal.
    :rtype: Literal["community", "standard"]
    """
    if approach == "Community":
        return "community"
    return "standard"


def _wire_json_loader(
    params_json_comp: gr.File,
    fit_comps: dict[str, gr.Component],
    fit_keys: tuple[str, ...],
    approach_comp: gr.Radio,
    x_unit_comp: gr.Dropdown,
) -> None:
    """Wire the JSON param-loader upload event with auto-detection of approach.

    Detects Community vs Standard from the uploaded JSON using
    :func:`~logic.compute.detect_approach_from_json`, switches the approach
    radio and section visibility, then populates the correct parameter form.

    **Performance design notes:**

    - **Spot-param fields are included in outputs.**  The loader now restores
      spot slots directly from result JSON data so previous-fit spot
      configuration is reproduced in the form.
    - **Interactive state included for active approach.**  Each
      ``constraint`` / ``min`` / ``max`` field is updated with both ``value``
      and the correct ``interactive`` flag derived from the loaded mode.
      Svelte only recreates a DOM node when ``interactive`` *actually changes*,
      so only params whose mode differs from the form default incur a lifecycle
      event - not all 22.  The inactive approach receives plain no-op
      ``gr.update()`` calls because its column is hidden.
    - **``show_progress="hidden"`` + ``show_progress_on=[]``.**  Without
      ``show_progress_on=[]``, Gradio's loading-state machinery defaults to
      ``show_progress_on=None``, which the JS frontend interprets as
      ``null || dep.outputs`` and therefore registers *every output component*
      for loading-status tracking.  ``update_loading_stati_state()`` is then
      called multiple times per event and invokes ``find_node_by_id`` once per
      registered component each time - for 231 outputs that produces hundreds
      of redundant tree-walk calls even though the progress spinner is hidden.
      Passing an explicit empty list sets ``dep.show_progress_on = []`` (truthy
      in JS), making ``[] || dep.outputs`` evaluate to ``[]``, so zero
      components are registered and zero extra ``find_node_by_id`` calls occur.

    :param params_json_comp: The file upload component for the result JSON.
    :type params_json_comp: gr.File
    :param fit_comps: Unified parameter component mapping.
    :type fit_comps: dict[str, gr.Component]
    :param fit_keys: Ordered unified param keys.
    :type fit_keys: tuple[str, ...]
    :param approach_comp: The approach radio button component.
    :type approach_comp: gr.Radio
    :param x_unit_comp: X-axis unit dropdown component.
    :type x_unit_comp: gr.Dropdown
    """
    loader_keys: tuple[str, ...] = fit_keys
    label_keys_list = [k for k in param_inputs.APPROACH_TOGGLED_LABEL_KEYS if k in fit_comps]
    primary_min_time_label_key = f"{_PRIMARY_MIN_TIME_PREFIX}_label"
    if primary_min_time_label_key in fit_comps and primary_min_time_label_key not in label_keys_list:
        label_keys_list.append(primary_min_time_label_key)
    label_keys: tuple[str, ...] = tuple(label_keys_list)

    all_outputs: list[gr.Component] = (
        [approach_comp] + [fit_comps[k] for k in loader_keys] + [fit_comps[k] for k in label_keys]
    )

    def _load_json_handler(json_file: object, x_unit: object) -> list[object]:
        """Populate the parameter form from an uploaded LC result JSON.

        Auto-detects Community vs Standard, switches the approach selector
        and section visibility, then loads parameter values into the form.
        Spot parameters are not written to the form; a notification is shown
        when the JSON contains them.

        :param json_file: Gradio file object from the upload component.
        :type json_file: object
        :returns: Updates for approach selector and all non-spot param components.
        :rtype: list[object]
        """
        path: str | None = getattr(json_file, "name", None)
        if path is None:
            msg = "No file uploaded."
            raise gr.Error(msg)

        try:
            approach = compute.detect_approach_from_json(path)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        ui_approach = cast("Literal['community', 'standard']", "community" if approach == "Community" else "standard")

        try:
            params = compute.load_params_from_json(path)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        x_unit_str = str(x_unit)
        updates: list[object] = [_build_active_param_update(k, params, ui_approach, x_unit_str) for k in loader_keys]

        label_updates: list[object] = []
        for label_key in label_keys:
            prefix = label_key.removesuffix("_label")
            base_label = param_inputs.get_label_for_prefix(prefix)
            if base_label is None:
                label_updates.append(gr.update())
                continue
            interactive = param_inputs.is_key_interactive_for_approach(f"{prefix}_value", ui_approach)
            styled = param_inputs.style_param_label(base_label, disabled=not interactive)
            label_updates.append(gr.update(value=styled))

        primary_min_time_label_key_ = f"{_PRIMARY_MIN_TIME_PREFIX}_label"
        if primary_min_time_label_key_ in label_keys:
            primary_min_time_label_index = label_keys.index(primary_min_time_label_key_)
            label_updates[primary_min_time_label_index] = _primary_min_time_label_update(
                disabled=not _is_jd_x_unit(x_unit_str),
            )

        return [
            gr.update(value=approach),
            *updates,
            *label_updates,
        ]

    params_json_comp.upload(
        fn=_load_json_handler,
        inputs=[params_json_comp, x_unit_comp],
        outputs=all_outputs,
        show_progress="hidden",
        show_progress_on=[],
    )


# ---------------------------------------------------------------------------
# UI section builders (called inside the Tab context from build())
# ---------------------------------------------------------------------------


def _build_data_accordion() -> tuple[
    data_inputs.LCDataComponents,
    gr.Number,
    gr.Number,
    gr.Number,
    gr.Number,
    gr.Number,
    gr.Button,
    gr.Image,
    gr.Accordion,
]:
    """Render accordion 1 (Observational data) and return component refs.

    :returns: Dataclass with all passband row and control components.
    :rtype: data_inputs.LCDataComponents
    """
    with gr.Accordion("1 · Observational data", open=True):
        data_comps = data_inputs.build()

        # Observed data plotting controls header
        gr.Markdown("#### Observed data plotting controls")

        # Allow supplying period and T0 directly in the data section
        with gr.Row():
            # Default period and T0 come from the LC parameter spec defaults.
            # Start as non-interactive because the default X unit is Phases -
            # they are only needed when the user switches to Julian days (JD).
            period_comp = gr.Number(
                value=param_inputs.SYSTEM_COMMON_SPEC.get("period", (None, None))[1],
                label="Orbital period (P) [d] - used to phase JD data",
                info="Optional - if set here it will be used instead of the System Period parameter",
                interactive=False,
                scale=2,
            )
            t0_comp = gr.Number(
                value=param_inputs.SYSTEM_COMMON_SPEC.get("primary_minimum_time", (None, None))[1],
                label="Primary minimum time (T0) - used to phase JD data",
                info="Enable and set when X-axis unit is Julian days (JD)",
                interactive=False,
                scale=2,
            )

            start_phase_comp = gr.Number(
                value=-0.6,
                label="Start phase - left boundary for plotted interval",
                info="Phase interval start, used to expand observations across cycles",
                interactive=True,
                scale=2,
            )
            stop_phase_comp = gr.Number(
                value=0.6,
                label="Stop phase - right boundary for plotted interval",
                info="Phase interval end, used to expand observations across cycles",
                interactive=True,
                scale=2,
            )

            centre_comp = gr.Number(
                value=0.0,
                label="Phase centre - centre used when folding JD to phases",
                info="Centre value passed to DataSet.convert_to_phases(centre=...)",
                interactive=True,
                scale=2,
            )

            plot_obs_btn = gr.Button("📈 Plot observed data", variant="secondary", scale=1)

        # Toggle period and T0 interactivity based on selected x-unit.
        # Both are only meaningful when data are in Julian days - they are
        # not needed (and make no sense) when data are already in phases.
        def _on_x_unit_change(val: str) -> tuple[object, object]:
            is_jd = val == "Julian days (JD)"
            return gr.update(interactive=is_jd), gr.update(interactive=is_jd)

        data_comps.x_unit.change(
            fn=_on_x_unit_change,
            inputs=[data_comps.x_unit],
            outputs=[period_comp, t0_comp],
            show_progress="hidden",
            show_progress_on=[],
        )

        # Collapsible plot area for observed light curves
        with gr.Accordion("Observed data plot", open=False) as obs_plot_accordion:
            observed_data_plot = gr.Image(label="Observed data", type="pil")

        return (
            data_comps,
            period_comp,
            t0_comp,
            start_phase_comp,
            stop_phase_comp,
            centre_comp,
            plot_obs_btn,
            observed_data_plot,
            obs_plot_accordion,
        )


def _build_mcmc_accordion() -> tuple[gr.Number, gr.Number, gr.Number, gr.Textbox, gr.Checkbox, gr.Checkbox]:
    """Render accordion 3 (MCMC settings) and return all control components.

    :returns: Tuple of ``(nwalkers, nsteps, burn_in, fit_id, save_chain, progress)``.
    :rtype: tuple[gr.Number, gr.Number, gr.Number, gr.Textbox, gr.Checkbox, gr.Checkbox]
    """
    with gr.Accordion("3 · MCMC settings", open=False):
        gr.Markdown(
            "**MCMC sampling configuration** - adjust these parameters "
            "to control the Markov Chain Monte Carlo sampling process.",
        )
        with gr.Row():
            nwalkers_comp = gr.Number(
                value=120,
                label="Number of walkers",
                info="Number of MCMC walkers (should be >= 2 x free parameters).",
                precision=0,
                scale=1,
                minimum=10,
            )
            nsteps_comp = gr.Number(
                value=100,
                label="Sampling steps",
                info="Total number of sampling steps per walker (after burn-in).",
                precision=0,
                scale=1,
                minimum=10,
            )
            burn_in_comp = gr.Number(
                value=10,
                label="Burn-in / Warmup steps",
                info="Number of initial steps to discard for chain equilibration.",
                precision=0,
                scale=1,
                minimum=0,
            )
        with gr.Row():
            fit_id_comp = gr.Textbox(
                value="mcmc_lc_fit",
                label="Chain file ID / Path",
                info=(
                    "Filename (saved to ELISa home), relative path, or absolute path "
                    "ending with .json.  Example: 'my_fit', 'results/fit.json'."
                ),
                scale=3,
            )
            save_chain_comp = gr.Checkbox(
                value=True,
                label="Save chain to disk",
                scale=1,
            )
            progress_comp = gr.Checkbox(
                value=True,
                label="Show progress bar",
                scale=1,
            )
    return nwalkers_comp, nsteps_comp, burn_in_comp, fit_id_comp, save_chain_comp, progress_comp


def _build_chain_load_accordion() -> tuple[
    gr.File,
    gr.File,
    gr.Number,
    gr.Checkbox,
    gr.Button,
    gr.Markdown,
]:
    """Render optional chain-loading controls used for MCMC resume.

    :returns: Tuple of chain loading components.
    :rtype: tuple[gr.File, gr.File, gr.Number, gr.Checkbox, gr.Button, gr.Markdown]
    """
    with gr.Accordion("3.1 · Load previous MCMC chain (optional)", open=False):
        gr.Markdown(
            "**Load a previously saved MCMC chain** to visualize earlier results "
            "or seed a new run. When loaded, corner and traces diagnostics are "
            "generated automatically. If 'Use as initial state' is enabled, the "
            "last nwalkers samples are used as starting walker positions for the "
            "next MCMC fit.",
        )
        with gr.Row():
            chain_file_comp = gr.File(
                label="Chain file (flat-chain JSON)",
                file_types=[".json"],
                scale=2,
            )
            result_file_comp = gr.File(
                label="Result file (result JSON)",
                file_types=[".json"],
                scale=2,
            )
        with gr.Row():
            discard_comp = gr.Number(
                value=0,
                label="Discard (burn-in samples per walker)",
                info="Number of initial samples to discard before generating diagnostics.",
                precision=0,
                scale=2,
                minimum=0,
            )
            use_initial_state_comp = gr.Checkbox(
                value=False,
                label="Use as initial state for next MCMC run",
                info="When checked, the last nwalkers samples seed the next MCMC fit.",
                scale=2,
            )
        load_chain_btn = gr.Button(
            "📂 Load chain & plot diagnostics",
            variant="secondary",
            scale=1,
        )
        chain_load_status = gr.Markdown(value="Ready to load chain.", visible=True)

    return chain_file_comp, result_file_comp, discard_comp, use_initial_state_comp, load_chain_btn, chain_load_status


def _build_action_buttons() -> tuple[gr.Button, gr.Button, gr.Button]:
    """Render the action button row and return all button components.

    :returns: Tuple of ``(lsqrt_btn, transfer_btn, mcmc_btn)``.
    :rtype: tuple[gr.Button, gr.Button, gr.Button]
    """
    with gr.Row():
        lsqrt_btn = gr.Button("⚡ Run Least Squares", variant="primary", scale=2)
        transfer_btn = gr.Button("↓ Transfer LSQRT → MCMC", variant="secondary", scale=1)
        mcmc_btn = gr.Button("🔬 Run MCMC", variant="primary", scale=2)
    return lsqrt_btn, transfer_btn, mcmc_btn


def _build_results_section() -> tuple[
    gr.Tabs,
    gr.Image,
    gr.DataFrame,
    gr.DownloadButton,
    gr.Image,
    gr.DataFrame,
    gr.Image,
    gr.Image,
    gr.DownloadButton,
]:
    """Render the results tabs section and return all output components.

    :returns: Tuple of
        ``(results_tabs, lsqrt_model_plot, lsqrt_table, lsqrt_download,
        mcmc_model_plot, mcmc_table, corner_plot, traces_plot, mcmc_download)``.
    :rtype: tuple
    """
    with gr.Tabs() as results_tabs:
        with gr.Tab("LSQRT Results", id=_TAB_LSQRT):
            lsqrt_model_plot, lsqrt_table, lsqrt_download = _build_lsqrt_results()
        with gr.Tab("MCMC Results", id=_TAB_MCMC):
            mcmc_model_plot, mcmc_table, corner_plot, traces_plot, mcmc_download = _build_mcmc_results()
    return (
        results_tabs,
        lsqrt_model_plot,
        lsqrt_table,
        lsqrt_download,
        mcmc_model_plot,
        mcmc_table,
        corner_plot,
        traces_plot,
        mcmc_download,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build() -> None:  # noqa: C901, PLR0915
    """Build the LC Fitting tab inside the active ``gr.Blocks`` context.

    Must be called from within a ``gr.Blocks`` context manager.  Wires all
    Gradio components to the computation back-end and registers the following
    event handlers:

    - **+ Add / - Remove passband** - show/hide pre-rendered passband rows.
    - **semi_major_axis mode** radio - toggles constraint textbox visibility
      and min/max interactivity.
    - **Fixed checkboxes** (all regular params) - toggle min/max interactivity.
    - **Run The Least Squares** - runs :func:`~logic.compute.run_lsqrt` and
      populates the *LSQRT Results* sub-tab.
    - **Transfer to MCMC** - copies fitted values from session state into the
      parameter form.
    - **Run MCMC** - runs :func:`~logic.compute.run_mcmc` and populates the
      *MCMC Results* sub-tab including corner and traces plots.

    :returns: ``None``
    :rtype: None
    """
    with gr.Tab("LC Fitting"):
        gr.Markdown(
            "## Light Curve Fitting\n"
            "Upload observed light curves (one file per passband), configure initial "
            "parameters, then run **Least Squares** for a quick fit or **MCMC** for "
            "a full posterior.  Use **Transfer to MCMC** to seed the sampler with "
            "the LSQRT solution.",
        )

        # Session state holding analytics tasks keyed by method.
        # Keys: 'lsqrt' -> LCBinaryAnalyticsTask (populated after each LSQRT run)
        tasks_state: gr.State = gr.State(value={})
        chain_state: gr.State = gr.State(value={})
        (
            data_comps,
            period_comp,
            t0_comp,
            start_phase_comp,
            stop_phase_comp,
            centre_comp,
            plot_obs_btn,
            observed_data_plot,
            obs_plot_accordion,
        ) = _build_data_accordion()

        # Build approach selector
        with gr.Accordion("2 · Initial parameters", open=True):
            with gr.Row():
                morphology_comp = gr.Radio(
                    choices=["detached", "over-contact"],
                    value="detached",
                    label="Binary morphology",
                    info="Expected system morphology - affects valid surface potential ranges.",
                    scale=1,
                )
                approach_comp = gr.Radio(
                    choices=["Community", "Standard"],
                    value="Community",
                    label="Fitting approach",
                    info="Community: mass_ratio + semi_major_axis. Standard: individual masses.",
                    scale=1,
                )

            fit_comps, _fit_sections = param_inputs.build(approach="community")

            with gr.Accordion("Load Parameters from Previous Fit", open=False):
                gr.Markdown(
                    "Upload a result JSON saved by a previous LSQRT or MCMC run to restore "
                    "all parameter values, bounds, modes, and constraints into the form below.",
                )
                with gr.Row():
                    params_json_comp = gr.File(
                        label="Result JSON",
                        file_types=[".json"],
                        scale=1,
                    )

        nwalkers_comp, nsteps_comp, burn_in_comp, fit_id_comp, save_chain_comp, progress_comp = _build_mcmc_accordion()
        (
            chain_file_comp,
            result_file_comp,
            discard_comp,
            use_initial_state_comp,
            load_chain_btn,
            chain_load_status,
        ) = _build_chain_load_accordion()
        lsqrt_btn, transfer_btn, mcmc_btn = _build_action_buttons()
        (
            results_tabs,
            lsqrt_model_plot,
            lsqrt_table,
            lsqrt_download,
            mcmc_model_plot,
            mcmc_table,
            corner_plot,
            traces_plot,
            mcmc_download,
        ) = _build_results_section()

        # Wire approach selector to toggle interactivity for exclusive fields.
        approach_toggle_keys = param_inputs.APPROACH_TOGGLED_KEYS
        approach_toggle_outputs = [fit_comps[k] for k in approach_toggle_keys]
        approach_toggle_prefixes = param_inputs.APPROACH_TOGGLED_PREFIXES
        approach_toggle_label_keys = [k for k in param_inputs.APPROACH_TOGGLED_LABEL_KEYS if k in fit_comps]
        approach_toggle_label_outputs = [fit_comps[k] for k in approach_toggle_label_keys]

        def _on_approach_change(approach: str) -> list[object]:
            normalized = _normalize_ui_approach(approach)
            field_updates = [
                gr.update(interactive=param_inputs.is_key_interactive_for_approach(k, normalized))
                for k in approach_toggle_keys
            ]

            label_updates: list[object] = []
            for prefix in approach_toggle_prefixes:
                base_label = param_inputs.get_label_for_prefix(prefix)
                if base_label is None:
                    label_updates.append(gr.update())
                    continue
                interactive = param_inputs.is_key_interactive_for_approach(f"{prefix}_value", normalized)
                styled = param_inputs.style_param_label(base_label, disabled=not interactive)
                label_updates.append(gr.update(value=styled))

            return field_updates + label_updates

        approach_comp.change(
            fn=_on_approach_change,
            inputs=[approach_comp],
            outputs=approach_toggle_outputs + approach_toggle_label_outputs,
            show_progress="hidden",
            show_progress_on=[],
        )

        fit_keys = FIELD_ORDER_UNIFIED
        fit_inputs = [fit_comps[k] for k in fit_keys]
        lc_data_inputs = (
            [data_comps.x_unit, data_comps.passband_count]
            + [
                comp
                for i in range(MAX_PASSBAND_ROWS)
                for comp in (
                    data_comps.row_passbands[i],
                    data_comps.row_files[i],
                    data_comps.row_y_units[i],
                    data_comps.row_ref_mags[i],
                )
            ]
            + [morphology_comp]
        )

        mcmc_keys: tuple[str, ...] = ("nwalkers", "nsteps", "burn_in", "fit_id", "save_chain", "progress")

        pb_outputs = [data_comps.passband_count, *data_comps.row_groups]
        data_comps.add_btn.click(  # type: ignore[union-attr]
            fn=_add_passband,
            inputs=[data_comps.passband_count],
            outputs=pb_outputs,
            show_progress="hidden",
            show_progress_on=[],
        )
        data_comps.remove_btn.click(  # type: ignore[union-attr]
            fn=_remove_passband,
            inputs=[data_comps.passband_count],
            outputs=pb_outputs,
            show_progress="hidden",
            show_progress_on=[],
        )

        _wire_json_loader(
            params_json_comp,
            fit_comps,
            fit_keys,
            approach_comp,
            data_comps.x_unit,
        )
        _wire_json_loader(
            result_file_comp,
            fit_comps,
            fit_keys,
            approach_comp,
            data_comps.x_unit,
        )

        primary_min_time_outputs = [fit_comps[k] for k in _PRIMARY_MIN_TIME_KEYS]
        primary_min_time_label_key = f"{_PRIMARY_MIN_TIME_PREFIX}_label"
        primary_min_time_label_output = fit_comps.get(primary_min_time_label_key)

        def _on_x_unit_fit_param_change(x_unit: str, mode: str) -> list[object]:
            enabled = _is_jd_x_unit(x_unit)
            _value_i, _mode_i, _constraint_i, _min_i, _max_i = _primary_min_time_interactivity(mode, enabled=enabled)
            return [
                gr.update(interactive=_value_i),
                gr.update(interactive=_mode_i),
                gr.update(interactive=_constraint_i),
                gr.update(interactive=_min_i),
                gr.update(interactive=_max_i),
                _primary_min_time_label_update(disabled=not enabled),
            ]

        if primary_min_time_label_output is not None:
            data_comps.x_unit.change(
                fn=_on_x_unit_fit_param_change,
                inputs=[data_comps.x_unit, fit_comps[f"{_PRIMARY_MIN_TIME_PREFIX}_mode"]],
                outputs=[*primary_min_time_outputs, primary_min_time_label_output],
                show_progress="hidden",
                show_progress_on=[],
            )

            # Ensure default render matches the default x-unit (Phases).
            default_enabled = _is_jd_x_unit(data_inputs.X_UNIT_DEFAULT)
            default_mode = str(fit_comps[f"{_PRIMARY_MIN_TIME_PREFIX}_mode"].value)
            value_i, mode_i, constraint_i, min_i, max_i = _primary_min_time_interactivity(
                default_mode,
                enabled=default_enabled,
            )
            fit_comps[f"{_PRIMARY_MIN_TIME_PREFIX}_value"].interactive = value_i
            fit_comps[f"{_PRIMARY_MIN_TIME_PREFIX}_mode"].interactive = mode_i
            fit_comps[f"{_PRIMARY_MIN_TIME_PREFIX}_constraint"].interactive = constraint_i
            fit_comps[f"{_PRIMARY_MIN_TIME_PREFIX}_min"].interactive = min_i
            fit_comps[f"{_PRIMARY_MIN_TIME_PREFIX}_max"].interactive = max_i
            primary_label_base = param_inputs.get_label_for_prefix(_PRIMARY_MIN_TIME_PREFIX)
            if primary_label_base is not None:
                fit_comps[primary_min_time_label_key].value = param_inputs.style_param_label(
                    primary_label_base,
                    disabled=not default_enabled,
                )

        # Unified LSQRT - one parameter form, approach-aware serialization.
        def _unified_lsqrt(*values: object) -> tuple:
            n_lc = len(lc_data_inputs)
            approach = str(values[-1])
            normalized_approach = _normalize_ui_approach(approach)

            x_unit_str, lc_rows, morphology, _ = _parse_lc_inputs_simple(
                values[:n_lc],
                MAX_PASSBAND_ROWS,
            )
            fit_vals = collect_param_values(fit_keys, values, n_lc)

            # Load data here so the task is constructed with real LCData objects
            # (mirroring the RV fitting pattern where the tab owns the task instance).
            try:
                lc_data = compute.load_lc_dataset(lc_rows, x_unit_str)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

            task_obj = LCBinaryAnalyticsTask(data=lc_data, method="least_squares", expected_morphology=morphology)

            try:
                _, fig, df, json_path = compute.run_lsqrt(
                    lc_rows,
                    x_unit_str,
                    fit_vals,
                    morphology,
                    approach=normalized_approach,
                    task=task_obj,
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

            return (
                gr.update(selected=_TAB_LSQRT),
                {"lsqrt": task_obj},
                fig,
                df,
                gr.update(value=json_path, visible=True),
            )

        def _extract_and_validate_initial_state(
            chain_state_dict: dict | None,
            nwalkers: Int,
        ) -> object | None:
            """Extract and validate initial MCMC state from a loaded chain.

            :param chain_state_dict: State dict with ``chain_task`` and ``use_initial_state`` keys.
            :type chain_state_dict: dict | None
            :param nwalkers: Number of walkers for the next MCMC run.
            :type nwalkers: elisa.types.Int
            :returns: Initial state array or ``None`` when reuse is not enabled.
            :rtype: object | None
            :raises gr.Error: If extraction fails or shape is invalid.
            """
            if not chain_state_dict or not isinstance(chain_state_dict, dict):
                return None

            use_initial_state = chain_state_dict.get("use_initial_state", False)
            chain_task = chain_state_dict.get("chain_task")
            if not use_initial_state or chain_task is None:
                return None

            try:
                initial_state: NDArray = compute.extract_initial_state_from_chain(
                    task=chain_task,
                    nwalkers=nwalkers,
                )
            except gr.Error:
                raise
            except Exception as exc:
                msg = f"Failed to extract initial state from loaded chain: {exc}"
                raise gr.Error(msg) from exc

            expected_n_params = len(chain_task.fit_cls.variable_labels)
            if initial_state.shape != (nwalkers, expected_n_params):
                msg = (
                    f"Invalid shape for initial MCMC state: expected ({nwalkers}, {expected_n_params}), "
                    f"got {initial_state.shape}."
                )
                raise gr.Error(msg)

            return initial_state

        # Unified MCMC - same approach
        def _unified_mcmc(
            *values: object,
            chain_state_dict: dict | None = None,
        ) -> tuple:
            n_lc = len(lc_data_inputs)
            n_fit = len(fit_keys)
            approach = str(values[-1])
            normalized_approach = _normalize_ui_approach(approach)

            x_unit_str, lc_rows, morphology, _ = _parse_lc_inputs_simple(
                values[:n_lc],
                MAX_PASSBAND_ROWS,
            )
            fit_vals = collect_param_values(fit_keys, values, n_lc)
            mcmc_vals = collect_param_values(mcmc_keys, values, n_lc + n_fit)

            nwalkers = int(mcmc_vals.get("nwalkers") or 100)
            nsteps = int(mcmc_vals.get("nsteps") or 100)
            burn_in = int(mcmc_vals.get("burn_in") or 20)
            fit_id = str(mcmc_vals.get("fit_id") or "mcmc_lc_fit")
            save = bool(mcmc_vals.get("save_chain", True))
            progress = bool(mcmc_vals.get("progress", True))
            initial_state = _extract_and_validate_initial_state(chain_state_dict, nwalkers)

            try:
                _result, model_fig, corner_fig, traces_fig, df, json_path = compute.run_mcmc(
                    lc_rows,
                    x_unit_str,
                    fit_vals,
                    morphology,
                    approach=normalized_approach,
                    nwalkers=nwalkers,
                    nsteps=nsteps,
                    burn_in=burn_in,
                    fit_id=fit_id,
                    save=save,
                    progress=progress,
                    initial_state=initial_state,
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
            return (
                gr.update(selected=_TAB_MCMC),
                model_fig,
                corner_fig,
                traces_fig,
                df,
                gr.update(value=json_path, visible=True),
            )

        def _load_chain_handler(
            chain_file: object | None,
            result_file: object | None,
            discard: float,
            x_unit: str,
            current_approach: str,
            *,
            use_initial_state: bool = False,
        ) -> Any:
            """Load an LC MCMC chain, generate diagnostics, and store reuse state.

            :param chain_file: Gradio file object for the chain JSON.
            :type chain_file: object | None
            :param result_file: Gradio file object for the result JSON.
            :type result_file: object | None
            :param discard: Number of burn-in samples to discard per walker.
            :type discard: float
            :param use_initial_state: Whether to reuse loaded samples as initial state.
            :type use_initial_state: bool
            :returns: Tuple of state, status, corner figure, traces figure, and results dataframe.
            :rtype: typing.Any
            """
            chain_path: str | None = getattr(chain_file, "name", None)
            result_path: str | None = getattr(result_file, "name", None)
            if chain_path is None or result_path is None:
                msg = "Both chain file and result file are required."
                raise gr.Error(msg)

            discard_int = int(discard) if discard else 0

            try:
                task, corner_fig, traces_fig, results_df = compute.load_chain(
                    chain_path,
                    result_path,
                    discard=discard_int,
                )
            except gr.Error:
                raise
            except Exception as exc:
                msg = f"Failed to load chain: {exc}"
                raise gr.Error(msg) from exc

            new_state: dict[str, object] = {
                "chain_task": task,
                "use_initial_state": use_initial_state,
            }

            if use_initial_state:
                status_msg = (
                    "✓ Chain loaded successfully. Corner and traces plots generated. "
                    "Initial state will be extracted when you run MCMC."
                )
            else:
                status_msg = (
                    "✓ Chain loaded successfully. Corner and traces plots generated. "
                    "Initial state reuse is disabled."
                )

            try:
                detected_approach = compute.detect_approach_from_json(result_path)
            except ValueError as exc:
                detected_approach = current_approach
                msg = f"Could not detect approach from result file, using current selection: {exc}"
                gr.Warning(msg)

            try:
                params = compute.load_params_from_json(result_path)
                normalized_approach = _normalize_ui_approach(detected_approach)
                param_updates = [
                    _build_active_param_update(k, params, normalized_approach, x_unit)
                    for k in fit_keys
                ]

                label_updates: list[object] = []
                for label_key in approach_toggle_label_keys:
                    prefix = label_key.removesuffix("_label")
                    base_label = param_inputs.get_label_for_prefix(prefix)
                    if base_label is None:
                        label_updates.append(gr.update())
                        continue
                    interactive = param_inputs.is_key_interactive_for_approach(
                        f"{prefix}_value",
                        normalized_approach,
                    )
                    styled = param_inputs.style_param_label(base_label, disabled=not interactive)
                    label_updates.append(gr.update(value=styled))

                primary_min_time_label_key_ = f"{_PRIMARY_MIN_TIME_PREFIX}_label"
                if primary_min_time_label_key_ in approach_toggle_label_keys:
                    primary_min_time_label_index = approach_toggle_label_keys.index(primary_min_time_label_key_)
                    label_updates[primary_min_time_label_index] = _primary_min_time_label_update(
                        disabled=not _is_jd_x_unit(x_unit),
                    )

                approach_update = gr.update(value=detected_approach)
            except ValueError as exc:
                msg = f"Could not extract parameters from result file: {exc}"
                gr.Warning(msg)
                approach_update = gr.update()
                param_updates = [gr.update() for _ in fit_keys]
                label_updates = [gr.update() for _ in approach_toggle_label_keys]

            return (
                new_state,
                status_msg,
                corner_fig,
                traces_fig,
                results_df,
                approach_update,
                *param_updates,
                *label_updates,
            )

        transfer_value_keys = (
            [f"system_{name}_value" for name in SYSTEM_REGULAR_PARAMS]
            + ["system_semi_major_axis_value", "system_mass_ratio_value"]
            + ["primary_mass_value"]
            + [f"primary_{n}_value" for n in COMPONENT_PARAMS]
            + ["secondary_mass_value"]
            + [f"secondary_{n}_value" for n in COMPONENT_PARAMS]
            + ["nuisance_ln_f_value"]
        )
        transfer_value_outputs = [fit_comps[k] for k in transfer_value_keys]

        def _transfer_values(state: dict | None, approach: str) -> list[object]:
            if not state:
                msg = "No LSQRT result available yet - run Least Squares first."
                raise gr.Error(msg)
            task = state.get("lsqrt") if isinstance(state, dict) else None
            if task is None:
                msg = "No LSQRT task stored in session state - run Least Squares first."
                raise gr.Error(msg)
            try:
                result = task.get_result()
            except Exception as exc:
                msg = f"Failed to extract result from stored task: {exc}"
                raise gr.Error(msg) from exc

            values = compute.extract_values_for_transfer(result)
            normalized = _normalize_ui_approach(approach)

            updates: list[object] = []
            for key in transfer_value_keys:
                if key in values and param_inputs.is_key_interactive_for_approach(key, normalized):
                    updates.append(gr.update(value=values[key]))
                else:
                    updates.append(gr.update())
            return updates

        mcmc_inputs_list: list[gr.Component] = [
            nwalkers_comp,
            nsteps_comp,
            burn_in_comp,
            fit_id_comp,
            save_chain_comp,
            progress_comp,
        ]

        unified_lsqrt_inputs = lc_data_inputs + fit_inputs + [approach_comp]
        unified_mcmc_inputs = lc_data_inputs + fit_inputs + mcmc_inputs_list + [approach_comp]

        lsqrt_btn.click(
            fn=_unified_lsqrt,
            inputs=unified_lsqrt_inputs,
            outputs=[results_tabs, tasks_state, lsqrt_model_plot, lsqrt_table, lsqrt_download],
            show_progress="hidden",
            show_progress_on=[],
        )

        # Observed-data plotting handler - plot uploaded LC files (does not run fit)
        def _plot_loaded_lc_handler(  # noqa: C901, PLR0912, PLR0915
            *values: SupportsIndex,
        ) -> tuple[object, dict]:
            # values correspond to lc_data_inputs order: x_unit, passband_count, then per-row comps
            x_unit_str = str(values[0])
            passband_count = int(values[1])

            # collect active row data into list of dicts (TypedDict)
            lc_rows: list[dict] = []
            offset = 2
            for i in range(MAX_PASSBAND_ROWS):
                if i >= passband_count:
                    break
                base = offset + i * 4
                pb = str(values[base])
                file_obj = values[base + 1]
                yu = str(values[base + 2])
                rm_raw = values[base + 3]
                rm = float(rm_raw) if rm_raw is not None else None
                lc_rows.append(
                    {
                        "passband": pb,
                        "file_path": getattr(file_obj, "name", None),
                        "y_unit": yu,
                        "reference_magnitude": rm,
                    },
                )

            # Tail inputs start immediately after the lc_data inputs.
            n_lc = len(lc_data_inputs)
            tail = values[n_lc : n_lc + 8]
            period_val = tail[0]
            fit_period_val = tail[1]
            fit_t0_val = tail[2]
            t0_val = tail[3]
            start_val = tail[4]
            stop_val = tail[5]
            centre_val = tail[6]
            lsqrt_state_val = tail[7]

            if not lc_rows:
                msg = "No light curve files uploaded"
                raise gr.Error(msg)

            # Choose ephemeris values - prefer data-section period, then form period, then LSQRT result
            def _extract_from_lsqr(key: str) -> object | None:  # noqa: PLR0911
                if not isinstance(lsqrt_state_val, dict):
                    return None
                task = lsqrt_state_val.get("lsqrt")
                if task is None:
                    return None
                # noinspection PyBroadException
                try:
                    result = task.get_result()
                except Exception:  # noqa: BLE001
                    return None
                if not isinstance(result, dict):
                    return None
                system = result.get("system")
                if not isinstance(system, dict):
                    return None
                meta = system.get(key)
                if not isinstance(meta, dict):
                    return None
                return meta.get("value")

            # final fallback: use parameter defaults from the LC param spec
            _lc_spec = param_inputs.SYSTEM_COMMON_SPEC
            param_default_period = _lc_spec.get("period", (None, None))[1]
            param_default_t0 = _lc_spec.get("primary_minimum_time", (None, None))[1]

            chosen_period = (
                period_val
                if period_val is not None
                else fit_period_val
                if fit_period_val is not None
                else _extract_from_lsqr("period") or param_default_period
            )

            chosen_t0 = (
                t0_val
                if t0_val is not None
                else fit_t0_val
                if fit_t0_val is not None
                else _extract_from_lsqr("primary_minimum_time") or param_default_t0
            )

            use_phases = x_unit_str == "Julian days (JD)"
            if use_phases and chosen_period is None:
                msg = "Orbital period is required to phase JD data."
                raise gr.Error(msg)
            if use_phases and chosen_t0 is None:
                msg = "Primary minimum time (T0) is required to phase JD data."
                raise gr.Error(msg)

            # Build figure and plot all uploaded passbands
            fig, ax = plt.subplots(figsize=(10, 4))
            for row in lc_rows:
                fp = row.get("file_path")
                pb = row.get("passband")
                if fp is None:
                    # skip empty rows
                    continue
                try:
                    lc = compute.load_lc_data(fp, pb, x_unit_str, row.get("y_unit"), row.get("reference_magnitude"))
                except Exception as exc:
                    msg = f"Failed to load LC file for passband {pb}: {exc}"
                    raise gr.Error(msg) from exc

                if use_phases:
                    centre_arg = float(centre_val) if centre_val is not None else 0.0
                    try:
                        lc.convert_to_phases(float(chosen_period), float(chosen_t0), centre=centre_arg)
                    except Exception as exc:
                        msg = f"Failed to phase LC data for passband {pb}: {exc}"
                        raise gr.Error(msg) from exc

                x = np.asarray(lc.x_data)
                y = np.asarray(lc.y_data)
                yerr = None if lc.y_err is None else np.asarray(lc.y_err)

                if use_phases and start_val is not None and stop_val is not None:
                    sp = float(start_val)
                    ep = float(stop_val)
                    if sp >= ep:
                        msg = "Start phase must be less than stop phase"
                        raise gr.Error(msg)
                    x_out, y_out, yerr_out = extend_observations_to_desired_interval(
                        sp,
                        ep,
                        {pb: x},
                        {pb: y},
                        {pb: yerr},
                    )
                    px = np.asarray(x_out.get(pb, np.empty(0, dtype=float)))
                    py = np.asarray(y_out.get(pb, np.empty(0, dtype=float)))
                    perr = yerr_out.get(pb) if isinstance(yerr_out, dict) else None
                    if perr is not None:
                        perr = np.asarray(perr)
                    if px.size:
                        order = np.argsort(px)
                        px = px[order]
                        py = py[order]
                        perr = None if perr is None else perr[order]
                else:
                    px = x
                    py = y
                    perr = yerr

                if perr is not None:
                    ax.errorbar(px, py, yerr=perr, fmt="o", markersize=4, label=pb)
                else:
                    ax.plot(px, py, "o", markersize=4, label=pb)

            ax.set_xlabel("Phase" if use_phases else "Time")
            ax.set_ylabel("Flux")
            ax.grid(visible=True)
            ax.legend(loc="best")
            fig.tight_layout()
            return figure_to_pil(fig), gr.update(open=True)

        # Wire observed-data plot button: only the data inputs and the handful of
        # tail values are actually read by the handler.  The full community/standard
        # fit_inputs (708 components) were previously included as positional padding
        # which caused Gradio to register 750+ components for loading-stati tracking,
        # firing find_node_by_id ~750 times on every click.
        tail_inputs = [
            period_comp,
            fit_comps.get("system_period_value"),
            fit_comps.get("system_primary_minimum_time_value"),
            t0_comp,
            start_phase_comp,
            stop_phase_comp,
            centre_comp,
            tasks_state,
        ]
        plot_obs_btn.click(
            fn=_plot_loaded_lc_handler,
            inputs=[*lc_data_inputs, *tail_inputs],
            outputs=[observed_data_plot, obs_plot_accordion],
            show_progress="hidden",
            show_progress_on=[],
        )

        transfer_btn.click(
            fn=_transfer_values,
            inputs=[tasks_state, approach_comp],
            outputs=transfer_value_outputs,
            show_progress="hidden",
            show_progress_on=[],
        )

        load_chain_btn.click(
            fn=lambda cf, rf, d, uis, xu, app: _load_chain_handler(
                cf,
                rf,
                d,
                xu,
                app,
                use_initial_state=uis,
            ),
            inputs=[
                chain_file_comp,
                result_file_comp,
                discard_comp,
                use_initial_state_comp,
                data_comps.x_unit,
                approach_comp,
            ],
            outputs=[
                chain_state,
                chain_load_status,
                corner_plot,
                traces_plot,
                mcmc_table,
                approach_comp,
                *[fit_comps[k] for k in fit_keys],
                *approach_toggle_label_outputs,
            ],
            show_progress="hidden",
            show_progress_on=[],
        )

        mcmc_btn.click(
            fn=lambda cstate, *vals: _unified_mcmc(*vals, chain_state_dict=cstate),
            inputs=[chain_state, *unified_mcmc_inputs],
            outputs=[
                results_tabs,
                mcmc_model_plot,
                corner_plot,
                traces_plot,
                mcmc_table,
                mcmc_download,
            ],
            show_progress="hidden",
            show_progress_on=[],
        )
