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

from typing import TYPE_CHECKING, SupportsIndex

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure  # noqa: TC002

from elisa.analytics.binary_fit.shared import extend_observations_to_desired_interval
from elisa.ui.tabs.lc_fitting.components import data_inputs, param_inputs
from elisa.ui.tabs.lc_fitting.components.data_inputs import MAX_PASSBAND_ROWS
from elisa.ui.tabs.lc_fitting.components.param_inputs import (
    COMPONENT_PARAMS,
    FIELD_ORDER_COMMUNITY,
    FIELD_ORDER_STANDARD,
    SYSTEM_REGULAR_PARAMS,
)
from elisa.ui.tabs.lc_fitting.logic import compute

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Tab ID constants
# ---------------------------------------------------------------------------

_TAB_LSQRT = "lc_lsqrt_results"
_TAB_MCMC = "lc_mcmc_results"


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


def _collect_param_values(
    param_keys: tuple[str, ...],
    values: tuple[object, ...],
    offset: int,
) -> dict[str, object]:
    """Slice *values* starting at *offset* into a named dict.

    :param param_keys: Ordered key names.
    :type param_keys: tuple[str, ...]
    :param values: Full flat value tuple from Gradio.
    :type values: tuple[object, ...]
    :param offset: Start index within *values*.
    :type offset: int
    :returns: Dict mapping each key to its corresponding value.
    :rtype: dict[str, object]
    """
    return dict(zip(param_keys, values[offset : offset + len(param_keys)], strict=True))


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


def _build_lsqrt_results() -> tuple[gr.Plot, gr.DataFrame, gr.DownloadButton]:
    """Render the LSQRT results sub-tab and return output components.

    :returns: Tuple of ``(model_plot, table, download_button)``.
    :rtype: tuple[gr.Plot, gr.DataFrame, gr.DownloadButton]
    """
    with gr.Row():
        model_plot = gr.Plot(label="Model fit")
    with gr.Row():
        table = gr.DataFrame(label="Fitted parameters", wrap=True)
    with gr.Row():
        download = gr.DownloadButton(label="💾 Download result JSON", visible=False)
    return model_plot, table, download


def _build_mcmc_results() -> tuple[gr.Plot, gr.DataFrame, gr.Plot, gr.Plot, gr.DownloadButton]:
    """Render the MCMC results sub-tab and return output components.

    :returns: Tuple of
        ``(model_plot, table, corner_plot, traces_plot, download_button)``.
    :rtype: tuple[gr.Plot, gr.DataFrame, gr.Plot, gr.Plot, gr.DownloadButton]
    """
    with gr.Row():
        model_plot = gr.Plot(label="Model fit (MCMC median)")
    with gr.Row():
        table = gr.DataFrame(label="Fitted parameters", wrap=True)
    with gr.Row():
        corner_plot = gr.Plot(label="Corner plot")
    with gr.Row():
        traces_plot = gr.Plot(label="Parameter traces")
    with gr.Row():
        download = gr.DownloadButton(label="💾 Download result JSON", visible=False)
    return model_plot, table, corner_plot, traces_plot, download


# ---------------------------------------------------------------------------
# Event wiring helpers
# ---------------------------------------------------------------------------


def _wire_json_loader(  # noqa: C901
    params_json_comp: gr.File,
    community_fit_comps: dict[str, gr.Component],
    standard_fit_comps: dict[str, gr.Component],
    community_fit_keys: tuple[str, ...],
    standard_fit_keys: tuple[str, ...],
    approach_comp: gr.Radio,
    community_group: gr.Column,
    standard_group: gr.Column,
) -> None:
    """Wire the JSON param-loader upload event with auto-detection of approach.

    Detects Community vs Standard from the uploaded JSON using
    :func:`~logic.compute.detect_approach_from_json`, switches the approach
    radio and section visibility, then populates the correct parameter form.

    :param params_json_comp: The file upload component for the result JSON.
    :type params_json_comp: gr.File
    :param community_fit_comps: Community component mapping.
    :type community_fit_comps: dict[str, gr.Component]
    :param standard_fit_comps: Standard component mapping.
    :type standard_fit_comps: dict[str, gr.Component]
    :param community_fit_keys: Ordered param keys for Community approach.
    :type community_fit_keys: tuple[str, ...]
    :param standard_fit_keys: Ordered param keys for Standard approach.
    :type standard_fit_keys: tuple[str, ...]
    :param approach_comp: The approach radio button component.
    :type approach_comp: gr.Radio
    :param community_group: The community parameters Column component.
    :type community_group: gr.Column
    :param standard_group: The standard parameters Column component.
    :type standard_group: gr.Column
    """
    all_outputs: list[gr.Component] = (
        [approach_comp, community_group, standard_group]
        + [community_fit_comps[k] for k in community_fit_keys]
        + [standard_fit_comps[k] for k in standard_fit_keys]
    )

    def _load_json_handler(json_file: object) -> list[object]:
        """Populate the parameter form from an uploaded LC result JSON.

        Auto-detects Community vs Standard, switches the approach selector
        and section visibility, then loads parameters into the correct form.

        :param json_file: Gradio file object from the upload component.
        :type json_file: object
        :returns: Updates for approach_comp, both groups, and all param components.
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

        is_community = approach == "Community"
        fit_keys = community_fit_keys if is_community else standard_fit_keys

        try:
            params = compute.load_params_from_json(path)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        # Build updates for the active approach's parameters.
        # For _constraint keys: set interactive=True only when mode=="constrained".
        # For _min / _max keys: set interactive=True only when mode=="free".
        # For all others (_value, _mode): set value if present.
        active_updates: list[object] = []
        for key in fit_keys:
            if key.endswith("_constraint"):
                base_prefix = key[: -len("_constraint")]
                mode = str(params.get(f"{base_prefix}_mode", "free"))
                val = params.get(key)
                upd: dict[str, object] = {"interactive": mode == "constrained"}
                if val is not None:
                    upd["value"] = val
                active_updates.append(gr.update(**upd))
            elif key.endswith(("_min", "_max")):
                base_prefix = key.rsplit("_", 1)[0]
                mode = str(params.get(f"{base_prefix}_mode", "free"))
                is_interactive = mode == "free"
                val = params.get(key)
                upd = {"interactive": is_interactive}
                if val is not None:
                    upd["value"] = val
                active_updates.append(gr.update(**upd))
            elif key in params:
                active_updates.append(gr.update(value=params[key]))
            else:
                active_updates.append(gr.update())

        # Inactive approach gets empty updates
        inactive_keys = standard_fit_keys if is_community else community_fit_keys
        inactive_updates = [gr.update() for _ in inactive_keys]

        community_updates = active_updates if is_community else inactive_updates
        standard_updates = inactive_updates if is_community else active_updates

        return [
            gr.update(value=approach),
            gr.update(visible=is_community),
            gr.update(visible=not is_community),
            *community_updates,
            *standard_updates,
        ]

    params_json_comp.upload(
        fn=_load_json_handler,
        inputs=[params_json_comp],
        outputs=all_outputs,
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
    gr.Plot,
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
        )

        # Collapsible plot area for observed light curves
        with gr.Accordion("Observed data plot", open=False) as obs_plot_accordion:
            observed_data_plot = gr.Plot(label="Observed data")

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
                value=120,
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
    gr.Plot,
    gr.DataFrame,
    gr.DownloadButton,
    gr.Plot,
    gr.DataFrame,
    gr.Plot,
    gr.Plot,
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

        lsqrt_result_state: gr.State = gr.State()
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

            # Build Community parameters
            with gr.Column(visible=True) as community_group:
                community_fit_comps, _community_sections = param_inputs.build(approach="community")

            # Build Standard parameters
            with gr.Column(visible=False) as standard_group:
                standard_fit_comps, _standard_sections = param_inputs.build(approach="standard")

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

        # Wire approach selector to toggle visibility
        approach_comp.change(
            fn=lambda a: (gr.update(visible=a == "Community"), gr.update(visible=a == "Standard")),
            inputs=[approach_comp],
            outputs=[community_group, standard_group],
        )

        # Setup for Community approach
        community_fit_keys = FIELD_ORDER_COMMUNITY
        community_fit_inputs = [community_fit_comps[k] for k in community_fit_keys]
        community_lc_data_inputs = (
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

        # Setup for Standard approach
        standard_fit_keys = FIELD_ORDER_STANDARD
        standard_fit_inputs = [standard_fit_comps[k] for k in standard_fit_keys]
        # Note: lc_data_inputs are identical for both approaches - reuse community_lc_data_inputs

        mcmc_keys: tuple[str, ...] = ("nwalkers", "nsteps", "burn_in", "fit_id", "save_chain", "progress")

        pb_outputs = [data_comps.passband_count, *data_comps.row_groups]
        data_comps.add_btn.click(fn=_add_passband, inputs=[data_comps.passband_count], outputs=pb_outputs)  # type: ignore[union-attr]
        data_comps.remove_btn.click(fn=_remove_passband, inputs=[data_comps.passband_count], outputs=pb_outputs)  # type: ignore[union-attr]

        _wire_json_loader(
            params_json_comp,
            community_fit_comps,
            standard_fit_comps,
            community_fit_keys,
            standard_fit_keys,
            approach_comp,
            community_group,
            standard_group,
        )

        # Unified LSQRT - routes to correct approach based on selection.
        # Inputs: lc_data + community_fit_inputs + standard_fit_inputs + approach_comp
        # (lc_data is identical for both approaches - same components)
        def _unified_lsqrt(*values: object) -> tuple:
            n_lc = len(community_lc_data_inputs)
            n_cf = len(community_fit_keys)
            approach = str(values[-1])
            is_community = approach == "Community"

            x_unit_str, lc_rows, morphology, _ = _parse_lc_inputs_simple(
                values[:n_lc],
                MAX_PASSBAND_ROWS,
            )
            if is_community:
                fit_vals = _collect_param_values(community_fit_keys, values, n_lc)
            else:
                fit_vals = _collect_param_values(standard_fit_keys, values, n_lc + n_cf)

            try:
                result, fig, df, json_path = compute.run_lsqrt(
                    lc_rows,
                    x_unit_str,
                    fit_vals,
                    morphology,
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
            return result, fig, df, gr.DownloadButton(value=json_path, visible=True)

        # Unified MCMC - same approach
        def _unified_mcmc(*values: object) -> tuple:
            n_lc = len(community_lc_data_inputs)
            n_cf = len(community_fit_keys)
            n_sf = len(standard_fit_keys)
            approach = str(values[-1])
            is_community = approach == "Community"

            x_unit_str, lc_rows, morphology, _ = _parse_lc_inputs_simple(
                values[:n_lc],
                MAX_PASSBAND_ROWS,
            )
            if is_community:
                fit_vals = _collect_param_values(community_fit_keys, values, n_lc)
                mcmc_vals = _collect_param_values(mcmc_keys, values, n_lc + n_cf)
            else:
                fit_vals = _collect_param_values(standard_fit_keys, values, n_lc + n_cf)
                mcmc_vals = _collect_param_values(mcmc_keys, values, n_lc + n_cf + n_sf)

            nwalkers = int(mcmc_vals.get("nwalkers") or 50)
            nsteps = int(mcmc_vals.get("nsteps") or 500)
            burn_in = int(mcmc_vals.get("burn_in") or 50)
            fit_id = str(mcmc_vals.get("fit_id") or "mcmc_lc_fit")
            save = bool(mcmc_vals.get("save_chain", True))
            progress = bool(mcmc_vals.get("progress", True))

            try:
                result, model_fig, corner_fig, traces_fig, df, json_path = compute.run_mcmc(
                    lc_rows,
                    x_unit_str,
                    fit_vals,
                    morphology,
                    nwalkers=nwalkers,
                    nsteps=nsteps,
                    burn_in=burn_in,
                    fit_id=fit_id,
                    save=save,
                    progress=progress,
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
            return result, model_fig, corner_fig, traces_fig, df, gr.DownloadButton(value=json_path, visible=True)

        # Create transfer handlers for each approach
        def _make_community_transfer() -> Callable[[dict | None, str], list[object]]:
            def _transfer(result: dict | None, approach: str) -> list[object]:
                if approach != "Community":
                    return [gr.update() for _ in community_value_outputs]
                if result is None:
                    msg = "No LSQRT result available yet - run Least Squares first."
                    raise gr.Error(msg)
                values = compute.extract_values_for_transfer(result)

                def _upd(key: str) -> object:
                    return gr.update(value=values[key]) if key in values else gr.update()

                return (
                    [_upd(f"system_{name}_value") for name in SYSTEM_REGULAR_PARAMS]
                    + [_upd("system_semi_major_axis_value")]
                    + [_upd("system_mass_ratio_value")]
                    + [_upd(f"primary_{n}_value") for n in COMPONENT_PARAMS]
                    + [_upd(f"secondary_{n}_value") for n in COMPONENT_PARAMS]
                    + [_upd("nuisance_ln_f_value")]
                )

            return _transfer

        def _make_standard_transfer() -> Callable[[dict | None, str], list[object]]:
            def _transfer(result: dict | None, approach: str) -> list[object]:
                if approach != "Standard":
                    return [gr.update() for _ in standard_value_outputs]
                if result is None:
                    msg = "No LSQRT result available yet - run Least Squares first."
                    raise gr.Error(msg)
                values = compute.extract_values_for_transfer(result)

                def _upd(key: str) -> object:
                    return gr.update(value=values[key]) if key in values else gr.update()

                return (
                    [_upd(f"system_{name}_value") for name in SYSTEM_REGULAR_PARAMS]
                    + [_upd("primary_mass_value")]
                    + [_upd(f"primary_{n}_value") for n in COMPONENT_PARAMS if n != "mass"]
                    + [_upd("secondary_mass_value")]
                    + [_upd(f"secondary_{n}_value") for n in COMPONENT_PARAMS if n != "mass"]
                    + [_upd("nuisance_ln_f_value")]
                )

            return _transfer

        community_value_outputs = (
            [community_fit_comps[f"system_{n}_value"] for n in SYSTEM_REGULAR_PARAMS]
            + [community_fit_comps["system_semi_major_axis_value"]]
            + [community_fit_comps["system_mass_ratio_value"]]
            + [community_fit_comps[f"primary_{n}_value"] for n in COMPONENT_PARAMS]
            + [community_fit_comps[f"secondary_{n}_value"] for n in COMPONENT_PARAMS]
            + [community_fit_comps["nuisance_ln_f_value"]]
        )

        standard_value_outputs = (
            [standard_fit_comps[f"system_{n}_value"] for n in SYSTEM_REGULAR_PARAMS]
            + [standard_fit_comps["primary_mass_value"]]
            + [standard_fit_comps[f"primary_{n}_value"] for n in COMPONENT_PARAMS if n != "mass"]
            + [standard_fit_comps["secondary_mass_value"]]
            + [standard_fit_comps[f"secondary_{n}_value"] for n in COMPONENT_PARAMS if n != "mass"]
            + [standard_fit_comps["nuisance_ln_f_value"]]
        )

        mcmc_inputs_list: list[gr.Component] = [
            nwalkers_comp,
            nsteps_comp,
            burn_in_comp,
            fit_id_comp,
            save_chain_comp,
            progress_comp,
        ]

        # Single combined input list for both unified handlers:
        # lc_data + community_fit + standard_fit + [approach]
        # lc_data is identical for both approaches (same component refs)
        unified_lsqrt_inputs = community_lc_data_inputs + community_fit_inputs + standard_fit_inputs + [approach_comp]
        unified_mcmc_inputs = (
            community_lc_data_inputs + community_fit_inputs + standard_fit_inputs + mcmc_inputs_list + [approach_comp]
        )

        lsqrt_btn.click(
            fn=lambda: gr.update(selected=_TAB_LSQRT),
            outputs=[results_tabs],
        ).then(
            fn=_unified_lsqrt,
            inputs=unified_lsqrt_inputs,
            outputs=[lsqrt_result_state, lsqrt_model_plot, lsqrt_table, lsqrt_download],
            show_progress="full",
        )

        # Observed-data plotting handler - plot uploaded LC files (does not run fit)
        def _plot_loaded_lc_handler(  # noqa: C901, PLR0912, PLR0915
            *values: SupportsIndex,
        ) -> tuple[Figure, dict]:
            # values correspond to community_lc_data_inputs order: x_unit, passband_count, then per-row comps
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

            # Determine indices for tail inputs (period/t0/etc). The prefix length is the number of
            # lc_data inputs + community and standard fit inputs + approach radio.
            prefix_len = len(community_lc_data_inputs) + len(community_fit_inputs) + len(standard_fit_inputs) + 1
            tail = values[prefix_len : prefix_len + 8]
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
            def _extract_from_lsqr(key: str) -> object | None:
                if not isinstance(lsqrt_state_val, dict):
                    return None
                system = lsqrt_state_val.get("system")
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
                        sp, ep, {pb: x}, {pb: y}, {pb: yerr},
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
            return fig, gr.update(open=True)

        # Wire observed-data plot button: unified inputs share the same lc_data inputs prefix
        plot_inputs = [*community_lc_data_inputs, *community_fit_inputs, *standard_fit_inputs, approach_comp]
        tail_inputs = [
            period_comp,
            community_fit_comps.get("system_period_value"),
            community_fit_comps.get("system_primary_minimum_time_value"),
            t0_comp,
            start_phase_comp,
            stop_phase_comp,
            centre_comp,
            lsqrt_result_state,
        ]
        plot_obs_btn.click(
            fn=_plot_loaded_lc_handler,
            inputs=[*plot_inputs, *tail_inputs],
            outputs=[observed_data_plot, obs_plot_accordion],
        )

        transfer_btn.click(
            fn=_make_community_transfer(),
            inputs=[lsqrt_result_state, approach_comp],
            outputs=community_value_outputs,
        ).then(
            fn=_make_standard_transfer(),
            inputs=[lsqrt_result_state, approach_comp],
            outputs=standard_value_outputs,
        )

        mcmc_btn.click(
            fn=lambda: gr.update(selected=_TAB_MCMC),
            outputs=[results_tabs],
        ).then(
            fn=_unified_mcmc,
            inputs=unified_mcmc_inputs,
            outputs=[
                gr.State(),
                mcmc_model_plot, corner_plot, traces_plot, mcmc_table, mcmc_download,
            ],
            show_progress="full",
        )

