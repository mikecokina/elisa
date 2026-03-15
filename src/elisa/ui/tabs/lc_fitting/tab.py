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

from typing import TYPE_CHECKING, cast

import gradio as gr
import pandas as pd  # noqa: TC002 - needed at runtime for Gradio type hints
from matplotlib.figure import Figure  # noqa: TC002 - same reason

from elisa.ui.tabs.lc_fitting.components import data_inputs, param_inputs
from elisa.ui.tabs.lc_fitting.components.data_inputs import MAX_PASSBAND_ROWS
from elisa.ui.tabs.lc_fitting.components.param_inputs import (
    COMPONENT_PARAMS,
    FIELD_ORDER,
    SYSTEM_PARAMS,
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


def _parse_lc_inputs(
    values: tuple[object, ...],
    n_rows: int,
) -> tuple[str, list[compute.LCRowData], str, int]:
    """Parse the flat input tuple from a fit handler into structured data.

    Expects inputs in the order laid out by :func:`build`:
    ``[x_unit, passband_count, row_0_pb, row_0_file, row_0_yu, row_0_rm, ...]``
    followed by ``morphology``.

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


def _lsqrt_handler(
    n_rows: int,
    fit_keys: tuple[str, ...],
) -> Callable[..., tuple[dict, Figure, pd.DataFrame, gr.DownloadButton]]:
    """Return the LSQRT Gradio event-handler.

    :param n_rows: Total number of pre-rendered passband rows.
    :type n_rows: int
    :param fit_keys: Ordered parameter keys (from :data:`FIELD_ORDER`).
    :type fit_keys: tuple[str, ...]
    :returns: Gradio-compatible handler.
    :rtype: Callable[..., tuple[dict, Figure, pandas.DataFrame, gr.DownloadButton]]
    """
    def handler(
        *values: object,
    ) -> tuple[dict, Figure, pd.DataFrame, gr.DownloadButton]:
        x_unit_str, lc_rows, morphology, fit_offset = _parse_lc_inputs(values, n_rows)
        fit_vals = _collect_param_values(fit_keys, values, fit_offset)
        try:
            result, fig, df, json_path = compute.run_lsqrt(
                lc_rows, x_unit_str, fit_vals, morphology,
            )
        except Exception as exc:
            msg = str(exc)
            raise gr.Error(msg) from exc
        return result, fig, df, gr.DownloadButton(value=json_path, visible=True)

    return handler


def _mcmc_handler(
    n_rows: int,
    fit_keys: tuple[str, ...],
    mcmc_keys: tuple[str, ...],
) -> Callable[..., tuple[dict, Figure, Figure | None, Figure | None, pd.DataFrame, gr.DownloadButton]]:
    """Return the MCMC Gradio event-handler.

    :param n_rows: Total number of pre-rendered passband rows.
    :type n_rows: int
    :param fit_keys: Ordered parameter keys (from :data:`FIELD_ORDER`).
    :type fit_keys: tuple[str, ...]
    :param mcmc_keys: Ordered MCMC control keys.
    :type mcmc_keys: tuple[str, ...]
    :returns: Gradio-compatible handler.
    :rtype: Callable
    """
    def handler(
        *values: object,
    ) -> tuple[dict, Figure, Figure | None, Figure | None, pd.DataFrame, gr.DownloadButton]:
        n_fit = len(fit_keys)
        x_unit_str, lc_rows, morphology, fit_offset = _parse_lc_inputs(values, n_rows)
        fit_vals = _collect_param_values(fit_keys, values, fit_offset)
        mcmc_vals = _collect_param_values(mcmc_keys, values, fit_offset + n_fit)

        nwalkers = int(mcmc_vals.get("nwalkers") or 50)
        nsteps = int(mcmc_vals.get("nsteps") or 500)
        burn_in = int(mcmc_vals.get("burn_in") or 50)
        fit_id = str(mcmc_vals.get("fit_id") or "mcmc_lc_fit")
        save = bool(mcmc_vals.get("save_chain", True))

        try:
            result, model_fig, corner_fig, traces_fig, df, json_path = compute.run_mcmc(
                lc_rows, x_unit_str, fit_vals, morphology,
                nwalkers, nsteps, burn_in, fit_id, save=save,
            )
        except Exception as exc:
            msg = str(exc)
            raise gr.Error(msg) from exc

        return (
            result,
            model_fig,
            corner_fig,
            traces_fig,
            df,
            gr.DownloadButton(value=json_path, visible=True),
        )

    return handler


def _make_transfer_handler() -> Callable[[dict | None], list[object]]:
    """Return a transfer handler that populates value fields from the LSQRT result.

    :returns: Handler function.
    :rtype: Callable[[dict | None], list[object]]
    """
    def _transfer(result: dict | None) -> list[object]:
        """Populate value fields from the stored LSQRT result.

        :param result: Nested fit result dict stored in session state.
        :type result: dict | None
        :returns: One ``gr.update`` per value component.
        :rtype: list[object]
        """
        if result is None:
            msg = "No LSQRT result available yet - run Least Squares first."
            raise gr.Error(msg)
        values = compute.extract_values_for_transfer(result)

        def _upd(key: str) -> object:
            return gr.update(value=values[key]) if key in values else gr.update()

        updates: list[object] = [
            _upd(f"system_{name}_value") for name in SYSTEM_REGULAR_PARAMS
        ]
        updates.append(_upd("system_semi_major_axis_value"))
        updates.extend(
            _upd(f"{section}_{name}_value")
            for section in ("primary", "secondary")
            for name in COMPONENT_PARAMS
        )
        updates.append(_upd("nuisance_ln_f_value"))
        return updates

    return _transfer


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


def _build_mcmc_results() -> (
    tuple[gr.Plot, gr.DataFrame, gr.Plot, gr.Plot, gr.DownloadButton]
):
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


def _wire_param_mode_changes(fit_comps: dict[str, gr.Component]) -> None:
    """Wire all param mode radios to toggle constraint/min/max fields.

    :param fit_comps: Component mapping returned by :func:`param_inputs.build`.
    :type fit_comps: dict[str, gr.Component]
    """
    # Wire all system params (including semi_major_axis)
    for name in SYSTEM_PARAMS:
        cast("gr.Radio", fit_comps[f"system_{name}_mode"]).change(
            fn=_sma_mode_changed,
            inputs=[fit_comps[f"system_{name}_mode"]],
            outputs=[
                fit_comps[f"system_{name}_constraint"],
                fit_comps[f"system_{name}_min"],
                fit_comps[f"system_{name}_max"],
            ],
        )

    # Wire primary and secondary params
    for section in ("primary", "secondary"):
        for name in COMPONENT_PARAMS:
            cast("gr.Radio", fit_comps[f"{section}_{name}_mode"]).change(
                fn=_sma_mode_changed,
                inputs=[fit_comps[f"{section}_{name}_mode"]],
                outputs=[
                    fit_comps[f"{section}_{name}_constraint"],
                    fit_comps[f"{section}_{name}_min"],
                    fit_comps[f"{section}_{name}_max"],
                ],
            )

    # Wire nuisance param
    cast("gr.Radio", fit_comps["nuisance_ln_f_mode"]).change(
        fn=_sma_mode_changed,
        inputs=[fit_comps["nuisance_ln_f_mode"]],
        outputs=[
            fit_comps["nuisance_ln_f_constraint"],
            fit_comps["nuisance_ln_f_min"],
            fit_comps["nuisance_ln_f_max"],
        ],
    )


def _wire_json_loader(
    params_json_comp: gr.File,
    fit_comps: dict[str, gr.Component],
    fit_keys: tuple[str, ...],
) -> None:
    """Wire the JSON param-loader upload event.

    :param params_json_comp: The file upload component for the result JSON.
    :type params_json_comp: gr.File
    :param fit_comps: Component mapping returned by :func:`param_inputs.build`.
    :type fit_comps: dict[str, gr.Component]
    :param fit_keys: Ordered param keys matching ``fit_comps``.
    :type fit_keys: tuple[str, ...]
    """
    fit_inputs_list: list[gr.Component] = [fit_comps[k] for k in fit_keys]

    def _load_json_handler(json_file: object) -> list[object]:
        """Populate the parameter form from an uploaded LC result JSON.

        :param json_file: Gradio file object from the upload component.
        :type json_file: object
        :returns: One ``gr.update`` per component in :data:`FIELD_ORDER`.
        :rtype: list[object]
        """
        path: str | None = getattr(json_file, "name", None)
        if path is None:
            msg = "No file uploaded."
            raise gr.Error(msg)
        try:
            params = compute.load_params_from_json(path)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        updates: list[object] = []
        for key in fit_keys:
            if key == "system_semi_major_axis_constraint":
                mode = str(params.get("system_semi_major_axis_mode", "constrained"))
                val = params.get(key)
                updates.append(
                    gr.update(visible=mode == "constrained")
                    if val is None
                    else gr.update(value=val, visible=mode == "constrained"),
                )
            elif key == "system_semi_major_axis_mode":
                updates.append(gr.update(value=str(params.get(key, "constrained"))))
            elif key.endswith(("_min", "_max")):
                base = key.rsplit("_", 1)[0]
                if "semi_major_axis" in key:
                    mode = str(params.get("system_semi_major_axis_mode", "constrained"))
                    is_interactive = mode == "free"
                else:
                    is_interactive = not bool(params.get(f"{base}_fixed", False))
                val = params.get(key)
                updates.append(
                    gr.update(interactive=is_interactive)
                    if val is None
                    else gr.update(value=val, interactive=is_interactive),
                )
            elif key in params:
                updates.append(gr.update(value=params[key]))
            else:
                updates.append(gr.update())
        return updates

    params_json_comp.upload(
        fn=_load_json_handler,
        inputs=[params_json_comp],
        outputs=fit_inputs_list,
    )


# ---------------------------------------------------------------------------
# UI section builders (called inside the Tab context from build())
# ---------------------------------------------------------------------------


def _build_data_accordion() -> data_inputs.LCDataComponents:
    """Render accordion 1 (Observational data) and return component refs.

    :returns: Dataclass with all passband row and control components.
    :rtype: data_inputs.LCDataComponents
    """
    with gr.Accordion("1 · Observational data", open=True):
        return data_inputs.build()


def _build_param_accordion() -> tuple[dict[str, gr.Component], gr.Radio, gr.File]:
    """Render accordion 2 (Initial parameters) and return component refs.

    :returns: Tuple of ``(fit_comps, morphology_comp, params_json_comp)``.
    :rtype: tuple[dict[str, gr.Component], gr.Radio, gr.File]
    """
    with gr.Accordion("2 · Initial parameters", open=True):
        with gr.Row():
            morphology_comp = gr.Radio(
                choices=["detached", "over-contact"],
                value="detached",
                label="Binary morphology",
                info="Expected system morphology - affects valid surface potential ranges.",
            )
        fit_comps = param_inputs.build()
        with gr.Accordion("Load Parameters from Previous Fit", open=False):
            gr.Markdown(
                "Upload a result JSON saved by a previous LSQRT or MCMC run to restore "
                "all parameter values, bounds, modes, and constraints into the form below.",
            )
            with gr.Row():
                params_json_comp = gr.File(
                    label="Result JSON", file_types=[".json"], scale=1,
                )
    return fit_comps, morphology_comp, params_json_comp


def _build_mcmc_accordion() -> tuple[gr.Number, gr.Number, gr.Number, gr.Textbox, gr.Checkbox]:
    """Render accordion 3 (MCMC settings) and return all control components.

    :returns: Tuple of ``(nwalkers, nsteps, burn_in, fit_id, save_chain)``.
    :rtype: tuple[gr.Number, gr.Number, gr.Number, gr.Textbox, gr.Checkbox]
    """
    with gr.Accordion("3 · MCMC settings", open=False):
        gr.Markdown(
            "**MCMC sampling configuration** - adjust these parameters "
            "to control the Markov Chain Monte Carlo sampling process.",
        )
        with gr.Row():
            nwalkers_comp = gr.Number(
                value=120, label="Number of walkers",
                info="Number of MCMC walkers (should be >= 2 x free parameters).",
                precision=0, scale=1, minimum=10,
            )
            nsteps_comp = gr.Number(
                value=120, label="Sampling steps",
                info="Total number of sampling steps per walker (after burn-in).",
                precision=0, scale=1, minimum=10,
            )
            burn_in_comp = gr.Number(
                value=10, label="Burn-in / Warmup steps",
                info="Number of initial steps to discard for chain equilibration.",
                precision=0, scale=1, minimum=0,
            )
        with gr.Row():
            fit_id_comp = gr.Textbox(
                value="mcmc_lc_fit", label="Chain file ID / Path",
                info=(
                    "Filename (saved to ELISa home), relative path, or absolute path "
                    "ending with .json.  Example: 'my_fit', 'results/fit.json'."
                ),
                scale=3,
            )
            save_chain_comp = gr.Checkbox(
                value=True, label="Save chain to disk", scale=1,
            )
    return nwalkers_comp, nsteps_comp, burn_in_comp, fit_id_comp, save_chain_comp


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
    gr.Tabs, gr.Plot, gr.DataFrame, gr.DownloadButton,
    gr.Plot, gr.DataFrame, gr.Plot, gr.Plot, gr.DownloadButton,
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
            mcmc_model_plot, mcmc_table, corner_plot, traces_plot, mcmc_download = (
                _build_mcmc_results()
            )
    return (
        results_tabs,
        lsqrt_model_plot, lsqrt_table, lsqrt_download,
        mcmc_model_plot, mcmc_table, corner_plot, traces_plot, mcmc_download,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build() -> None:
    """Build the LC Fitting tab inside the active ``gr.Blocks`` context.

    Must be called from within a ``gr.Blocks`` context manager.  Wires all
    Gradio components to the computation back-end and registers the following
    event handlers:

    - **+ Add / - Remove passband** - show/hide pre-rendered passband rows.
    - **semi_major_axis mode** radio - toggles constraint textbox visibility
      and min/max interactivity.
    - **Fixed checkboxes** (all regular params) - toggle min/max interactivity.
    - **Run Least Squares** - runs :func:`~logic.compute.run_lsqrt` and
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

        lsqrt_result_state: gr.State = gr.State(value=None)
        data_comps = _build_data_accordion()
        fit_comps, morphology_comp, params_json_comp = _build_param_accordion()
        nwalkers_comp, nsteps_comp, burn_in_comp, fit_id_comp, save_chain_comp = (
            _build_mcmc_accordion()
        )
        lsqrt_btn, transfer_btn, mcmc_btn = _build_action_buttons()
        (
            results_tabs,
            lsqrt_model_plot, lsqrt_table, lsqrt_download,
            mcmc_model_plot, mcmc_table, corner_plot, traces_plot, mcmc_download,
        ) = _build_results_section()

        fit_keys = FIELD_ORDER
        mcmc_keys: tuple[str, ...] = ("nwalkers", "nsteps", "burn_in", "fit_id", "save_chain")

        pb_outputs = [data_comps.passband_count, *data_comps.row_groups]
        data_comps.add_btn.click(fn=_add_passband, inputs=[data_comps.passband_count], outputs=pb_outputs)  # type: ignore[union-attr]
        data_comps.remove_btn.click(fn=_remove_passband, inputs=[data_comps.passband_count], outputs=pb_outputs)  # type: ignore[union-attr]

        # Wire mode changes for all params
        _wire_param_mode_changes(fit_comps)
        _wire_json_loader(params_json_comp, fit_comps, fit_keys)

        fit_inputs_list: list[gr.Component] = [fit_comps[k] for k in fit_keys]
        lc_data_inputs_list: list[gr.Component] = (
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
        mcmc_inputs_list: list[gr.Component] = [
            nwalkers_comp, nsteps_comp, burn_in_comp, fit_id_comp, save_chain_comp,
        ]
        lsqrt_all_inputs = lc_data_inputs_list + fit_inputs_list
        mcmc_all_inputs = lc_data_inputs_list + fit_inputs_list + mcmc_inputs_list

        lsqrt_btn.click(
            fn=lambda: gr.update(selected=_TAB_LSQRT),
            outputs=[results_tabs],
        ).then(
            fn=_lsqrt_handler(MAX_PASSBAND_ROWS, fit_keys),
            inputs=lsqrt_all_inputs,
            outputs=[lsqrt_result_state, lsqrt_model_plot, lsqrt_table, lsqrt_download],
            show_progress="full",
        )

        _value_outputs: list[gr.Component] = (
            [fit_comps[f"system_{n}_value"] for n in SYSTEM_REGULAR_PARAMS]
            + [fit_comps["system_semi_major_axis_value"]]
            + [fit_comps[f"primary_{n}_value"] for n in COMPONENT_PARAMS]
            + [fit_comps[f"secondary_{n}_value"] for n in COMPONENT_PARAMS]
            + [fit_comps["nuisance_ln_f_value"]]
        )
        transfer_btn.click(
            fn=_make_transfer_handler(),
            inputs=[lsqrt_result_state],
            outputs=_value_outputs,
        )

        mcmc_btn.click(
            fn=lambda: gr.update(selected=_TAB_MCMC),
            outputs=[results_tabs],
        ).then(
            fn=_mcmc_handler(MAX_PASSBAND_ROWS, fit_keys, mcmc_keys),
            inputs=mcmc_all_inputs,
            outputs=[
                gr.State(),
                mcmc_model_plot, corner_plot, traces_plot, mcmc_table, mcmc_download,
            ],
            show_progress="full",
        )

