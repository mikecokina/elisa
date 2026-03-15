"""RV Fitting tab for the ELISa Gradio UI.

Provides a full workflow for fitting radial velocity curves using either
the least-squares (LSQRT) or MCMC method:

1. Upload primary (and optionally secondary) RV data files.
2. Set initial parameter values, bounds, and fixed flags.
3. Run LSQRT - fast deterministic fit, result stored in session state.
4. Optionally transfer the LSQRT result as MCMC starting point.
5. Run MCMC - full posterior sampling, produces corner and traces plots.
6. Inspect the summary table, model plot, and MCMC diagnostics.
7. Download the JSON result file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr
import pandas as pd  # noqa: TC002 - needed at runtime for Gradio type hints
from matplotlib.figure import Figure  # noqa: TC002 - same reason

from elisa.ui.tabs.rv_fitting.components import data_inputs, param_inputs
from elisa.ui.tabs.rv_fitting.logic import compute

if TYPE_CHECKING:
    from collections.abc import Callable

    from elisa.ui.tabs.rv_fitting.components.data_inputs import DataInputComponents

# ---------------------------------------------------------------------------
# Tab ID constants - must match id= on gr.Tab definitions
# ---------------------------------------------------------------------------

_TAB_LSQRT = "rv_lsqrt_results"
_TAB_MCMC = "rv_mcmc_results"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _lsqrt_handler(
    data_keys: tuple[str, ...],
    fit_keys: tuple[str, ...],
) -> Callable[..., tuple[dict, Figure, pd.DataFrame, gr.DownloadButton]]:
    """Return the LSQRT Gradio event-handler bound to the given key tuples.

    :param data_keys: Keys for data-input components in order.
    :type data_keys: tuple[str, ...]
    :param fit_keys: Keys for parameter-input components in order.
    :type fit_keys: tuple[str, ...]
    :returns: Gradio-compatible handler function.
    """

    def handler(
        *values: object,
    ) -> tuple[dict, Figure, pd.DataFrame, gr.DownloadButton]:
        n_data = len(data_keys)
        data_vals = _collect_param_values(data_keys, values, 0)
        fit_vals = _collect_param_values(fit_keys, values, n_data)

        primary_path: str | None = getattr(data_vals.get("primary_file"), "name", None)
        secondary_path: str | None = getattr(data_vals.get("secondary_file"), "name", None)
        x_unit_str: str = str(data_vals.get("x_unit", "Julian days (JD)"))

        try:
            result, fig, df, json_path = compute.run_lsqrt(
                primary_path,
                secondary_path,
                x_unit_str,
                fit_vals,
            )
        except Exception as exc:
            msg = str(exc)
            raise gr.Error(msg) from exc

        return result, fig, df, gr.DownloadButton(value=json_path, visible=True)

    return handler


def _mcmc_handler(
    data_keys: tuple[str, ...],
    fit_keys: tuple[str, ...],
    mcmc_keys: tuple[str, ...],
) -> Callable[..., tuple[dict, Figure, Figure | None, Figure | None, pd.DataFrame, gr.DownloadButton]]:
    """Return the MCMC Gradio event-handler bound to the given key tuples.

    :param data_keys: Keys for data-input components in order.
    :type data_keys: tuple[str, ...]
    :param fit_keys: Keys for parameter-input components in order.
    :type fit_keys: tuple[str, ...]
    :param mcmc_keys: Keys for MCMC control components in order.
    :type mcmc_keys: tuple[str, ...]
    :returns: Gradio-compatible handler function.
    """

    def handler(
        *values: object,
    ) -> tuple[dict, Figure, Figure | None, Figure | None, pd.DataFrame, gr.DownloadButton]:
        n_data = len(data_keys)
        n_fit = len(fit_keys)
        data_vals = _collect_param_values(data_keys, values, 0)
        fit_vals = _collect_param_values(fit_keys, values, n_data)
        mcmc_vals = _collect_param_values(mcmc_keys, values, n_data + n_fit)

        primary_path: str | None = getattr(data_vals.get("primary_file"), "name", None)
        secondary_path: str | None = getattr(data_vals.get("secondary_file"), "name", None)
        x_unit_str: str = str(data_vals.get("x_unit", "Julian days (JD)"))

        nwalkers = int(mcmc_vals.get("nwalkers") or 50)
        nsteps = int(mcmc_vals.get("nsteps") or 500)
        burn_in = int(mcmc_vals.get("burn_in") or 50)
        fit_id = str(mcmc_vals.get("fit_id") or "mcmc_rv_fit")
        save = bool(mcmc_vals.get("save_chain", True))

        try:
            result, model_fig, corner_fig, traces_fig, df, json_path = compute.run_mcmc(
                primary_path,
                secondary_path,
                x_unit_str,
                fit_vals,
                nwalkers,
                nsteps,
                burn_in,
                fit_id,
                save=save,
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_lsqrt_results() -> tuple[gr.Plot, gr.DataFrame, gr.DownloadButton]:
    """Render the LSQRT results sub-tab content and return output components.

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
    """Render the MCMC results sub-tab content and return output components.

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


def build() -> None:  # noqa: PLR0915
    """Build the RV Fitting tab inside the active ``gr.Blocks`` context.

    Must be called from within a ``gr.Blocks`` context manager.  Wires all
    Gradio components to the computation back-end and registers the
    following event handlers:

    - **Run The Least Squares** - runs :func:`~logic.compute.run_lsqrt` and
      populates the *LSQRT Results* sub-tab.
    - **Transfer to MCMC** - copies fitted values from the session state
      into the parameter form so the MCMC run starts from the LSQRT
      solution.
    - **Run MCMC** - runs :func:`~logic.compute.run_mcmc` and populates the
      *MCMC Results* sub-tab including corner and traces plots.

    :returns: ``None``
    :rtype: None
    """
    with gr.Tab("RV Fitting"):
        gr.Markdown(
            "## Radial Velocity Fitting\n"
            "Upload observed RV curves, configure initial parameters, then run "
            "**Least Squares** for a quick fit or **MCMC** for a full posterior. "
            "Use **Transfer to MCMC** to seed the sampler with the LSQRT solution.",
        )

        # ------------------------------------------------------------------ #
        # Session state                                                         #
        # ------------------------------------------------------------------ #
        lsqrt_result_state: gr.State = gr.State(value=None)

        # ------------------------------------------------------------------ #
        # Section 1 - Data upload                                              #
        # ------------------------------------------------------------------ #
        with gr.Accordion("1 · Observational data", open=True):
            data_comps: DataInputComponents = data_inputs.build()

        # ------------------------------------------------------------------ #
        # Section 2 - Initial parameters                                       #
        # ------------------------------------------------------------------ #
        with gr.Accordion("2 · Initial parameters", open=True):
            fit_comps = param_inputs.build()
            with gr.Accordion("Load Parameters from Previous Fit", open=False):
                gr.Markdown(
                    "Upload a result JSON saved by a previous LSQRT or MCMC run to restore "
                    "all parameter values, bounds, and fixed flags into the form below.",
                )
                with gr.Row():
                    params_json_comp = gr.File(
                        label="Result JSON",
                        file_types=[".json"],
                        scale=1,
                    )

        # ------------------------------------------------------------------ #
        # Section 3 - MCMC settings                                            #
        # ------------------------------------------------------------------ #
        with gr.Accordion("3 · MCMC settings", open=False):
            gr.Markdown(
                "**MCMC sampling configuration** - adjust these parameters "
                "to control the Markov Chain Monte Carlo sampling process.",
            )
            with gr.Row():
                nwalkers_comp = gr.Number(
                    value=120,
                    label="Number of walkers",
                    info="Number of MCMC walkers (should be >= 2 x number of free parameters).",
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
                    value="mcmc_rv_fit",
                    label="Chain file ID / Path",
                    info=(
                        "Filename (saved to ELISa home), relative path, or absolute path ending with .json. "
                        "If parent directory exists, saves there. "
                        "Example: 'my_fit', 'results/fit.json', or '/path/to/fit.json'."
                    ),
                    scale=3,
                )
                save_chain_comp = gr.Checkbox(
                    value=True,
                    label="Save chain to disk",
                    scale=1,
                )

        # ------------------------------------------------------------------ #
        # Section 4 - Action buttons                                           #
        # ------------------------------------------------------------------ #
        with gr.Row():
            lsqrt_btn = gr.Button(
                "⚡ Run Least Squares",
                variant="primary",
                scale=2,
            )
            transfer_btn = gr.Button(
                "↓ Transfer LSQRT → MCMC",
                variant="secondary",
                scale=1,
            )
            mcmc_btn = gr.Button(
                "🔬 Run MCMC",
                variant="primary",
                scale=2,
            )

        # ------------------------------------------------------------------ #
        # Section 5 - Results                                                  #
        # ------------------------------------------------------------------ #
        with gr.Tabs() as results_tabs:
            with gr.Tab("LSQRT Results", id=_TAB_LSQRT):
                lsqrt_model_plot, lsqrt_table, lsqrt_download = _build_lsqrt_results()
            with gr.Tab("MCMC Results", id=_TAB_MCMC):
                mcmc_model_plot, mcmc_table, corner_plot, traces_plot, mcmc_download = _build_mcmc_results()

        # ------------------------------------------------------------------ #
        # Event wiring                                                         #
        # ------------------------------------------------------------------ #
        data_keys = tuple(data_comps.keys())
        fit_keys = param_inputs.FIELD_ORDER
        mcmc_keys = ("nwalkers", "nsteps", "burn_in", "fit_id", "save_chain")

        # Build flat input lists for each button
        # data_comps is a TypedDict (subclass of dict) - values() gives components in insertion order
        data_inputs_list: list[gr.Component] = list(data_comps.values())  # type: ignore[arg-type]
        fit_inputs_list = [fit_comps[k] for k in fit_keys]
        mcmc_inputs_list = [
            nwalkers_comp,
            nsteps_comp,
            burn_in_comp,
            fit_id_comp,
            save_chain_comp,
        ]

        lsqrt_all_inputs = data_inputs_list + fit_inputs_list
        mcmc_all_inputs = data_inputs_list + fit_inputs_list + mcmc_inputs_list

        # Load parameters from uploaded result JSON
        def _load_json_handler(json_file: object) -> list[object]:
            """Populate the parameter form from an uploaded result JSON.

            Calls :func:`~logic.compute.load_params_from_json` and returns a
            ``gr.update`` for every component in ``fit_inputs_list``, setting
            ``value`` (and ``interactive`` for min/max fields based on the
            fixed flag).

            :param json_file: Gradio file object from the upload component.
            :type json_file: object
            :returns: One ``gr.update`` per component in :data:`param_inputs.FIELD_ORDER`.
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
                if key.endswith(("_min", "_max")):
                    param_name = key.rsplit("_", 1)[0]
                    is_fixed = bool(params.get(f"{param_name}_fixed", False))
                    val = params.get(key)
                    updates.append(
                        gr.update(interactive=not is_fixed)
                        if val is None
                        else gr.update(value=val, interactive=not is_fixed),
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

        # LSQRT run - immediately switch tab, then run computation
        lsqrt_btn.click(
            fn=lambda: gr.update(selected=_TAB_LSQRT),
            outputs=[results_tabs],
        ).then(
            fn=_lsqrt_handler(data_keys, fit_keys),
            inputs=lsqrt_all_inputs,
            outputs=[
                lsqrt_result_state,
                lsqrt_model_plot,
                lsqrt_table,
                lsqrt_download,
            ],
            show_progress="full",
        )

        # Transfer LSQRT result values into the param form
        def _transfer(result: dict | None) -> list[object]:
            """Populate the parameter value fields from the stored LSQRT result.

            :param result: Nested fit result dict stored in session state.
            :type result: dict | None
            :returns: List of ``gr.update`` calls - one per ``*_value``
                component in :data:`~components.param_inputs.FIELD_ORDER`.
            :rtype: list[object]
            """
            if result is None:
                msg = "No LSQRT result available yet - run Least Squares first."
                raise gr.Error(msg)
            values = compute.extract_values_for_transfer(result)
            # Return updates only for the _value components (every 4th in FIELD_ORDER)
            from elisa.ui.tabs.rv_fitting.components.param_inputs import PARAMS  # noqa: PLC0415

            return [
                gr.update(value=values[f"{name}_value"]) if f"{name}_value" in values else gr.update()
                for name in PARAMS
            ]

        value_outputs = [fit_comps[f"{name}_value"] for name in param_inputs.PARAMS]
        transfer_btn.click(
            fn=_transfer,
            inputs=[lsqrt_result_state],
            outputs=value_outputs,
        )

        # MCMC run - immediately switch tab, then run computation
        mcmc_btn.click(
            fn=lambda: gr.update(selected=_TAB_MCMC),
            outputs=[results_tabs],
        ).then(
            fn=_mcmc_handler(data_keys, fit_keys, mcmc_keys),
            inputs=mcmc_all_inputs,
            outputs=[
                gr.State(),  # mcmc result (not stored - download covers it)
                mcmc_model_plot,
                corner_plot,
                traces_plot,
                mcmc_table,
                mcmc_download,
            ],
            show_progress="full",
        )
