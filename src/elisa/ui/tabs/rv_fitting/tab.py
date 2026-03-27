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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa: TC002 - needed at runtime for Gradio type hints
from matplotlib.figure import Figure  # noqa: TC002 - same reason

from elisa import units as u
from elisa.analytics.binary_fit.shared import extend_observations_to_desired_interval
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
        progress = bool(mcmc_vals.get("progress", True))

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
                progress=progress,
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


def _plot_loaded_data_handler(_data_keys: tuple[str, ...]) -> Callable[..., Figure]:  # noqa: C901, PLR0915
    """Return a handler that plots uploaded RV files (observational points).

    If the uploaded data are in Julian days, the handler will phase the time
    series using the supplied orbital *period* and *T0* (primary minimum time).
    Otherwise, the raw x-axis is used (time/phase as uploaded).

    The function returns a matplotlib Figure showing primary and optional
    secondary data with error bars.
    """

    def _plot_loaded_data(  # noqa: C901, PLR0912, PLR0915
        primary_file: object | None,
        secondary_file: object | None,
        x_unit_str: str,
        data_period: float | None,
        fit_period: float | None,
        t0_value: float | None,
        start_phase: float | None,
        stop_phase: float | None,
        centre_value: float | None,
        lsqrt_result: dict | None = None,
    ) -> Figure:
        primary_path: str | None = getattr(primary_file, "name", None)
        secondary_path: str | None = getattr(secondary_file, "name", None)

        try:
            rv_primary = compute.load_rv_data(primary_path, x_unit_str)
            rv_secondary = compute.load_rv_data(secondary_path, x_unit_str)
        except Exception as exc:
            msg = str(exc)
            raise gr.Error(msg) from exc

        if rv_primary is None:
            msg = "Primary RV data file is required."
            raise gr.Error(msg)

        # Decide whether to phase JD times
        use_phases = x_unit_str == "Julian days (JD)"
        # initialize x arrays so linters don't complain about potential
        # uninitialized usage later
        secondary_x = None
        sp = None
        ep = None

        if use_phases:
            # prefer period supplied in data section, otherwise fall back to
            # the value from the parameter form
            period_value = data_period if data_period is not None else fit_period
            # fallback to stored LSQRT result if available
            if period_value is None and lsqrt_result is not None:
                try:
                    period_value = lsqrt_result.get("system", {}).get("period", {}).get("value")
                except (AttributeError, KeyError, TypeError):
                    period_value = None

            if period_value is None:
                msg = (
                    "Orbital period is required to phase JD data. Provide it in the data section, "
                    "in System Parameters, or run Least Squares to store a result."
                )
                raise gr.Error(msg)
            if t0_value is None:
                msg = "Primary minimum time (T0) is required to phase JD data."
                raise gr.Error(msg)

            # Use the RVData.convert_to_phases method so phasing logic is
            # consistent with LC phasing (see demo12 example). This updates
            # the dataset x_data in-place and sets x_unit accordingly.
            try:
                centre_arg = float(centre_value) if centre_value is not None else 0.0
                rv_primary.convert_to_phases(float(period_value), float(t0_value), centre=centre_arg)
                primary_x = rv_primary.x_data
            except (ValueError, TypeError) as exc:
                msg = f"Failed to phase primary data: {exc}"
                raise gr.Error(msg) from exc

            if rv_secondary is not None:
                try:
                    centre_arg = float(centre_value) if centre_value is not None else 0.0
                    rv_secondary.convert_to_phases(float(period_value), float(t0_value), centre=centre_arg)
                    secondary_x = rv_secondary.x_data
                except (ValueError, TypeError):
                    # if secondary phasing fails, fall back to raw times
                    secondary_x = rv_secondary.x_data
        else:
            primary_x = rv_primary.x_data
            secondary_x = rv_secondary.x_data if rv_secondary is not None else None

        # Convert velocities to km/s
        primary_y = (rv_primary.y_data * u.VELOCITY_UNIT).to(u.km / u.s).value
        primary_yerr = None
        if rv_primary.y_err is not None:
            try:
                primary_yerr = (rv_primary.y_err * u.VELOCITY_UNIT).to(u.km / u.s).value
            except (ValueError, TypeError):
                primary_yerr = None

        # If folded to phases and an interval is provided, expand observations
        if use_phases and start_phase is not None and stop_phase is not None:
            try:
                sp = float(start_phase)
                ep = float(stop_phase)
            except (TypeError, ValueError) as exc:
                msg = f"Invalid start/stop phase values: {exc}"
                raise gr.Error(msg) from exc
            if sp >= ep:
                msg = "Start phase must be less than stop phase"
                raise gr.Error(msg)

            x_out, y_out, yerr_out = extend_observations_to_desired_interval(
                sp,
                ep,
                {"primary": np.asarray(primary_x)},
                {"primary": np.asarray(primary_y)},
                {"primary": (np.asarray(primary_yerr) if primary_yerr is not None else None)},
            )
            px = np.asarray(x_out.get("primary", np.empty(0, dtype=float)))
            py = np.asarray(y_out.get("primary", np.empty(0, dtype=float)))
            perr = yerr_out.get("primary") if isinstance(yerr_out, dict) else None
            if perr is not None:
                perr = np.asarray(perr)
            if px.size:
                order = np.argsort(px)
                px = px[order]
                py = py[order]
                perr = None if perr is None else perr[order]
        else:
            px = np.asarray(primary_x)
            py = np.asarray(primary_y)
            perr = None if primary_yerr is None else np.asarray(primary_yerr)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(px, py, yerr=perr, fmt="o", label="Primary", markersize=4)

        if rv_secondary is not None:
            secondary_y = (rv_secondary.y_data * u.VELOCITY_UNIT).to(u.km / u.s).value
            secondary_yerr = None
            if rv_secondary.y_err is not None:
                try:
                    secondary_yerr = (rv_secondary.y_err * u.VELOCITY_UNIT).to(u.km / u.s).value
                except (ValueError, TypeError):
                    secondary_yerr = None
            # secondary - apply same expansion when phasing and interval provided
            if use_phases and start_phase is not None and stop_phase is not None:
                x_out_s, y_out_s, yerr_out_s = extend_observations_to_desired_interval(
                    sp,
                    ep,
                    {"secondary": np.asarray(secondary_x)},
                    {"secondary": np.asarray(secondary_y)},
                    {"secondary": (np.asarray(secondary_yerr) if secondary_yerr is not None else None)},
                )
                sx = np.asarray(x_out_s.get("secondary", np.empty(0, dtype=float)))
                sy = np.asarray(y_out_s.get("secondary", np.empty(0, dtype=float)))
                serr = yerr_out_s.get("secondary") if isinstance(yerr_out_s, dict) else None
                if serr is not None:
                    serr = np.asarray(serr)
                if sx.size:
                    order = np.argsort(sx)
                    sx = sx[order]
                    sy = sy[order]
                    serr = None if serr is None else serr[order]
            else:
                sx = secondary_x if secondary_x is not None else rv_secondary.x_data
                sy = secondary_y
                serr = None if secondary_yerr is None else np.asarray(secondary_yerr)

            ax.errorbar(sx, sy, yerr=serr, fmt="o", label="Secondary", markersize=4)

        xlabel = "Phase" if use_phases else "Time"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Radial velocity  (km s$^{-1}$)")
        ax.set_title("Observed radial velocity data", pad=10)
        ax.grid(visible=True)
        ax.legend(loc="best")
        fig.tight_layout()

        return fig

    return _plot_loaded_data





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


def build() -> None:  # noqa: C901, PLR0915
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
            # T0 input - only used when uploaded data are in Julian days (JD)
            with gr.Row():
                # Allow supplying period and T0 directly in the data section
                period_comp = gr.Number(
                    value=param_inputs.PARAM_SPEC.get("period", (None, None))[1],
                    label="Orbital period (P) [d] - used to phase JD data",
                    info="Optional - if set here it will be used instead of the System Period parameter",
                    interactive=True,
                    scale=2,
                )

                # Make T0 interactive by default (JD is the UI default)
                t0_comp = gr.Number(
                    value=param_inputs.PARAM_SPEC.get("primary_minimum_time", (None, None))[1],
                    label="Primary minimum time (T0) - used to phase JD data",
                    info="Enable and set when X-axis unit is Julian days (JD)",
                    interactive=True,
                    scale=2,
                )

                # Phase window controls - used when plotting phased data
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

                # Centre used when folding JD to phases
                centre_comp = gr.Number(
                    value=0.0,
                    label="Phase centre - centre used when folding JD to phases",
                    info="Centre value passed to DataSet.convert_to_phases(centre=...)",
                    interactive=True,
                    scale=2,
                )

                plot_obs_btn = gr.Button(
                    "📈 Plot observed data",
                    variant="secondary",
                    scale=1,
                )
            # Toggle T0 interactivity based on selected x-unit
            data_comps["x_unit"].change(
                fn=lambda val: gr.update(interactive=(val == "Julian days (JD)")),
                inputs=[data_comps["x_unit"]],
                outputs=[t0_comp],
            )

            # Inline, collapsible plot area for the uploaded observational data -
            # placed inside the data accordion so the plot appears near the inputs.
            # Default to collapsed so the data controls remain the primary focus.
            with gr.Accordion("Observed data plot", open=False):
                observed_data_plot = gr.Plot(label="Observed data")

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
                progress_comp = gr.Checkbox(
                    value=True,
                    label="Show progress bar",
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
        mcmc_keys = ("nwalkers", "nsteps", "burn_in", "fit_id", "save_chain", "progress")

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
            progress_comp,
        ]

        lsqrt_all_inputs = data_inputs_list + fit_inputs_list
        mcmc_all_inputs = data_inputs_list + fit_inputs_list + mcmc_inputs_list

        # Load parameters from uploaded result JSON
        def _load_json_handler(json_file: object) -> list[object]:
            """Populate the parameter form from an uploaded result JSON.

            Calls :func:`~logic.compute.load_params_from_json` and returns a
            ``gr.update`` for every component in ``fit_inputs_list``, setting
            ``value``, ``mode``, ``constraint``, and interactivity of min/max
            based on the loaded mode.

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
                    param_name = key.rsplit("_", maxsplit=1)[0]
                    mode = str(params.get(f"{param_name}_mode", "free"))
                    val = params.get(key)
                    upd = gr.update(interactive=mode == "free")
                    if val is not None:
                        upd = gr.update(value=val, interactive=mode == "free")
                    updates.append(upd)
                elif key.endswith("_constraint"):
                    param_name = key.rsplit("_", maxsplit=1)[0]
                    mode = str(params.get(f"{param_name}_mode", "free"))
                    val = params.get(key, "")
                    updates.append(gr.update(value=val, interactive=mode == "constrained"))
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

        # Keep the data-section period input in sync with the System Period value
        # so users who only set the system period don't need to copy it manually.
        # Ensure the component is seen as a concrete gr.Number so static
        # analyzers recognize the `.change` method. We assign it to a
        # local variable with an explicit type annotation and use that.
        period_value_comp: gr.Number = fit_comps["period_value"]  # type: ignore[assignment]
        period_value_comp.change(
            fn=lambda val: gr.update(value=val),
            inputs=[period_value_comp],
            outputs=[period_comp],
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

        # Plot loaded observational data without fitting - include period and T0
        plot_obs_btn.click(
            fn=_plot_loaded_data_handler(data_keys),
            inputs=[
                data_inputs_list[0],  # primary_file
                data_inputs_list[1],  # secondary_file
                data_inputs_list[2],  # x_unit
                period_comp,          # optional period provided in data section
                fit_comps["period_value"],
                t0_comp,
                start_phase_comp,
                stop_phase_comp,
                centre_comp,
                lsqrt_result_state,
            ],
            outputs=[observed_data_plot],
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
