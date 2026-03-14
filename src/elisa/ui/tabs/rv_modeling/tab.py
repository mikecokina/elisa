"""Radial Velocity Modeling tab for the ELISa Gradio UI."""
from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr
import pandas as pd  # noqa: TC002  - needed at runtime for Gradio get_type_hints()
from matplotlib.figure import Figure  # noqa: TC002  - same reason

from elisa.ui.components import star_inputs, system_inputs
from elisa.ui.tabs.rv_modeling.components import observer_inputs
from elisa.ui.tabs.rv_modeling.logic import compute

if TYPE_CHECKING:
    from collections.abc import Callable
_PRIMARY_DEFAULTS: dict[str, object] = {
    "mass": 2.15,
    "t_eff": 10000,
    "surface_potential": 3.6,
    "synchronicity": 1.0,
    "metallicity": 0.0,
    "discretization_factor": 5,
    "atmosphere": "bb",
}
_SECONDARY_DEFAULTS: dict[str, object] = {
    "mass": 1.2,
    "t_eff": 7000,
    "surface_potential": 4.0,
    "synchronicity": 1.0,
    "metallicity": 0.0,
    "atmosphere": "bb",
}
_SYSTEM_DEFAULTS: dict[str, object] = {
    "inclination": 85.0,
    "period": 2.5,
    "eccentricity": 0.0,
    "argument_of_periastron": 58.0,
    "gamma": 0.0,
    "phase_shift": 0.0,
    "additional_light": 0.0,
    "primary_minimum_time": 2440000.0,
    "distance": 155.0,
}
_OBSERVER_DEFAULTS: dict[str, object] = {
    "from_phase": -0.6,
    "to_phase": 0.6,
    "phase_step": 0.01,
    "method": "kinematic",
}
def _make_handler(
    prim_keys: tuple[str, ...],
    sec_keys: tuple[str, ...],
    sys_keys: tuple[str, ...],
    obs_keys: tuple[str, ...],
) -> Callable[..., tuple[Figure, pd.DataFrame, gr.DownloadButton]]:
    """Return a Gradio event-handler function for RV computation."""
    def handler(*values: object) -> tuple[Figure, pd.DataFrame, gr.DownloadButton]:
        idx = 0
        n_prim = len(prim_keys)
        n_sec = len(sec_keys)
        n_sys = len(sys_keys)
        primary_params = dict(zip(prim_keys, values[idx : idx + n_prim], strict=True))
        idx += n_prim
        secondary_params = dict(zip(sec_keys, values[idx : idx + n_sec], strict=True))
        idx += n_sec
        system_params = dict(zip(sys_keys, values[idx : idx + n_sys], strict=True))
        idx += n_sys
        observer_params = dict(zip(obs_keys, values[idx:], strict=True))
        try:
            fig, df, csv_path = compute.run_rv(
                primary_params, secondary_params, system_params, observer_params,
            )
            return fig, df, gr.DownloadButton(value=csv_path, visible=True)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
    return handler
def build() -> None:
    """Build the RV Modeling tab inside the active gr.Blocks context."""
    with gr.Tab("Radial Velocity Modeling"):
        gr.Markdown(
            "## Radial Velocity Modeling\n"
            "Configure both stellar components, the binary-system geometry, and the "
            "observational settings, then click **Compute** to synthesize RV curves.",
        )
        with gr.Row():
            with gr.Column(scale=1):
                prim_comps = star_inputs.build("Primary Star", defaults=_PRIMARY_DEFAULTS)
            with gr.Column(scale=1):
                sec_comps = star_inputs.build("Secondary Star", defaults=_SECONDARY_DEFAULTS)
            with gr.Column(scale=1):
                sys_comps = system_inputs.build(defaults=_SYSTEM_DEFAULTS)
            with gr.Column(scale=1):
                obs_comps = observer_inputs.build(defaults=_OBSERVER_DEFAULTS)
        with gr.Row():
            compute_btn = gr.Button("🚀 Compute RV Curves", variant="primary", scale=2)
            clear_btn = gr.Button("🗑 Clear outputs", variant="secondary", scale=1)
        with gr.Row():
            rv_plot = gr.Plot(label="Radial Velocity Curves", scale=3)
            with gr.Column(scale=2):
                rv_table = gr.DataFrame(label="Phase / RV data (km/s)")
                dl_btn = gr.DownloadButton(
                    label="⬇ Download CSV",
                    variant="secondary",
                    visible=False,
                )
        prim_keys = star_inputs.FIELD_ORDER
        sec_keys = star_inputs.FIELD_ORDER
        sys_keys = system_inputs.FIELD_ORDER
        obs_keys = observer_inputs.FIELD_ORDER
        all_inputs: list[gr.Component] = (
            list(prim_comps.values())
            + list(sec_comps.values())
            + list(sys_comps.values())
            + list(obs_comps.values())
        )
        compute_btn.click(
            fn=_make_handler(prim_keys, sec_keys, sys_keys, obs_keys),
            inputs=all_inputs,
            outputs=[rv_plot, rv_table, dl_btn],
        )
        clear_btn.click(
            fn=lambda: (None, None, gr.DownloadButton(visible=False)),
            inputs=None,
            outputs=[rv_plot, rv_table, dl_btn],
        )
