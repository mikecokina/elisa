"""Light Curve Modeling tab for the ELISa Gradio UI.

Call :func:`build` from inside a ``gr.Blocks`` context to register
the tab.  The function creates the full layout, wires all inputs to
the computation back-end (:mod:`~elisa.ui.tabs.lc_modeling.logic.compute`),
and registers event handlers.  No state is held at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr
import pandas as pd  # noqa: TC002  - needed at runtime for Gradio get_type_hints()
from matplotlib.figure import Figure  # noqa: TC002  - same reason

from elisa.ui.components import star_inputs, system_inputs
from elisa.ui.shared.const import ATMOSPHERE_CHOICES
from elisa.ui.shared.fit_json import make_json_load_handler
from elisa.ui.tabs.lc_modeling.components import observer_inputs
from elisa.ui.tabs.lc_modeling.logic import compute

if TYPE_CHECKING:
    from collections.abc import Callable

    from elisa.types import Number


# Default parameter presets that match the example in ``main.py``.
_PRIMARY_DEFAULTS: dict[str, Number | str] = {
    "mass": 2.15,
    "t_eff": 10000,
    "surface_potential": 3.6,
    "synchronicity": 1.0,
    "metallicity": 0.0,
    "discretization_factor": 5,
    "atmosphere": ATMOSPHERE_CHOICES[1],
}

_SECONDARY_DEFAULTS: dict[str, Number | str] = {
    "mass": 1.2,
    "t_eff": 7000,
    "surface_potential": 4.0,
    "synchronicity": 1.0,
    "metallicity": 0.0,
    "atmosphere": ATMOSPHERE_CHOICES[1],
}

_SYSTEM_DEFAULTS: dict[str, Number | str] = {
    "inclination": 90.0,
    "period": 2.5,
    "eccentricity": 0.0,
    "argument_of_periastron": 90.0,
    "gamma": 0.0,
    "phase_shift": 0.0,
    "additional_light": 0.0,
    "primary_minimum_time": 2440000.0,
    "distance": 155.0,
}

_OBSERVER_DEFAULTS: dict[str, Number | str] = {
    "passband": ["Generic.Bessell.U"],
    "from_phase": -0.6,
    "to_phase": 0.6,
    "phase_step": 0.01,
    "normalize": False,
}


# ---------------------------------------------------------------------------
# Internal handler
# ---------------------------------------------------------------------------



def _make_handler(
    prim_keys: tuple[str, ...],
    sec_keys: tuple[str, ...],
    sys_keys: tuple[str, ...],
    obs_keys: tuple[str, ...],
) -> Callable[..., tuple[Figure, pd.DataFrame, gr.DownloadButton]]:
    """Return a Gradio event-handler function bound to the given key sequences.

    The handler unpacks the flat list of values that Gradio passes to the
    callback into named parameter dicts and delegates to
    :func:`~elisa.ui.tabs.lc_modeling.logic.compute.run_lc`.

    :param prim_keys: Ordered keys for primary star parameters.
    :type prim_keys: tuple[str, ...]
    :param sec_keys: Ordered keys for secondary star parameters.
    :type sec_keys: tuple[str, ...]
    :param sys_keys: Ordered keys for binary system parameters.
    :type sys_keys: tuple[str, ...]
    :param obs_keys: Ordered keys for observer parameters.
    :type obs_keys: tuple[str, ...]
    :returns: A callable suitable for ``gr.Button.click(fn=...)``.
    :rtype: Callable[..., tuple[Figure, pandas.DataFrame]]
    """

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
            fig, df, csv_path = compute.run_lc(
                primary_params, secondary_params, system_params, observer_params,
            )
            return fig, df, gr.DownloadButton(value=csv_path, visible=True)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    return handler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build() -> None:
    """Build the Light Curve Modeling tab inside the active ``gr.Blocks`` context.

    Must be called from within a ``gr.Blocks`` (or ``gr.Tab`` inside
    ``gr.Blocks``) context manager.  Creates the full form layout,
    output area, and event bindings.

    :returns: ``None``
    :rtype: None
    """
    with gr.Tab("Light Curve Modeling"):
        gr.Markdown(
            "## Light Curve Modeling\n"
            "Configure both stellar components, the binary-system geometry, and the "
            "observational settings, then click **Compute** to synthesize the light curve.",
        )

        # ------------------------------------------------------------------ #
        # Load from fit-result JSON                                            #
        # ------------------------------------------------------------------ #
        with gr.Accordion("Load Parameters from Fit Result JSON", open=False):
            gr.Markdown(
                "Upload a result JSON saved by a previous **LSQRT or MCMC** run to "
                "pre-fill all star and system parameters below.  "
                "Both **Standard** and **Community** fit JSONs are accepted - "
                "community parameters (semi-major axis + mass ratio) are automatically "
                "converted to individual masses via Kepler's third law.",
            )
            json_file_input = gr.File(
                label="Fit result JSON",
                file_types=[".json"],
            )

        # ------------------------------------------------------------------ #
        # Input forms - four equal columns                                     #
        # ------------------------------------------------------------------ #
        with gr.Row():
            with gr.Column(scale=1):
                prim_comps = star_inputs.build("Primary Star", defaults=_PRIMARY_DEFAULTS)

            with gr.Column(scale=1):
                sec_comps = star_inputs.build("Secondary Star", defaults=_SECONDARY_DEFAULTS)

            with gr.Column(scale=1):
                sys_comps = system_inputs.build(defaults=_SYSTEM_DEFAULTS)

            with gr.Column(scale=1):
                obs_comps = observer_inputs.build(defaults=_OBSERVER_DEFAULTS)

        # ------------------------------------------------------------------ #
        # Action                                                               #
        # ------------------------------------------------------------------ #
        with gr.Row():
            compute_btn = gr.Button("🚀 Compute Light Curve", variant="primary", scale=2)
            clear_btn = gr.Button("🗑 Clear outputs", variant="secondary", scale=1)

        # ------------------------------------------------------------------ #
        # Outputs                                                              #
        # ------------------------------------------------------------------ #
        with gr.Row():
            lc_plot = gr.Plot(label="Light Curve", scale=3)
            with gr.Column(scale=2):
                lc_table = gr.DataFrame(label="Phase / Flux data")
                dl_btn = gr.DownloadButton(
                    label="⬇ Download CSV",
                    variant="secondary",
                    visible=False,
                )

        # ------------------------------------------------------------------ #
        # Event wiring                                                         #
        # ------------------------------------------------------------------ #
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
            outputs=[lc_plot, lc_table, dl_btn],
        )

        clear_btn.click(
            fn=lambda: (None, None, gr.DownloadButton(visible=False)),
            inputs=None,
            outputs=[lc_plot, lc_table, dl_btn],
        )

        all_form_outputs: list[gr.Component] = (
            list(prim_comps.values())
            + list(sec_comps.values())
            + list(sys_comps.values())
        )
        json_file_input.upload(
            fn=make_json_load_handler(prim_keys, sec_keys, sys_keys),
            inputs=[json_file_input],
            outputs=all_form_outputs,
        )

