"""System Visualization tab for the ELISa Gradio UI.

Call :func:`build` from inside a ``gr.Blocks`` context to register
the tab. The function creates the full layout, wires all inputs to
the computation back-end (:mod:`~elisa.ui.tabs.system_visualization.logic.compute`),
and registers event handlers. No state is held at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

from elisa.ui.components import star_inputs, system_inputs
from elisa.ui.shared.const import ATMOSPHERE_CHOICES
from elisa.ui.shared.fit_json import make_json_load_handler
from elisa.ui.tabs.lc_modeling.components import pulsation_inputs
from elisa.ui.tabs.system_visualization.components import observer_inputs
from elisa.ui.tabs.system_visualization.logic import compute

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

_OBSERVER_DEFAULTS: dict[str, Number | str | None] = {
    "visualization_mode": None,
    "phase": 0.0,
    "components_to_plot": "both",
    "plane": "xy",
    "frame_of_reference": "primary",
    "colormap": None,
}


# ---------------------------------------------------------------------------
# Internal handler
# ---------------------------------------------------------------------------


def _make_handler(
    prim_keys: tuple[str, ...],
    sec_keys: tuple[str, ...],
    sys_keys: tuple[str, ...],
    obs_keys: tuple[str, ...],
    puls_keys: tuple[str, ...],
) -> Callable[..., tuple[dict, dict, dict, dict]]:
    """Return a Gradio event-handler function bound to the given key sequences.

    The handler unpacks the flat list of values that Gradio passes to the
    callback into named parameter dicts and delegates to
    :func:`~elisa.ui.tabs.system_visualization.logic.compute.run_visualization`.

    Each output is returned as a ``gr.update`` dict that sets both ``value``
    and ``visible`` so that a single click event is the authoritative source
    of truth for plot state, independent of the separate ``.change()`` event
    that drives the mode selector.

    :param prim_keys: Ordered keys for primary star parameters.
    :type prim_keys: tuple[str, ...]
    :param sec_keys: Ordered keys for secondary star parameters.
    :type sec_keys: tuple[str, ...]
    :param sys_keys: Ordered keys for binary system parameters.
    :type sys_keys: tuple[str, ...]
    :param obs_keys: Ordered keys for observer parameters.
    :type obs_keys: tuple[str, ...]
    :param puls_keys: Ordered keys for per-component pulsation parameters.
    :type puls_keys: tuple[str, ...]
    :returns: A callable suitable for ``gr.Button.click(fn=...)``.
    :rtype: Callable[..., tuple[dict, dict, dict, dict]]
    """

    def handler(*values: object) -> tuple[dict, dict, dict, dict]:
        idx = 0
        n_prim = len(prim_keys)
        n_sec = len(sec_keys)
        n_sys = len(sys_keys)
        n_puls = len(puls_keys)

        primary_params = dict(zip(prim_keys, values[idx : idx + n_prim], strict=True))
        idx += n_prim
        secondary_params = dict(zip(sec_keys, values[idx : idx + n_sec], strict=True))
        idx += n_sec
        system_params = dict(zip(sys_keys, values[idx : idx + n_sys], strict=True))
        idx += n_sys
        primary_puls_params = dict(zip(puls_keys, values[idx : idx + n_puls], strict=True))
        idx += n_puls
        secondary_puls_params = dict(zip(puls_keys, values[idx : idx + n_puls], strict=True))
        idx += n_puls
        observer_params = dict(zip(obs_keys, values[idx:], strict=True))

        try:
            mesh_fig, orbit_fig, equip_fig, surface_fig = compute.run_visualization(
                primary_params,
                secondary_params,
                system_params,
                observer_params,
                primary_puls_params,
                secondary_puls_params,
            )
        except Exception as exc:
            msg = str(exc)
            raise gr.Error(msg) from exc

        mode: str | None = observer_params.get("visualization_mode") or None
        return (
            gr.update(value=mesh_fig,    visible=mode == "mesh"),
            gr.update(value=orbit_fig,   visible=mode == "orbit"),
            gr.update(value=equip_fig,   visible=mode == "equipotential"),
            gr.update(value=surface_fig, visible=mode == "surface"),
        )

    return handler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build() -> None:
    """Build the System Visualization tab inside the active ``gr.Blocks`` context.

    Must be called from within a ``gr.Blocks`` (or ``gr.Tab`` inside
    ``gr.Blocks``) context manager. Creates the full form layout,
    output area, and event bindings.

    The visualization mode dropdown controls which input controls and output
    plot widgets are visible - switching to ``"mesh"`` hides the orbital frame
    control and the orbit plot; switching to ``"orbit"`` hides the phase/
    component controls and the mesh plot; switching to ``"surface"`` shows
    the shared phase/component controls and the colormap selector.

    :returns: ``None``
    :rtype: None
    """
    with gr.Tab("System Visualization"):
        gr.Markdown(
            "## System Visualization\n"
            "Configure both stellar components and the binary-system geometry, "
            "then click **Visualize**.",
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
                prim_puls_comps = pulsation_inputs.build("Primary")

            with gr.Column(scale=1):
                sec_comps = star_inputs.build("Secondary Star", defaults=_SECONDARY_DEFAULTS)
                sec_puls_comps = pulsation_inputs.build("Secondary")

            with gr.Column(scale=1):
                sys_comps = system_inputs.build(defaults=_SYSTEM_DEFAULTS)

            with gr.Column(scale=1):
                obs_comps = observer_inputs.build(defaults=_OBSERVER_DEFAULTS)

        # ------------------------------------------------------------------ #
        # Action                                                               #
        # ------------------------------------------------------------------ #
        with gr.Row():
            visualize_btn = gr.Button("🎨 Visualize System", variant="primary", scale=2)
            clear_btn = gr.Button("🗑 Clear outputs", variant="secondary", scale=1)

        # ------------------------------------------------------------------ #
        # Outputs - initial visibility matches the default mode               #
        # ------------------------------------------------------------------ #
        with gr.Row():
            mesh_plot = gr.Plot(label="Mesh Visualization", visible=False, scale=1)
            orbit_plot = gr.Plot(label="Orbital Motion", visible=False, scale=1)
            equipotential_plot = gr.Plot(label="Equipotential", visible=False, scale=1)
            surface_plot = gr.Plot(label="Surface Visualization", visible=False, scale=1)

        # ------------------------------------------------------------------ #
        # Event wiring                                                         #
        # ------------------------------------------------------------------ #
        prim_keys = star_inputs.FIELD_ORDER
        sec_keys = star_inputs.FIELD_ORDER
        sys_keys = system_inputs.FIELD_ORDER
        obs_keys = observer_inputs.FIELD_ORDER
        puls_keys = pulsation_inputs.FIELD_ORDER

        all_inputs: list[gr.Component] = (
            [prim_comps[k] for k in prim_keys]
            + [sec_comps[k] for k in sec_keys]
            + [sys_comps[k] for k in sys_keys]
            + [prim_puls_comps[k] for k in puls_keys]
            + [sec_puls_comps[k] for k in puls_keys]
            + [obs_comps[k] for k in obs_keys]
        )

        visualize_btn.click(
            fn=_make_handler(prim_keys, sec_keys, sys_keys, obs_keys, puls_keys),
            inputs=all_inputs,
            outputs=[mesh_plot, orbit_plot, equipotential_plot, surface_plot],
        )

        clear_btn.click(
            fn=lambda: (None, None, None, None),
            inputs=None,
            outputs=[mesh_plot, orbit_plot, equipotential_plot, surface_plot],
        )

        all_form_outputs: list[gr.Component] = (
            [prim_comps[k] for k in prim_keys]
            + [sec_comps[k] for k in sec_keys]
            + [sys_comps[k] for k in sys_keys]
        )
        json_file_input.upload(
            fn=make_json_load_handler(prim_keys, sec_keys, sys_keys),
            inputs=[json_file_input],
            outputs=all_form_outputs,
        )
