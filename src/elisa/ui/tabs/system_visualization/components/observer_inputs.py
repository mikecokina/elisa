"""Gradio component builder for visualization observer parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from elisa.types import Float


FIELD_ORDER: tuple[str, ...] = (
    "visualization_mode",
    "phase",
    "components_to_plot",
    "plane",
    "frame_of_reference",
    "colormap",
    "elevation",
    "azimuth",
)

_VISUALIZATION_MODES: list[str] = ["mesh", "orbit", "equipotential", "surface"]
_COMPONENTS_CHOICES: list[str] = ["both", "primary", "secondary"]
_PLANE_CHOICES: list[str] = ["xy", "yz", "zx"]
_FRAME_CHOICES: list[str] = ["primary", "barycentric"]
_COLORMAP_CHOICES: list[str | None] = [
    None,
    "temperature",
    "gravity_acceleration",
    "radius",
    "velocity",
    "radial_velocity",
    "normal_radiance",
    "radiance",
]

# Ordered tuple of the 7 mode-controlled component keys.
# :func:`update_ui` returns one ``gr.update(interactive=...)`` per key in
# this order, followed by four ``gr.update(value=None)`` plot clears.
# Tab event-wiring uses this constant to build the outputs list dynamically.
MODE_CONTROLLED_KEYS: tuple[str, ...] = (
    "phase",
    "components_to_plot",
    "plane",
    "frame_of_reference",
    "colormap",
    "elevation",
    "azimuth",
)


def update_ui(mode: str | None) -> tuple[dict, ...]:
    """Return ``gr.update`` dicts to enable/disable individual inputs and clear plots.

    Uses ``interactive=True/False`` on each mode-controlled component -
    the same mechanism used by parameter rows in RV/LC fitting tabs -
    because this is the only update path that Gradio 6.x reliably propagates
    to every component type at runtime.

    All four output plots are cleared (``value=None``) on every mode change
    so the 2x2 grid always starts blank before the user clicks **Visualize**.

    The caller must list outputs in the same order as the return tuple:

    ``[phase, components_to_plot, plane, frame_of_reference,``
    ``colormap, elevation, azimuth,``
    ``mesh_plot, orbit_plot, equipotential_plot, surface_plot]``

    which is exactly ``[obs_comps[k] for k in MODE_CONTROLLED_KEYS]``
    followed by the four plot components.

    :param mode: Selected visualization mode - one of ``"mesh"``,
        ``"orbit"``, ``"equipotential"``, ``"surface"``, or ``None``
        (nothing selected).
    :type mode: str | None
    :returns: Eleven ``gr.update`` dicts (7 interactive + 4 plot clears).
    :rtype: tuple[dict, ...]
    """
    show_mesh = mode == "mesh"
    show_orbit = mode == "orbit"
    show_equip = mode == "equipotential"
    show_surface = mode == "surface"
    show_shared = show_mesh or show_equip or show_surface
    return (
        gr.update(interactive=show_shared),   # phase
        gr.update(interactive=show_shared),   # components_to_plot
        gr.update(interactive=show_equip),    # plane
        gr.update(interactive=show_orbit),    # frame_of_reference
        gr.update(interactive=show_surface),  # colormap
        gr.update(interactive=show_surface),  # elevation
        gr.update(interactive=show_surface),  # azimuth
        gr.update(value=None),                # mesh_plot
        gr.update(value=None),                # orbit_plot
        gr.update(value=None),                # equipotential_plot
        gr.update(value=None),                # surface_plot
    )


def build(*, defaults: dict[str, Float | str | None] | None = None) -> dict[str, gr.Component]:
    """Render visualization parameter inputs and return a component mapping.

    Creates a mode dropdown at the top followed by four ``gr.Group`` sections
    (one per parameter set) that are always visible. Each mode-controlled
    component is rendered with the correct ``interactive=`` state for the
    initial mode so the form is already in the right state on first load.
    When the mode dropdown changes, :func:`update_ui` issues
    ``gr.update(interactive=...)`` for every mode-controlled component.

    The ``gr.Group`` containers serve only as visual layout wrappers (they
    add a subtle border that groups related inputs); they are **not** used
    for any runtime state update.

    :param defaults: Optional mapping of field name - default value.
        Unrecognised keys are silently ignored.
    :type defaults: dict[str, Float | str | None] | None
    :returns: Dict keyed by :data:`FIELD_ORDER` entries.
    :rtype: dict[str, gr.Component]
    """
    if defaults is None:
        defaults = {}

    mode_default: str | None = defaults.get("visualization_mode")  # type: ignore[assignment]

    show_mesh = mode_default == "mesh"
    show_orbit = mode_default == "orbit"
    show_equip = mode_default == "equipotential"
    show_surface = mode_default == "surface"
    show_shared = show_mesh or show_equip or show_surface

    components: dict[str, gr.Component] = {}

    gr.Markdown("### Visualization Settings")

    components["visualization_mode"] = gr.Dropdown(
        choices=_VISUALIZATION_MODES,
        value=mode_default,
        label="What to plot",
        info="Select a plot type - relevant controls appear automatically.",
    )

    # --------------------------------------------------------------------------
    # Phase & Components - active for: mesh, equipotential, surface
    # --------------------------------------------------------------------------
    with gr.Group():
        gr.Markdown("**Phase & Components**", elem_classes=["section-header"])
        phase_default = defaults.get("phase", 0.0)
        _phase_cls = [] if show_shared else ["viz-control-disabled"]
        components["phase"] = gr.Slider(
            minimum=-0.5,
            maximum=1.5,
            step=0.01,
            value=phase_default,
            label="Orbital Phase",
            info="Phase at which to display the system geometry.",
            interactive=True,
            elem_id="viz-ctrl-phase",
            elem_classes=_phase_cls,
        )
        comp_default = defaults.get("components_to_plot", "both")
        components["components_to_plot"] = gr.Dropdown(
            choices=_COMPONENTS_CHOICES,
            value=comp_default,
            label="Components to Plot",
            info="Which stellar components to include.",
            interactive=True,
            elem_id="viz-ctrl-components_to_plot",
            elem_classes=_phase_cls,
        )

    # --------------------------------------------------------------------------
    # Cross-section plane - active for: equipotential
    # --------------------------------------------------------------------------
    with gr.Group():
        gr.Markdown("**Cross-section Plane**", elem_classes=["section-header"])
        plane_default = defaults.get("plane", "xy")
        _equip_cls = [] if show_equip else ["viz-control-disabled"]
        components["plane"] = gr.Dropdown(
            choices=_PLANE_CHOICES,
            value=plane_default,
            label="Cross-section Plane",
            info="Plane for the equipotential cross-section.",
            interactive=True,
            elem_id="viz-ctrl-plane",
            elem_classes=_equip_cls,
        )

    # --------------------------------------------------------------------------
    # Frame of reference - active for: orbit
    # --------------------------------------------------------------------------
    with gr.Group():
        gr.Markdown("**Frame of Reference**", elem_classes=["section-header"])
        frame_default = defaults.get("frame_of_reference", "primary")
        _orbit_cls = [] if show_orbit else ["viz-control-disabled"]
        components["frame_of_reference"] = gr.Dropdown(
            choices=_FRAME_CHOICES,
            value=frame_default,
            label="Orbital Frame of Reference",
            info="Coordinate frame for the orbital motion plot.",
            interactive=True,
            elem_id="viz-ctrl-frame_of_reference",
            elem_classes=_orbit_cls,
        )

    # --------------------------------------------------------------------------
    # Surface colormap & camera - active for: surface
    # --------------------------------------------------------------------------
    with gr.Group():
        gr.Markdown("**Surface Colormap & Camera**", elem_classes=["section-header"])
        colormap_default = defaults.get("colormap")
        _surface_cls = [] if show_surface else ["viz-control-disabled"]
        components["colormap"] = gr.Dropdown(
            choices=_COLORMAP_CHOICES,
            value=colormap_default,
            label="Surface Colormap",
            info="Physical quantity to map onto the surface color. Leave blank for solid colors.",
            interactive=True,
            elem_id="viz-ctrl-colormap",
            elem_classes=_surface_cls,
        )
        elevation_default = defaults.get("elevation", 0.0)
        components["elevation"] = gr.Slider(
            minimum=-90,
            maximum=90,
            step=1,
            value=elevation_default,
            label="Camera Elevation",
            info="Vertical angle of the camera in degrees (0 = equator, 90 = top).",
            interactive=True,
            elem_id="viz-ctrl-elevation",
            elem_classes=_surface_cls,
        )
        azimuth_default = defaults.get("azimuth", 90.0)
        components["azimuth"] = gr.Slider(
            minimum=-180,
            maximum=180,
            step=1,
            value=azimuth_default,
            label="Camera Azimuth",
            info="Horizontal rotation of the camera in degrees.",
            interactive=True,
            elem_id="viz-ctrl-azimuth",
            elem_classes=_surface_cls,
        )

    return components
