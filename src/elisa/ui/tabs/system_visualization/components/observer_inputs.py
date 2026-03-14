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

# Keys used inside the returned component dict for the column handles.
# They are NOT part of FIELD_ORDER so the computation handler ignores them.
COL_SHARED = "_col_shared"  # phase + components_to_plot (mesh, equipotential & surface)
COL_EQUIP = "_col_equip"  # plane (equipotential only)
COL_ORBIT = "_col_orbit"  # frame_of_reference (orbit only)
COL_SURFACE = "_col_surface"  # colormap (surface only)


def update_ui(mode: str | None) -> tuple[dict, dict, dict, dict, dict, dict, dict, dict]:
    """Return ``gr.update`` dicts to show/hide column wrappers and output plots.

    Toggles four ``gr.Column`` containers (one per control group) rather
    than individual components - Gradio handles container visibility more
    reliably than per-component visibility.

    The caller must list outputs in the same order as the return tuple:
    ``[_col_shared, _col_equip, _col_orbit, _col_surface,
    mesh_plot, orbit_plot, equipotential_plot, surface_plot]``.

    :param mode: Selected visualization mode - one of ``"mesh"``,
        ``"orbit"``, ``"equipotential"``, ``"surface"``, or ``None``
        (nothing selected).
    :type mode: str | None
    :returns: Eight ``gr.update`` dicts for
        (_col_shared, _col_equip, _col_orbit, _col_surface,
        mesh_plot, orbit_plot, equipotential_plot, surface_plot).
    :rtype: tuple[dict, dict, dict, dict, dict, dict, dict, dict]
    """
    show_mesh = mode == "mesh"
    show_orbit = mode == "orbit"
    show_equip = mode == "equipotential"
    show_surface = mode == "surface"
    show_shared = show_mesh or show_equip or show_surface
    return (
        gr.update(visible=show_shared),  # _col_shared
        gr.update(visible=show_equip),  # _col_equip
        gr.update(visible=show_orbit),  # _col_orbit
        gr.update(visible=show_surface),  # _col_surface
        gr.update(visible=show_mesh, value=None),  # mesh_plot
        gr.update(visible=show_orbit, value=None),  # orbit_plot
        gr.update(visible=show_equip, value=None),  # equipotential_plot
        gr.update(visible=show_surface, value=None),  # surface_plot
    )


def build(*, defaults: dict[str, Float | str | None] | None = None) -> dict[str, gr.Component]:
    """Render visualization parameter inputs and return a component mapping.

    Creates a mode dropdown at the top.  Mode-specific controls are
    grouped in ``gr.Column`` containers (stored under :data:`COL_SHARED`,
    :data:`COL_EQUIP`, :data:`COL_ORBIT`, and :data:`COL_SURFACE` in the
    returned dict) whose visibility is toggled by :func:`update_ui`.  The
    columns start hidden when no mode is selected so the page loads clean.

    The returned dict contains both the actual input components (keyed by
    :data:`FIELD_ORDER`) and the column handles (keyed by the ``COL_*``
    constants).  Only :data:`FIELD_ORDER` entries should be used as form
    inputs; the column handles are only for visibility wiring.

    :param defaults: Optional mapping of field name - default value.
        Unrecognised keys are silently ignored.
    :type defaults: dict[str, Float | str | None] | None
    :returns: Dict with FIELD_ORDER entries plus column handle entries.
    :rtype: dict[str, gr.Component]
    """
    if defaults is None:
        defaults = {}

    mode_default: str | None = defaults.get("visualization_mode")  # type: ignore[assignment]

    components: dict[str, gr.Component] = {}

    gr.Markdown("### Visualization Settings")

    components["visualization_mode"] = gr.Dropdown(
        choices=_VISUALIZATION_MODES,
        value=mode_default,
        label="What to plot",
        info="Select a plot type - relevant controls appear automatically.",
    )

    # ALL columns are created WITHOUT visible= parameter so they're always
    # fully mounted in the DOM from the start. The .change() event will
    # control their visibility. This is the only reliable way to avoid
    # Gradio's visibility toggling bugs.
    with gr.Column() as col_shared:
        phase_default = defaults.get("phase", 0.0)
        components["phase"] = gr.Slider(
            minimum=-0.5,
            maximum=1.5,
            step=0.01,
            value=phase_default,
            label="Orbital Phase",
            info="Phase at which to display the system geometry.",
        )
        comp_default = defaults.get("components_to_plot", "both")
        components["components_to_plot"] = gr.Dropdown(
            choices=_COMPONENTS_CHOICES,
            value=comp_default,
            label="Components to Plot",
            info="Which stellar components to include.",
        )

    with gr.Column() as col_equip:
        plane_default = defaults.get("plane", "xy")
        components["plane"] = gr.Dropdown(
            choices=_PLANE_CHOICES,
            value=plane_default,
            label="Cross-section Plane",
            info="Plane for the equipotential cross-section.",
        )

    with gr.Column() as col_orbit:
        frame_default = defaults.get("frame_of_reference", "primary")
        components["frame_of_reference"] = gr.Dropdown(
            choices=_FRAME_CHOICES,
            value=frame_default,
            label="Orbital Frame of Reference",
            info="Coordinate frame for the orbital motion plot.",
        )

    with gr.Column() as col_surface:
        colormap_default = defaults.get("colormap")
        components["colormap"] = gr.Dropdown(
            choices=_COLORMAP_CHOICES,
            value=colormap_default,
            label="Surface Colormap",
            info=(
                "Physical quantity to map onto the surface color. "
                "Leave blank for solid colors."
            ),
        )
        elevation_default = defaults.get("elevation", 0.0)
        components["elevation"] = gr.Slider(
            minimum=-90,
            maximum=90,
            step=1,
            value=elevation_default,
            label="Camera Elevation",
            info="Vertical angle of the camera in degrees (0 = equator, 90 = top).",
        )
        azimuth_default = defaults.get("azimuth", 0.0)
        components["azimuth"] = gr.Slider(
            minimum=-180,
            maximum=180,
            step=1,
            value=azimuth_default,
            label="Camera Azimuth",
            info="Horizontal rotation of the camera in degrees.",
        )

    components[COL_SHARED]  = col_shared
    components[COL_EQUIP]   = col_equip
    components[COL_ORBIT]   = col_orbit
    components[COL_SURFACE] = col_surface

    return components
