"""Core computation logic for system visualization.

This module is a pure-logic layer with no Gradio dependency. It translates
raw parameter dictionaries into ELISa objects, generates mesh and orbit
visualizations, and returns figures that can be rendered by the UI layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elisa import BinarySystem, Star
from elisa import units as u

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from elisa.types import Float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Keys that should be converted to int rather than float.
_INT_KEYS: frozenset[str] = frozenset({"discretization_factor"})

# Keys that are plain strings and must never be coerced to a number.
_STRING_KEYS: frozenset[str] = frozenset({"atmosphere"})


def _ensure_float(value: Float | str | None) -> Float | None:
    """Convert value to float, handling string inputs from Gradio.

    :param value: Numeric value, string representation, or ``None``.
    :type value: Float | str | None
    :returns: Parsed float or ``None``.
    :rtype: Float | None
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        return float(stripped)
    return float(value)


def _ensure_int(value: Float | str | None) -> int | None:
    """Convert value to int, handling string inputs from Gradio.

    :param value: Numeric value, string representation, or ``None``.
    :type value: Float | str | None
    :returns: Parsed int or ``None``.
    :rtype: int | None
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        return int(float(stripped))
    return int(value)


def _convert_params(params: dict) -> dict:
    """Convert all numeric string parameters to proper types.

    Processes a parameter dictionary, converting any string values to
    their appropriate numeric types. Empty strings are converted to
    ``None`` so that optional parameters can be cleanly filtered out
    before passing to ELISa. Non-numeric string values (e.g. atmosphere
    names) are left unchanged.

    :param params: Parameter dictionary (potentially with string values
        as returned by Gradio ``gr.Textbox`` components).
    :type params: dict
    :returns: Parameter dictionary with converted numeric values and
        empty strings replaced by ``None``.
    :rtype: dict
    """
    converted = {}
    for key, value in params.items():
        if key in _STRING_KEYS:
            converted[key] = value
        elif key in _INT_KEYS:
            converted[key] = _ensure_int(value)
        elif isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                # Empty optional textbox - let ELISa apply its own default
                converted[key] = None
            else:
                converted[key] = _ensure_float(stripped)
        else:
            converted[key] = value
    return converted


def _filter_none(params: dict) -> dict:
    """Return a copy of *params* with all ``None`` values removed.

    ELISa constructors treat missing keyword arguments as "use default",
    so we must not pass explicit ``None`` for optional fields.

    :param params: Parameter dictionary possibly containing ``None`` values.
    :type params: dict
    :returns: New dict with ``None`` entries dropped.
    :rtype: dict
    """
    return {k: v for k, v in params.items() if v is not None}


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def run_visualization(
    primary_params: dict,
    secondary_params: dict,
    system_params: dict,
    observer_params: dict,
) -> tuple[Figure | None, Figure | None, Figure | None, Figure | None]:
    """Generate mesh, orbit, equipotential, and/or surface visualizations for a binary system.

    Creates a binary system from the given parameters and produces matplotlib
    figures based on the visualization mode:

    - ``"mesh"``: 3D surface geometry at a given phase
    - ``"orbit"``: 2D orbital motion trajectory
    - ``"equipotential"``: 2D cross-section of Hill surface potentials
    - ``"surface"``: 3D shaded surface with an optional physical colormap

    :param primary_params: Primary star parameters (mass, T_eff, etc.).
    :type primary_params: dict
    :param secondary_params: Secondary star parameters.
    :type secondary_params: dict
    :param system_params: Binary system parameters (inclination, period, etc.).
    :type system_params: dict
    :param observer_params: Visualization parameters (mode, phase, components,
        plane, frame, colormap).
    :type observer_params: dict
    :returns: Tuple of (mesh_figure, orbit_figure, equipotential_figure,
        surface_figure) where any element may be ``None`` depending on the
        visualization mode.
    :rtype: tuple[Figure | None, Figure | None, Figure | None, Figure | None]
    :raises ValueError: If parameter values are invalid or binary construction fails.
    """
    primary_params = _filter_none(_convert_params(primary_params))
    secondary_params = _filter_none(_convert_params(secondary_params))
    system_params = _filter_none(_convert_params(system_params))

    visualization_mode: str = observer_params.get("visualization_mode") or ""
    phase: Float = _ensure_float(observer_params["phase"])
    components_to_plot: str = observer_params["components_to_plot"]
    plane: str = observer_params.get("plane", "xy")
    frame_of_reference: str = observer_params["frame_of_reference"]
    colormap: str | None = observer_params.get("colormap") or None
    elevation: Float | None = _ensure_float(observer_params.get("elevation"))
    azimuth: Float | None = _ensure_float(observer_params.get("azimuth"))

    if not visualization_mode:
        msg = "No visualization mode selected. Please choose one from the 'What to plot' dropdown."
        raise ValueError(msg)

    primary = Star(**primary_params)
    secondary = Star(**secondary_params)
    binary = BinarySystem(primary=primary, secondary=secondary, **system_params)

    mesh_fig = None
    orbit_fig = None
    equipotential_fig = None
    surface_fig = None

    if visualization_mode == "mesh":
        mesh_fig = binary.plot.mesh(
            phase=phase,
            components_to_plot=components_to_plot,
            return_figure_instance=True,
        )
    elif visualization_mode == "orbit":
        orbit_fig = binary.plot.orbit(
            start_phase=-0.5,
            stop_phase=1.5,
            number_of_points=300,
            axis_units=u.solRad,
            frame_of_reference=frame_of_reference,
            return_figure_instance=True,
        )
    elif visualization_mode == "equipotential":
        equipotential_fig = binary.plot.equipotential(
            plane=plane,
            phase=phase,
            components_to_plot=components_to_plot,
            return_figure_instance=True,
        )
    elif visualization_mode == "surface":
        surface_fig = binary.plot.surface(
            phase=phase,
            components_to_plot=components_to_plot,
            colormap=colormap,
            elevation=elevation,
            azimuth=azimuth,
            return_figure_instance=True,
        )

    return mesh_fig, orbit_fig, equipotential_fig, surface_fig

