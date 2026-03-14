"""Core computation logic for system visualization.

This module is a pure-logic layer with no Gradio dependency. It translates
raw parameter dictionaries into ELISa objects, generates mesh and orbit
visualizations, and returns figures that can be rendered by the UI layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from elisa import BinarySystem, Star
from elisa import units as u

if TYPE_CHECKING:
    from elisa.types import Float


def run_visualization(
    primary_params: dict,
    secondary_params: dict,
    system_params: dict,
    observer_params: dict,
) -> tuple[Figure, Figure]:
    """Generate mesh and orbit visualizations for a binary system.

    Creates a binary system from the given parameters and produces two
    matplotlib figures:
    1. Mesh visualization - 3D surface geometry of the stars at a given phase
    2. Orbit visualization - 2D orbital motion trajectory

    :param primary_params: Primary star parameters (mass, T_eff, etc.).
    :type primary_params: dict
    :param secondary_params: Secondary star parameters.
    :type secondary_params: dict
    :param system_params: Binary system parameters (inclination, period, etc.).
    :type system_params: dict
    :param observer_params: Visualization parameters (phase, components, frame).
    :type observer_params: dict
    :returns: Tuple of (mesh_figure, orbit_figure).
    :rtype: tuple[Figure, Figure]
    :raises ValueError: If parameter values are invalid or binary construction fails.
    """
    # Extract visualization parameters
    phase: Float = observer_params["phase"]
    components_to_plot: str = observer_params["components_to_plot"]
    frame_of_reference: str = observer_params["frame_of_reference"]

    # Build stellar components
    primary = Star(**primary_params)
    secondary = Star(**secondary_params)

    # Build binary system
    binary = BinarySystem(
        primary=primary,
        secondary=secondary,
        **system_params,
    )

    # Generate mesh visualization
    mesh_fig = binary.plot.mesh(
        phase=phase,
        components_to_plot=components_to_plot,
        return_figure_instance=True,
    )

    # Generate orbit visualization
    orbit_fig = binary.plot.orbit(
        start_phase=-0.5,
        stop_phase=1.5,
        number_of_points=300,
        axis_units=u.solRad,
        frame_of_reference=frame_of_reference,
        return_figure_instance=True,
    )

    return mesh_fig, orbit_fig

