from __future__ import annotations

from typing import TYPE_CHECKING

from elisa import const
from elisa.base.error import (
    raise_elisa_unexpected_error,
)
from elisa.base.types import FLOAT
from elisa.binary_system._radius_solver import RADIUS_LIMIT, solve_radius
from elisa.conf.enums import BinaryComponent

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import ComponentName, Float

MAXIMAL_RADIUS_BOUNDARY = RADIUS_LIMIT


def calculate_radius(
    synchronicity: Float,
    mass_ratio: Float,
    surface_potential: Float,
    component: ComponentName,
    components_distance: Float,
    phi: Float,
    theta: Float,
) -> Float:
    """Calculate the stellar radius in an arbitrary direction.

    The radius is measured from the center of the selected component in the
    direction defined by spherical coordinates ``phi`` and ``theta``. The
    direction is evaluated in the Roche geometry by numerically solving for the
    first intersection on the descending branch of the local radial potential.
    The interval ends at the first radial minimum (before the companion on the
    forward axis), or at ``MAXIMAL_RADIUS_BOUNDARY`` for a decreasing ray.
    A rounded critical potential may return the limiting stationary radius;
    a potential demonstrably below the minimum raises ``ElisaNumericError``.
    The result is independent of previous calls and uses the supplied instantaneous
    separation and spin, including eccentric and asynchronous geometries.

    The spherical coordinates are interpreted as follows:

    - ``phi`` is the longitudinal angle of the direction vector measured from
      the point under :math:`L_1` in the positive direction, in radians.
    - ``theta`` is the latitudinal angle of the direction vector measured from
      the North Pole, in radians.

    :param synchronicity: Rotational synchronicity factor of the component.
    :type synchronicity: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param surface_potential: Surface potential of the component.
    :type surface_potential: Float
    :param component: Component identifier, either ``"primary"`` or
        ``"secondary"``.
    :type component: Literal["primary", "secondary"]
    :param components_distance: Distance between components in semi-major-axis
        units.
    :type components_distance: Float
    :param phi: Longitudinal angle of the direction vector in radians.
    :type phi: Float
    :param theta: Latitudinal angle of the direction vector in radians.
    :type theta: Float
    :return: Radius in the requested direction.
    :rtype: Float
    :raises ElisaUnexpectedError: If the internal component selector is invalid.
    :raises ElisaMaxIterationError: If the radius solver does not converge.
    :raises ElisaNumericError: If the computed radius is outside the accepted range.
    """
    if component not in (BinaryComponent.PRIMARY, BinaryComponent.SECONDARY):
        message = f"Radius calculation received an unexpected component {component!r}; expected primary or secondary."
        raise_elisa_unexpected_error(message)
    return solve_radius(
        synchronicity, mass_ratio, components_distance, surface_potential,
        secondary=component == BinaryComponent.SECONDARY, phi=phi, theta=theta,
    )


def calculate_polar_radius(
    synchronicity: Float,
    mass_ratio: Float,
    components_distance: Float,
    surface_potential: Float,
    component: ComponentName,
) -> Float:
    """Calculate the stellar radius in the direction of the pole.

    :param synchronicity: Rotational synchronicity factor of the component.
    :type synchronicity: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param components_distance: Distance between components.
    :type components_distance: Float
    :param surface_potential: Surface potential of the component.
    :type surface_potential: Float
    :param component: Component identifier, either ``"primary"`` or
        ``"secondary"``.
    :type component: Literal["primary", "secondary"]
    :return: Polar radius.
    :rtype: Float
    """
    return calculate_radius(
        synchronicity=synchronicity,
        mass_ratio=mass_ratio,
        surface_potential=surface_potential,
        component=component,
        components_distance=components_distance,
        phi=0.0,
        theta=0.0,
    )


def calculate_side_radius(
    synchronicity: Float,
    mass_ratio: Float,
    components_distance: Float,
    surface_potential: Float,
    component: ComponentName,
) -> Float:
    """Calculate the stellar radius perpendicular to pole and join vector.

    This radius is evaluated in the direction perpendicular to both the pole
    direction and the component-joining vector.

    :param synchronicity: Rotational synchronicity factor of the component.
    :type synchronicity: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param components_distance: Distance between components.
    :type components_distance: Float
    :param surface_potential: Surface potential of the component.
    :type surface_potential: Float
    :param component: Component identifier, either ``"primary"`` or
        ``"secondary"``.
    :type component: Literal["primary", "secondary"]
    :return: Side radius.
    :rtype: Float
    """
    return calculate_radius(
        synchronicity=synchronicity,
        mass_ratio=mass_ratio,
        surface_potential=surface_potential,
        component=component,
        components_distance=components_distance,
        phi=const.HALF_PI,
        theta=const.HALF_PI,
    )


def calculate_backward_radius(
    synchronicity: Float,
    mass_ratio: Float,
    components_distance: Float,
    surface_potential: Float,
    component: ComponentName,
) -> Float:
    """Calculate the stellar radius in the direction away from the companion.

    :param synchronicity: Rotational synchronicity factor of the component.
    :type synchronicity: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param components_distance: Distance between components.
    :type components_distance: Float
    :param surface_potential: Surface potential of the component.
    :type surface_potential: Float
    :param component: Component identifier, either ``"primary"`` or
        ``"secondary"``.
    :type component: Literal["primary", "secondary"]
    :return: Backward radius.
    :rtype: Float
    """
    return calculate_radius(
        synchronicity=synchronicity,
        mass_ratio=mass_ratio,
        surface_potential=surface_potential,
        component=component,
        components_distance=components_distance,
        phi=const.PI,
        theta=const.HALF_PI,
    )


def calculate_forward_radius(
    synchronicity: Float,
    mass_ratio: Float,
    components_distance: Float,
    surface_potential: Float,
    component: ComponentName,
) -> Float:
    """Calculate the stellar radius in the direction toward the companion.

    :param synchronicity: Rotational synchronicity factor of the component.
    :type synchronicity: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param components_distance: Distance between components.
    :type components_distance: Float
    :param surface_potential: Surface potential of the component.
    :type surface_potential: Float
    :param component: Component identifier, either ``"primary"`` or
        ``"secondary"``.
    :type component: Literal["primary", "secondary"]
    :return: Forward radius.
    :rtype: Float
    """
    return calculate_radius(
        synchronicity=synchronicity,
        mass_ratio=mass_ratio,
        surface_potential=surface_potential,
        component=component,
        components_distance=components_distance,
        phi=0.0,
        theta=const.HALF_PI,
    )


def calculate_forward_radii(
    distances: NDArray[Float],
    surface_potential: NDArray[Float],
    mass_ratio: Float,
    synchronicity: Float,
    component: ComponentName,
) -> list[Float]:
    """Calculate forward radii for an array of component distances.

    The function evaluates forward radii of the selected component for each
    supplied component distance and matching surface potential entry.

    :param distances: Component distances at which to calculate forward radii.
    :type distances: NDArray
    :param surface_potential: Surface potential values corresponding to each
        distance.
    :type surface_potential: NDArray
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param synchronicity: Rotational synchronicity factor of the component.
    :type synchronicity: Float
    :param component: Component identifier, either ``"primary"`` or
        ``"secondary"``.
    :type component: Literal["primary", "secondary"]
    :return: Forward radii for all supplied distances.
    :rtype: list[Float]
    """
    return [
        calculate_forward_radius(
            synchronicity=synchronicity,
            mass_ratio=mass_ratio,
            components_distance=FLOAT(distance),
            surface_potential=FLOAT(surface_potential[index]),
            component=component,
        )
        for index, distance in enumerate(distances)
    ]
