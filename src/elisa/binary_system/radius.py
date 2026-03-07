from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from elisa import const
from elisa import umpy as up
from elisa.base.types import FLOAT
from elisa.binary_system import model
from elisa.opt.fsolver import fsolve

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from elisa.types import Float

ComponentName: TypeAlias = Literal["primary", "secondary"]

MAXIMAL_RADIUS_BOUNDARY = 30.0


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
    radius at which the potential equals ``surface_potential``.

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
    :raises ValueError: If ``component`` is invalid or if the computed radius
        is outside the accepted range.
    """
    if component == "primary":
        potential_fn = model.potential_primary_fn
        precalculate_fn = model.pre_calculate_for_potential_value_primary
    elif component == "secondary":
        potential_fn = model.potential_secondary_fn
        precalculate_fn = model.pre_calculate_for_potential_value_secondary
    else:
        message = (
            f"Invalid value of `component` argument {component}. "
            "Expecting `primary` or `secondary`."
        )
        raise ValueError(message)

    precalc_args = (
        synchronicity,
        mass_ratio,
        components_distance,
        phi,
        theta,
    )
    solver_init_value = np.array([1e-4], dtype=np.float64)
    solver_args = (
        (mass_ratio, *precalculate_fn(*precalc_args)),
        surface_potential,
    )

    solution, _, ier, _ = fsolve(
        potential_fn,
        solver_init_value,
        full_output=True,
        args=solver_args,
        xtol=1e-10,
    )

    radius = solution[0]

    if ier == 1 and not up.isnan(radius) and MAXIMAL_RADIUS_BOUNDARY >= radius >= 0.0:
        return radius

    if not (0.0 < radius < 1.0):
        message = f"Invalid value of radius {solution} was calculated."
        raise ValueError(message)

    return radius


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
        distances: ArrayLike[Float],
        surface_potential: ArrayLike[Float],
        mass_ratio: Float,
        synchronicity: Float,
        component: ComponentName,
) -> list[Float]:
    """Calculate forward radii for an array of component distances.

    The function evaluates forward radii of the selected component for each
    supplied component distance and matching surface potential entry.

    :param distances: Component distances at which to calculate forward radii.
    :type distances: ArrayLike
    :param surface_potential: Surface potential values corresponding to each
        distance.
    :type surface_potential: ArrayLike
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
