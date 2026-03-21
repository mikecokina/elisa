from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import const
from elisa.base.types import FLOAT
from elisa.opt.fsolver import fsolve
from elisa.single_system import model

if TYPE_CHECKING:
    from elisa.types import Float


def calculate_radius(mass: Float, angular_velocity: Float, surface_potential: Float, *args: Float) -> Float:
    """Calculate radius along a direction given in spherical coordinates.

    The function solves for the radius along an arbitrary direction
    (provided as additional positional ``args``, typically ``theta``)
    by finding a root of the radial potential equation using the
    project-specific solver.

    :param mass: Stellar mass.
    :type mass: elisa.types.Float
    :param angular_velocity: Angular speed of rotation.
    :type angular_velocity: elisa.types.Float
    :param surface_potential: Target surface potential value.
    :type surface_potential: elisa.types.Float
    :param args: Additional positional floats describing direction (e.g. theta).
    :type args: tuple[elisa.types.Float, ...]
    :returns: Computed radius value.
    :rtype: elisa.types.Float
    :raises ValueError: When solver returns an invalid radius.
    """
    fn = model.potential_fn
    precalc = model.pre_calculate_for_potential_value

    precalc_args: tuple[Float, ...] = (mass, angular_velocity, *args)
    init_val = -const.G * mass / surface_potential
    scipy_solver_init_value = np.array([init_val], dtype=float)
    argss = (precalc(*precalc_args), surface_potential)
    solution, _, ier, _ = fsolve(fn, scipy_solver_init_value, full_output=True, args=argss, xtol=1e-10)

    # check for regular solution
    if ier == 1 and not np.isnan(solution[0]) and 5 * init_val >= solution[0] >= 0:
        return FLOAT(solution[0])

    # assign message before raising as per project convention
    if not (0 < solution[0] < 5 * init_val):
        msg = f"Invalid value of radius {solution} was calculated."
        raise ValueError(msg)

    return FLOAT(solution[0])


def calculate_polar_radius(mass: Float, angular_velocity: Float, surface_potential: Float) -> Float:
    """Return polar radius (theta == 0).

    :param mass: Stellar mass.
    :type mass: elisa.types.Float
    :param angular_velocity: Angular speed of rotation.
    :type angular_velocity: elisa.types.Float
    :param surface_potential: Target surface potential value.
    :type surface_potential: elisa.types.Float
    :returns: Polar radius value.
    :rtype: elisa.types.Float
    """
    return calculate_radius(mass, angular_velocity, surface_potential, 0.0)


def calculate_equatorial_radius(mass: Float, angular_velocity: Float, surface_potential: Float) -> Float:
    """Return equatorial radius (theta == pi/2).

    :param mass: Stellar mass.
    :type mass: elisa.types.Float
    :param angular_velocity: Angular speed of rotation.
    :type angular_velocity: elisa.types.Float
    :param surface_potential: Target surface potential value.
    :type surface_potential: elisa.types.Float
    :returns: Equatorial radius value.
    :rtype: elisa.types.Float
    """
    return calculate_radius(mass, angular_velocity, surface_potential, const.HALF_PI)
