import numpy as np

from . import model
from .. import const
from .. opt.fsolver import fsolve


def calculate_radius(mass, angular_velocity, surface_potential, *args):
    """
    Function calculates radius of the star in given direction of arbitrary direction vector (in spherical
    coordinates) starting from the centre of the star.

    :param mass: float;
    :param angular_velocity: float;
    :param surface_potential: float;
    :param args: Tuple;

    ::

        (
            theta: float - latitudonal angle of direction vector measured from north pole (in radians)
        )

    :return: float; radius
    """
    fn = model.potential_fn
    precalc = model.pre_calculate_for_potential_value

    precalc_args = (mass, angular_velocity) + args
    init_val = - const.G * mass / surface_potential
    scipy_solver_init_value = np.array([init_val])
    argss = (precalc(*precalc_args), surface_potential)
    solution, _, ier, _ = fsolve(fn, scipy_solver_init_value, full_output=True, args=argss, xtol=1e-10)
    # check for regular solution
    if ier == 1 and not np.isnan(solution[0]) and 5 * init_val >= solution[0] >= 0:
        return solution[0]
    else:
        if not (0 < solution[0] < 5 * init_val):
            raise ValueError(f'Invalid value of radius {solution} was calculated.')
        return solution[0]


def calculate_polar_radius(mass, angular_velocity, surface_potential):
    args = (0.0, )
    return calculate_radius(mass, angular_velocity, surface_potential, *args)


def calculate_equatorial_radius(mass, angular_velocity, surface_potential):
    args = (const.HALF_PI, )
    return calculate_radius(mass, angular_velocity, surface_potential, *args)
