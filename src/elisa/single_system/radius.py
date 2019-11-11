import numpy as np
import scipy

from scipy import optimize
from elisa.single_system import model
from elisa import (
    const,
)


def calculate_radius(mass, angular_velocity, surface_potential, *args):
    fn = model.potential_fn
    precalc = model.pre_calculate_for_potential_value

    precalc_args = (mass, angular_velocity) + args
    scipy_solver_init_value = np.array([- const.G * mass / surface_potential])


def calculate_polar_radius():
    args = ()
    return calculate_radius()


def calculate_equatorial_radius():
    args = ()
    return calculate_radius()
