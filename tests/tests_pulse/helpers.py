# The tests package selects Astropy constants before test-module imports.


from elisa import units as u
from tests import set_astropy_units

set_astropy_units()


STAR_PARAMS = {
    "mass": 2.0 * u.solMass,
    "t_eff": 10000 * u.K,
    "gravity_darkening": 1.0,
    "discretization_factor": 5,
    "albedo": 1.0,
    "metallicity": 0.0,
    "polar_log_g": 4.0 * u.dex(u.cm / u.s**2),
}

SYSTEM_PARMAS = {
    "gamma": 0 * u.km / u.s,
    "inclination": 80 * u.deg,
    "rotation_period": 30 * u.d,
}

BINARY_STAR = {
    "mass": 2.0 * u.solMass,
    "t_eff": 10000 * u.K,
    "gravity_darkening": 1.0,
    "discretization_factor": 5,
    "albedo": 1.0,
    "metallicity": 0.0,
    "synchronicity": 1.0,
    "surface_potential": 10,
}

BINARY_SYSTEM = {
    "argument_of_periastron": 0 * u.deg,
    "gamma": 0 * u.km / u.s,
    "period": 2.0 * u.d,
    "eccentricity": 0.1,
    "inclination": 87 * u.deg,
    "primary_minimum_time": 0 * u.d,
    "phase_shift": 0.0,
}

TOL = 1e-3
