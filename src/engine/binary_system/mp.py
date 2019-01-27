import numpy as np
import scipy

from engine.binary_system import model


def prepare_get_surface_points_args(preacalc_vals_args, mass_ratio, surface_potential):
    return ((idx, mass_ratio, surface_potential) +
            preacalc_vals_arg for idx, preacalc_vals_arg in enumerate(preacalc_vals_args))


def get_surface_points_worker(*args):
    """
    function solves radius for given azimuths that are passed in *args
    """

    potential_fn, xargs = args
    potential_fn = getattr(model, potential_fn)

    _idx, _args = xargs[0], xargs[1:]
    solver_init_value = np.array([1. / 10000.])
    solution, _, ier, _ = scipy.optimize.fsolve(potential_fn, solver_init_value, full_output=True,
                                                args=_args, xtol=1e-12)
    return _idx, solution[0]
