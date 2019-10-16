import numpy as np

from elisa import umpy as up
from scipy.optimize import fsolve


def solver(fn, condition, *args, **kwargs):
    """
    Will solve `fn` implicit function taking args by using scipy.optimize.fsolve method and return
    solution if satisfy conditional function.

    :param fn: function
    :param condition: function
    :param args: tuple
    :return: float, bool
    """
    # precalculation of auxiliary values
    solution, use = np.nan, False
    scipy_solver_init_value = np.array([1e-4])
    try:
        solution, _, ier, mesg = fsolve(fn, scipy_solver_init_value, full_output=True, args=args, xtol=1e-10)
        if ier == 1 and not up.isnan(solution[0]):
            solution = solution[0]
            use = True if 1e15 > solution > 0 else False
    except Exception:
        use = False

    args_to_use = kwargs.get('original_kwargs', args)
    return (solution, use) if condition(solution, *args_to_use) else (np.nan, False)
