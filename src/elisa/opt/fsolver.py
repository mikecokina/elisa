import numpy as np

from elisa import umpy as up, logger
from scipy.optimize import fsolve
from elisa.conf import config

config.set_up_logging()
__logger__ = logger.getLogger(__name__)


def fsolver(fn, condition, *args, **kwargs):
    """
    Will solve `fn` implicit function taking args by using scipy.optimize.fsolve method and return
    solution if satisfy conditional function.

    :param fn: function
    :param condition: function
    :param args: Tuple
    :param kwargs: Dict
        * **original_kwargs** *
    :return: Tuple
    """
    # precalculation of auxiliary values
    solution, use = np.nan, False
    scipy_solver_init_value = np.array([1e-4])
    try:
        solution, _, ier, msg = fsolve(fn, scipy_solver_init_value, full_output=True, args=args, xtol=1e-10)
        if ier == 1 and not up.isnan(solution[0]):
            solution = solution[0]
            use = True if 1e15 > solution > 0 else False
        else:
            __logger__.warning(f'solution in implicit solver was not found, cause: {msg}')
    except ValueError:
        raise
    except Exception as e:
        __logger__.debug(f"attempt to solve function {fn.__name__} finished w/ exception: {str(e)}")
        use = False

    args_to_use = kwargs.get('original_kwargs', args)
    return (solution, use) if condition(solution, *args_to_use) else (np.nan, False)
