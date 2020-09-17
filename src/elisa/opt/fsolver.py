import numpy as np

from elisa import umpy as up
from scipy.optimize import fsolve
from .. logger import getLogger

logger = getLogger('opt.fsolver')


def fsolver(fn, condition, *args, **kwargs):
    """
    Will solve `fn` implicit function taking args by using scipy.optimize.fsolve method and return
    solution if satisfy conditional function.

    :param fn: function;
    :param condition: function;
    :param args: Tuple;
    :param kwargs: Dict;
        * **original_kwargs** *
    :return: Tuple;
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
            logger.warning(f'solution in implicit solver was not found, cause: {msg}')
    except Exception as e:
        logger.debug(f"attempt to solve function {fn.__name__} finished w/ exception: {str(e)}")
        use = False
        raise

    args_to_use = kwargs.get('original_kwargs', args)
    return (solution, use) if condition(solution, *args_to_use) else (np.nan, False)
