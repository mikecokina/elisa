from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import settings
from elisa import umpy as up
from elisa.logger import getLogger

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from elisa.types import Float

logger = getLogger("opt.fsolver")

MAX_EXPECTED_SOLUTION_BOUNDARY = 1e15


def fsolve(
        func: Callable,
        x0: NDArray,
        args: tuple = (),
        fprime: Callable | None = None,
        xtol: Float = 1.49012e-8,
        maxfev: int = 0,
        band: tuple[int, int] | None = None,
        epsfcn: Float | None = None,
        factor: Float = 100,
        diag: NDArray | None = None,
        *,
        full_output: bool = False,
        col_deriv: bool = False,
) -> NDArray[Float] | tuple[NDArray[Float], dict, int, str]:
    """Wrap :func:`scipy.optimize.fsolve` via :mod:`elisa.umpy`.

    This helper preserves the original behavior of returning only the solution
    unless ``full_output`` is requested.

    :param func: Function whose roots are to be found.
    :param x0: Initial guess for the roots.
    :param args: Extra positional arguments passed to ``func``.
    :param fprime: Optional Jacobian function.
    :param full_output: Whether to return the full SciPy solver output.
    :param col_deriv: Whether the Jacobian computes derivatives down columns.
    :param xtol: Relative error tolerance for convergence.
    :param maxfev: Maximum number of function evaluations.
    :param band: Optional banded Jacobian specification.
    :param epsfcn: Step length for forward-difference Jacobian approximation.
    :param factor: Parameter determining the initial step bound.
    :param diag: Optional scaling factors for the variables.
    :returns: Solution array, or full SciPy solver output if ``full_output`` is ``True``.
    """
    solution, info, ier, msg = up.optimize.fsolve(
        func,
        x0,
        args=args,
        fprime=fprime,
        full_output=True,
        col_deriv=col_deriv,
        xtol=xtol,
        maxfev=maxfev,
        band=band,
        epsfcn=epsfcn,
        factor=factor,
        diag=diag,
    )
    if not full_output:
        return solution
    return solution, info, ier, msg


def fsolver(
        fn: Callable,
        condition: Callable,
        *args: tuple,
        **kwargs: dict,
) -> tuple[Float, bool]:
    """Solve an implicit function and validate the result with a condition.

    The function uses :func:`scipy.optimize.fsolve` through the local
    :func:`fsolve` wrapper. If the solver converges and the returned value
    satisfies ``condition``, the solution is returned together with ``True``.
    Otherwise, ``numpy.nan`` and ``False`` are returned.

    The keyword argument ``original_kwargs`` may be used to override the
    arguments passed to ``condition``.

    :param fn: Implicit function to solve.
    :param condition: Validation function applied to the computed solution.
    :param args: Positional arguments passed to ``fn``.
    :param kwargs: Optional keyword arguments. Supported key is
                   ``original_kwargs`` for condition validation arguments.
    :returns: Tuple of ``(solution, use)``.
    :raises Exception: Re-raises any exception from the underlying solver.
    """
    solution: Float = np.nan
    use = False
    scipy_solver_init_value = np.array([1e-4])

    try:
        solution_arr, _, ier, msg = fsolve(
            fn,
            scipy_solver_init_value,
            full_output=True,
            args=args,
            xtol=1e-10,
        )

        if ier == 1 and not up.isnan(solution_arr[0]):
            solution = solution_arr[0]
            use = 0 < solution < MAX_EXPECTED_SOLUTION_BOUNDARY
        elif not settings.SUPPRESS_WARNINGS:
            logger.warning("solution in implicit solver was not found, cause: %s", msg)
    except Exception as exc:
        logger.debug(
            "attempt to solve function %s finished w/ exception: %s",
            fn.__name__,
            str(exc),
        )
        raise

    args_to_use = kwargs.get("original_kwargs", args)
    return (solution, use) if condition(solution, *args_to_use) else (np.nan, False)
