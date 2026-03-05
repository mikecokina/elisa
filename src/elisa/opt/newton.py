from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import umpy as up
from elisa.base import error

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

    from elisa.types import Float


def newton(
        func: Callable,
        x0: ArrayLike | Float,
        fprime: Callable,
        args: tuple = (),
        maxiter: int = 50,
        rtol: Float = 0.0,
) -> NDArray[Float] | Float:
    # noinspection GrazieInspection
    r"""Solve for a root using the Newton method.

    The iteration follows:

    math:

        x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}

    :param func: callable
        Function whose root is sought.
        Must accept the current iterate as the first argument,
        followed by ``args``.
    :param x0: ArrayLike | Float
        Initial estimate of the solution.
    :param fprime: callable
        Derivative of ``func``.
    :param args: tuple
        Extra positional arguments passed to ``func`` and ``fprime``.
    :param maxiter: int
        Maximum number of iterations.
    :param rtol: Float
        Relative tolerance for convergence.
    :returns: NDArray[Float] | Float
        Computed root. The return type follows the shape of ``x0``.
    :raises error.MaxIterationError:
        If the solver does not converge within ``maxiter`` iterations.
    """
    x_n = np.copy(x0) if isinstance(x0, np.ndarray) else x0

    for _ in range(maxiter):
        difference = func(x_n, *args) / fprime(x_n, *args[0])
        x_m = x_n - difference

        if np.max(up.abs(difference / x_n)) <= rtol:
            return x_m

        x_n = x_m

    msg = f"Max iteration limit - {maxiter} - exceeded"
    raise error.MaxIterationError(msg)
