import numpy as np

from .. import umpy as up
from .. base import error


def newton(func, x0, fprime, args=(), maxiter=50, rtol=0.0):
    """
    Newton root solver.
    Solved by approach::

        x_n+1 = x_n - (func(x_n) / fprime(x_n))

    Starts from x0.

    :param func: callable; the function whose zero is wanted. It must be a function of a single variable
                           of the form f(x,a,b,c...), where a, b, c... are extra arguments that can be passed
                           in the args parameter.
    :param x0: Union[numpy.array, float]; an initial estimate of the solution
    :param fprime: callable; the derivative of the function
    :param args: Tuple; extra arguments to be used in the function call
    :param maxiter: int; maximum number of iterations
    :param rtol: tolerance (relative) for termination
    :return: Union[numpy.array, float]; depends on the `x0` shape
    """
    x_n = np.copy(x0) if isinstance(x0, np.ndarray) else x0
    for _ in range(maxiter):
        difference = func(x_n, *args) / fprime(x_n, *args[0])
        x_m = x_n - difference
        if np.max(up.abs(difference / x_n)) <= rtol:
            return x_m
        x_n = x_m
    raise error.MaxIterationError(f"Max iteration limit - {maxiter} - exceeded")
