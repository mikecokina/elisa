import numpy as np
from elisa import const as c, opt
from elisa.conf import config


def potential(radius, *args):
    """
    function calculates potential on the given point of the star

    :param radius: (np.)float; spherical variable
    :param args: tuple; (A, B) - such that: Psi = -a/r - b*r**2
    :return: (np.)float
    """
    a, b = args

    return - a / radius - b * np.power(radius, 2.0)


def radial_potential_derivative(radius, *args):
    """
    function calculate radial derivative of potential function in spherical coordinates

    :param radius: float; radius of given point(s) in spherical coordinates
    :param args: tuple; (A, B) - such that: Psi = -a/r**2 - 2*b*r
    :type args: tuple
    :return:
    """
    a, b = args

    return a / np.power(radius, 2) - 2 * b * radius


def pre_calc_latitudes(alpha):
    """
    function pre-calculates latitudes of stellar surface with exception of pole and equator
    :param alpha: angular distance of points
    :return:
    """
    num = int((c.HALF_PI - 2 * alpha) // alpha)
    return np.linspace(alpha, c.HALF_PI - alpha, num=num, endpoint=True)


def get_surface_points_radii(*args):
    """
    Function solves radius for given latitudes that are passed in `args`.
    Function to solve is specified as last parameter in `args` Tuple.

    :param args: Tuple;

    ::

        Tuple[
                theta: numpy.array,
                x0: float,
                precalc: callable,
                fn: callable,
                derivative_fn: callable]

    :return: numpy.array
    """
    theta, x0, precalc_fn, potential_fn, potential_derivative_fn, surface_potential = args
    precalc_vals = precalc_fn(*(theta,), return_as_tuple=True)
    x0 = x0 * np.ones(theta)
    radius = opt.newton.newton(potential_fn, x0, fprime=potential_derivative_fn,
                               maxiter=config.MAX_SOLVER_ITERS, args=(precalc_vals, surface_potential), rtol=1e-10)
    return radius
