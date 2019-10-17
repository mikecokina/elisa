import numpy as np
from elisa import const as c, units as U, opt, utils
from astropy import units as u
from elisa.conf import config


def angular_velocity(rotation_period):
    """
    rotational angular velocity of the star
    :return:
    """
    return c.FULL_ARC / (rotation_period * U.PERIOD_UNIT).to(u.s).value


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
    a, b = args[0]

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
    Function solves radius for given latitudes that are passed in *args.
    Function to solve is specified as last parameter in *args Tuple.

    :param args: Tuple;

    ::

        Tuple[
                theta: numpy.array,
                x0: float,
                precalc: method,
                fn: method,
                derivative_fn: method]

    :return: numpy.array
    """
    theta, x0, precalc_fn, potential_fn, potential_derivative_fn, surface_potential = args
    precalc_vals = precalc_fn(*(theta,), return_as_tuple=True)
    x0 = x0 * np.ones(theta.shape)
    radius = opt.newton.newton(potential_fn, x0, fprime=potential_derivative_fn,
                               maxiter=config.MAX_SOLVER_ITERS, args=(precalc_vals, surface_potential), rtol=1e-10)
    return radius


def get_surface_points(*args):
    """
    Function solves radius for given azimuths that are passed in *args.
    Function to solve is specified as last parameter in *args Tuple.

    :param args: Tuple;

    ::

        Tuple[
                phi: numpy.array,
                theta: numpy.array,
                x0: float,
                precalc: method,
                fn: method,
                derivative_fn: method]

    :return: numpy.array
    """
    phi, theta, x0, precalc_fn, potential_fn, potential_derivative_fn, surface_potential = args
    precalc_vals = precalc_fn(*(theta,), return_as_tuple=True)
    x0 = x0 * np.ones(phi.shape)
    radius = opt.newton.newton(potential_fn, x0, fprime=potential_derivative_fn,
                               maxiter=config.MAX_SOLVER_ITERS, args=(precalc_vals, surface_potential), rtol=1e-10)
    return utils.spherical_to_cartesian(np.column_stack((radius, phi, theta)))


def calculate_points_on_quarter_surface(radius, thetas, characteristic_distance):
    """
    generates points on the quarter of the stellar surface in cartesian coordinates

    :param radius: array; radius(theta)
    :param thetas: array; set of latitudes at which to calculate surface points
    :param characteristic_distance: desired angular distance between points
    :return: tuple; x, y, z corrdinates of the surface points
    """
    r_q, phi_q, theta_q = np.array([]), np.array([]), np.array([])
    for ii, theta in enumerate(thetas):
        num = int(c.HALF_PI * radius[ii] * np.sin(theta) // characteristic_distance)
        if num <= 1:
            continue

        r_aux = radius[ii] * np.ones(num - 1)
        r_q = np.concatenate((r_q, r_aux))

        M = c.HALF_PI / num
        phi_aux = np.linspace(M, c.HALF_PI - M, num=num - 1, endpoint=True)
        phi_q = np.concatenate((phi_q, phi_aux))

        theta_aux = theta * np.ones(num - 1)
        theta_q = np.concatenate((theta_q, theta_aux))

    quarter = utils.spherical_to_cartesian(np.column_stack((r_q, phi_q, theta_q)))
    return quarter[:, 0], quarter[:, 1], quarter[:, 2]


def calculate_points_on_meridian(radius, theta):
    phi_mer = np.zeros(radius.shape)
    meridian = utils.spherical_to_cartesian(np.column_stack((radius, phi_mer, theta)))
    return meridian[:, 0], meridian[:, 1], meridian[:, 2]
