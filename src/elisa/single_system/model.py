import numpy as np
from .. import const


def surface_potential_from_polar_log_g(polar_log_g, mass):
    """
    calculation of surface potential based from polar gravity acceleration

    :param polar_log_g: float; polar gravity acceleration
    :param mass: float; stellar mass
    :return:
    """
    polar_gravity_acceleration = np.power(10, polar_log_g)
    return - np.power(const.G * mass * polar_gravity_acceleration, 0.5)


def potential_fn(radius, *args):
    """
    implicit potential function

    :param radius: (np.)float; spherical variable
    :param args: (array), float; (polar angles) and desired value of potential
    :return:
    """
    return potential(radius, *args[0]) - args[1]


def potential(radius, *args):
    """
    function calculates potential on the given point of the star

    :param radius: (np.)float; spherical variable
    :param args: Tuple; (A, B) - such that: Psi = -a/r - b*r**2
    :return: (np.)float
    """
    a, b = args

    return - a / radius - b * np.power(radius, 2.0)


def pre_calculate_for_potential_value(*args, return_as_tuple=False):
    """
    Function calculates auxiliary values for calculation of primary component potential,
    and therefore they don't need to be wastefully recalculated every iteration in solver.

    :param return_as_tuple: return coefficients as a tuple of numpy vectors instead of numpy matrix
    :type return_as_tuple: bool
    :param args: Tuple; (mass, angular velocity of rotation, latitude angle (0, pi))
    :return: Tuple; (a, b) such that: Psi = -a/r - b*r**2
    """
    mass, angular_velocity, theta, = args

    a = const.G * mass
    b = 0.5 * np.power(angular_velocity * np.sin(theta), 2)

    if np.isscalar(theta):
        return a, b
    else:
        aa = a * np.ones(np.shape(theta))
        return (aa, b) if return_as_tuple else np.column_stack((aa, b))


def radial_potential_derivative(radius, *args):
    """
    function calculate radial derivative of potential function in spherical coordinates

    :param radius: float; radius of given point(s) in spherical coordinates
    :param args: Tuple; (A, B) - such that: Psi = -a/r**2 - 2*b*r
    :type args: Tuple;
    :return:
    """
    a, b = args

    return a / np.power(radius, 2) - 2 * b * radius

