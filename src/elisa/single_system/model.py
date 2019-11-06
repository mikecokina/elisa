import numpy as np
from elisa import const


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
    :param args: tuple; (A, B) - such that: Psi = -a/r - b*r**2
    :return: (np.)float
    """
    a, b = args

    return - a / radius - b * np.power(radius, 2.0)


def pre_calculate_for_potential_value(self, *args, return_as_tuple=False):
    """
    Function calculates auxiliary values for calculation of primary component potential,
    and therefore they don't need to be wastefully recalculated every iteration in solver.

    :param return_as_tuple: return coefficients as a tuple of numpy vectors instead of numpy matrix
    :type return_as_tuple: bool
    :param args: tuple; (mass, angular velocity of rotation, latitude angle (0, pi))
    :return: tuple: (a, b) such that: Psi = -a/r - b*r**2
    """
    mass, angular_velocity, theta, = args

    a = const.G * mass
    b = 0.5 * np.power(angular_velocity * np.sin(theta), 2)

    if np.isscalar(theta):
        return a, b
    else:
        aa = a * np.ones(np.shape(theta))
        return (aa, b) if return_as_tuple else np.column_stack((aa, b))


