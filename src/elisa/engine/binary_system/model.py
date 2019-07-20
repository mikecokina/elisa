import numpy as np


def static_potential_primary_fn(radius, *args):
    """
    Pontetial function which defines surface of primary component of binary system in spherical coordinates.
    It is exposed for multiprocessing to avoid pickleing of classes, loggers, etc.

    :param radius: float
    :param args: Tuple; (mass_ratio, surface_potential, b, c, d, e)

    (b, c, d, e) are precalculated values to decrease runtime::

        (b, c, d, e) such that: Psi1 = 1/r + a/sqrt(b+r^2+c*r) - d*r + e*x^2

    :return: float
    """
    mass_ratio, surface_potential, b, c, d, e = args
    radius2 = np.power(radius, 2)
    a = 1 / radius + mass_ratio / np.sqrt(b + radius2 - c * radius) - d * radius + e * radius2
    return a - surface_potential


def static_potential_secondary_fn(radius, *args):
    """
    Pontetial function which defines surface of primary component of binary system in spherical coordinates.
    It is exposed for multiprocessing to avoid pickleing of classes, loggers, etc.

    :param radius: float
    :param args: Tuple; (mass_ratio, surface_potential, b, c, d, e, f)

    (b, c, d, e, f) are precalculated values to decrease runtime::

        (b, c, d, e, f) such that: Psi2 = q/r + 1/sqrt(b+r^2-Cr) - d*r + e*x^2 + f

    :return: float
    """
    mass_ratio, surface_potential, b, c, d, e, f = args
    radius2 = np.power(radius, 2)
    a = mass_ratio / radius + 1. / np.sqrt(b + radius2 - c * radius) - d * radius + e * radius2 + f
    return a - surface_potential


def static_potential_primary_cylindrical_fn(radius, *args):
    """
    Pontetial function which defines surface of primary component of binary system in cylindrical coordinates.
    Usefull for W UMa systems.

    :param radius: float
    :param args: Tuple; (mass_ratio, surface_potential, a, b, c, d, e, f)


    (a, b, c, d, e, f) are precalculated values to decrease runtime::

        (a, b, c, d, e, f) such that: Psi1 = 1/sqrt(a+r^2) + q/sqrt(b + r^2) - c + d*(e+f*r^2)

    :return: float
    """
    mass_ratio, surface_potential, a, b, c, d, e, f = args
    radius2 = np.power(radius, 2)
    return 1 / np.sqrt(a + radius2) + mass_ratio / np.sqrt(b + radius2) - c + d * (e + f * radius2) - surface_potential


def static_potential_secondary_cylindrical_fn(radius, *args):
    """
    Pontetial function which defines surface of primary component of binary system in cylindrical coordinates.
    Usefull for W UMa systems.

    :param radius: float
    :param args: Tuple; (mass_ratio, surface_potential, a, b, c, d, e, f)


    (a, b, c, d, e, f) are precalculated values to decrease runtime::

        (a, b, c, d, e, f) such that: Psi1 = 1/sqrt(a+r^2) + q/sqrt(b + r^2) - c + d*(e+f*r^2)

    :return: float
    """
    mass_ratio, surface_potential, a, b, c, d, e, f = args
    radius2 = np.power(radius, 2)
    return mass_ratio / np.sqrt(a + radius2) + 1. / np.sqrt(b + radius2) + c * (d + e * radius2) + f - surface_potential

