import numpy as np

from elisa import umpy as up


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
    radius2 = up.power(radius, 2)
    a = 1 / radius + mass_ratio / up.sqrt(b + radius2 - c * radius) - d * radius + e * radius2
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
    radius2 = up.power(radius, 2)
    a = mass_ratio / radius + 1. / up.sqrt(b + radius2 - c * radius) - d * radius + e * radius2 + f
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
    radius2 = up.power(radius, 2)
    return 1 / up.sqrt(a + radius2) + mass_ratio / up.sqrt(b + radius2) - c + d * (e + f * radius2) - surface_potential


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
    radius2 = up.power(radius, 2)
    return mass_ratio / up.sqrt(a + radius2) + 1. / up.sqrt(b + radius2) + c * (d + e * radius2) + f - surface_potential


def potential_value_primary(radius, *args):
    """
    Calculates modified Kopal's potential from point of view of primary component.
    :param radius: (numpy.)float; spherical variable
    :param args: tuple: (mass_ratio, B, C, D, E) such that: Psi1 = 1/r + q/sqrt(B+r^2+Cr) - D*r + E*x^2
    :return: (numpy.)float
    """
    mass_ratio, b, c, d, e = args
    radius2 = up.power(radius, 2)
    return 1 / radius + mass_ratio / up.sqrt(b + radius2 - c * radius) - d * radius + e * radius2


def potential_value_secondary(radius, *args):
    """
    Calculates modified Kopal's potential from point of view of secondary component.
    :param radius: up.float; spherical variable
    :param args: tuple: (mass_ratio, b, c, d, e, f) such that: Psi2 = q/r + 1/sqrt(b+r^2-Cr) - d*r + e*r^2 + f
    :return:
    """
    mass_ratio, b, c, d, e, f = args
    radius2 = up.power(radius, 2)
    return mass_ratio / radius + 1. / up.sqrt(b + radius2 - c * radius) - d * radius + e * radius2 + f


def potential_primary_fn(radius, *args):
    """
    Implicit potential function from perspective of primary component.
    :param radius: numpy.float; spherical variable
    :param args: tuple; pre calculated values for potential function and desired value of potential
    :return:
    """
    return potential_value_primary(radius, *args[0]) - args[1]


def potential_secondary_fn(radius, *args):
    """
    Implicit potential function from perspective of secondary component.
    :param radius: numpy.float; spherical variable
    :param args: pre calculated values for potential function and desired value of potential
    :return: numpy.float
    """
    # return self.potential_value_secondary(radius, *args) - self.secondary.surface_potential
    return potential_value_secondary(radius, *args[0]) - args[1]


def potential_value_primary_cylindrical(radius, *args):
    """
    Calculates modified Kopal's potential from point of view of primary component
    in cylindrical coordinates r_n, phi_n, z_n, where z_n = x and heads along z axis.

    This function is intended for generation of ``necks``
    of W UMa systems, therefore components distance = 1 an synchronicity = 1 is assumed.

    :param radius: up.float
    :param args: tuple: (a, b, c, d, e) such that: Psi1 = 1/sqrt(a+r^2) + q/sqrt(b + r^2) - c + d*(a+e*r^2)
    :return:
    """
    mass_ratio, a, b, c, d, e = args

    radius2 = up.power(radius, 2)
    return 1 / up.sqrt(a + radius2) + mass_ratio / up.sqrt(b + radius2) - c + d * (a + e * radius2)


def potential_value_secondary_cylindrical(radius, *args):
    """
    Calculates modified kopal potential from point of view of secondary
    component in cylindrical coordinates r_n, phi_n, z_n, where z_n = x and heads along z axis.
    :param radius: up.float
    :param args: tuple: (a, b, c, d, e, f) such that: Psi2 = q/sqrt(a+r^2) + 1/sqrt(b + r^2) - c + d*(a+e*r^2)
    :return:
    """
    mass_ratio, a, b, c, d, e, f = args

    radius2 = up.power(radius, 2)
    return mass_ratio / up.sqrt(a + radius2) + 1. / up.sqrt(b + radius2) - c + d * (e * radius2 + a) + f


def potential_primary_cylindrical_fn(radius, *args):
    """
    Implicit potential function from perspective of primary component given in cylindrical coordinates.
    :param radius: numpy.float
    :param args: tuple; pre calculated values for potential function and desired value of potential
    :return:
    """
    return potential_value_primary_cylindrical(radius, *args[0]) - args[1]


def potential_secondary_cylindrical_fn(radius, *args):
    """
    Implicit potential function from perspective of secondary component given in cylindrical coordinates.
    :param radius: numpy.float
    :param args: tuple: pre calculated values for potential function and desired value of potential
    :return: numpy.float
    """
    # return self.potential_value_secondary_cylindrical(radius, *args) - self.secondary.surface_potential
    return potential_value_secondary_cylindrical(radius, *args[0]) - args[1]


def radial_primary_potential_derivative(radius, *args):
    """
    Function calculate radial derivative of primary potential function in spherical coordinates
    :param radius: radius of given point(s) in spherical coordinates
    :type radius: float or numpy array
    :param args: b, c, d, e - such that: dPsi1/dr = -1/r^2 + 0.5*q*(c-2r)/(b-cr+r^2)^(3/2) - d +2er
    :type args: tuple
    :return:
    """
    # auxiliary values pre-calculated in pre_calculate_for_potential_value_primary()
    mass_ratio, b, c, d, e = args[0]
    radius2 = up.power(radius, 2)

    return - 1 / radius2 + 0.5 * mass_ratio * (c - 2 * radius) / up.power(b - c * radius + radius2, 1.5) \
           - d + 2 * e * radius


def radial_secondary_potential_derivative(radius, *args):
    """
    Function calculate radial derivative of secondary potential function in spherical coordinates
    :param radius: radius of given point(s) in spherical coordinates
    :type radius: float or numpy array
    :param args: b, c, d, e, f - such that: dPsi2/dr = -q/r^2 + (0.5c - x)/(b - cx + r^2)^(3/2) - d + 2er
    :type args: tuple
    :return:
    """
    # auxiliary values pre-calculated in pre_calculate_for_potential_value_primary()
    mass_ratio, b, c, d, e, f = args[0]
    radius2 = up.power(radius, 2)

    return - mass_ratio / radius2 + (0.5 * c - radius) / up.power(b - c * radius + radius2, 1.5) \
           - d + 2 * e * radius


def pre_calculate_for_potential_value_primary(*args, return_as_tuple=False):
    """
    Function calculates auxiliary values for calculation of primary component potential,
    and therefore they don't need to be wastefully recalculated every iteration in solver.
    :param return_as_tuple: return coefficients as a tuple of numpy vectors instead of numpy matrix
    :type return_as_tuple: bool
    :param args: (component distance, azimut angle (0, 2pi), latitude angle (0, pi)
    :return: tuple: (b, c, d, e) such that: Psi1 = 1/r + q/sqrt(b+r^2-c*r) - d*r + e*r^2
    """
    # synchronicity, mass_ratio, distance between components, azimuth angle, latitude angle (0,180)
    synchronicity, mass_ratio, distance, phi, theta = args

    cs = up.cos(phi) * up.sin(theta)

    b = up.power(distance, 2)
    c = 2 * distance * cs
    d = (mass_ratio * cs) / b
    e = 0.5 * up.power(synchronicity, 2) * (1 + mass_ratio) * up.power(up.sin(theta), 2)

    if np.isscalar(phi):
        return b, c, d, e
    else:
        bb = b * np.ones(np.shape(phi))
        return (bb, c, d, e) if return_as_tuple else np.column_stack((bb, c, d, e))


def pre_calculate_for_potential_value_secondary(*args, return_as_tuple=False):
    """
    Function calculates auxiliary values for calculation of secondary component potential,
    and therefore they don't need to be wastefully recalculated every iteration in solver.
    :param return_as_tuple: return coefficients as a tuple of numpy vectors instead of numpy matrix
    :type return_as_tuple: bool
    :param args: (component distance, azimut angle (0, 2pi), latitude angle (0, pi)
    :return: tuple: (b, c, d, e, f) such that: Psi2 = q/r + 1/sqrt(b+r^2+Cr) - d*r + e*r^2 + f
    """
    # synchronicity, mass_ratio, distance between components, azimuth angle, latitude angle (0,180)
    synchronicity, mass_ratio, distance, phi, theta = args

    cs = up.cos(phi) * up.sin(theta)

    b = up.power(distance, 2)
    c = 2 * distance * cs
    d = cs / b
    e = 0.5 * up.power(synchronicity, 2) * (1 + mass_ratio) * up.power(up.sin(theta), 2)
    f = 0.5 - 0.5 * mass_ratio

    if np.isscalar(phi):
        return b, c, d, e, f
    else:
        bb = b * np.ones(np.shape(phi))
        ff = f * np.ones(np.shape(phi))
        return (bb, c, d, e, ff) if return_as_tuple else np.column_stack((bb, c, d, e, ff))


def radial_primary_potential_derivative_cylindrical(radius, *args):
    """
    Function calculate radial derivative of primary potential function in cylindrical coordinates
    :param radius: radius of given point(s) in cylindrical coordinates
    :type radius: float or numpy array
    :param args: a, b, c, d, e such that: dPsi1/dr = - r/(a+r^2)^(3/2) - rq/(b+r^2)^(3/2) + 2der
    :type args: tuple
    :return:
    """
    mass_ratio, a, b, c, d, e = args[0]

    radius2 = up.power(radius, 2)
    return + 0 - radius / up.power(a + radius2, 1.5) - radius * mass_ratio / up.power(b + radius2, 1.5) \
           + 2 * d * e * radius


def radial_secondary_potential_derivative_cylindrical(radius, *args):
    """
    Function calculate radial derivative of secondary potential function in cylindrical coordinates
    :param radius: radius of given point(s) in cylindrical coordinates
    :type radius: float or numpy array
    :param args: a, b, c, d, e, f such that: dPsi2/dr = - qr/(a+r^2)^(3/2) - r/(b+r^2)^(3/2) + 2cer
    :type args: tuple
    :return:
    """
    mass_ratio, a, b, c, d, e, f = args[0]

    radius2 = up.power(radius, 2)
    return + 0 - radius * mass_ratio / up.power(a + radius2, 1.5) - radius / up.power(b + radius2, 1.5) \
           + 2 * d * e * radius
