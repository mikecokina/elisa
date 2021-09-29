import numpy as np
from .. import umpy as up


def static_potential_primary_fn(radius, *args):
    """
    Potential function which defines surface of primary component of binary system in spherical coordinates.
    It is exposed for multiprocessing to avoid pickling of classes, loggers, etc.

    :param radius: float;
    :param args: Tuple; (mass_ratio, surface_potential, b, c, d, e)

    (b, c, d, e) are precalculated values to decrease runtime::

        (b, c, d, e) such that: Psi1 = 1/r + a/sqrt(b+r^2+c*r) - d*r + e*x^2

    :return: float;
    """
    mass_ratio, surface_potential, b, c, d, e = args
    radius2 = up.power(radius, 2)
    a = 1 / radius + mass_ratio / up.sqrt(b + radius2 - c * radius) - d * radius + e * radius2
    return a - surface_potential


def static_potential_secondary_fn(radius, *args):
    """
    Potential function which defines surface of primary component of binary system in spherical coordinates.
    It is exposed for multiprocessing to avoid pickling of classes, loggers, etc.

    :param radius: float;
    :param args: Tuple; (mass_ratio, surface_potential, b, c, d, e, f)

    (b, c, d, e, f) are precalculated values to decrease runtime::

        (b, c, d, e, f) such that: Psi2 = q/r + 1/sqrt(b+r^2-Cr) - d*r + e*x^2 + f

    :return: float;
    """
    mass_ratio, surface_potential, b, c, d, e, f = args
    radius2 = up.power(radius, 2)
    a = mass_ratio / radius + 1. / up.sqrt(b + radius2 - c * radius) - d * radius + e * radius2 + f
    return a - surface_potential


def static_potential_primary_cylindrical_fn(radius, *args):
    """
    Potential function which defines surface of primary component of binary system in cylindrical coordinates.
    Useful for W UMa systems.

    :param radius: float;
    :param args: Tuple; (mass_ratio, surface_potential, a, b, c, d, e, f)

    (a, b, c, d, e, f) are precalculated values to decrease runtime::

        (a, b, c, d, e, f) such that: Psi1 = 1/sqrt(a+r^2) + q/sqrt(b + r^2) - c + d*(e+f*r^2)

    :return: float;
    """
    mass_ratio, surface_potential, a, b, c, d, e, f = args
    radius2 = up.power(radius, 2)
    return 1 / up.sqrt(a + radius2) + mass_ratio / up.sqrt(b + radius2) - c + d * (e + f * radius2) - surface_potential


def static_potential_secondary_cylindrical_fn(radius, *args):
    """
    Potential function which defines surface of primary component of binary system in cylindrical coordinates.
    Useful for W UMa systems.

    :param radius: float;
    :param args: Tuple; (mass_ratio, surface_potential, a, b, c, d, e, f)

    (a, b, c, d, e, f) are precalculated values to decrease runtime::

        (a, b, c, d, e, f) such that: Psi1 = 1/sqrt(a+r^2) + q/sqrt(b + r^2) - c + d*(e+f*r^2)

    :return: float;
    """
    mass_ratio, surface_potential, a, b, c, d, e, f = args
    radius2 = up.power(radius, 2)
    return mass_ratio / up.sqrt(a + radius2) + 1. / up.sqrt(b + radius2) + c * (d + e * radius2) + f - surface_potential


def potential_value_primary(radius, *args):
    """
    Calculates modified Kopal's potential from point of view of primary component.

    :param radius: (numpy.)float; spherical variable
    :param args: Tuple; (mass_ratio, B, C, D, E) such that: Psi1 = 1/r + q/sqrt(B+r^2+Cr) - D*r + E*r^2
    :return: (numpy.)float;
    """
    mass_ratio, b, c, d, e = args
    radius2 = up.power(radius, 2)
    return 1 / radius + mass_ratio / up.sqrt(b + radius2 - c * radius) - d * radius + e * radius2


def potential_value_secondary(radius, *args):
    """
    Calculates modified Kopal's potential from point of view of secondary component.
    :param radius: up.float; spherical variable
    :param args: Tuple; (mass_ratio, b, c, d, e, f) such that: Psi2 = q/r + 1/sqrt(b+r^2-Cr) - d*r + e*r^2 + f
    :return: float;
    """
    mass_ratio, b, c, d, e, f = args
    radius2 = up.power(radius, 2)
    return mass_ratio / radius + 1. / up.sqrt(b + radius2 - c * radius) - d * radius + e * radius2 + f


def potential_primary_fn(radius, *args):
    """
    Implicit potential function from perspective of primary component.

    :param radius: float; spherical variable
    :param args: Tuple; pre calculated values for potential function and desired value of potential
    :return: float;
    """
    return potential_value_primary(radius, *args[0]) - args[1]


def potential_secondary_fn(radius, *args):
    """
    Implicit potential function from perspective of secondary component.

    :param radius: float; spherical variable
    :param args: pre calculated values for potential function and desired value of potential
    :return: float;
    """
    # return self.potential_value_secondary(radius, *args) - self.secondary.surface_potential
    return potential_value_secondary(radius, *args[0]) - args[1]


def potential_value_primary_cylindrical(radius, *args):
    """
    Calculates modified Kopal's potential from point of view of primary component
    in cylindrical coordinates r_n, phi_n, z_n, where z_n = x and heads along z axis.

    This function is intended for generation of ``necks``
    of W UMa systems, therefore components distance = 1 an synchronicity = 1 is assumed.

    :param radius: float;
    :param args: Tuple; (a, b, c, d, e) such that: Psi1 = 1/sqrt(a+r^2) + q/sqrt(b + r^2) - c + d*(a+e*r^2)
    :return: float;
    """
    mass_ratio, a, b, c, d, e = args

    radius2 = up.power(radius, 2)
    return 1 / up.sqrt(a + radius2) + mass_ratio / up.sqrt(b + radius2) - c + d * (a + e * radius2)


def potential_value_secondary_cylindrical(radius, *args):
    """
    Calculates modified kopal potential from point of view of secondary
    component in cylindrical coordinates r_n, phi_n, z_n, where z_n = x and heads along z axis.

    :param radius: float;
    :param args: Tuple; (a, b, c, d, e, f) such that: Psi2 = q/sqrt(a+r^2) + 1/sqrt(b + r^2) - c + d*(a+e*r^2)
    :return: float;
    """
    mass_ratio, a, b, c, d, e, f = args

    radius2 = up.power(radius, 2)
    return mass_ratio / up.sqrt(a + radius2) + 1. / up.sqrt(b + radius2) - c + d * (e * radius2 + a) + f


def potential_primary_cylindrical_fn(radius, *args):
    """
    Implicit potential function from perspective of primary component given in cylindrical coordinates.

    :param radius: float;
    :param args: Tuple; pre calculated values for potential function and desired value of potential
    :return: float;
    """
    return potential_value_primary_cylindrical(radius, *args[0]) - args[1]


def potential_secondary_cylindrical_fn(radius, *args):
    """
    Implicit potential function from perspective of secondary component given in cylindrical coordinates.

    :param radius: float;
    :param args: Tuple: pre calculated values for potential function and desired value of potential
    :return: float;
    """
    return potential_value_secondary_cylindrical(radius, *args[0]) - args[1]


def radial_primary_potential_derivative(radius, *args):
    """
    Function calculate radial derivative of primary potential function in spherical coordinates.

    :param radius: radius of given point(s) in spherical coordinates
    :type radius: Union[float, numpy.array];
    :param args: b, c, d, e - such that: dPsi1/dr = -1/r^2 + 0.5*q*(c-2r)/(b-cr+r^2)^(3/2) - d +2er
    :type args: Tuple;
    :return: Union[float, numpy.array];
    """
    # auxiliary values pre-calculated in pre_calculate_for_potential_value_primary()
    mass_ratio, b, c, d, e = args
    radius2 = up.power(radius, 2)

    return - 1 / radius2 + 0.5 * mass_ratio * (c - 2 * radius) / up.power(b - c * radius + radius2, 1.5) \
           - d + 2 * e * radius


def radial_secondary_potential_derivative(radius, *args):
    """
    Function calculate radial derivative of secondary potential function in spherical coordinates

    :param radius: radius of given point(s) in spherical coordinates
    :type radius: Union[float, numpy.array];
    :param args: b, c, d, e, f - such that: dPsi2/dr = -q/r^2 + (0.5c - x)/(b - cx + r^2)^(3/2) - d + 2er
    :type args: Tuple;
    :return: float;
    """
    # auxiliary values pre-calculated in pre_calculate_for_potential_value_primary()
    mass_ratio, b, c, d, e, f = args
    radius2 = up.power(radius, 2)

    return - mass_ratio / radius2 + (0.5 * c - radius) / up.power(b - c * radius + radius2, 1.5) \
           - d + 2 * e * radius


def pre_calculate_for_potential_value_primary(*args, return_as_tuple=False):
    """
    Function calculates auxiliary values for calculation of primary component potential,
    and therefore they don't need to be wastefully recalculated every iteration in solver.

    :param return_as_tuple: return coefficients as a tuple of numpy vectors instead of numpy matrix
    :type return_as_tuple: bool;
    :param args: (component distance, azimut angle (0, 2pi), latitude angle (0, pi)
    :return: Tuple; (b, c, d, e) such that: Psi1 = 1/r + q/sqrt(b+r^2-c*r) - d*r + e*r^2
    """
    # synchronicity, mass_ratio, distance between components, azimuth angle, latitude angle (0,180)
    synchronicity, mass_ratio, distance, phi, theta = args

    sin_theta = up.sin(theta)
    cs = up.cos(phi) * sin_theta

    b = up.power(distance, 2)
    c = 2 * distance * cs
    d = (mass_ratio * cs) / b
    e = 0.5 * up.power(synchronicity, 2) * (1 + mass_ratio) * up.power(sin_theta, 2)

    if np.isscalar(phi):
        return b, c, d, e
    else:
        bb = b * np.ones(np.shape(phi))
        return (bb, c, d, e) if return_as_tuple else np.column_stack((bb, c, d, e))


def pre_calculate_for_potential_value_secondary(*args, return_as_tuple=False):
    """
    Function calculates auxiliary values for calculation of secondary component potential,
    and therefore they don't need to be wastefully recalculated every iteration in solver.

    :param return_as_tuple: return coefficients as a Tuple of numpy vectors instead of numpy matrix
    :type return_as_tuple: bool
    :param args: (component distance, azimut angle (0, 2pi), latitude angle (0, pi)
    :return: Tuple: (b, c, d, e, f) such that: Psi2 = q/r + 1/sqrt(b+r^2+Cr) - d*r + e*r^2 + f
    """
    # synchronicity, mass_ratio, distance between components, azimuth angle, latitude angle (0,180)
    synchronicity, mass_ratio, distance, phi, theta = args

    sin_theta = up.sin(theta)
    cs = up.cos(phi) * sin_theta

    b = up.power(distance, 2)
    c = 2 * distance * cs
    d = cs / b
    e = 0.5 * up.power(synchronicity, 2) * (1 + mass_ratio) * up.power(sin_theta, 2)
    f = 0.5 - 0.5 * mass_ratio

    if np.isscalar(phi):
        return b, c, d, e, f
    else:
        bb = b * np.ones(np.shape(phi))
        ff = f * np.ones(np.shape(phi))
        return (bb, c, d, e, ff) if return_as_tuple else np.column_stack((bb, c, d, e, ff))


def radial_primary_potential_derivative_cylindrical(radius, *args):
    """
    Function calculate radial derivative of primary potential function in cylindrical coordinates.

    :param radius: radius of given point(s) in cylindrical coordinates
    :type radius: Union[float, numpy.array]
    :param args: a, b, c, d, e such that: dPsi1/dr = - r/(a+r^2)^(3/2) - rq/(b+r^2)^(3/2) + 2der
    :type args: Tuple;
    :return: float;
    """
    mass_ratio, a, b, c, d, e = args

    radius2 = up.power(radius, 2)
    return + 0 - radius / up.power(a + radius2, 1.5) - radius * mass_ratio / up.power(b + radius2, 1.5) \
           + 2 * d * e * radius


def radial_secondary_potential_derivative_cylindrical(radius, *args):
    """
    Function calculate radial derivative of secondary potential function in cylindrical coordinates.

    :param radius: radius of given point(s) in cylindrical coordinates
    :type radius: Union[float, numpy.array]
    :param args: a, b, c, d, e, f such that: dPsi2/dr = - qr/(a+r^2)^(3/2) - r/(b+r^2)^(3/2) + 2cer
    :type args: Tuple;
    :return: float;
    """
    mass_ratio, a, b, c, d, e, f = args

    radius2 = up.power(radius, 2)
    return + 0 - radius * mass_ratio / up.power(a + radius2, 1.5) - radius / up.power(b + radius2, 1.5) \
           + 2 * d * e * radius


def pre_calculate_for_potential_value_primary_cylindrical(*args, return_as_tuple=False):
    """
    Function calculates auxiliary values for calculation of primary component potential
    in cylindrical symmetry. Therefore they don't need to be wastefully recalculated every iteration in solver.

    :param return_as_tuple: return coefficients as a tuple of numpy vectors instead of numpy matrix
    :type return_as_tuple: bool
    :param args: (azimut angle (0, 2pi), z_n (cylindrical, identical with cartesian x)), components distance
    :return: Tuple; (a, b, c, d, e) such that: Psi1 = 1/sqrt(a+r^2) + q/sqrt(b + r^2) - c + d*(a+e*r^2)
    """
    synchronicity, mass_ratio, phi, z, distance = args

    a = up.power(z, 2)
    b = up.power(distance - z, 2)
    c = mass_ratio * z / up.power(distance, 2)
    d = 0.5 * up.power(synchronicity, 2) * (1 + mass_ratio)
    e = up.power(up.sin(phi), 2)

    if np.isscalar(phi):
        return a, b, c, d, e
    else:
        dd = d * np.ones(np.shape(phi))
        return (a, b, c, dd, e) if return_as_tuple else np.column_stack((a, b, c, dd, e))


def pre_calculate_for_potential_value_secondary_cylindrical(*args, return_as_tuple=False):
    """
    Function calculates auxiliary values for calculation of secondary
    component potential in cylindrical symmetry, and therefore they don't need
    to be wastefully recalculated every iteration in solver.

    :param return_as_tuple: return coefficients as a tuple of numpy vectors instead of numpy matrix
    :type return_as_tuple: bool;
    :param args: (azimut angle (0, 2pi), z_n (cylindrical, identical with cartesian x)), components distance
    :return: Tuple; (a, b, c, d, e, f) such that: Psi2 = q/sqrt(a+r^2) + 1/sqrt(b + r^2) - c + d*(a+e*r^2)
    """
    synchronicity, mass_ratio, phi, z, distance = args

    a = up.power(z, 2)
    b = up.power(distance - z, 2)
    c = z / up.power(distance, 2)
    d = 0.5 * up.power(synchronicity, 2) * (1 + mass_ratio)
    e = up.power(up.sin(phi), 2)
    f = 0.5 * (1 - mass_ratio)

    if np.isscalar(phi):
        return a, b, c, d, e, f
    else:
        dd = d * np.ones(np.shape(phi))
        ff = f * np.ones(np.shape(phi))
        return (a, b, c, dd, e, ff) if return_as_tuple else np.column_stack((a, b, c, dd, e, ff))


def primary_potential_derivative_x(x, *args):
    """
    Derivative of potential function perspective of primary component along the x axis.

    :param x: (numpy.)float;
    :param args: Tuple (float, float, float); (synchronicity of primary component, mass ratio, components distance)
    :return: (numpy.)float;
    """
    synchronicity, mass_ratio, d = args
    r_sqr, rw_sqr = x ** 2, (d - x) ** 2
    return - (x / r_sqr ** (3.0 / 2.0)) + ((mass_ratio * (d - x)) / rw_sqr ** (
            3.0 / 2.0)) + synchronicity ** 2 * (mass_ratio + 1) * x - mass_ratio / d ** 2


def secondary_potential_derivative_x(x, *args):
    """
    Derivative of potential function perspective of secondary component along the x axis.

    :param x: (numpy.)float;
    :param args: Tuple (float, float, float); (synchronicity of secondary component, mass ratio, components distance)
    :return: (numpy.)float;
    """
    synchronicity, mass_ratio, d = args
    r_sqr, rw_sqr = x ** 2, (d - x) ** 2
    return - (x / r_sqr ** (3.0 / 2.0)) + ((mass_ratio * (d - x)) / rw_sqr ** (
            3.0 / 2.0)) - synchronicity ** 2 * (mass_ratio + 1) * (d - x) + (1.0 / d ** 2)
