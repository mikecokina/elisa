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
    :param radius: np.float
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
