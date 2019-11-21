import numpy as np
from ... import umpy as up


def calculate_effective_temperatures(star_container, gradient_magnitudes):
    """
    Calculates effective temperatures for given gradient magnitudes.
    If None is given, star surface t_effs are calculated.

    :param star_container: elisa.base.container.StarContainer;
    :param gradient_magnitudes: numpy.array;
    :return: numpy.array;
    """

    t_eff_polar = calculate_polar_effective_temperature(star_container)
    t_eff = t_eff_polar * up.power(gradient_magnitudes / star_container.polar_potential_gradient_magnitude,
                                   0.25 * star_container.gravity_darkening)
    return t_eff if star_container.symmetry_test() else t_eff[star_container.face_symmetry_vector]


def calculate_polar_effective_temperature(star_container):
    """
    Returns polar effective temperature.

    :param star_container: elisa.base.container.StarContainer;
    :return:
    """
    return star_container.t_eff * up.power(np.sum(star_container.areas) /
                                           np.sum(star_container.areas * up.power(
                                               star_container.potential_gradient_magnitudes /
                                               star_container.polar_potential_gradient_magnitude,
                                               star_container.gravity_darkening)),
                                           0.25)
