import numpy as np
from copy import copy
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
    return t_eff[star_container.face_symmetry_vector] if star_container.symmetry_test() else t_eff


def calculate_polar_effective_temperature(star_container):
    """
    Returns polar effective temperature.

    :param star_container: elisa.base.container.StarContainer;
    :return:
    """
    areas = copy(star_container.areas)
    potential_gradient_magnitudes = star_container.potential_gradient_magnitudes
    if star_container.has_spots():
        for idx, spot in star_container.spots.items():
            areas = up.concatenate((areas, spot.areas), axis=0)
            potential_gradient_magnitudes = \
                up.concatenate((potential_gradient_magnitudes, spot.potential_gradient_magnitudes), axis=0)

    return star_container.t_eff * up.power(np.sum(areas) /
                                           np.sum(areas * up.power(
                                               potential_gradient_magnitudes /
                                               star_container.polar_potential_gradient_magnitude,
                                               star_container.gravity_darkening)),
                                           0.25)
