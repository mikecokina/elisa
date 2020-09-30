import numpy as np
from copy import copy
from ... import umpy as up
from ... logger import getLogger


logger = getLogger("base.surface.temperature")


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


def renormalize_temperatures(star):
    """
    In case of spot presence, renormalize temperatures to fit effective temperature again,
    since spots disrupt effective temperature of Star as entity.
    """
    # no need to calculate surfaces they had to be calculated already, otherwise there is nothing to renormalize
    total_surface = np.sum(star.areas)
    if star.has_spots():
        for spot_index, spot in star.spots.items():
            total_surface += np.sum(spot.areas)
    desired_flux_value = total_surface * star.t_eff**4

    current_flux = np.sum(star.areas * star.temperatures**4)
    if star.spots:
        for spot_index, spot in star.spots.items():
            current_flux += np.sum(spot.areas * spot.temperatures**4)

    coefficient = up.power(desired_flux_value / current_flux, 0.25)
    logger.debug(f'surface temperature map renormalized by a factor {coefficient}')
    star.temperatures *= coefficient
    if star.spots:
        for spot_index, spot in star.spots.items():
            spot.temperatures *= coefficient
