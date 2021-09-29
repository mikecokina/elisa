from ... logger import getLogger
from ... base.surface import temperature as btemperature
from ... pulse import pulsations
from elisa.base.surface.temperature import renormalize_temperatures

logger = getLogger("single_system.surface.temperature")


def build_temperature_distribution(system_container):
    """
    Function calculates temperature distribution on across all faces.

    :param system_container: elisa.single_system.container.SinglePositionContainer;
    :return: elisa.single_system.container.SinglePositionContainer;
    """
    star_container = system_container.star

    logger.debug('Computing effective temperature distribution on the star.')
    star_container.temperatures = \
        btemperature.calculate_effective_temperatures(star_container, star_container.potential_gradient_magnitudes)

    if star_container.has_spots():
        for spot_index, spot in star_container.spots.items():
            logger.debug('Computing temperature distribution of {} spot'.format(spot_index))

            pgms = spot.potential_gradient_magnitudes
            spot.temperatures = \
                spot.temperature_factor * btemperature.calculate_effective_temperatures(star_container, pgms)

    # renormalize_temperatures(star_container)
    return system_container


def build_temperature_perturbations(system_container):
    """
    adds position perturbations to container mesh

    :param system_container: elisa.single_system.container.SinglePositionContainer;
    :return: elisa.single_system.container.SinglePositionContainer;
    """
    if system_container.has_pulsations():
        star = getattr(system_container, 'star')
        star = pulsations.incorporate_temperature_perturbations(
            star, com_x=0.0,
            phase=system_container.position.phase,
            time=system_container.time
        )
    return system_container
