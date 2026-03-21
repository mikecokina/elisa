from __future__ import annotations

from typing import TYPE_CHECKING

from elisa.base.surface import temperature as btemperature
from elisa.logger import getLogger

if TYPE_CHECKING:
    from elisa.single_system.container import SinglePositionContainer


logger = getLogger("single_system.surface.temperature")


def build_temperature_distribution(system_container: SinglePositionContainer) -> SinglePositionContainer:
    """Compute temperature distribution for the star inside a position container.

    This function calculates effective temperatures across all surface faces
    of the star stored in ``system_container`` and stores the result back into
    the container. If the star contains spots, their temperature distributions
    are computed using each spot's potential gradient magnitudes and temperature
    factor.

    :param system_container: Container with single-system position data and star.
    :type system_container: elisa.single_system.container.SinglePositionContainer
    :return: The same container instance with ``temperatures`` populated for the star
        (and for spots, if present).
    :rtype: elisa.single_system.container.SinglePositionContainer
    """
    star_container = system_container.star

    logger.debug("Computing effective temperature distribution on the star.")
    star_container.temperatures = btemperature.calculate_effective_temperatures(
        star_container, star_container.potential_gradient_magnitudes,
    )

    if star_container.has_spots():
        for spot_index, spot in star_container.spots.items():
            logger.debug("Computing temperature distribution of %s spot", spot_index)

            pgms = spot.potential_gradient_magnitudes
            spot.temperatures = spot.temperature_factor * btemperature.calculate_effective_temperatures(
                star_container,
                pgms,
            )

    return system_container
