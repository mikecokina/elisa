from elisa.logger import getLogger
from elisa.base.surface import temperature as btemperature

logger = getLogger("single_system.surface.temperature")


def build_temperature_distribution(system_container, do_pulsations=False, phase=None):
    phase = 0 if phase is None else phase
    star_container = system_container.star

    logger.debug('Computing effective temperature distribution on the star.')
    star_container.temperatures = btemperature.calculate_effective_temperatures()

    if star_container.has_spots():
        for spot_index, spot in star_container.spots.items():
            logger.debug('Computing temperature distribution of {} spot'.format(spot_index))

            pgms = spot.potential_gradient_magnitudes
            spot.temperatures = \
                spot.temperature_factor * btemperature.calculate_effective_temperatures(star_container, pgms)

    # logger.debug('Computing effective temprature distibution on the star.')
    # self.star.temperatures = self.star.calculate_effective_temperatures()
    # if self.star.pulsations:
    #     logger.debug('Adding pulsations to surface temperature distribution ')
    #     self.star.temperatures = self.star.add_pulsations()
    #
    # if self.star.has_spots():
    #     for spot_index, spot in self.star.spots.items():
    #         logger.debug('Computing temperature distribution of {} spot'.format(spot_index))
    #         spot.temperatures = spot.temperature_factor * self.star.calculate_effective_temperatures(
    #             gradient_magnitudes=spot.potential_gradient_magnitudes)
    #         if self.star.pulsations:
    #             logger.debug('Adding pulsations to temperature distribution of {} spot'.format(spot_index))
    #             spot.temperatures = self.star.add_pulsations(points=spot.points, faces=spot.faces,
    #                                                          temperatures=spot.temperatures)

    return system_container
