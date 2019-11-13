import numpy as np

from copy import copy
from elisa.logger import getLogger

logger = getLogger('single_system.build')



# def build_surface_map(self, colormap=None, return_map=False):
#     """
#     function calculates surface maps (temperature or gravity acceleration) for star and spot faces and it can return
#     them as one array if return_map=True
#
#     :param return_map: if True function returns arrays with surface map including star and spot segments
#     :param colormap: str - `temperature` or `gravity`
#     :return:
#     """
#     if colormap is None:
#         raise ValueError('Specify colormap to calculate (`temperature` or `gravity_acceleration`).')
#
#     build_surface_gravity(self)
#
#     if colormap == 'temperature':
#         build_temperature_distribution(self)
#         # logger.debug('Computing effective temprature distibution of stellar surface.')
#         # self.star.temperatures = self.star.calculate_effective_temperatures()
#         if self.star.pulsations:
#             logger.debug('Adding pulsations to surface temperature distribution of the star.')
#             self.star.temperatures = self.star.add_pulsations()
#
#     if self.star.spots:
#         for spot_index, spot in self.star.spots.items():
#             if colormap == 'temperature':
#                 if self.star.pulsations:
#                     logger.debug('Adding pulsations to temperature distribution of spot: '
#                                        '{}'.format(spot_index))
#                     spot.temperatures = self.star.add_pulsations(points=spot.points, faces=spot.faces,
#                                                                  temperatures=spot.temperatures)
#         logger.debug('Renormalizing temperature map of star surface.')
#         self.star.renormalize_temperatures()
#
#     if return_map:
#         if colormap == 'temperature':
#             ret_list = copy(self.star.temperatures)
#         elif colormap == 'gravity_acceleration':
#             ret_list = copy(self.star.log_g)
#
#         if self.star.spots:
#             for spot_index, spot in self.star.spots.items():
#                 if colormap == 'temperature':
#                     ret_list = np.append(ret_list, spot.temperatures)
#                 elif colormap == 'gravity_acceleration':
#                     ret_list = np.append(ret_list, spot.log_g)
#         return ret_list
#     return

            
def build_temperature_distribution(self):
    """
    function calculates temperature distribution on across all faces

    :return:
    """
    logger.debug('Computing effective temprature distibution on the star.')
    self.star.temperatures = self.star.calculate_effective_temperatures()
    if self.star.pulsations:
        logger.debug('Adding pulsations to surface temperature distribution ')
        self.star.temperatures = self.star.add_pulsations()

    if self.star.spots:
        for spot_index, spot in self.star.spots.items():
            logger.debug('Computing temperature distribution of {} spot'.format(spot_index))
            spot.temperatures = spot.temperature_factor * self.star.calculate_effective_temperatures(
                gradient_magnitudes=spot.potential_gradient_magnitudes)
            if self.star.pulsations:
                logger.debug('Adding pulsations to temperature distribution of {} spot'.format(spot_index))
                spot.temperatures = self.star.add_pulsations(points=spot.points, faces=spot.faces,
                                                             temperatures=spot.temperatures)

