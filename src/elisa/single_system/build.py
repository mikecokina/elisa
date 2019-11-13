import numpy as np

from copy import copy
from elisa.logger import getLogger

logger = getLogger('single_system.build')






def build_surface(self, return_surface=False):
    """
    function for building of general system component points and surfaces including spots

    :param return_surface: bool - if true, function returns arrays with all points and faces (surface + spots)
    :type: str
    :return:
    """
    self.build_mesh()

    # build surface if there is no spot specified
    if not self.star.spots:
        build_surface_with_no_spots(self)
        if return_surface:
            return self.star.points, self.star.faces
        else:
            return

    # saving one eighth of the star without spots to be used as reference for faces unaffected by spots
    # self.star.base_symmetry_points = copy(self.star.points[:self.star.base_symmetry_points_number])
    # self.star.base_symmetry_faces = copy(self.star.faces[:self.star.base_symmetry_faces_number])
    build_surface_with_spots(self)

    if return_surface:
        ret_points = copy(self.star.points)
        ret_faces = copy(self.star.faces)
        for spot_index, spot in self.star.spots.items():
            n_points = np.shape(ret_points)[0]
            ret_faces = np.append(ret_faces, spot.faces+n_points, axis=0)
            ret_points = np.append(ret_points, spot.points, axis=0)
        return ret_points, ret_faces


def build_surface_map(self, colormap=None, return_map=False):
    """
    function calculates surface maps (temperature or gravity acceleration) for star and spot faces and it can return
    them as one array if return_map=True

    :param return_map: if True function returns arrays with surface map including star and spot segments
    :param colormap: str - `temperature` or `gravity`
    :return:
    """
    if colormap is None:
        raise ValueError('Specify colormap to calculate (`temperature` or `gravity_acceleration`).')

    build_surface_gravity(self)

    if colormap == 'temperature':
        build_temperature_distribution(self)
        # logger.debug('Computing effective temprature distibution of stellar surface.')
        # self.star.temperatures = self.star.calculate_effective_temperatures()
        if self.star.pulsations:
            logger.debug('Adding pulsations to surface temperature distribution of the star.')
            self.star.temperatures = self.star.add_pulsations()

    if self.star.spots:
        for spot_index, spot in self.star.spots.items():
            if colormap == 'temperature':
                if self.star.pulsations:
                    logger.debug('Adding pulsations to temperature distribution of spot: '
                                       '{}'.format(spot_index))
                    spot.temperatures = self.star.add_pulsations(points=spot.points, faces=spot.faces,
                                                                 temperatures=spot.temperatures)
        logger.debug('Renormalizing temperature map of star surface.')
        self.star.renormalize_temperatures()

    if return_map:
        if colormap == 'temperature':
            ret_list = copy(self.star.temperatures)
        elif colormap == 'gravity_acceleration':
            ret_list = copy(self.star.log_g)

        if self.star.spots:
            for spot_index, spot in self.star.spots.items():
                if colormap == 'temperature':
                    ret_list = np.append(ret_list, spot.temperatures)
                elif colormap == 'gravity_acceleration':
                    ret_list = np.append(ret_list, spot.log_g)
        return ret_list
    return

def build_surface_gravity(self):
    """
    function calculates gravity potential gradient magnitude (surface gravity) for each face

    :return:
    """

    logger.debug('computing surface areas of star')
    self.star.areas = self.star.calculate_areas()

    # compute and assign potential gradient magnitudes for elements if missing
    logger.debug('computing potential gradient magnitudes distribution of a star')
    self.star.potential_gradient_magnitudes = self.calculate_face_magnitude_gradient()

    logger.debug('computing magnitude of polar potential gradient')
    self.star.polar_potential_gradient_magnitude = self.calculate_polar_potential_gradient_magnitude()
    gravity_scalling_factor = np.power(10, self.star.polar_log_g) / self.star.polar_potential_gradient_magnitude
    self.star.log_g = np.log10(gravity_scalling_factor * self.star.potential_gradient_magnitudes)

    if self.star.spots:
        for spot_index, spot in self.star.spots.items():
            logger.debug('calculating surface areas of {} spot'.format(spot_index))
            spot.areas = spot.calculate_areas()

            logger.debug('calculating distribution of potential '
                               'gradient magnitudes of {} spot'.format(spot_index))
            spot.potential_gradient_magnitudes = self.calculate_face_magnitude_gradient(points=spot.points,
                                                                                        faces=spot.faces)
            spot.log_g = np.log10(gravity_scalling_factor * spot.potential_gradient_magnitudes)

            
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

