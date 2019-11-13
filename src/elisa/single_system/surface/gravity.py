import numpy as np

from elisa.logger import getLogger

logger = getLogger("binary-system-gravity-module")


def build_surface_gravity(system_container):
    """
        function calculates gravity potential gradient magnitude (surface gravity) for each face

        :return:
        """
    star_container = system_container.star

    # polar_gravity = np.log10(star_container.polar_log_g)
    # logger.debug('computing potential gradient magnitudes distribution of a star')
    # star_container.potential_gradient_magnitudes = self.calculate_face_magnitude_gradient()
    #
    # logger.debug('computing magnitude of polar potential gradient')
    # self.star.polar_potential_gradient_magnitude = self.calculate_polar_potential_gradient_magnitude()
    # gravity_scalling_factor = np.power(10, self.star.polar_log_g) / self.star.polar_potential_gradient_magnitude
    # self.star.log_g = np.log10(gravity_scalling_factor * self.star.potential_gradient_magnitudes)
    #
    # if self.star.spots:
    #     for spot_index, spot in self.star.spots.items():
    #         logger.debug('calculating surface areas of {} spot'.format(spot_index))
    #         spot.areas = spot.calculate_areas()
    #
    #         logger.debug('calculating distribution of potential '
    #                      'gradient magnitudes of {} spot'.format(spot_index))
    #         spot.potential_gradient_magnitudes = self.calculate_face_magnitude_gradient(points=spot.points,
    #                                                                                     faces=spot.faces)
    #         spot.log_g = np.log10(gravity_scalling_factor * spot.potential_gradient_magnitudes)
