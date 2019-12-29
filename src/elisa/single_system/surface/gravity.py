import numpy as np

from elisa import const as c
from elisa.logger import getLogger
from elisa.base.surface import gravity as bgravity
from elisa.utils import is_empty

logger = getLogger("binary-system-gravity-module")


def build_surface_gravity(system_container):
    """
        function calculates gravity potential gradient magnitude (surface gravity) for each face

        :return:
        """
    star_container = system_container.star

    polar_gravity = np.power(10, star_container.polar_log_g)

    pgm = calculate_polar_potential_gradient_magnitude(star_container.polar_radius, star_container.mass)
    star_container.polar_potential_gradient_magnitude = pgm
    gravity_scalling_factor = polar_gravity / pgm

    logger.debug('computing potential gradient magnitudes distribution on a star')
    pgms_args = bgravity.eval_args_for_magnitude_gradient(star_container) + \
                (system_container.angular_velocity, star_container.mass)
    pgms_kwargs = dict(
        **{"face_symmetry_vector": star_container.face_symmetry_vector} if not star_container.has_spots() else {})

    star_container.potential_gradient_magnitudes = calculate_face_magnitude_gradient(*pgms_args, **pgms_kwargs)

    star_container.log_g = np.log10(gravity_scalling_factor * star_container.potential_gradient_magnitudes)

    if star_container.has_spots():
        for spot_index, spot in star_container.spots.items():
            logger.debug(f'calculating surface SI unit gravity of {spot_index} spot')
            logger.debug(f'calculating distribution of potential gradient '
                         f'magnitudes of spot index: {spot_index} component')
            spot.potential_gradient_magnitudes = calculate_face_magnitude_gradient(spot.points, spot.faces,
                                                                                   system_container.angular_velocity,
                                                                                   star_container.mass,
                                                                                   face_symmetry_vector=None)
            spot.log_g = np.log10(gravity_scalling_factor * spot.potential_gradient_magnitudes)
    return system_container


def calculate_polar_potential_gradient_magnitude(polar_radius, mass):
    """
    returns magnitude of polar gradient of gravitational potential

    :param polar_radius: float;
    :param mass: float;
    :return: float;
    """
    points_z = polar_radius
    r3 = np.power(points_z, 3)
    domega_dz = c.G * mass * points_z / r3
    return domega_dz


def calculate_face_magnitude_gradient(points, faces, angular_velocity, mass, face_symmetry_vector=None):
    """
    returns array of absolute values of potential gradients for corresponding faces

    :param points: numpy.array; points in which to calculate magnitude of gradient, if False/None take star points
    :param faces: numpy.array; faces corresponding to given points
    :param angular_velocity: float;
    :param mass: float;
    :param face_symmetry_vector: Union[numpy.array, None];
    :return: numpy.array;
    """

    r3 = np.power(np.linalg.norm(points, axis=1), 3)
    domega_dx = c.G * mass * points[:, 0] / r3 \
                - np.power(angular_velocity, 2) * points[:, 0]
    domega_dy = c.G * mass * points[:, 1] / r3 \
                - np.power(angular_velocity, 2) * points[:, 1]
    domega_dz = c.G * mass * points[:, 2] / r3
    points_gradients = np.power(np.power(domega_dx, 2) + np.power(domega_dy, 2) + np.power(domega_dz, 2), 0.5)

    return np.mean(points_gradients[faces], axis=1) if is_empty(face_symmetry_vector) \
        else np.mean(points_gradients[faces], axis=1)[face_symmetry_vector]

