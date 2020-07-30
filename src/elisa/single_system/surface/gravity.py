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

    star_container.polar_potential_gradient_magnitude = polar_gravity

    logger.debug('computing potential gradient magnitudes distribution on a star')
    points, faces = bgravity.eval_args_for_magnitude_gradient(star_container)

    g_acc_vector = - calculate_potential_gradient(points, system_container.angular_velocity, star_container.mass)

    # TODO: here implement pulsations
    if star_container.has_pulsations():
        pass

    gravity = np.mean(np.linalg.norm(g_acc_vector, axis=1)[faces], axis=1)
    setattr(star_container, 'potential_gradient_magnitudes', gravity[star_container.face_symmetry_vector]) \
        if star_container.symmetry_test() else setattr(star_container, 'potential_gradient_magnitudes', gravity)
    setattr(star_container, 'log_g', np.log10(star_container.potential_gradient_magnitudes))

    if star_container.has_spots():
        for spot_index, spot in star_container.spots.items():
            logger.debug(f'calculating surface SI unit gravity of {spot_index} spot')
            logger.debug(f'calculating distribution of potential gradient '
                         f'magnitudes of spot index: {spot_index} component')

            g_acc_vector = - calculate_potential_gradient(spot.points, system_container.angular_velocity,
                                                          star_container.mass)

            # TODO: here implement pulsations
            if star_container.has_pulsations():
                pass

            setattr(spot, 'potential_gradient_magnitudes',
                    np.mean(np.linalg.norm(g_acc_vector, axis=1)[spot.faces], axis=1))
            setattr(spot, 'log_g', np.log10(spot.potential_gradient_magnitudes))

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


# deprecated
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
    point_gradient_magnitudes = np.linalg.norm(calculate_potential_gradient(points, angular_velocity, mass), axis=1)
    face_gradient_magnitudes = np.mean(point_gradient_magnitudes[faces], axis=1)

    return face_gradient_magnitudes if is_empty(face_symmetry_vector) else \
        face_gradient_magnitudes[face_symmetry_vector]


def calculate_potential_gradient(points, angular_velocity, mass):
    """
    returns array of gravity potential gradients for corresponding faces

    :param points: np.array; (N * 3) array of surface points
    :param faces: np.array; simplices of triangulated points
    :param angular_velocity: float; angular velocity of rotation
    :param mass: float; stellar mass
    :return: np.array; gravity gradients for each face
    """
    r3 = np.power(np.linalg.norm(points, axis=1), 3)
    points_gradients = np.empty(points.shape)
    points_gradients[:, 0] = c.G * mass * points[:, 0] / r3 - np.power(angular_velocity, 2) * points[:, 0]
    points_gradients[:, 1] = c.G * mass * points[:, 1] / r3 - np.power(angular_velocity, 2) * points[:, 1]
    points_gradients[:, 2] = c.G * mass * points[:, 2] / r3

    return points_gradients

