from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import const as c
from elisa.base.surface import gravity as bgravity
from elisa.logger import getLogger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.single_system.container import SinglePositionContainer
    from elisa.types import Float


logger = getLogger("single_system.surface.gravity")


def build_surface_gravity(system_container: SinglePositionContainer) -> SinglePositionContainer:
    """Calculate surface gravity (potential gradient magnitudes) for a star container.

    The function computes the magnitude of the gradient of the gravitational
    potential (including centrifugal contributions) for every surface face
    of the star stored in ``system_container`` and saves the results back into
    the container. If the star has surface spots, the same quantity is
    calculated for each spot and stored on the spot object.

    :param system_container: Container with single-system position and star data.
    :type system_container: elisa.single_system.container.SinglePositionContainer
    :return: The same container with ``potential_gradient_magnitudes`` and ``log_g``
        assigned for the star and for spots (if present).
    :rtype: elisa.single_system.container.SinglePositionContainer
    """
    star_container = system_container.star

    polar_gravity = np.power(10, star_container.polar_log_g)
    star_container.polar_potential_gradient_magnitude = polar_gravity

    logger.debug("computing potential gradient magnitudes distribution on a star")
    points, faces = bgravity.eval_args_for_magnitude_gradient(star_container)

    g_acc_vector = calculate_potential_gradient(
        points, system_container.angular_velocity, star_container.mass,
    )
    gravity = np.mean(np.linalg.norm(g_acc_vector, axis=1)[faces], axis=1)

    if star_container.symmetry_test():
        star_container.potential_gradient_magnitudes = star_container.mirror_face_values(gravity)
    else:
        star_container.potential_gradient_magnitudes = gravity

    star_container.log_g = np.log10(star_container.potential_gradient_magnitudes)

    if star_container.has_spots():
        g_acc_vector_spot: dict[str, NDArray[Float]] = {}
        for spot_index, spot in star_container.spots.items():
            logger.debug("calculating surface SI unit gravity of %s spot", spot_index)
            logger.debug(
                "calculating distribution of potential gradient magnitudes of spot index: %s component",
                spot_index,
            )

            g_acc_vector_spot[spot_index] = calculate_potential_gradient(
                spot.points, system_container.angular_velocity, star_container.mass,
            )

            spot_vals = np.mean(np.linalg.norm(g_acc_vector_spot[spot_index], axis=1)[spot.faces], axis=1)
            spot.potential_gradient_magnitudes = spot_vals
            spot.log_g = np.log10(spot_vals)

    return system_container


def calculate_polar_potential_gradient_magnitude(polar_radius: Float, mass: Float) -> Float:
    """Calculate magnitude of polar gradient of gravitational potential.

    :param polar_radius: Polar radius value.
    :type polar_radius: elisa.types.Float
    :param mass: Stellar mass.
    :type mass: elisa.types.Float
    :return: Magnitude of the polar potential gradient.
    :rtype: elisa.types.Float
    """
    return c.G * mass * polar_radius / np.power(polar_radius, 3)


def calculate_potential_gradient(points: NDArray, angular_velocity: Float, mass: Float) -> NDArray[Float]:
    """Compute gravity potential gradient vectors for the supplied points.

    The returned array contains the (x, y, z) gradient vector for each input
    point. The gravitational contribution is computed from Newtonian gravity
    and the centrifugal term from rotation is subtracted for the x/y
    components.

    :param points: Array-like of shape (N, 3) with point coordinates.
    :type points: numpy.typing.NDArray
    :param angular_velocity: Angular velocity of rotation.
    :type angular_velocity: elisa.types.Float
    :param mass: Stellar mass.
    :type mass: elisa.types.Float
    :return: Array with shape (N, 3) containing potential gradient vectors.
    :rtype: numpy.typing.NDArray[elisa.types.Float]
    """
    pts = np.asarray(points)
    r3 = np.power(np.linalg.norm(pts, axis=1), 3)
    # ensure float dtype to avoid integer division surprises
    points_gradients = np.empty(pts.shape, dtype=float)
    points_gradients[:, 0] = c.G * mass * pts[:, 0] / r3 - (angular_velocity ** 2) * pts[:, 0]
    points_gradients[:, 1] = c.G * mass * pts[:, 1] / r3 - (angular_velocity ** 2) * pts[:, 1]
    points_gradients[:, 2] = c.G * mass * pts[:, 2] / r3

    return -points_gradients
