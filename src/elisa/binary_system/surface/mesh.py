from __future__ import annotations

import functools
from typing import TYPE_CHECKING, cast

import numpy as np

from elisa import const, opt, settings, utils
from elisa import umpy as up
from elisa.base.error import MaxIterationError, SpotError
from elisa.base.spot import incorporate_spots_mesh
from elisa.base.surface.mesh import correct_component_mesh
from elisa.base.types import FLOAT
from elisa.binary_system import model
from elisa.binary_system import radius as bsradius
from elisa.binary_system import utils as bsutils
from elisa.logger import getLogger
from elisa.opt.fsolver import fsolver
from elisa.utils import is_empty

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.binary_system.system import BinarySystem
    from elisa.types import ComponentName, ComponentSelection, Float, Int

logger = getLogger("binary_system.surface.mesh")


MESH_NUM_POINTS_TRAPEZOIDAL_OVECONTACT = 4
MESH_NUM_POINTS_OVERCONTACT_NECK = 5
SPOT_POINTS_NDIM = 2


@functools.cache
def _load_correction_factors() -> dict[str, NDArray[Float]]:
    """Load mesh correction factor tables.

    :return: Mapping of morphology names to correction-factor tables.
    :rtype: dict[str, NDArray[Float]]
    """
    detached = np.load(settings.PATH_TO_DETACHED_CORRECTIONS, allow_pickle=False)
    over_contact = np.load(
        settings.PATH_TO_OVER_CONTACT_CORRECTIONS,
        allow_pickle=False,
    )
    return {
        "detached": detached,
        "semi-detached": detached,
        "double-contact": detached,
        "over-contact": over_contact,
    }


def build_mesh(
    system: OrbitalPositionContainer,
    components_distance: Float,
    component: ComponentSelection = "all",
) -> OrbitalPositionContainer:
    """Build surface points for selected binary-system components.

    In case of spots, the spot point mesh is incorporated into the model and
    the resulting points are assigned back to the system container.

    :param system: Orbital position container instance.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param component: Component selector.
    :type component: ComponentSelection
    :return: Updated orbital position container.
    :rtype: OrbitalPositionContainer
    """
    components = bsutils.component_to_list(component)

    for component_name in components:
        star = getattr(system, component_name)
        if system.morphology == "over-contact":
            points, base_points_count, inverse_symmetry_matrix = mesh_over_contact(
                system,
                component_name,
                symmetry_output=True,
            )
        else:
            points, base_points_count, inverse_symmetry_matrix = mesh_detached(
                system,
                components_distance,
                component_name,
                symmetry_output=True,
            )

        star.points = points
        star.base_symmetry_points_number = base_points_count
        star.inverse_point_symmetry_matrix = inverse_symmetry_matrix

    add_spots_to_mesh(system, components_distance, component="all")
    return system


def rebuild_symmetric_detached_mesh(
    system: OrbitalPositionContainer,
    components_distance: Float,
    component: ComponentSelection,
) -> OrbitalPositionContainer:
    """Rebuild a symmetric detached mesh using the previous mesh azimuths.

    This preserves the number of points and faces.

    :param system: Orbital position container instance.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param component: Component selector.
    :type component: ComponentSelection
    :return: Updated orbital position container.
    :rtype: OrbitalPositionContainer
    """
    components = bsutils.component_to_list(component)

    for component_name in components:
        star = getattr(system, component_name)
        star.points = rebuild_mesh_detached(system, components_distance, component_name)

    return system


def pre_calc_azimuths_for_detached_points(
    discretization: Float,
    star: StarContainer,
) -> tuple[NDArray[Float], NDArray[Float], list[int]]:
    """Route to the detached-surface discretization method.

    :param discretization: Discretization factor.
    :type discretization: Float
    :param star: Stellar container.
    :type star: StarContainer
    :return: Tuple ``(phi, theta, separator)``.
    :rtype: tuple[NDArray[Float], NDArray[Float], list[int]]
    """
    if settings.MESH_GENERATOR in ["auto", "improved_trapezoidal"]:
        rel_radii = (star.forward_radius - star.polar_radius) / star.polar_radius
        if rel_radii > settings.DEFORMATION_TOL or settings.MESH_GENERATOR == "improved_trapezoidal":
            args = (
                discretization,
                star.forward_radius,
                star.polar_radius,
                star.side_radius,
                star.backward_radius,
            )
            return improved_trapezoidal_mesh(*args)

    return trapezoidal_mesh(discretization)


def trapezoidal_mesh(
    discretization: Float,
) -> tuple[NDArray[Float], NDArray[Float], list[int]]:
    """Calculate azimuths using trapezoidal discretization.

    This method works well for nearly spherical stars.

    :param discretization: Discretization factor.
    :type discretization: Float
    :return: Tuple ``(phi, theta, separator)``.
    :rtype: tuple[NDArray[Float], NDArray[Float], list[int]]
    """
    vertical_alpha = discretization
    separator: list[int] = []

    num = int(const.PI // (const.SEAM_CONST * discretization))
    phi = np.linspace(0.0, const.PI, num=num + 1)
    theta = np.full(phi.shape, const.HALF_PI)
    separator.append(np.shape(theta)[0])

    v_num = int(const.HALF_PI // (const.SEAM_CONST * vertical_alpha))
    phi_meridian = np.concatenate((const.PI * np.ones(v_num - 1), np.zeros(v_num)))
    theta_meridian = up.concatenate(
        (
            np.linspace(const.HALF_PI, 0, num=v_num + 1)[1:-1],
            np.linspace(0.0, const.HALF_PI, num=v_num, endpoint=False),
        ),
    )

    phi = up.concatenate((phi, phi_meridian))
    theta = up.concatenate((theta, theta_meridian))
    separator.append(np.shape(theta)[0])

    v_num = int(const.HALF_PI // vertical_alpha)
    thetas = np.linspace(discretization, const.HALF_PI, num=v_num - 1, endpoint=False)
    phi_parts = [phi]
    theta_parts = [theta]
    for theta_value in thetas:
        alpha_corrected = discretization / up.sin(theta_value)
        num = int(const.PI // alpha_corrected)
        alpha_corrected = const.PI / (num + 1)
        phi_q_add = alpha_corrected * np.arange(1, num + 1)
        phi_parts.append(phi_q_add)
        theta_parts.append(np.full(num, theta_value))
    phi = np.concatenate(phi_parts)
    theta = np.concatenate(theta_parts)

    return phi, theta, separator


def improved_trapezoidal_mesh(
    discretization: Float,
    forward_radius: Float,
    polar_radius: Float,
    side_radius: Float,
    backward_radius: Float,
) -> tuple[NDArray[Float], NDArray[Float], list[int]]:
    """Calculate azimuths using improved trapezoidal discretization.

    This method conserves triangle areas better than the standard trapezoidal
    method.

    :param discretization: Discretization factor.
    :type discretization: Float
    :param forward_radius: Forward radius.
    :type forward_radius: Float
    :param polar_radius: Polar radius.
    :type polar_radius: Float
    :param side_radius: Side radius.
    :type side_radius: Float
    :param backward_radius: Backward radius.
    :type backward_radius: Float
    :return: Tuple ``(phi, theta, separator)``.
    :rtype: tuple[NDArray[Float], NDArray[Float], list[int]]
    """
    vertical_alpha = discretization
    separator: list[int] = []

    num = int(const.PI // (const.SEAM_CONST * discretization))
    phi = np.linspace(0.0, const.PI, num=num + 1)
    theta = np.full(phi.shape, const.HALF_PI)
    separator.append(np.shape(theta)[0])

    inner_mask = (phi < const.HALF_PI) & (phi > 0)
    outer_mask = (phi > const.HALF_PI) & (phi < const.PI)
    inner_phis = phi[inner_mask]
    outer_phis = phi[outer_mask]
    tan_phs1 = up.tan(inner_phis)
    tan_phs2 = up.tan(outer_phis)
    inner_corr = up.arctan(
        (side_radius - forward_radius) * tan_phs1 / (side_radius + forward_radius * tan_phs1**2),
    )
    outer_corr = up.arctan(
        (side_radius - backward_radius) * tan_phs2 / (side_radius + backward_radius * tan_phs2**2),
    )
    phi[inner_mask] += inner_corr
    phi[outer_mask] += outer_corr

    v_num = int(const.HALF_PI // (1.07 * vertical_alpha))
    phi_meridian = np.concatenate((const.PI * np.ones(v_num - 1), np.zeros(v_num)))
    theta_meridian = up.concatenate(
        (
            np.linspace(const.HALF_PI, 0, num=v_num + 1)[1:-1],
            np.linspace(0.0, const.HALF_PI, num=v_num, endpoint=False),
        ),
    )

    v_num = int(const.HALF_PI // vertical_alpha)
    thetas_lin = np.linspace(discretization, const.HALF_PI, num=v_num - 1, endpoint=False)

    est_eqt_r = (side_radius + forward_radius + backward_radius) / 3.0
    tan_tht = np.tan(theta_meridian)
    theta_meridian += up.arctan(
        (est_eqt_r - polar_radius) * tan_tht / (polar_radius + est_eqt_r * tan_tht**2),
    )

    phi = up.concatenate((phi, phi_meridian))
    theta = up.concatenate((theta, theta_meridian))
    separator.append(np.shape(theta)[0])

    tan_tht = np.tan(thetas_lin)
    thetas = thetas_lin + up.arctan(
        (est_eqt_r - polar_radius) * tan_tht / (polar_radius + est_eqt_r * tan_tht**2),
    )

    phi_parts, theta_parts = [phi], [theta]
    for theta_value in thetas:
        alpha_corrected = discretization / up.sin(theta_value)
        num = int(const.PI // alpha_corrected)
        alpha_corrected = const.PI / (num + 1)
        phi_q_add = alpha_corrected * np.arange(1, num + 1)

        inner_mask = phi_q_add < const.HALF_PI
        outer_mask = phi_q_add > const.HALF_PI
        inner_phis = phi_q_add[inner_mask]
        outer_phis = phi_q_add[outer_mask]

        tan_phs1 = up.tan(inner_phis)
        tan_phs2 = up.tan(outer_phis)
        scaling_factor = np.sin(theta_value)
        inner_corr = np.arctan(
            scaling_factor * (side_radius - forward_radius) * tan_phs1 / (side_radius + forward_radius * tan_phs1**2),
        )
        outer_corr = np.arctan(
            scaling_factor * (side_radius - backward_radius) * tan_phs2 / (side_radius + backward_radius * tan_phs2**2),
        )
        phi_q_add[inner_mask] += inner_corr
        phi_q_add[outer_mask] += outer_corr

        phi_parts.append(phi_q_add)
        theta_parts.append(np.full(num, theta_value))
    phi, theta = np.concatenate(phi_parts), np.concatenate(theta_parts)

    return phi, theta, separator


def pre_calc_azimuths_for_overcontact_points(
    discretization: Float,
    star: StarContainer,
    component: ComponentName,
    neck_position: Float,
    neck_polynomial: NDArray[Float],
) -> tuple[
    tuple[NDArray[Float], NDArray[Float], list[int]],
    tuple[NDArray[Float], NDArray[Float], list[int]],
]:
    """Route to the over-contact discretization methods.

    :param discretization: Discretization factor.
    :type discretization: Float
    :param star: Stellar container.
    :type star: StarContainer
    :param component: Component selector.
    :type component: ComponentName
    :param neck_position: x coordinate of the neck.
    :type neck_position: Float
    :param neck_polynomial: Polynomial coefficients describing the neck.
    :type neck_polynomial: NDArray[Float]
    :return: Tuples for far-side and neck azimuth arguments.
    :rtype: tuple[tuple[NDArray[Float], NDArray[Float], list[int]], tuple[NDArray[Float], NDArray[Float], list[int]]]
    """
    if settings.MESH_GENERATOR in ["auto", "improved_trapezoidal"]:
        rel_radii = (star.backward_radius - star.polar_radius) / star.polar_radius
        if rel_radii > settings.DEFORMATION_TOL or settings.MESH_GENERATOR == "improved_trapezoidal":
            far_azim = improved_trapezoidal_overcontact_farside_points(
                discretization,
                star.polar_radius,
                star.side_radius,
                star.backward_radius,
            )
            near_azim = improved_trapezoidal_overcontact_neck_points(
                discretization,
                neck_position,
                neck_polynomial,
                star.polar_radius,
                star.side_radius,
                component,
            )
            return far_azim, near_azim

    far_azim = trapezoidal_overcontact_farside_points(discretization)
    near_azim = trapezoidal_overcontact_neck_points(
        discretization,
        neck_position,
        neck_polynomial,
        star.polar_radius,
        component,
    )
    return far_azim, near_azim


def trapezoidal_overcontact_farside_points(
    discretization: Float,
) -> tuple[NDArray[Float], NDArray[Float], list[int]]:
    """Calculate far-side azimuths for over-contact components.

    :param discretization: Discretization factor.
    :type discretization: Float
    :return: Tuple ``(phi, theta, separator)``.
    :rtype: tuple[NDArray[Float], NDArray[Float], list[int]]
    """
    vertical_alpha = discretization
    separator: list[int] = []

    num = int(const.HALF_PI // (const.SEAM_CONST * discretization))
    phi = np.linspace(const.HALF_PI, const.PI, num=num + 2)
    theta = np.full(phi.shape, const.HALF_PI)
    separator.append(np.shape(theta)[0])

    v_num = int(const.HALF_PI / (const.SEAM_CONST * vertical_alpha))
    phi_meridian1 = np.full(v_num - 1, const.PI)
    theta_meridian1 = np.linspace(0.0, const.HALF_PI, num=v_num - 1, endpoint=False)
    phi = up.concatenate((phi, phi_meridian1))
    theta = up.concatenate((theta, theta_meridian1))
    separator.append(np.shape(theta)[0])

    v_num -= 1
    phi_meridian2 = np.full(v_num - 1, const.HALF_PI)
    theta_meridian2 = np.linspace(0, const.HALF_PI, num=v_num, endpoint=False)[1:]
    phi = up.concatenate((phi, phi_meridian2))
    theta = up.concatenate((theta, theta_meridian2))
    separator.append(np.shape(theta)[0])

    v_num = int(const.HALF_PI / vertical_alpha)
    theta_meridian = np.linspace(0.0, const.HALF_PI, num=v_num - 1, endpoint=False)
    phi_parts = [phi]
    theta_parts = [theta]
    for theta_value in theta_meridian[1:]:
        alpha_corrected = discretization / up.sin(theta_value)
        num = int(const.HALF_PI // alpha_corrected)
        alpha_corrected = const.HALF_PI / (num + 1)
        phi_q_add = const.HALF_PI + alpha_corrected * np.arange(1, num + 1)
        phi_parts.append(phi_q_add)
        theta_parts.append(np.full(phi_q_add.shape[0], theta_value))
    phi = np.concatenate(phi_parts)
    theta = np.concatenate(theta_parts)

    separator.append(len(theta))
    return phi, theta, separator


def improved_trapezoidal_overcontact_farside_points(
    discretization: Float,
    polar_radius: Float,
    side_radius: Float,
    backward_radius: Float,
) -> tuple[NDArray[Float], NDArray[Float], list[int]]:
    """Calculate far-side azimuths using improved trapezoidal discretization.

    :param discretization: Discretization factor.
    :type discretization: Float
    :param polar_radius: Polar radius.
    :type polar_radius: Float
    :param side_radius: Side radius.
    :type side_radius: Float
    :param backward_radius: Backward radius.
    :type backward_radius: Float
    :return: Tuple ``(phi, theta, separator)``.
    :rtype: tuple[NDArray[Float], NDArray[Float], list[int]]
    """
    vertical_alpha = discretization
    separator: list[int] = []

    num = int(const.HALF_PI / (const.SEAM_CONST * discretization))
    phi = np.linspace(const.HALF_PI, const.PI, num=num + 2)
    theta = np.full(phi.shape, const.HALF_PI)
    separator.append(np.shape(theta)[0])

    tan_phs = up.tan(phi)
    corr = up.arctan(
        (side_radius - backward_radius) * tan_phs / (side_radius + backward_radius * tan_phs**2),
    )
    phi += corr

    v_num = int(const.HALF_PI / (const.SEAM_CONST * vertical_alpha))
    phi_meridian1 = np.full(v_num - 1, const.PI)
    theta_meridian1 = np.linspace(0.0, const.HALF_PI, num=v_num - 1, endpoint=False)
    est_eqt_r = (side_radius + 2 * backward_radius) / 3.0
    tan_tht = np.tan(theta_meridian1)
    theta_meridian1 += up.arctan(
        (est_eqt_r - polar_radius) * tan_tht / (polar_radius + est_eqt_r * tan_tht**2),
    )

    phi = up.concatenate((phi, phi_meridian1))
    theta = up.concatenate((theta, theta_meridian1))
    separator.append(np.shape(theta)[0])

    v_num -= 1
    phi_meridian2 = np.full(v_num - 1, const.HALF_PI)
    theta_meridian2 = theta_meridian1[1:]
    phi = up.concatenate((phi, phi_meridian2))
    theta = up.concatenate((theta, theta_meridian2))
    separator.append(np.shape(theta)[0])

    v_num = int(const.HALF_PI / vertical_alpha)
    theta_meridian = np.linspace(0.0, const.HALF_PI, num=v_num - 1, endpoint=False)
    tan_tht = np.tan(theta_meridian)
    theta_meridian += up.arctan(
        (est_eqt_r - polar_radius) * tan_tht / (polar_radius + est_eqt_r * tan_tht**2),
    )
    phi_parts = [phi]
    theta_parts = [theta]
    for theta_value in theta_meridian[1:]:
        alpha_corrected = discretization / up.sin(theta_value)
        num = int(const.HALF_PI // alpha_corrected)
        alpha_corrected = const.HALF_PI / (num + 1)
        phi_q_add = const.HALF_PI + alpha_corrected * np.arange(1, num + 1)
        scaling_factor = np.sin(theta_value)
        tan_phs = up.tan(phi_q_add)
        corr = np.arctan(
            scaling_factor * (side_radius - backward_radius) * tan_phs / (side_radius + backward_radius * tan_phs**2),
        )
        phi_q_add += corr

        phi_parts.append(phi_q_add)
        theta_parts.append(np.full(phi_q_add.shape[0], theta_value))
    phi = np.concatenate(phi_parts)
    theta = np.concatenate(theta_parts)

    separator.append(len(theta))
    return phi, theta, separator


def _generate_neck_zs(
    delta_z: Float,
    component: ComponentName,
    neck_position: Float,
    neck_polynomial: NDArray[Float],
) -> tuple[
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    list[Int],
]:
    """Generate sampling positions on the neck of an over-contact system.

    :param delta_z: Sampling step in the neck axial direction.
    :type delta_z: Float
    :param component: Component selector.
    :type component: ComponentName
    :param neck_position: x coordinate of the neck.
    :type neck_position: Float
    :param neck_polynomial: Polynomial coefficients describing neck profile in
        the ``xz`` plane.
    :type neck_polynomial: NDArray[Float]
    :return: Tuple ``(phi, z, z_ns, r_neck, separator)``.
    :rtype: tuple[NDArray[Float], NDArray[Float], NDArray[Float], NDArray[Float], list[int]]
    """
    delta_z_polar = const.SEAM_CONST * delta_z
    separator: list[Int] = []

    if component == "primary":
        num = 100 * int(neck_position // delta_z)
        x_curve = np.linspace(0.0, neck_position, num=num, endpoint=True)
        z_curve = np.polyval(neck_polynomial, x_curve)
        curve = np.column_stack((x_curve, z_curve))
        lengths = up.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
        neck_lengths = np.cumsum(lengths)
        num_z = int(neck_lengths[-1] // delta_z)
        num_z_polar = int(neck_lengths[-1] // delta_z_polar)
        segments = np.linspace(0, neck_lengths[-1], num=num_z)[1:]
        segments_polar = np.linspace(0, neck_lengths[-1], num=num_z_polar)[1:]
        z_ns = np.interp(segments, neck_lengths, x_curve[1:])
        z_ns_polar = np.interp(segments_polar, neck_lengths, x_curve[1:])
        # Ensure interpolated positions and evaluated radii are float arrays
        z_ns = np.asarray(z_ns, dtype=FLOAT)
        z_ns_polar = np.asarray(z_ns_polar, dtype=FLOAT)
        r_neck = np.asarray(np.polyval(neck_polynomial, z_ns), dtype=FLOAT)
    else:
        num = 100 * int((1 - neck_position) // delta_z)
        x_curve = np.linspace(neck_position, 1, num=num, endpoint=True)
        z_curve = np.polyval(neck_polynomial, x_curve)
        curve = np.column_stack((x_curve, z_curve))
        lengths = up.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
        neck_lengths = np.cumsum(lengths)
        num_z = int(neck_lengths[-1] // delta_z)
        num_z_polar = int(neck_lengths[-1] // delta_z_polar)
        segments = np.linspace(0, neck_lengths[-1], num=num_z)[:-1]
        segments_polar = np.linspace(0, neck_lengths[-1], num=num_z_polar)[:-1]
        z_ns = np.interp(segments, neck_lengths, x_curve[:-1])
        z_ns_polar = np.interp(segments_polar, neck_lengths, x_curve[:-1])
        # Ensure interpolated positions and evaluated radii are float arrays
        z_ns = np.asarray(z_ns, dtype=FLOAT)
        z_ns_polar = np.asarray(z_ns_polar, dtype=FLOAT)
        r_neck = np.asarray(np.polyval(neck_polynomial, z_ns), dtype=FLOAT)
        z_ns = 1 - z_ns
        z_ns_polar = 1 - z_ns_polar

    phi: NDArray[Float] = np.full(z_ns_polar.shape, const.HALF_PI)
    z: NDArray[Float] = z_ns_polar
    separator.append(np.shape(z)[0])
    phi = up.concatenate((phi, np.zeros(z_ns_polar.shape)))
    z = up.concatenate((z, z_ns_polar))
    separator.append(np.shape(z)[0])

    return phi, z, z_ns, r_neck, separator


def trapezoidal_overcontact_neck_points(
    discretization: Float,
    neck_position: Float,
    neck_polynomial: NDArray[Float],
    polar_radius: Float,
    component: ComponentName,
) -> tuple[NDArray[Float], NDArray[Float], list[int]]:
    """Calculate azimuths to neck surface points of an over-contact component.

    :param discretization: Discretization factor.
    :type discretization: Float
    :param neck_position: x position of the neck.
    :type neck_position: Float
    :param neck_polynomial: Polynomial coefficients defining the neck profile in
        plane ``xz``.
    :type neck_polynomial: NDArray[Float]
    :param polar_radius: Polar radius.
    :type polar_radius: Float
    :param component: Component selector.
    :type component: ComponentName
    :return: Tuple ``(phi, z, separator)``.
    :rtype: tuple[NDArray[Float], NDArray[Float], list[int]]
    """
    delta_z = discretization * polar_radius
    phi, z, z_ns, r_neck, separator = _generate_neck_zs(
        delta_z,
        component,
        neck_position,
        neck_polynomial,
    )

    phi_parts = [phi]
    z_parts = [z]
    for index, zz in enumerate(z_ns):
        num = int(const.HALF_PI * r_neck[index] // delta_z)
        num = num + 1 if num < MESH_NUM_POINTS_OVERCONTACT_NECK else num
        phis = np.linspace(0, const.HALF_PI, num=int(num), endpoint=False)[1:]
        phi_parts.append(phis)
        z_parts.append(np.full(phis.shape[0], zz))
    phi = np.concatenate(phi_parts)
    z = np.concatenate(z_parts)

    separator.append(len(z))
    return phi, z, separator


def improved_trapezoidal_overcontact_neck_points(
    discretization: Float,
    neck_position: Float,
    neck_polynomial: NDArray[Float],
    polar_radius: Float,
    side_radius: Float,
    component: ComponentName,
) -> tuple[NDArray[Float], NDArray[Float], list[int]]:
    """Calculate neck azimuths using improved trapezoidal discretization.

    :param discretization: Discretization factor.
    :type discretization: Float
    :param neck_position: x position of the neck.
    :type neck_position: Float
    :param neck_polynomial: Polynomial coefficients defining the neck profile in
        plane ``xz``.
    :type neck_polynomial: NDArray[Float]
    :param polar_radius: Polar radius.
    :type polar_radius: Float
    :param side_radius: Side radius.
    :type side_radius: Float
    :param component: Component selector.
    :type component: ComponentName
    :return: Tuple ``(phi, z, separator)``.
    :rtype: tuple[NDArray[Float], NDArray[Float], list[int]]
    """
    delta_z = discretization * polar_radius
    phi, z, z_ns, r_neck, separator = _generate_neck_zs(
        delta_z,
        component,
        neck_position,
        neck_polynomial,
    )

    eq_coeff = side_radius / polar_radius
    phi_parts = [phi]
    z_parts = [z]
    for index, zz in enumerate(z_ns):
        num = const.HALF_PI * r_neck[index] // delta_z
        num = num + 1 if num < MESH_NUM_POINTS_TRAPEZOIDAL_OVECONTACT else num
        phis = np.linspace(0, const.HALF_PI, num=int(num), endpoint=False)[1:]
        tan_phis = np.tan(phis)
        phis += up.arctan((eq_coeff - 1) * tan_phis / (1 + eq_coeff * tan_phis**2))
        phi_parts.append(phis)
        z_parts.append(np.full(phis.shape[0], zz))
    phi = np.concatenate(phi_parts)
    z = np.concatenate(z_parts)

    separator.append(len(z))
    return phi, z, separator


def get_surface_points(
    phi: NDArray,
    theta: NDArray,
    x0: Float,
    components_distance: Float,
    precalc_fn: Callable[..., tuple[NDArray, ...]],
    potential_fn: Callable[..., NDArray],
    fprime: Callable[..., NDArray],
    potential: Float,
    mass_ratio: Float,
    synchronicity: Float,
) -> NDArray[Float]:
    """Solve surface radii for the supplied spherical directions.

    The radii are solved using the Newton solver and then converted to Cartesian
    coordinates.

    :param phi: Azimuth angles.
    :type phi: NDArray
    :param theta: Polar angles.
    :type theta: NDArray
    :param x0: Initial radial guess.
    :type x0: Float
    :param components_distance: Component separation in SMA units.
    :type components_distance: Float
    :param precalc_fn: Precalculation helper for the potential function.
    :type precalc_fn: collections.abc.Callable[..., tuple[NDArray, ...]]
    :param potential_fn: Potential function.
    :type potential_fn: collections.abc.Callable[..., NDArray]
    :param fprime: Derivative of the potential function.
    :type fprime: collections.abc.Callable[..., NDArray]
    :param potential: Surface potential.
    :type potential: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param synchronicity: Component synchronicity factor.
    :type synchronicity: Float
    :return: Surface points in Cartesian coordinates.
    :rtype: NDArray[Float]
    """
    phi_arr = np.asarray(phi)
    theta_arr = np.asarray(theta)
    max_iter = settings.MAX_SOLVER_ITERS
    precalc_vals = precalc_fn(
        *(synchronicity, mass_ratio, components_distance, phi_arr, theta_arr),
        return_as_tuple=True,
    )
    x0_arr = x0 * np.ones(phi_arr.shape)
    radius_kwargs = {
        "fprime": fprime,
        "maxiter": max_iter,
        "args": ((mass_ratio, *precalc_vals), potential),
        "rtol": 1e-10,
    }
    radius = opt.newton.newton(potential_fn, x0_arr, **radius_kwargs)
    if (radius < 0.0).any():
        msg = "Solver found at least one point in the opposite direction. Check your points."
        raise ValueError(msg)
    return utils.spherical_to_cartesian(np.column_stack((radius, phi_arr, theta_arr)))


def get_surface_points_cylindrical(
    phi: NDArray,
    z: NDArray,
    components_distance: Float,
    x0: Float,
    precalc_fn: Callable[..., tuple[NDArray, ...]],
    potential_fn: Callable[..., NDArray],
    fprime: Callable[..., NDArray],
    potential: Float,
    mass_ratio: Float,
    synchronicity: Float,
) -> NDArray[Float]:
    """Solve surface radii for the supplied cylindrical directions.

    :param phi: Azimuth angles.
    :type phi: NDArray
    :param z: Cylindrical axial coordinates.
    :type z: NDArray
    :param components_distance: Component separation in SMA units.
    :type components_distance: Float
    :param x0: Initial radial guess.
    :type x0: Float
    :param precalc_fn: Precalculation helper for the potential function.
    :type precalc_fn: collections.abc.Callable[..., tuple[NDArray, ...]]
    :param potential_fn: Potential function.
    :type potential_fn: collections.abc.Callable[..., NDArray]
    :param fprime: Derivative of the potential function.
    :type fprime: collections.abc.Callable[..., NDArray]
    :param potential: Surface potential.
    :type potential: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param synchronicity: Component synchronicity factor.
    :type synchronicity: Float
    :return: Surface points in Cartesian coordinates.
    :rtype: NDArray[Float]
    """
    phi_arr = np.asarray(phi)
    z_arr = np.asarray(z)
    max_iter = settings.MAX_SOLVER_ITERS
    precalc_vals = precalc_fn(
        *(synchronicity, mass_ratio, phi_arr, z_arr, components_distance),
        return_as_tuple=True,
    )
    x0_arr = np.full(phi_arr.shape, x0)
    radius_kwargs = {
        "fprime": fprime,
        "maxiter": max_iter,
        "rtol": 1e-10,
        "args": ((mass_ratio, *precalc_vals), potential),
    }
    radius = opt.newton.newton(potential_fn, x0_arr, **radius_kwargs)
    return utils.cylindrical_to_cartesian(
        np.column_stack((up.abs(radius), phi_arr, z_arr)),
    )


def mesh_detached(
    system: OrbitalPositionContainer,
    components_distance: Float,
    component: ComponentName,
    *,
    symmetry_output: bool = False,
) -> NDArray[Float] | tuple[NDArray[Float], Int, NDArray[np.int64]]:
    """Create the surface mesh for a detached or semi-detached component.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param component: Component selector.
    :type component: ComponentName
    :param symmetry_output: If ``True``, also return symmetry metadata.
    :type symmetry_output: bool
    :return: Surface points or ``(points, base_symmetry_points_number,
        inverse_symmetry_matrix)``.
    :rtype: NDArray[Float] | tuple[NDArray[Float], Int, NDArray[int]]
    """
    star = getattr(system, component)
    discretization_factor = star.discretization_factor
    synchronicity = star.synchronicity
    mass_ratio = system.mass_ratio
    potential = star.surface_potential

    potential_fn = getattr(model, f"potential_{component}_fn")
    precalc_fn = getattr(model, f"pre_calculate_for_potential_value_{component}")
    fprime = getattr(model, f"radial_{component}_potential_derivative")

    # Recompute position-specific radii from the current surface potential and
    # component distance.  This makes mesh_detached self-contained and ensures
    # correctness when the container is reused across orbital positions (e.g.
    # integrate_eccentric_curve_exactly), where set_on_position_params updates
    # the surface potential but not the cached radii.
    _rad_kwargs = {
        "synchronicity": synchronicity,
        "mass_ratio": mass_ratio,
        "components_distance": components_distance,
        "surface_potential": potential,
        "component": component,
    }
    star.polar_radius = bsradius.calculate_polar_radius(**_rad_kwargs)
    star.side_radius = bsradius.calculate_side_radius(**_rad_kwargs)
    star.backward_radius = bsradius.calculate_backward_radius(**_rad_kwargs)
    star.forward_radius = bsradius.calculate_forward_radius(**_rad_kwargs)

    phi, theta, separator = pre_calc_azimuths_for_detached_points(
        discretization_factor,
        star,
    )
    star.azimuth_args = phi, theta, separator
    args = (
        phi[1:],
        theta[1:],
        star.side_radius,
        components_distance,
        precalc_fn,
        potential_fn,
        fprime,
        potential,
        mass_ratio,
        synchronicity,
    )

    logger.debug(
        "calculating surface points of %s component in mesh_detached function using single process method",
        component,
    )
    points_q = get_surface_points(*args)
    points_q = np.insert(points_q, 0, [star.forward_radius, 0.0, 0.0], axis=0)
    points = stitch_quarters_in_detached(
        points_q,
        separator,
        component,
        components_distance,
    )

    if not symmetry_output:
        return points

    equator_length = separator[0] - 2
    meridian_length = separator[1] - separator[0]
    quarter_length = np.shape(points_q)[0] - separator[1]
    base_symmetry_points_number = 2 + equator_length + quarter_length + meridian_length

    points_length = np.shape(points)[0]
    inverse_symmetry_matrix = np.array(
        [
            up.arange(base_symmetry_points_number),
            up.concatenate(
                (
                    [0, 1],
                    up.arange(
                        base_symmetry_points_number + quarter_length,
                        base_symmetry_points_number + quarter_length + equator_length,
                    ),
                    up.arange(
                        base_symmetry_points_number,
                        base_symmetry_points_number + quarter_length,
                    ),
                    up.arange(
                        base_symmetry_points_number - meridian_length,
                        base_symmetry_points_number,
                    ),
                ),
            ),
            up.concatenate(
                (
                    [0, 1],
                    up.arange(
                        base_symmetry_points_number + quarter_length,
                        base_symmetry_points_number + quarter_length + equator_length,
                    ),
                    up.arange(
                        base_symmetry_points_number + quarter_length + equator_length,
                        base_symmetry_points_number + 2 * quarter_length + equator_length + meridian_length,
                    ),
                ),
            ),
            up.concatenate(
                (
                    up.arange(2 + equator_length),
                    up.arange(points_length - quarter_length, points_length),
                    up.arange(
                        base_symmetry_points_number + 2 * quarter_length + equator_length,
                        base_symmetry_points_number + 2 * quarter_length + equator_length + meridian_length,
                    ),
                ),
            ),
        ],
    )
    return points, base_symmetry_points_number, inverse_symmetry_matrix


def rebuild_mesh_detached(
    system: OrbitalPositionContainer,
    components_distance: Float,
    component: ComponentName,
) -> NDArray[Float]:
    """Rebuild a symmetric detached mesh from previously stored azimuths.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param component: Component selector.
    :type component: ComponentName
    :return: Rebuilt surface points.
    :rtype: NDArray[Float]
    """
    star = getattr(system, component)
    synchronicity = star.synchronicity
    mass_ratio = system.mass_ratio
    potential = star.surface_potential

    if is_empty(star.points):
        msg = "This function can be used only on container with already built mesh."
        raise RuntimeError(msg)
    if star.base_symmetry_points_number == 0:
        msg = "This function can be used only on symmetrical meshes."
        raise RuntimeError(msg)

    potential_fn = getattr(model, f"potential_{component}_fn")
    precalc_fn = getattr(model, f"pre_calculate_for_potential_value_{component}")
    fprime = getattr(model, f"radial_{component}_potential_derivative")
    phi, theta, separator = star.azimuth_args
    args = (
        phi,
        theta,
        star.side_radius,
        components_distance,
        precalc_fn,
        potential_fn,
        fprime,
        potential,
        mass_ratio,
        synchronicity,
    )

    logger.debug("re calculating surface points of %s component in rebuild_mesh_detached", component)
    points_q = np.round(get_surface_points(*args), 15)
    return stitch_quarters_in_detached(points_q, separator, component, components_distance)


def stitch_quarters_in_detached(
    points_q: NDArray[Float],
    separator: list[int],
    component: ComponentName,
    components_distance: Float,
) -> NDArray[Float]:
    """Stitch a detached stellar surface from a symmetric quarter.

    :param points_q: Points on the symmetric quarter including the equator and
        meridian.
    :type points_q: NDArray[Float]
    :param separator: Separator indices dividing equator, meridian, and inner
        points.
    :type separator: list[int]
    :param component: Component selector.
    :type component: ComponentName
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :return: Stitched full surface.
    :rtype: NDArray[Float]
    """
    equator = points_q[: separator[0], :]
    x_a, x_eq, x_b = equator[0, 0], equator[1:-1, 0], equator[-1, 0]
    y_a, y_eq, y_b = equator[0, 1], equator[1:-1, 1], equator[-1, 1]
    z_a, z_eq, z_b = equator[0, 2], equator[1:-1, 2], equator[-1, 2]

    meridian = points_q[separator[0] : separator[1], :]
    x_meridian, y_meridian, z_meridian = meridian[:, 0], meridian[:, 1], meridian[:, 2]

    quarter = points_q[separator[1] :, :]
    x_q, y_q, z_q = quarter[:, 0], quarter[:, 1], quarter[:, 2]

    x = np.array([x_a, x_b])
    y = np.array([y_a, y_b])
    z = np.array([z_a, z_b])
    x = up.concatenate((x, x_eq, x_q, x_meridian, x_q, x_eq, x_q, x_meridian, x_q))
    y = up.concatenate((y, y_eq, y_q, y_meridian, -y_q, -y_eq, -y_q, -y_meridian, y_q))
    z = up.concatenate((z, z_eq, z_q, z_meridian, z_q, z_eq, -z_q, -z_meridian, -z_q))

    x = -x + components_distance if component == "secondary" else x
    return np.column_stack((x, y, z))


def mesh_over_contact(  # noqa: PLR0915
    system: OrbitalPositionContainer,
    component: ComponentName = "primary",
    *,
    symmetry_output: bool = False,
) -> NDArray[Float] | tuple[NDArray[Float], Int, NDArray[np.int64]]:
    """Create the surface mesh for an over-contact component.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param component: Component selector.
    :type component: ComponentName
    :param symmetry_output: If ``True``, also return symmetry metadata.
    :type symmetry_output: bool
    :return: Surface points or ``(points, base_symmetry_points_number,
        inverse_symmetry_matrix)``.
    :rtype: NDArray[Float] | tuple[NDArray[Float], Int, NDArray[int]]
    """
    star = getattr(system, component)
    discretization_factor = star.discretization_factor
    synchronicity = star.synchronicity
    mass_ratio = system.mass_ratio
    potential = star.surface_potential
    r_polar = star.polar_radius
    components_distance = 1.0

    fn = getattr(model, f"potential_{component}_fn")
    fn_cylindrical = getattr(model, f"potential_{component}_cylindrical_fn")
    precalc = getattr(model, f"pre_calculate_for_potential_value_{component}")
    precalc_cylindrical = getattr(
        model,
        f"pre_calculate_for_potential_value_{component}_cylindrical",
    )
    fprime = getattr(model, f"radial_{component}_potential_derivative")
    cylindrical_fprime = getattr(
        model,
        f"radial_{component}_potential_derivative_cylindrical",
    )

    neck_position, neck_polynomial = calculate_neck_position(
        system,
        return_polynomial=True,
    )
    (phi_farside, theta_farside, separator_farside), (phi_neck, z_neck, separator_neck) = (
        pre_calc_azimuths_for_overcontact_points(
            discretization_factor,
            star,
            component,
            neck_position,
            neck_polynomial,
        )
    )

    args = (
        phi_farside,
        theta_farside,
        r_polar,
        components_distance,
        precalc,
        fn,
        fprime,
        potential,
        mass_ratio,
        synchronicity,
    )
    logger.debug(
        "calculating farside points of %s component in mesh_overcontact function using single process method",
        component,
    )
    points_farside = get_surface_points(*args)

    equator_farside = points_farside[: separator_farside[0], :]
    x_eq1, x_a = equator_farside[:-1, 0], equator_farside[-1, 0]
    y_eq1, y_a = equator_farside[:-1, 1], equator_farside[-1, 1]
    z_eq1, z_a = equator_farside[:-1, 2], equator_farside[-1, 2]

    meridian_farside1 = points_farside[separator_farside[0] : separator_farside[1], :]
    x_meridian1, y_meridian1, z_meridian1 = (
        meridian_farside1[:, 0],
        meridian_farside1[:, 1],
        meridian_farside1[:, 2],
    )

    meridian_farside2 = points_farside[separator_farside[1] : separator_farside[2], :]
    x_meridian2, y_meridian2, z_meridian2 = (
        meridian_farside2[:, 0],
        meridian_farside2[:, 1],
        meridian_farside2[:, 2],
    )

    quarter = points_farside[separator_farside[2] :, :]
    x_q1, y_q1, z_q1 = quarter[:, 0], quarter[:, 1], quarter[:, 2]

    args = (
        phi_neck,
        z_neck,
        components_distance,
        0.25 * star.polar_radius,
        precalc_cylindrical,
        fn_cylindrical,
        cylindrical_fprime,
        star.surface_potential,
        system.mass_ratio,
        synchronicity,
    )
    logger.debug(
        "calculating neck points of %s component in mesh_overcontact function using single process method",
        component,
    )
    points_neck = get_surface_points_cylindrical(*args)

    r_eqn = points_neck[: separator_neck[0], :]
    z_eqn, y_eqn, x_eqn = r_eqn[:, 0], r_eqn[:, 1], r_eqn[:, 2]

    r_meridian_n = points_neck[separator_neck[0] : separator_neck[1], :]
    z_meridian_n, y_meridian_n, x_meridian_n = (
        r_meridian_n[:, 0],
        r_meridian_n[:, 1],
        r_meridian_n[:, 2],
    )

    r_n = points_neck[separator_neck[1] :, :]
    z_n, y_n, x_n = r_n[:, 0], r_n[:, 1], r_n[:, 2]

    x_eq = up.concatenate((x_eqn, x_eq1), axis=0)
    y_eq = up.concatenate((y_eqn, y_eq1), axis=0)
    z_eq = up.concatenate((z_eqn, z_eq1), axis=0)
    x_q = up.concatenate((x_n, x_meridian2, x_q1), axis=0)
    y_q = up.concatenate((y_n, y_meridian2, y_q1), axis=0)
    z_q = up.concatenate((z_n, z_meridian2, z_q1), axis=0)
    x_meridian = up.concatenate((x_meridian_n, x_meridian1), axis=0)
    y_meridian = up.concatenate((y_meridian_n, y_meridian1), axis=0)
    z_meridian = up.concatenate((z_meridian_n, z_meridian1), axis=0)

    x = up.concatenate((np.array([x_a]), x_eq, x_q, x_meridian, x_q, x_eq, x_q, x_meridian, x_q))
    y = up.concatenate((np.array([y_a]), y_eq, y_q, y_meridian, -y_q, -y_eq, -y_q, -y_meridian, y_q))
    z = up.concatenate((np.array([z_a]), z_eq, z_q, z_meridian, z_q, z_eq, -z_q, -z_meridian, -z_q))

    x = -x + components_distance if component == "secondary" else x
    points = np.column_stack((x, y, z))

    if not symmetry_output:
        return points

    equator_length = np.shape(x_eq)[0]
    meridian_length = np.shape(x_meridian)[0]
    quarter_length = np.shape(x_q)[0]
    base_symmetry_points_number = 1 + equator_length + quarter_length + meridian_length

    points_length = np.shape(x)[0]
    inverse_symmetry_matrix = np.array(
        [
            up.arange(base_symmetry_points_number),
            up.concatenate(
                (
                    [0],
                    up.arange(
                        base_symmetry_points_number + quarter_length,
                        base_symmetry_points_number + quarter_length + equator_length,
                    ),
                    up.arange(
                        base_symmetry_points_number,
                        base_symmetry_points_number + quarter_length,
                    ),
                    up.arange(
                        base_symmetry_points_number - meridian_length,
                        base_symmetry_points_number,
                    ),
                ),
            ),
            up.concatenate(
                (
                    [0],
                    up.arange(
                        base_symmetry_points_number + quarter_length,
                        base_symmetry_points_number + quarter_length + equator_length,
                    ),
                    up.arange(
                        base_symmetry_points_number + quarter_length + equator_length,
                        base_symmetry_points_number + 2 * quarter_length + equator_length + meridian_length,
                    ),
                ),
            ),
            up.concatenate(
                (
                    up.arange(1 + equator_length),
                    up.arange(points_length - quarter_length, points_length),
                    up.arange(
                        base_symmetry_points_number + 2 * quarter_length + equator_length,
                        base_symmetry_points_number + 2 * quarter_length + equator_length + meridian_length,
                    ),
                ),
            ),
        ],
    )
    return points, base_symmetry_points_number, inverse_symmetry_matrix


def mesh_spots(  # noqa: C901, PLR0912, PLR0915
    system: OrbitalPositionContainer,
    components_distance: Float,
    component: ComponentSelection = "all",
) -> None:
    """Compute spot meshes and assign them to spot containers.

    If any spot point cannot be obtained, the entire spot is omitted.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param component: Component selector.
    :type component: ComponentSelection
    :return: ``None``.
    :rtype: None
    """

    def solver_condition(x: Float | NDArray[Float], *_args: Float) -> bool:
        if isinstance(x, np.ndarray):
            x = cast("Float", x[0])
        point = utils.spherical_to_cartesian(np.asarray([x, _args[1], _args[2]], dtype=FLOAT))
        point[0] = point[0] if component_name == "primary" else components_distance - point[0]
        # Use a single combined `if` rather than nested `if` statements.
        is_overcontanct = system.morphology == "over-contact"
        is_primary_positive = component_name == "primary" and point[0] >= neck_position
        is_secondary_negative = component_name == "secondary" and point[0] <= neck_position
        return not (is_overcontanct and (is_primary_positive or is_secondary_negative))

    components = bsutils.component_to_list(component)
    fns = {
        "primary": (
            model.potential_primary_fn,
            model.pre_calculate_for_potential_value_primary,
            model.radial_primary_potential_derivative,
        ),
        "secondary": (
            model.potential_secondary_fn,
            model.pre_calculate_for_potential_value_secondary,
            model.radial_secondary_potential_derivative,
        ),
    }
    selected_fns = {name: fns[name] for name in components}
    neck_position = calculate_neck_position(system) if system.morphology == "over-contact" else 1e10

    for component_name, functions in selected_fns.items():
        logger.debug("evaluating spots for %s component", component_name)
        potential_fn, precalc_fn, fprime = functions
        component_instance = getattr(system, component_name)

        if not component_instance.spots:
            logger.debug("no spots to evaluate for %s component - continue", component_name)
            continue

        for spot_index, spot_instance in list(component_instance.spots.items()):
            lon, lat = spot_instance.longitude, spot_instance.latitude
            alpha = min(spot_instance.angular_radius, spot_instance.discretization_factor)
            spot_radius = spot_instance.angular_radius
            synchronicity = component_instance.synchronicity
            mass_ratio = system.mass_ratio
            potential = component_instance.surface_potential

            radial_vector = np.array([1.0, lon, lat])
            center_vector = utils.spherical_to_cartesian(np.asarray([1.0, lon, lat], dtype=FLOAT))
            args1 = (
                synchronicity,
                mass_ratio,
                components_distance,
                radial_vector[1],
                radial_vector[2],
            )
            args2 = ((system.mass_ratio, *precalc_fn(*args1)), potential)
            kwargs = {"original_kwargs": args1}
            solution, use = fsolver(potential_fn, solver_condition, *args2, **kwargs)

            if not use:
                if not settings.SUPPRESS_WARNINGS:
                    logger.warning(
                        "center of spot %s doesn't satisfy reasonable conditions and entire spot will be omitted",
                        spot_instance.kwargs_serializer(),
                    )
                component_instance.remove_spot(spot_index=spot_index)
                continue

            spot_center_r = solution
            spot_center = utils.spherical_to_cartesian(np.asarray([spot_center_r, lon, lat], dtype=FLOAT))

            args1 = (synchronicity, mass_ratio, components_distance, lon, lat + alpha)
            args2 = ((system.mass_ratio, *precalc_fn(*args1)), potential)
            kwargs = {"original_kwargs": args1}
            solution, use = fsolver(potential_fn, solver_condition, *args2, **kwargs)

            if not use:
                if not settings.SUPPRESS_WARNINGS:
                    logger.warning(
                        "first inner ring of spot %s doesn't satisfy reasonable conditions "
                        "and entire spot will be omitted",
                        spot_instance.kwargs_serializer(),
                    )
                component_instance.remove_spot(spot_index=spot_index)
                continue

            x0 = up.sqrt(
                spot_center_r**2 + solution**2 - (2.0 * spot_center_r * solution * up.cos(alpha)),
            )
            num_radial = int(np.round(spot_radius / alpha)) + 1
            logger.debug(
                "number of rings in spot %s is %s",
                spot_instance.kwargs_serializer(),
                num_radial,
            )
            thetas = np.linspace(lat, lat + spot_radius, num=num_radial, endpoint=True)
            num_azimuthal = [
                1 if index == 0 else int(index * 2.0 * const.PI * x0 // x0) for index in range(len(thetas))
            ]
            deltas = [np.linspace(0.0, const.FULL_ARC, num=num, endpoint=False) for num in num_azimuthal]

            spot_phi: list[Float] = []
            spot_theta: list[Float] = []
            for theta_index, theta_value in enumerate(thetas):
                default_spherical_vector = np.asarray([1.0, lon % const.FULL_ARC, theta_value], dtype=FLOAT)
                for delta in deltas[theta_index]:
                    delta_vector = utils.arbitrary_rotation(
                        theta=delta,
                        omega=center_vector,
                        vector=utils.spherical_to_cartesian(default_spherical_vector),
                        degrees=False,
                        omega_normalized=True,
                    )
                    spherical_delta_vector = utils.cartesian_to_spherical(delta_vector)
                    spot_phi.append(spherical_delta_vector[1])
                    spot_theta.append(spherical_delta_vector[2])

            spot_phi_arr = np.array(spot_phi)
            spot_theta_arr = np.array(spot_theta)
            args = (
                spot_phi_arr,
                spot_theta_arr,
                spot_center_r,
                components_distance,
                precalc_fn,
                potential_fn,
                fprime,
                potential,
                mass_ratio,
                synchronicity,
            )
            try:
                spot_points = get_surface_points(*args)
            except (MaxIterationError, ValueError) as exc:
                msg = (
                    "Solver could not find at least some surface points of spot "
                    f"{spot_instance.kwargs_serializer()}. Probable reason is that your "
                    "spot is intersecting neck which is currently not supported."
                )
                raise SpotError(msg) from exc

            if system.morphology == "over-contact":
                if spot_points.ndim == SPOT_POINTS_NDIM:
                    validity_test = (
                        (spot_points[:, 0] <= neck_position).all()
                        if component_name == "primary"
                        else (spot_points[:, 0] <= (1 - neck_position)).all()
                    )
                else:
                    validity_test = False

                if not validity_test:
                    msg = (
                        f"Your spot {spot_instance.kwargs_serializer()} is intersecting neck "
                        "which is currently not supported."
                    )
                    raise SpotError(msg)

            boundary_points = spot_points[-len(deltas[-1]) :]
            if component_name == "primary":
                spot_instance.points = np.array(spot_points)
                spot_instance.boundary = np.array(boundary_points)
                spot_instance.center = np.array(spot_center)
            else:
                # Vectorised transform: flip x and y, offset x by components_distance.
                # Equivalent to: [cd - x, -y, z] applied row-wise.
                _flip = np.array([-1.0, -1.0, 1.0])
                _offset = np.array([components_distance, 0.0, 0.0])
                spot_instance.points = spot_points * _flip + _offset
                spot_instance.boundary = boundary_points * _flip + _offset
                spot_instance.center = spot_center * _flip + _offset


def calculate_neck_position(
    system: BinarySystem | OrbitalPositionContainer,
    *,
    return_polynomial: bool = False,
) -> Float | tuple[Float, NDArray[Float]]:
    """Calculate the x coordinate of the neck of an over-contact system.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param return_polynomial: If ``True``, also return the fitted polynomial
        coefficients.
    :type return_polynomial: bool
    :return: Neck position or ``(neck_position, polynomial_fit)``.
    :rtype: Float | tuple[Float, NDArray[Float]]
    """
    n_points = int(100 * np.radians(5) / system.primary.discretization_factor)
    degree = 15
    components_distance = 1.0

    star = system.primary
    precalc_cylindrical = model.pre_calculate_for_potential_value_primary_cylindrical
    fn_cylindrical = model.potential_primary_cylindrical_fn
    cylindrical_fprime = model.radial_primary_potential_derivative_cylindrical

    phi = np.zeros(n_points)
    z = np.linspace(0, 1, num=n_points)
    args = (
        phi,
        z,
        components_distance,
        0.5 * star.polar_radius,
        precalc_cylindrical,
        fn_cylindrical,
        cylindrical_fprime,
        star.surface_potential,
        system.mass_ratio,
        1.0,
    )
    points_neck = get_surface_points_cylindrical(*args)
    x = np.abs(points_neck[:, 2])
    r_c = np.abs(points_neck[:, 0])

    polynomial_fit = np.polyfit(x, r_c, deg=degree)
    polynomial_fit_differentiation = np.polyder(polynomial_fit)
    roots = np.roots(polynomial_fit_differentiation)
    roots = np.real(roots[np.imag(roots) == 0])
    dist_to_cntr = np.abs(roots - 0.5)
    neck_position = roots[np.argmin(dist_to_cntr)]

    if return_polynomial:
        return neck_position, polynomial_fit
    return neck_position


def add_spots_to_mesh(
    system: OrbitalPositionContainer,
    components_distance: Float,
    component: ComponentSelection | None = "all",
) -> None:
    """Incorporate spot surface points into the clean stellar mesh.

    Overlapping stellar points and overlapping spot points are removed as part
    of the incorporation process.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param component: Component selector.
    :type component: ComponentSelection | None
    :return: ``None``.
    :rtype: None
    """
    components = bsutils.component_to_list(component)
    if is_empty(components):
        return

    component_com = {"primary": 0.0, "secondary": components_distance}
    for component_name in components:
        star = getattr(system, component_name)
        mesh_spots(system, components_distance=components_distance, component=component_name)
        incorporate_spots_mesh(star, component_com=component_com[component_name])


def correct_mesh(
    system: OrbitalPositionContainer,
    components_distance: Float | None = None,
    component: ComponentSelection = "all",
) -> OrbitalPositionContainer:
    """Correct surface underestimation caused by discretization.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float | None
    :param component: Component selector.
    :type component: ComponentSelection
    :return: Updated orbital position container.
    :rtype: OrbitalPositionContainer
    """
    correction_factors = _load_correction_factors()
    components = bsutils.component_to_list(component)
    com = {"primary": 0, "secondary": components_distance}

    for component_name in components:
        star = getattr(system, component_name)
        correct_component_mesh(
            star,
            com=com[component_name],
            correction_factors=correction_factors[system.morphology],
        )

    return system
