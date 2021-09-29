import os
import numpy as np
import elisa.umpy as up

from .. import model
from .. radius import calculate_radius
from ... logger import getLogger
from ... base.spot import incorporate_spots_mesh
from ... base.surface.mesh import correct_component_mesh
from ... base.error import MaxIterationError
from ... import settings
from ... import (
    opt,
    const,
    utils
)


logger = getLogger("single_system.surface.mesh")
CORRECTION_FACTORS = np.load(settings.PATH_TO_SINGLE_CORRECTIONS, allow_pickle=False)


def build_mesh(system):
    """
    Build surface point mesh including spots.

    :param system: elisa.single_system.contaier.PositionContainer; instance
    :return: elisa.single_system.container.SinglePositionContainer; instance
    """
    a, c, d = mesh(system_container=system, symmetry_output=True)

    system.star.points = a
    system.star.base_symmetry_points_number = c
    system.star.inverse_point_symmetry_matrix = d

    add_spots_to_mesh(system)
    return system


def mesh(system_container, symmetry_output=False):
    """
    Function for creating surface mesh of single star system.

    :return: numpy.array;

    ::

            numpy.array([[x1 y1 z1],
                          [x2 y2 z2],
                            ...
                          [xN yN zN]]) - array of surface points if symmetry_output = False, else:
            numpy.array([[x1 y1 z1],
                          [x2 y2 z2],
                            ...
                          [xN yN zN]]) - array of surface points,
            numpy.float - number of points included in symmetrical one eighth of surface,
            numpy.array([octants[indexes_of_remapped_points_in_octants]) - matrix of eight sub matrices that mapps
                                                                           base symmetry octant to all others
                                                                           octants

    """
    star_container = getattr(system_container, 'star')
    discretization_factor = star_container.discretization_factor
    if discretization_factor > const.HALF_PI:
        raise ValueError("Invalid value of alpha parameter. Use value less than 90.")

    potential_fn = model.potential_fn
    precalc_fn = model.pre_calculate_for_potential_value
    potential_derivative_fn = model.radial_potential_derivative

    characteristic_distance = discretization_factor * star_container.polar_radius

    # calculating equatorial part
    equator_points = calculate_equator_points(characteristic_distance, star_container.equatorial_radius)
    x_eq, y_eq, z_eq = equator_points[:, 0], equator_points[:, 1], equator_points[:, 2]

    # axial symmetry, therefore calculating latitudes
    thetas = pre_calc_latitudes(discretization_factor, star_container.polar_radius, star_container.equatorial_radius)
    thetas_meridian = pre_calc_latitudes(const.SEAM_CONST*discretization_factor, star_container.polar_radius,
                                         star_container.equatorial_radius)

    x0 = 0.5 * (star_container.equatorial_radius + star_container.polar_radius)
    args = thetas, x0, precalc_fn, potential_fn, potential_derivative_fn, star_container.surface_potential, \
        star_container.mass, system_container.angular_velocity
    args_meridian = thetas_meridian, x0, precalc_fn, potential_fn, potential_derivative_fn, \
                    star_container.surface_potential, star_container.mass, system_container.angular_velocity

    radius = get_surface_points_radii(*args)
    radius_meridian = get_surface_points_radii(*args_meridian)

    # converting this eighth of surface to cartesian coordinates
    quarter_points = calculate_points_on_quarter_surface(radius, thetas, characteristic_distance)
    x_q, y_q, z_q = quarter_points[:, 0], quarter_points[:, 1], quarter_points[:, 2]
    meridian_points = calculate_points_on_meridian(radius_meridian, thetas_meridian)
    x_mer, y_mer, z_mer = meridian_points[:, 0], meridian_points[:, 1], meridian_points[:, 2],

    x = np.concatenate((np.array([0]), x_mer, x_eq, x_q, -y_mer, -y_eq, -y_q, -x_mer, -x_eq, -x_q, y_mer, y_eq,
                        y_q, np.array([0]), x_mer, x_q, -y_mer, -y_q, -x_mer, -x_q, y_mer, y_q))
    y = np.concatenate((np.array([0]), y_mer, y_eq, y_q, x_mer, x_eq, x_q, -y_mer, -y_eq, -y_q, -x_mer, -x_eq,
                        -x_q, np.array([0]), y_mer, y_q, x_mer, x_q, -y_mer, -y_q, -x_mer, -x_q))
    z = np.concatenate((np.array([star_container.polar_radius]), z_mer, z_eq, z_q, z_mer, z_eq, z_q, z_mer, z_eq,
                        z_q, z_mer, z_eq, z_q, np.array([-star_container.polar_radius]), -z_mer, -z_q, -z_mer, -z_q,
                        -z_mer, -z_q, -z_mer, -z_q))

    if symmetry_output:
        quarter_equator_length = len(x_eq)
        meridian_length = len(x_mer)
        quarter_length = len(x_q)
        base_symmetry_points_number = 1 + meridian_length + quarter_equator_length + quarter_length + meridian_length

        south_pole_index = 4 * (base_symmetry_points_number - meridian_length) - 3
        reduced_bspn = base_symmetry_points_number - meridian_length  # auxiliary variable1
        reduced_bspn2 = base_symmetry_points_number - quarter_equator_length
        inverse_symmetry_matrix = \
            np.array([
                np.arange(base_symmetry_points_number + 1),  # 1st quadrant (north hem)
                # 2nd quadrant (north hem)
                np.concatenate(([0], np.arange(reduced_bspn, 2 * base_symmetry_points_number - meridian_length))),
                # 3rd quadrant (north hem)
                np.concatenate(([0], np.arange(2 * reduced_bspn - 1, 3 * reduced_bspn + meridian_length - 1))),
                # 4th quadrant (north hem)
                np.concatenate(([0], np.arange(3 * reduced_bspn - 2, 4 * reduced_bspn - 3),
                                np.arange(1, meridian_length + 2))),
                # 1st quadrant (south hemisphere)
                np.concatenate((np.arange(south_pole_index, meridian_length + 1 + south_pole_index),
                                np.arange(1 + meridian_length, 1 + meridian_length + quarter_equator_length),
                                np.arange(meridian_length + 1 + south_pole_index,
                                          base_symmetry_points_number - quarter_equator_length + south_pole_index),
                                [base_symmetry_points_number])),
                # 2nd quadrant (south hem)
                np.concatenate(([south_pole_index],
                                np.arange(reduced_bspn2 - meridian_length + south_pole_index,
                                          reduced_bspn2 + south_pole_index),
                                np.arange(base_symmetry_points_number,
                                          base_symmetry_points_number + quarter_equator_length),
                                np.arange(reduced_bspn2 + south_pole_index,
                                          2 * reduced_bspn2 - meridian_length - 1 +
                                          south_pole_index),
                                [2 * base_symmetry_points_number - meridian_length - 1])),
                # 3rd quadrant (south hem)
                np.concatenate(([south_pole_index],
                                np.arange(2 * reduced_bspn2 - 2 * meridian_length - 1 + south_pole_index,
                                          2 * reduced_bspn2 - meridian_length - 1 + south_pole_index),
                                np.arange(2 * base_symmetry_points_number - meridian_length - 1,
                                          2 * base_symmetry_points_number - meridian_length + quarter_equator_length
                                          - 1),
                                np.arange(2 * reduced_bspn2 - meridian_length - 1 + south_pole_index,
                                          3 * reduced_bspn2 - 2 * meridian_length - 2 + south_pole_index),
                                [3 * reduced_bspn + meridian_length - 2])),
                # 4th quadrant (south hem)
                np.concatenate(([south_pole_index],
                                np.arange(3 * reduced_bspn2 - 3 * meridian_length - 2 + south_pole_index,
                                          3 * reduced_bspn2 - 2 * meridian_length - 2 + south_pole_index),
                                np.arange(3 * reduced_bspn + meridian_length - 2,
                                          3 * reduced_bspn + meridian_length - 2 +
                                          quarter_equator_length),
                                np.arange(3 * reduced_bspn2 - 2 * meridian_length - 2 + south_pole_index, len(x)),
                                np.arange(1 + south_pole_index, meridian_length + south_pole_index + 1),
                                [1 + meridian_length]
                                ))
            ])

        return np.column_stack((x, y, z)), base_symmetry_points_number + 1, inverse_symmetry_matrix
    else:
        return np.column_stack((x, y, z))


def pre_calc_latitudes(alpha, polar_radius, equatorial_radius):
    """
    Function pre-calculates latitudes of stellar surface with exception of pole and equator.

    :param equatorial_radius: float;
    :param polar_radius: float;
    :param alpha: float; angular distance of points
    :return: numpy.array; latitudes for mesh
    """
    # alpha_corr = const.POINT_ROW_SEPARATION_FACTOR * alpha
    alpha_corr = alpha
    num = int(const.HALF_PI // alpha_corr)
    thetas = np.linspace(0, const.HALF_PI, num=num, endpoint=True)[1:-1]
    # solving non uniform sampling along theta coordinates for squashed stars
    auto_test = settings.MESH_GENERATOR == 'auto' and \
                (equatorial_radius - polar_radius) / polar_radius > settings.DEFORMATION_TOL
    if auto_test or settings.MESH_GENERATOR == 'improved_trapezoidal':
        thetas += up.arctan((equatorial_radius - polar_radius) * up.tan(thetas) /
                            (polar_radius + equatorial_radius * up.tan(thetas)**2))
    return thetas


def get_surface_points_radii(*args):
    """
    Function solves radius for given latitudes that are passed in `args`.
    Function to solve is specified as last parameter in `args` Tuple.

    :param args: Tuple;

    ::

        Tuple[
                theta: numpy.array,
                x0: float,
                precalc: callable,
                fn: callable,
                derivative_fn: callable
              ]

    :return: numpy.array;
    """
    theta, x0, precalc_fn, potential_fn, potential_derivative_fn, surface_potential, mass, angular_velocity = args
    precalc_vals = precalc_fn(*(mass, angular_velocity, theta,), return_as_tuple=True)
    x0 = x0 * np.ones(theta.shape)
    radius = opt.newton.newton(potential_fn, x0, fprime=potential_derivative_fn,
                               maxiter=settings.MAX_SOLVER_ITERS, args=(precalc_vals, surface_potential), rtol=1e-10)
    return radius


def calculate_points_on_quarter_surface(radius, thetas, characteristic_distance):
    """
    Function returns cartesian coordinates for points on the quarter of the surface.

    :param radius: numpy.array;
    :param thetas: numpy.array;
    :param characteristic_distance: float; mean distance between points
    :return: numpy.array; N * 3 array of x, y, z coordinates
    """
    r_q, phi_q, theta_q = [], [], []
    for ii, theta in enumerate(thetas):
        num = int(const.HALF_PI * radius[ii] * np.sin(theta) / characteristic_distance)
        alpha = (const.HALF_PI / num)
        num -= 1 if num > 0 else num
        r_q.append(radius[ii] * np.ones(num))
        theta_q.append(theta * np.ones(num))
        phi_q.append(np.linspace(alpha, const.HALF_PI-alpha, num=num, endpoint=True))
    r_q = np.concatenate(r_q)
    theta_q = np.concatenate(theta_q)
    phi_q = np.concatenate(phi_q)
    return utils.spherical_to_cartesian(np.column_stack((r_q, phi_q, theta_q)))


def calculate_points_on_meridian(radius, thetas):
    """
    Function returns cartesian coordinates for points on the surface meridian.

    :param radius: numpy.array;
    :param thetas: numpy.array;
    :return: numpy.array; N * 3 array of x, y, z coordinates
    """
    phi = 0.0 * np.ones(radius.shape)
    return utils.spherical_to_cartesian(np.column_stack((radius, phi, thetas)))


def calculate_equator_points(characteristic_distance, equatorial_radius):
    """
    Function returns cartesian coordinates for points on the equator.

    :param characteristic_distance: float; number of points on the quarter of the equator
    :param equatorial_radius: float;
    :return: numpy.array; N * 3 array of x, y, z coordinates
    """
    num = int(const.HALF_PI * equatorial_radius / (const.SEAM_CONST * characteristic_distance))
    radii = equatorial_radius * np.ones(num)
    thetas = const.HALF_PI * np.ones(num)
    phis = np.linspace(0, const.HALF_PI, num=num, endpoint=False)
    return utils.spherical_to_cartesian(np.column_stack((radii, phis, thetas)))


def mesh_spots(system_container):
    """
    Compute points of each spots and assigns values to spot container instance.
    """

    logger.info("evaluating spots")
    star_container = system_container.star
    if not star_container.has_spots():
        logger.info("no spots to evaluate")
        return

    potential_fn = model.potential_fn
    precalc_fn = model.pre_calculate_for_potential_value
    potential_derivative_fn = model.radial_potential_derivative

    # iterate over spots
    for spot_index, spot_instance in list(star_container.spots.items()):
        # lon -> phi, lat -> theta
        lon, lat = spot_instance.longitude, spot_instance.latitude

        alpha, spot_radius = spot_instance.discretization_factor, spot_instance.angular_radius

        # initial containers for current spot
        boundary_points, spot_points = [], []

        # initial radial vector
        radial_vector = np.array([1.0, lon, lat])  # unit radial vector to the center of current spot
        center_vector = utils.spherical_to_cartesian([1.0, lon, lat])

        args = (radial_vector[2],)

        solution = calculate_radius(star_container.mass, system_container.angular_velocity,
                                    star_container.surface_potential, *args)

        if solution > star_container.equatorial_radius or solution < star_container.polar_radius:
            # in case of spots, each point should be usefull, otherwise remove spot from
            # component spot list and skip current spot computation
            logger.info(f"center of spot {spot_instance.kwargs_serializer()} doesn't satisfy "
                        f"reasonable conditions and entire spot will be omitted")

            star_container.remove_spot(spot_index=spot_index)
            continue

        spot_center_r = solution
        spot_center = utils.spherical_to_cartesian([spot_center_r, lon, lat])

        # compute euclidean distance of two points on spot (x0)
        # we have to obtain distance between center and 1st point in 1st ring of spot
        args = (lat + alpha,)
        solution = calculate_radius(star_container.mass, system_container.angular_velocity,
                                    star_container.surface_potential, *args)

        if solution > star_container.equatorial_radius or solution < star_container.polar_radius:
            # in case of spots, each point should be usefull, otherwise remove spot from
            # component spot list and skip current spot computation
            logger.info(f"first ring of spot {spot_instance.kwargs_serializer()} doesn't satisfy "
                        f"reasonable conditions and entire spot will be omitted")
            star_container.remove_spot(spot_index=spot_index)
            continue

        x0 = np.sqrt(spot_center_r ** 2 + solution ** 2 - (2.0 * spot_center_r * solution * np.cos(alpha)))

        # number of points in latitudal direction
        num_radial = int(np.round(spot_radius / alpha)) + 1
        thetas = np.linspace(lat, lat + spot_radius, num=num_radial, endpoint=True)

        num_azimuthal = [1 if i == 0 else int(i * 2.0 * np.pi * x0 // x0) for i in range(0, len(thetas))]
        deltas = [np.linspace(0., const.FULL_ARC, num=num, endpoint=False) for num in num_azimuthal]

        spot_phi, spot_theta = [], []
        for theta_index, theta in enumerate(thetas):
            # first point of n-th ring of spot (counting start from center)
            default_spherical_vector = [1.0, lon % const.FULL_ARC, theta]

            for delta_index, delta in enumerate(deltas[theta_index]):
                # rotating default spherical vector around spot center vector and thus generating concentric
                # circle of points around centre of spot
                delta_vector = utils.arbitrary_rotation(theta=delta, omega=center_vector,
                                                        vector=utils.spherical_to_cartesian(default_spherical_vector),
                                                        degrees=False,
                                                        omega_normalized=True)

                spherical_delta_vector = utils.cartesian_to_spherical(delta_vector)

                spot_phi.append(spherical_delta_vector[1])
                spot_theta.append(spherical_delta_vector[2])

        spot_phi, spot_theta = np.array(spot_phi), np.array(spot_theta)
        args = spot_theta, spot_center_r, precalc_fn, potential_fn, potential_derivative_fn, \
            star_container.surface_potential, star_container.mass, system_container.angular_velocity
        try:
            spot_points_radii = get_surface_points_radii(*args)
        except MaxIterationError:
            if not settings.SUPPRESS_WARNINGS:
                logger.warning(f"at least 1 point of spot {spot_instance.kwargs_serializer()} "
                               f"doesn't satisfy reasonable conditions and entire spot will be omitted")
            star_container.remove_spot(spot_index=spot_index)
            continue

        spot_points = utils.spherical_to_cartesian(np.column_stack((spot_points_radii, spot_phi, spot_theta)))

        spot_instance.points = np.array(spot_points)
        spot_instance.boundary = np.array(boundary_points)
        spot_instance.boundary_center = spot_points[0]
        spot_instance.center = np.array(spot_center)


def add_spots_to_mesh(system):
    """
    Function implements surface points into clean mesh and removes stellar
    points and other spot points under the given spot if such overlapped spots exists.
    """
    mesh_spots(system)
    incorporate_spots_mesh(system.star, component_com=0.0)


def correct_mesh(system):
    """
    Correcting the underestimation of the surface due to the discretization.

    :param system: elisa.single_system.container.SystemContainer;
    :return: elisa.single_system.container.SystemContainer;
    """
    star = getattr(system, 'star')
    correct_component_mesh(star, com=0.0, correction_factors=CORRECTION_FACTORS)

    return system
