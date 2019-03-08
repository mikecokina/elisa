from queue import Empty

import numpy as np
from copy import copy

import scipy

from elisa.engine import utils, const


def visibility_test(centres, xlim, component):
    """
    tests if given faces are visible from the other star

    :param component:
    :param centres:
    :param xlim: visibility threshold in x axis for given component
    :return:
    """
    return centres[:, 0] >= xlim if component == 'primary' else centres[:, 0] <= xlim


def get_symmetrical_distance_matrix(shape, shape_reduced, centres, vis_test, vis_test_symmetry):
    """
    function uses symmetries of the stellar component in order to reduce time in calculation distance matrix
    :param shape: desired shape of join vector matrix
    :param shape_reduced: shape of the surface symmetries, (faces above those indices are symmetrical to the ones
    below)
    :param centres:
    :param vis_test:
    :param vis_test_symmetry:
    :return: distance - distance matrix
             join vector - matrix of unit vectors pointing between each two faces on opposite stars
    """
    distance = np.empty(shape=shape[:2], dtype=np.float)
    join_vector = np.empty(shape=shape, dtype=np.float)

    # in case of symmetries, you need to calculate only minority part of distance matrix connected with base
    # symmetry part of the both surfaces
    distance[:shape_reduced[0], :], join_vector[:shape_reduced[0], :, :] = \
        utils.calculate_distance_matrix(points1=centres['primary'][vis_test_symmetry['primary']],
                                        points2=centres['secondary'][vis_test['secondary']],
                                        return_join_vector_matrix=True)

    aux = centres['primary'][vis_test['primary']]
    distance[shape_reduced[0]:, :shape_reduced[1]], join_vector[shape_reduced[0]:, :shape_reduced[1], :] = \
        utils.calculate_distance_matrix(points1=aux[shape_reduced[0]:],
                                        points2=centres['secondary'][vis_test_symmetry['secondary']],
                                        return_join_vector_matrix=True)

    return distance, join_vector


def init_surface_variables(component_instance):
    """
    function copies basic parameters of the stellar surface (points, faces, normals, temperatures and areas) of
    given star instance into new arrays during calculation of reflection effect

    :param component_instance:
    :return:
    """
    points, faces = component_instance.return_whole_surface()
    centres = copy(component_instance.face_centres)
    normals = copy(component_instance.normals)
    temperatures = copy(component_instance.temperatures)
    areas = copy(component_instance.areas)
    return points, faces, centres, normals, temperatures, areas


def include_spot_to_surface_variables(centres, spot_centres, normals, spot_normals,
                                      temperatures, spot_temperatures, areas, spot_areas, vis_test, vis_test_spot):
        """
        function includes surface parameters of spot faces into global arrays containing parameters from whole surface
        used in reflection effect

        :param centres:
        :param spot_centres: spot centres to append to `centres`
        :param normals:
        :param spot_normals: spot normals to append to `normals`
        :param temperatures:
        :param spot_temperatures: spot temperatures to append to `temperatures`
        :param areas:
        :param spot_areas: spot areas to append to `areas`
        :param vis_test:
        :param vis_test_spot: spot visibility test to append to `vis_test`
        :return:
        """
        centres = np.append(centres, spot_centres, axis=0)
        normals = np.append(normals, spot_normals, axis=0)
        temperatures = np.append(temperatures, spot_temperatures, axis=0)
        areas = np.append(areas, spot_areas, axis=0)
        vis_test = np.append(vis_test, vis_test_spot, axis=0)

        return centres, normals, temperatures, areas, vis_test


def get_symmetrical_gammma(shape, shape_reduced, normals, join_vector, vis_test, vis_test_symmetry):
    """
    function uses surface symmetries to calculate cosine of angles between join vector and surface normals

    :param shape: desired shape of gamma
    :param shape_reduced: shape of the surface symmetries, (faces above those indices are symmetrical to the ones
    below)
    :param normals:
    :param join_vector:
    :param vis_test:
    :param vis_test_symmetry:
    :return: gamma - cos(angle(normal, join_vector))
    """
    gamma = {'primary': np.empty(shape=shape, dtype=np.float),
             'secondary': np.empty(shape=shape, dtype=np.float)}

    # calculating only necessary components of the matrix (near left and upper edge) because of surface symmetry
    gamma['primary'][:, :shape_reduced[1]] = \
        np.sum(np.multiply(normals['primary'][vis_test['primary']][:, None, :],
                           join_vector[:, :shape_reduced[1], :]), axis=2)
    gamma['primary'][:shape_reduced[0], shape_reduced[1]:] = \
        np.sum(np.multiply(normals['primary'][vis_test_symmetry['primary']][:, None, :],
                           join_vector[:shape_reduced[0], shape_reduced[1]:, :]), axis=2)

    gamma['secondary'][:shape_reduced[0], :] = \
        - np.sum(np.multiply(normals['secondary'][vis_test['secondary']][None, :, :],
                             join_vector[:shape_reduced[0], :, :]), axis=2)
    gamma['secondary'][shape_reduced[0]:, :shape_reduced[1]] = \
        - np.sum(np.multiply(normals['secondary'][vis_test_symmetry['secondary']][None, :, :],
                             join_vector[shape_reduced[0]:, :shape_reduced[1], :]), axis=2)
    return gamma


def check_symmetric_gamma_for_negative_num(gamma, shape_reduced):
        """
        if cos < 0 it will be redefined as 0
        :param gamma:
        :param shape_reduced:
        :return:
        """
        gamma['primary'][:, :shape_reduced[1]][gamma['primary'][:, :shape_reduced[1]] < 0] = 0.
        gamma['primary'][:shape_reduced[0], shape_reduced[1]:][gamma['primary'][:shape_reduced[0],
                                                               shape_reduced[1]:] < 0] = 0.
        gamma['secondary'][:shape_reduced[0], :][gamma['secondary'][:shape_reduced[0], :] < 0] = 0.
        gamma['secondary'][shape_reduced[0]:, :shape_reduced[1]][gamma['secondary'][shape_reduced[0]:,
                                                                 :shape_reduced[1]] < 0] = 0.


def get_symmetrical_q_ab(shape, shape_reduced, gamma, distance):
    """
    function uses surface symmetries to calculate parameter QAB = (cos gamma_a)*cos(gamma_b)/d**2 in reflection
    effect

    :param shape: desired shape of q_ab
    :param shape_reduced: shape of the surface symmetries, (faces above those indices are symmetrical to the ones
    below)
    :param gamma:
    :param distance:
    :return:
    """
    q_ab = np.empty(shape=shape, dtype=np.float)
    q_ab[:, :shape_reduced[1]] = \
        np.divide(np.multiply(gamma['primary'][:, :shape_reduced[1]],
                              gamma['secondary'][:, :shape_reduced[1]]),
                  np.power(distance[:, :shape_reduced[1]], 2))
    q_ab[:shape_reduced[0], shape_reduced[1]:] = \
        np.divide(np.multiply(gamma['primary'][:shape_reduced[0], shape_reduced[1]:],
                              gamma['secondary'][:shape_reduced[0], shape_reduced[1]:]),
                  np.power(distance[:shape_reduced[0], shape_reduced[1]:], 2))
    return q_ab


def compute_filling_factor(surface_potential, lagrangian_points):
    """

    :param surface_potential:
    :param lagrangian_points: list; lagrangian points in `order` (in order to ensure that L2)
    :return:
    """
    return (lagrangian_points[1] - surface_potential) / (lagrangian_points[1] - lagrangian_points[2])


def pre_calc_azimuths_for_detached_points(alpha):
    """
    returns azimuths for the whole quarter surface in specific order (near point, equator, far point and the rest)
    separator gives you information about position of these sections

    :param alpha:
    :return:
    """
    separator = []

    # azimuths for points on equator
    num = int(const.PI // alpha)
    phi = np.linspace(0., const.PI, num=num + 1)
    theta = np.array([const.HALF_PI for _ in phi])
    separator.append(np.shape(theta)[0])

    # azimuths for points on meridian
    num = int(const.HALF_PI // alpha)
    phi_meridian = np.array([const.PI for _ in range(num - 1)] + [0 for _ in range(num)])
    theta_meridian = np.concatenate((np.linspace(const.HALF_PI - alpha, alpha, num=num - 1),
                                     np.linspace(0., const.HALF_PI, num=num, endpoint=False)))

    phi = np.concatenate((phi, phi_meridian))
    theta = np.concatenate((theta, theta_meridian))
    separator.append(np.shape(theta)[0])

    # azimuths for rest of the quarter
    num = int(const.HALF_PI // alpha)
    thetas = np.linspace(alpha, const.HALF_PI, num=num - 1, endpoint=False)
    phi_q, theta_q = [], []
    for tht in thetas:
        alpha_corrected = alpha / np.sin(tht)
        num = int(const.PI // alpha_corrected)
        alpha_corrected = const.PI / (num + 1)
        phi_q_add = [alpha_corrected * ii for ii in range(1, num + 1)]
        phi_q += phi_q_add
        theta_q += [tht for _ in phi_q_add]

    phi = np.concatenate((phi, phi_q))
    theta = np.concatenate((theta, theta_q))

    return phi, theta, separator


def pre_calc_azimuths_for_overcontact_farside_points(alpha):
    """
    calculates azimuths (directions) to the surface points of over-contact component on its far-side

    :param alpha: discretization factor
    :return:
    """
    separator = []

    # calculating points on farside equator
    num = int(const.HALF_PI // alpha)
    phi = np.linspace(const.HALF_PI, const.PI, num=num + 1)
    theta = np.array([const.HALF_PI for _ in phi])
    separator.append(np.shape(theta)[0])

    # calculating points on phi = pi meridian
    phi_meridian1 = np.array([const.PI for _ in range(num)])
    theta_meridian1 = np.linspace(0., const.HALF_PI - alpha, num=num)
    phi = np.concatenate((phi, phi_meridian1))
    theta = np.concatenate((theta, theta_meridian1))
    separator.append(np.shape(theta)[0])

    # calculating points on phi = pi/2 meridian, perpendicular to component`s distance vector
    num -= 1
    phi_meridian2 = np.array([const.HALF_PI for _ in range(num)])
    theta_meridian2 = np.linspace(alpha, const.HALF_PI, num=num, endpoint=False)
    phi = np.concatenate((phi, phi_meridian2))
    theta = np.concatenate((theta, theta_meridian2))
    separator.append(np.shape(theta)[0])

    # calculating the rest of the surface on farside
    thetas = np.linspace(alpha, const.HALF_PI, num=num, endpoint=False)
    phi_q1, theta_q1 = [], []
    for tht in thetas:
        alpha_corrected = alpha / np.sin(tht)
        num = int(const.HALF_PI // alpha_corrected)
        alpha_corrected = const.HALF_PI / (num + 1)
        phi_q_add = [const.HALF_PI + alpha_corrected * ii for ii in range(1, num + 1)]
        phi_q1 += phi_q_add
        theta_q1 += [tht for _ in phi_q_add]
    phi = np.concatenate((phi, phi_q1))
    theta = np.concatenate((theta, theta_q1))
    separator.append(np.shape(theta)[0])

    return phi, theta, separator


def pre_calc_azimuths_for_overcontact_neck_points(alpha, neck_position, neck_polynomial, polar_radius, component):
    # generating the neck

    # lets define cylindrical coordinate system r_n, phi_n, z_n for our neck where z_n = x, phi_n = 0 heads along
    # z axis
    delta_z = alpha * polar_radius

    # test radii on neck_position
    r_neck = []
    separator = []

    if component == 'primary':
        num = 15 * int(neck_position // (polar_radius * alpha))
        # position of z_n adapted to the slope of the neck, gives triangles with more similar areas
        x_curve = np.linspace(0., neck_position, num=num, endpoint=True)
        z_curve = np.polyval(neck_polynomial, x_curve)

        # radius on the neck
        mid_r = np.min(z_curve)

        curve = np.column_stack((x_curve, z_curve))
        neck_lengths = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
        neck_length = np.sum(neck_lengths)
        segment = neck_length / (int(neck_length // delta_z))

        k = 1
        z_ns, line_sum = [], 0.0
        for ii in range(num - 2):
            line_sum += neck_lengths[ii]
            if line_sum > k * segment:
                z_ns.append(x_curve[ii + 1])
                r_neck.append(z_curve[ii])
                k += 1
        z_ns.append(neck_position)
        r_neck.append(mid_r)
        z_ns = np.array(z_ns)
    else:
        num = 15 * int((1 - neck_position) // (polar_radius * alpha))
        # position of z_n adapted to the slope of the neck, gives triangles with more similar areas
        x_curve = np.linspace(neck_position, 1, num=num, endpoint=True)
        z_curve = np.polyval(neck_polynomial, x_curve)

        # radius on the neck
        mid_r = np.min(z_curve)

        curve = np.column_stack((x_curve, z_curve))
        neck_lengths = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
        neck_length = np.sum(neck_lengths)
        segment = neck_length / (int(neck_length // delta_z))

        k = 1
        z_ns, line_sum = [1 - neck_position], 0.0
        r_neck.append(mid_r)
        for ii in range(num - 2):
            line_sum += neck_lengths[ii]
            if line_sum > k * segment:
                z_ns.append(1 - x_curve[ii + 1])
                r_neck.append(z_curve[ii])
                k += 1

        z_ns = np.array(z_ns)

    # equator azimuths
    phi = np.array([const.HALF_PI for _ in z_ns])
    z = z_ns
    separator.append(np.shape(z)[0])
    # meridian azimuths
    phi = np.concatenate((phi, np.array([0 for _ in z_ns])))
    z = np.concatenate((z, z_ns))
    separator.append(np.shape(z)[0])

    phi_n, z_n = [], []
    for ii, zz in enumerate(z_ns):
        num = int(const.HALF_PI * r_neck[ii] // delta_z)
        num = 1 if num == 0 else num
        start_val = const.HALF_PI / num
        phis = np.linspace(start_val, const.HALF_PI, num=num - 1, endpoint=False)
        z_n += [zz for _ in phis]
        phi_n += [phi for phi in phis]
    phi = np.concatenate((phi, np.array(phi_n)))
    z = np.concatenate((z, np.array(z_n)))
    separator.append(np.shape(z)[0])

    return phi, z, separator


def get_surface_points(*args):
    """
    function solves radius for given azimuths that are passed in *argss

    :param args:
    :return:
    """
    phi, theta, components_distance, precalc, fn = args

    pre_calc_vals = precalc(*(components_distance, phi, theta))

    solver_init_value = np.array([1. / 10000.])
    r = []
    for ii, phii in enumerate(phi):
        args = tuple(pre_calc_vals[ii, :])
        solution, _, ier, _ = scipy.optimize.fsolve(fn, solver_init_value, full_output=True, args=args, xtol=1e-12)
        r.append(solution[0])

    r = np.array(r)
    return utils.spherical_to_cartesian(np.column_stack((r, phi, theta)))


def get_surface_points_cylindrical(*args):
    """
    function solves radius for given azimuths that are passed in *argss

    :param args:
    :return:
    """
    phi, z, precalc, fn = args

    pre_calc_vals = precalc(*(phi, z))

    solver_init_value = np.array([1. / 10000.])
    r = []
    for ii, phii in enumerate(phi):
        args = tuple(pre_calc_vals[ii, :])
        solution, _, ier, _ = scipy.optimize.fsolve(fn, solver_init_value, full_output=True, args=args, xtol=1e-12)
        r.append(solution[0])

    r = np.array(r)
    return utils.cylindrical_to_cartesian(np.column_stack((r, phi, z)))


def component_to_list(component):
    """
    converts component name string into list

    :param component: if None, `['primary', 'secondary']` will be returned
                      otherwise `primary` and `secondary` will be converted into lists [`primary`] and [`secondary`]
    :return:
    """
    if not component:
        component = ['primary', 'secondary']
    elif component in ['primary', 'secondary']:
        component = [component]
    else:
        raise ValueError('Invalid name of the component. Use `primary` or `secondary`.')
    return component
