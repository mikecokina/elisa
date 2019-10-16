import numpy as np

from copy import copy
from elisa import utils, const, ld, opt
from elisa.utils import is_empty
from elisa.conf import config


def visibility_test(centres, xlim, component):
    """
    Tests if given faces are visible from the other star.

    :param component: str
    :param centres: numpy.array
    :param xlim: visibility threshold in x axis for given component
    :return: numpy.array[bool]
    """
    return centres[:, 0] >= xlim if component == 'primary' else centres[:, 0] <= xlim


def compute_filling_factor(surface_potential, lagrangian_points):
    """
    Compute filling factor of given BinaryStar system.
    Filling factor is computed as::

        (Omega_{inner} - Omega) / (Omega_{inner} - Omega_{outter}),

    where Omega_X denote potential value and `Omega` is potential of given Star.
    Inner and outter are critical inner and outter potentials for given binary star system.

    :param surface_potential:
    :param lagrangian_points: list; lagrangian points in `order` (in order to ensure that L2)
    :return:
    """
    return (lagrangian_points[1] - surface_potential) / (lagrangian_points[1] - lagrangian_points[2])





























def get_symmetrical_distance_matrix(shape, shape_reduced, centres, vis_test, vis_test_symmetry):
    """
    Function uses symmetries of the stellar component in order to reduce time in calculation distance matrix.

    :param shape: Tuple[int]; desired shape of join vector matrix
    :param shape_reduced: Tuple[int]; shape of the surface symmetries,
                         (faces above those indices are symmetrical to the ones below)
    :param centres: Dict
    :param vis_test: Dict[str, numpy.array]
    :param vis_test_symmetry: Dict[str, numpy.array]
    :return: Tuple; (distance, join vector)

    ::

        distance - distance matrix
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
    Function copies basic parameters of the stellar surface (points, faces, normals, temperatures, areas and log_g) of
    given star instance into new arrays during calculation of reflection effect.

    :param component_instance: Star instance
    :return: Tuple; (points, faces, centres, normals, temperatures, areas)
    """
    points, faces = component_instance.return_whole_surface()
    centres = copy(component_instance.face_centres)
    normals = copy(component_instance.normals)
    temperatures = copy(component_instance.temperatures)
    log_g = copy(component_instance.log_g)
    areas = copy(component_instance.areas)
    return points, faces, centres, normals, temperatures, areas, log_g


def include_spot_to_surface_variables(centres, spot_centres, normals, spot_normals, temperatures,
                                      spot_temperatures, areas, spot_areas, log_g, spot_log_g, vis_test, vis_test_spot):
    """
    Function includes surface parameters of spot faces into global arrays containing parameters from whole surface
    used in reflection effect.

    :param centres: numpy.array
    :param spot_centres: numpy.array; spot centres to append to `centres`
    :param normals: numpy.array;
    :param spot_normals: numpy.array; spot normals to append to `normals`
    :param temperatures: numpy.array;
    :param spot_temperatures: numpy.array; spot temperatures to append to `temperatures`
    :param areas: numpy.array;
    :param spot_areas: numpy.array; spot areas to append to `areas`
    :param vis_test: numpy.array;
    :param vis_test_spot: numpy.array; spot visibility test to append to `vis_test`
    :return: Tuple; (centres, normals, temperatures, areas, vis_test)
    """
    centres = np.append(centres, spot_centres, axis=0)
    normals = np.append(normals, spot_normals, axis=0)
    temperatures = np.append(temperatures, spot_temperatures, axis=0)
    areas = np.append(areas, spot_areas, axis=0)
    log_g = np.append(log_g, spot_log_g, axis=0)
    vis_test = np.append(vis_test, vis_test_spot, axis=0)

    return centres, normals, temperatures, areas, vis_test, log_g


def get_symmetrical_gammma(shape, shape_reduced, normals, join_vector, vis_test, vis_test_symmetry):
    """
    Function uses surface symmetries to calculate cosine of angles between join vector and surface normals.

    :param shape: Tuple[int]; desired shape of gamma
    :param shape_reduced: Tuple[int]; shape of the surface symmetries, (faces above those
                                      indices are symmetrical to the ones below)
    :param normals: Dict[str, numpy.array]
    :param join_vector: Dict[str, numpy.array]
    :param vis_test: Dict[str, numpy.array]
    :param vis_test_symmetry: Dict[str, numpy.array]
    :return: gamma: Dict[str, numpy.array]; cos(angle(normal, join_vector))
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
        np.sum(np.multiply(normals['secondary'][vis_test['secondary']][None, :, :],
                           -join_vector[:shape_reduced[0], :, :]), axis=2)
    gamma['secondary'][shape_reduced[0]:, :shape_reduced[1]] = \
        np.sum(np.multiply(normals['secondary'][vis_test_symmetry['secondary']][None, :, :],
                           -join_vector[shape_reduced[0]:, :shape_reduced[1], :]), axis=2)
    return gamma


def get_symmetrical_d_gamma(shape, shape_reduced, ldc, gamma):
    """
    Function uses surface symmetries to calculate limb darkening factor matrices
    for each components that are used in reflection effect.

    :param ldc: dict - arrays of limb darkening coefficients for each face of each component
    :param shape: desired shape of limb darkening matrices d_gamma
    :param shape_reduced: shape of the surface symmetries, (faces above those indices are symmetrical to the ones
    below)
    :param normals:
    :param join_vector:
    :param vis_test:
    :return:
    """
    # todo: important -fix LD COEFF to real
    d_gamma = {'primary': np.empty(shape=shape, dtype=np.float),
               'secondary': np.empty(shape=shape, dtype=np.float)}

    cos_theta = gamma['primary'][:, :shape_reduced[1]]
    d_gamma['primary'][:, :shape_reduced[1]] = ld.limb_darkening_factor(
        coefficients=ldc['primary'][:, :shape[0]].T,
        limb_darkening_law=config.LIMB_DARKENING_LAW,
        cos_theta=cos_theta)

    cos_theta = gamma['primary'][:shape_reduced[0], shape_reduced[1]:]
    d_gamma['primary'][:shape_reduced[0], shape_reduced[1]:] = ld.limb_darkening_factor(
        coefficients=ldc['primary'][:, :shape_reduced[0]].T,
        limb_darkening_law=config.LIMB_DARKENING_LAW,
        cos_theta=cos_theta)

    cos_theta = gamma['secondary'][:shape_reduced[0], :]
    d_gamma['secondary'][:shape_reduced[0], :] = ld.limb_darkening_factor(
        coefficients=ldc['secondary'][:, :shape[1]].T,
        limb_darkening_law=config.LIMB_DARKENING_LAW,
        cos_theta=cos_theta.T).T

    cos_theta = gamma['secondary'][shape_reduced[0]:, :shape_reduced[1]]
    d_gamma['secondary'][shape_reduced[0]:, :shape_reduced[1]] = ld.limb_darkening_factor(
        coefficients=ldc['secondary'][:, :shape_reduced[1]].T,
        limb_darkening_law=config.LIMB_DARKENING_LAW,
        cos_theta=cos_theta.T).T

    return d_gamma


def check_symmetric_gamma_for_negative_num(gamma, shape_reduced):
    """
    If cos < 0 it will be redefined as 0 are inplaced.

    :param gamma: Dict[str, numpy.array]
    :param shape_reduced: Tuple[int]
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
    Function uses surface symmetries to calculate parameter::

        QAB = (cos gamma_a)*cos(gamma_b)/d**2

    in reflection effect.

    :param shape: Tuple[int]; desired shape of q_ab
    :param shape_reduced: Tuple[int]; shape of the surface symmetries,
                                     (faces above those indices are symmetrical to the ones below)
    :param gamma: Dict[str, numpy.array]
    :param distance: numpy.array
    :return: numpy.array
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


def pre_calc_azimuths_for_detached_points(alpha):
    """
    Returns azimuths for the whole quarter surface in specific order::

        (near point, equator, far point and the rest)

    separator gives you information about position of these sections.

    :param alpha: float; discretization factor
    :return: Tuple; (phi: numpy.array, theta: numpy.array, separtor: numpy.array)
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
    Calculates azimuths (directions) to the surface points of over-contact component on its far-side.

    :param alpha: float; discretization factor
    :return: Tuple; (phi: numpy.array, theta: numpy.array, separtor: numpy.array)
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
    """
    Calculates azimuths (directions) to the surface points of over-contact component on neck.

    :param alpha: float; doscretiozation factor
    :param neck_position: float; x position of neck of over-contact binary
    :param neck_polynomial: scipy.Polynome; polynome that define neck profile in plane `xz`
    :param polar_radius: float
    :param component: str; `primary` or `secondary`
    :return: Tuple; (phi: numpy.array, z: numpy.array, separator: numpy.array)
    """
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
        num = int(0.93 * (const.HALF_PI * r_neck[ii] // delta_z))
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
    Function solves radius for given azimuths that are passed in *args.
    It use `scipy.optimize.fsolve` method. Function to solve is specified as last parameter in *args Tuple.

    :param args: Tuple;

    ::

        Tuple[
                phi: numpy.array,
                theta: numpy.array,
                x0: float,
                components_distance: float,
                precalc: method,
                fn: method,
                derivative_fn: method]

    :return: numpy.array
    """
    phi, theta, x0, components_distance, precalc_fn, potential_fn, potential_derivative_fn, surface_potential = args
    precalc_vals = precalc_fn(*(components_distance, phi, theta), return_as_tuple=True)
    x0 = x0 * np.ones(phi.shape)
    radius = opt.newton.newton(potential_fn, x0, fprime=potential_derivative_fn,
                               maxiter=config.MAX_SOLVER_ITERS, args=(precalc_vals, surface_potential), rtol=1e-10)
    return utils.spherical_to_cartesian(np.column_stack((radius, phi, theta)))


def get_surface_points_cylindrical(*args):
    """
    Function solves radius for given azimuths that are passed in *args.

    :param args: Tuple;

    ::

         Tuple[
                phi: numpy.array,
                z: numpy.array,
                components_distance: float,
                x0: float,
                precalc: method,
                fn: method
                fprime: method
              ]

    :return: numpy.array
    """
    phi, z, components_distance, x0, precalc_fn, potential_fn, potential_derivative_fn, surface_potential = args
    precalc_vals = precalc_fn(*(phi, z, components_distance), return_as_tuple=True)
    x0 = x0 * np.ones(phi.shape)

    radius = opt.newton.newton(potential_fn, x0, fprime=potential_derivative_fn, args=(precalc_vals, surface_potential),
                               maxiter=config.MAX_SOLVER_ITERS, rtol=1e-10)
    return utils.cylindrical_to_cartesian(np.column_stack((np.abs(radius), phi, z)))


def component_to_list(component):
    """
    Converts component name string into list.

    :param component: str;  If None, `['primary', 'secondary']` will be returned otherwise
                            `primary` and `secondary` will be converted into lists [`primary`] and [`secondary`].
    :return: List[str]
    """
    if component in ["all", "both"]:
        component = ['primary', 'secondary']
    elif component in ['primary', 'secondary']:
        component = [component]
    elif is_empty(component):
        return []
    else:
        raise ValueError('Invalid name of the component. Use `primary`, `secondary`, `all` or `both`')
    return component
