import numpy as np

from copy import copy, deepcopy
from elisa import utils, ld, opt, umpy as up
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


def darkside_filter(line_of_sight, normals):
    """
    Return indices for visible faces defined by given normals.
    Function assumes that `line_of_sight` ([1, 0, 0]) and `normals` are already normalized to one.

    :param line_of_sight: numpy.array
    :param normals: numpy.array
    :return: numpy.array
    """
    # todo: require to resolve self shadowing in case of W UMa
    # calculating normals utilizing the fact that normals and line of sight vector [1, 0, 0] are already normalized
    if (line_of_sight == np.array([1.0, 0.0, 0.0])).all():
        cosines = utils.calculate_cos_theta_los_x(normals=normals)
    else:
        cosines = utils.calculate_cos_theta(normals=normals, line_of_sight_vector=np.array([1, 0, 0]))
    # recovering indices of points on near-side (from the point of view of observer)
    return up.arange(np.shape(normals)[0])[cosines > 0]


def plane_projection(points, plane, keep_3d=False):
    """
    Function projects 3D points into given plane.

    :param keep_3d: bool; if True, the dimensions of the array is kept the same, with given column equal to zero
    :param points: numpy.array
    :param plane: str; ('xy', 'yz', 'zx')
    :return: numpy.array
    """
    rm_index = {"xy": 2, "yz": 0, "zx": 1}[plane]
    if not keep_3d:
        indices_to_keep = [0, 1, 2]
        del indices_to_keep[rm_index]
        return points[:, indices_to_keep]
    in_plane = deepcopy(points)
    in_plane[:, rm_index] = 0.0
    return in_plane
























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
