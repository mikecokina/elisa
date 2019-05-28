import numpy as np
import scipy as sp
import re

from queue import Empty
from copy import copy

from numpy.linalg import norm
from scipy.spatial import distance_matrix as dstm
from elisa.engine import const as c
from typing import Tuple

# temporary
from time import time


def polar_to_cartesian(radius, phi):
    """

    :param radius: (np.)float, (np.)int
    :param phi: (np.)float, (np.)int
    :return: tuple ((np.)float, (np.)float)
    """
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    return x, y


def invalid_kwarg_checker(kwargs, kwarglist, instance):
    invalid_kwargs = [kwarg for kwarg in kwargs if kwarg not in kwarglist]
    if len(invalid_kwargs) > 0:
        raise ValueError('Invalid keyword argument(s): {} in class instance {}.\n List of available parameters: {}'
                         ''.format(', '.join(invalid_kwargs), instance.__name__, format(', '.join(kwarglist))))


def is_plane(given, expected):
    pattern = r'^({0})|({1})$'.format(expected, expected[::-1])
    return re.search(pattern, given)


def find_nearest_dist_3d(data=None):
    """
    function finds the smallest distance between given set of points

    :param data: array like
    :return: (np.)float; minimal distance of points in dataset
    """
    from scipy.spatial import KDTree
    points = copy(data)
    test_points, distances = copy(points), []

    for i in range(0, len(test_points) - 1):
        points.remove(test_points[i])
        tree = KDTree(points)
        distance, ndx = tree.query([test_points[i]], k=1)
        distances.append(distance[0])
    return min(distances)


def cartesian_to_spherical(points, degrees=False):
    """
    convert cartesian to spherical coordinates if only 1 point is given input an output is only 1D vector

    :param points: numpy_array([[x1, y1, z1],
                                [x2, y2, z2],
                                 ...
                                [xn, yn, zn]])
    :param degrees: bool
    :return: numpy_array([[r1, phi1, theta1],
                          [r2, phi2, theta2],
                          ...
                          [rn, phin, thetan]])
    """
    points = np.array(points)
    points = np.expand_dims(points, axis=0) if len(np.shape(points)) == 1 else points
    r = np.linalg.norm(points, axis=1)

    np.seterr(divide='ignore', invalid='ignore')
    phi = np.arcsin(points[:, 1] / (np.linalg.norm(points[:, :2], axis=1)))  # vypocet azimutalneho (rovinneho) uhla
    phi[np.isnan(phi)] = 0

    theta = np.arccos(points[:, 2] / r)  # vypocet polarneho (elevacneho) uhla
    theta[np.isnan(theta)] = 0
    np.seterr(divide='print', invalid='print')

    signtest = points[:, 0] < 0
    phi[signtest] = (np.pi - phi[signtest])

    return_val = np.column_stack((r, phi, theta)) if not degrees else np.column_stack((r, np.degrees(phi),
                                                                                       np.degrees(theta)))
    return np.squeeze(return_val, axis=0) if np.shape(return_val)[0] == 1 else return_val


# def spherical_to_cartesian(radius, phi, theta):
def spherical_to_cartesian(spherical_points):
    """
    converts spherical coordinates into cartesian, if input is one point, output is 1D vector

    :param spherical_points: numpy_array([[r1, phi1, theta1],
                                          [r2, phi2, theta2],
                                           ...
                                          [rn, phin, thetan]])
    :return: numpy_array([[x1, y1, z1],
                          [x2, y2, z2],
                           ...
                          [xn, yn, zn]])
    """
    spherical_points = np.array(spherical_points)
    spherical_points = np.expand_dims(spherical_points, axis=0) if len(np.shape(spherical_points)) == 1 \
        else spherical_points
    x = spherical_points[:, 0] * np.cos(spherical_points[:, 1]) * np.sin(spherical_points[:, 2])
    y = spherical_points[:, 0] * np.sin(spherical_points[:, 1]) * np.sin(spherical_points[:, 2])
    z = spherical_points[:, 0] * np.cos(spherical_points[:, 2])
    points = np.column_stack((x, y, z))
    return np.squeeze(points, axis=0) if np.shape(points)[0] == 1 else points


def cylindrical_to_cartesian(cylindrical_points):
    """
    converts cylindrical coordinates into cartesian, if input is one point, output is 1D vector

    :param cylindrical_points: numpy_array([[r1, phi1, z1],
                                            [r2, phi2, z2],
                                             ...
                                            [rn, phin, zn]])
    :return: numpy_array([[x1, y1, z1],
                          [x2, y2, z2],
                           ...
                          [xn, yn, zn]])
    """
    cylindrical_points = np.array(cylindrical_points)
    cylindrical_points = np.expand_dims(cylindrical_points, axis=0) if len(np.shape(cylindrical_points)) == 1 \
        else cylindrical_points
    x = cylindrical_points[:, 0] * np.cos(cylindrical_points[:, 1])
    y = cylindrical_points[:, 0] * np.sin(cylindrical_points[:, 1])
    points = np.column_stack((x, y, cylindrical_points[:, 2]))
    return np.squeeze(points, axis=0) if np.shape(points)[0] == 1 else points


def arbitrary_rotation(theta, omega=None, vector=None, degrees=False, omega_normalized=False):
    """
    Rodrigues` Rotaion Formula
    function rotates `vector` around axis defined by `omega` vector by amount `theta`

    :param theta: float; radial vector of point of interest to rotate
    :param omega: 3d list of floats; arbitrary vector to rotate around
    :param vector: 3d list of floats;
    :param degrees: bool; units of incoming vector
    :param omega_normalized: if True, then in-function normalization of omega is not performed
    :return: np.array;
    """
    # this action normalizes the same vector over and over again during spot calculation, which is unnecessary
    if not omega_normalized:
        omega = np.array(omega) / np.linalg.norm(np.array(omega))

    theta = theta if not degrees else np.radians(theta)

    matrix = np.arange(9, dtype=np.float).reshape((3, 3))

    matrix[0, 0] = (np.cos(theta)) + (omega[0] ** 2 * (1. - np.cos(theta)))
    matrix[1, 0] = (omega[0] * omega[1] * (1. - np.cos(theta))) - (omega[2] * np.sin(theta))
    matrix[2, 0] = (omega[1] * np.sin(theta)) + (omega[0] * omega[2] * (1. - np.cos(theta)))

    matrix[0, 1] = (omega[2] * np.sin(theta)) + (omega[0] * omega[1] * (1. - np.cos(theta)))
    matrix[1, 1] = (np.cos(theta)) + (omega[1] ** 2 * (1. - np.cos(theta)))
    matrix[2, 1] = (- omega[0] * np.sin(theta)) + (omega[1] * omega[2] * (1. - np.cos(theta)))

    matrix[0, 2] = (- omega[1] * np.sin(theta)) + (omega[0] * omega[2] * (1. - np.cos(theta)))
    matrix[1, 2] = (omega[0] * np.sin(theta)) + (omega[1] * omega[2] * (1. - np.cos(theta)))
    matrix[2, 2] = (np.cos(theta)) + (omega[2] ** 2 * (1. - np.cos(theta)))

    return np.matmul(vector, matrix)


def axis_rotation(theta, vector, axis, inverse=False, degrees=False):
    # TODO: check if true. If yes I propose refactor, name suggests that axis are rotated, not points
    """
    rotation of `vector` around 'axis' by an amount `theta

    :param theta: degree of rotation
    :param vector: vector to rotate around
    :param axis: axis of rotation `x`, `y`, or `z`
    :param inverse: I HAVE NO CLUE...
    :param degrees: if True value theta is assumed to be in degrees
    :return: np.array - rotatet vector(s)
    """
    matrix = np.arange(9, dtype=np.float).reshape((3, 3))
    theta = theta if not degrees else np.radians(theta)
    vector = np.array(vector)

    if axis == "x":
        matrix[0][0], matrix[1][0], matrix[2][0] = 1, 0, 0
        matrix[0][1], matrix[1][1], matrix[2][1] = 0, np.cos(theta), - np.sin(theta)
        matrix[0][2], matrix[1][2], matrix[2][2] = 0, np.sin(theta), np.cos(theta)
        if inverse:
            matrix[2][1], matrix[1][2] = np.sin(theta), - np.sin(theta)
    if axis == "y":
        matrix[0][0], matrix[1][0], matrix[2][0] = np.cos(theta), 0, np.sin(theta)
        matrix[0][1], matrix[1][1], matrix[2][1] = 0, 1, 0
        matrix[0][2], matrix[1][2], matrix[2][2] = - np.sin(theta), 0, np.cos(theta)
        if inverse:
            matrix[0][2], matrix[2][0] = + np.sin(theta), - np.sin(theta)
    if axis == "z":
        matrix[0][0], matrix[1][0], matrix[2][0] = np.cos(theta), - np.sin(theta), 0
        matrix[0][1], matrix[1][1], matrix[2][1] = np.sin(theta), np.cos(theta), 0
        matrix[0][2], matrix[1][2], matrix[2][2] = 0, 0, 1
        if inverse:
            matrix[1][0], matrix[0][1] = + np.sin(theta), - np.sin(theta)
    # return np.matmul(matrix, vector.T).T
    return np.matmul(vector, matrix)


def average_spacing_cgal(data=None, neighbours=6):
    """
    Average Spacing - calculates average distance between points using average distances to `neighbours` number of
    points
    Match w/ CGAL average spacing function

    :param data: list (np.array); 3-dimensinal dataset
    :param neighbours: int; nearest neighbours to average
    :return:
    """
    if not isinstance(data, type(np.array)):
        data = np.array(data)

    dist = sp.spatial.distance.cdist(data, data, 'euclidean')
    total = 0
    for line in dist:
        total += np.sort(line)[1:1 + neighbours].sum() / (neighbours + 1)
    return total / dist.shape[0]


def average_spacing(data=None, mean_angular_distance=None):
    """
    calculates mean distance between points using mean radius of data points and mean angular distance between them
    :param data: numpy.array([[x1 y1 z1],
                              [x2 y2 z2],
                                ...
                              [xN yN zN]])
    :param mean_angular_distance: np.float - in radians
    :return:
    """
    average_radius = np.mean(np.linalg.norm(data, axis=1)) if not np.isscalar(data) else data
    return average_radius * mean_angular_distance


def remap(x, mapper):
    """
    function rearranges list of points indices according to indices in mapper, maper contains on the nth place new
    address of the nth point

    :param x: faces-like matrix
    :param mapper: transformation map - numpy.array(): new_index_of_point = mapper[old_index_of_point]
    :return: faces-like matrix
    """
    return list(map(lambda val: [mapper[val[0]], mapper[val[1]], mapper[val[2]]], x))


def poly_areas(polygons):
    """
    calculates areas of triangles, where `triangles` indexes of vertices which coordinates are stored in `points`
    :param polygons: np.array; 3d points
    :return: np.array
    """
    if polygons is None:
        raise ValueError('Faces from which to calculate areas were not supplied.')
    polygons = np.array(polygons)
    return 0.5 * np.linalg.norm(np.cross(polygons[:, 1] - polygons[:, 0],
                                         polygons[:, 2] - polygons[:, 0]), axis=1)


def triangle_areas(triangles=None, points=None):
    """
    calculates areas of triangles, where `triangles` indexes of vertices which coordinates are stored in `points`
    :param triangles: np.array; indices of triangulation
    :param points: np.array; 3d points
    :return: np.array
    """
    if triangles is None:
        raise ValueError('Faces from which to calculate areas were not supplied.')
    if points is None:
        raise ValueError('Vertices of faces from which to calculate areas were not supplied.')

    return 0.5 * np.linalg.norm(np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                                         points[triangles[:, 2]] - points[triangles[:, 0]]), axis=1)


def calculate_distance_matrix(points1=None, points2=None, return_join_vector_matrix=False):
    """
    function returns distance matrix between two sets of points
    :param points1:
    :param points2:
    :param return_join_vector_matrix: if True, function also returns normalized distance vectors useful for dot product
    during calculation of cos
    :return:
    """
    # pairwise distance vector matrix
    distance_vector_matrix = points2[None, :, :] - points1[:, None, :]
    distance_matrix = np.linalg.norm(distance_vector_matrix, axis=2)

    return distance_matrix, distance_vector_matrix / distance_matrix[:, :, None] if return_join_vector_matrix \
        else distance_matrix


def find_face_centres(faces=None):
    """
    function calculates centres of each supplied face
    :param faces: np.array([[[x11,y11,z11],
                             [x11,y11,z11],
                             [x12,y12,z12]]
                            [[x21,y21,z21],
                             [x21,y21,z21],
                             [x22,y22,z22]]
                             ...
                           ])
    :return:
    """
    if faces is None:
        raise ValueError('Faces were not supplied')
    return np.mean(faces, axis=1)


def check_missing_kwargs(kwargs=None, instance_kwargs=None, instance_of=None):
    """
    checks if all `kwargs` are all in `instance kwargs`
    :param kwargs: list
    :param instance_kwargs: list
    :param instance_of: object
    :return:
    """
    missing_kwargs = []
    for kwarg in kwargs:
        if kwarg not in instance_kwargs:
            missing_kwargs.append("`{}`".format(kwarg))

    if len(missing_kwargs) > 0:
        raise ValueError('Missing argument(s): {} in class instance {}'.format(', '.join(missing_kwargs),
                                                                               instance_of.__name__))


def numeric_logg_to_string(logg):
    return "g%02d" % (logg * 10)


def numeric_metallicity_to_string(metallicity):
    sign = "p" if metallicity >= 0 else "m"
    leadzeronum = "%02d" % (metallicity * 10) if metallicity >= 0 else "%02d" % (metallicity * -10)
    return "{sign}{leadzeronum}".format(sign=sign, leadzeronum=leadzeronum)


def find_nearest_value_as_matrix(look_in: np.array, look_for: np.array) -> Tuple[np.array, np.array]:
    """
    finds values and indices of elements in `look_in` that are the closest to the each value in `values`

    :param look_in:
    :param look_for: np.look_in - look_in of elements according to which the closest element in `look_in` is searched for
    :return:
    """
    val = np.array([look_for]) if np.isscalar(look_for) else look_for
    dif = np.abs(val[:, np.newaxis] - look_in)
    argmins = dif.argmin(axis=1)
    val = look_in[argmins]
    return val, argmins


def find_nearest_value(look_in, look_for):
    look_in = np.array(look_in)
    look_for = look_in[(np.abs(look_in - look_for)).argmin()]
    index = np.where(look_in == look_for)[0][0]
    return [look_for, index]


def find_surrounded_as_matrix(look_in, look_for):
    dif = look_for[:, np.newaxis] - look_in
    positive_mask = dif >= 0
    # for values on the left side of look_in array
    all_positive = np.all(positive_mask, axis=1)
    # add artificial sign change for right boundary value
    # switch 'fancy' indexing to integer index since in numpy, combined assigment can't be done by fancy indexing)
    all_positive_inline = np.arange(0, len(look_for))[all_positive]
    positive_mask[all_positive_inline, -1] = False
    # find signs switching columns
    sign_swith_mask = np.logical_xor(positive_mask[:, :-1], positive_mask[:, 1:])
    idx_array = np.ones(np.shape(dif), dtype=np.int) * np.arange(np.shape(look_in)[0])
    idx_array = idx_array[:, :-1][sign_swith_mask]
    ret_matrix = np.column_stack((look_in[idx_array], look_in[idx_array + 1]))
    # consider on place value as not surounded (surounded by itself)
    isin_look_in = np.isin(look_for, look_in)
    ret_matrix[isin_look_in] = np.array([look_for, look_for]).T[isin_look_in]
    return ret_matrix


def find_surrounded(look_in, look_for):
    # find surounded look_for in passed look_in

    arr, ret = np.array(look_in[:]), []
    f_nst = find_nearest_value(arr, look_for)
    ret.append(f_nst[0])

    new_arr = []
    if f_nst[0] > look_for:
        for i in range(0, len(arr)):
            if arr[i] < f_nst[0]:
                new_arr.append(arr[i])
    else:
        for i in range(0, len(arr)):
            if arr[i] > f_nst[0]:
                new_arr.append(arr[i])

    arr = new_arr[:]
    del new_arr

    # arr = np.delete(arr, f_nst[1], 0)
    ret.append(find_nearest_value(arr, look_for)[0])
    ret = sorted(ret)
    # test
    return ret if ret[0] < look_for < ret[1] else [look_for]


def calculate_cos_theta(normals, line_of_sight_vector):
    """
    calculates cosine between two set of normalized vectors
    - matrix(N * 3), matrix(3) - cosine between each matrix(N * 3) and matrix(3)
    - matrix(N * 3), matrix(M * 3) - cosine between each combination of matrix(N * 3) and matrix(M * 3)
    :param normals:
    :param line_of_sight_vector:
    :return:
    """
    return np.sum(np.multiply(normals, line_of_sight_vector[None, :]), axis=1) if np.ndim(line_of_sight_vector) == 1 else \
        np.sum(np.multiply(normals[:, None, :], line_of_sight_vector[None, :, :]))


def calculate_cos_theta_los_x(normals):
    """
    calculates cosine of an angle between normalized vectors and line of sight vector [1 ,0 ,0]
    :param normals: np.array
    :return:
    """
    return normals[:, 0]


def get_line_of_sight_single_system(phase=None, inclination=None):
    """
    returns line of sight vector for given phase, inclination of the system
    and period of the rotation of given system

    :param phase: list
    :param inclination:
    :return:
    """
    line_of_sight_spherical = np.empty((len(phase), 3), dtype=np.float)
    line_of_sight_spherical[:, 0] = 1
    line_of_sight_spherical[:, 1] = c.FULL_ARC * phase
    line_of_sight_spherical[:, 2] = inclination
    return spherical_to_cartesian(line_of_sight_spherical)


def convert_gravity_acceleration_array(colormap, units):
    """
    function converts gravity acceleration array from log_g(SI) units to other units such as `log_cgs`, `SI`, `cgs`

    :param colormap: array
    :param units: str - `log_cgs`, `SI`, `cgs`, `log_SI`
    :return:
    """
    if units == 'log_cgs':
        colormap += 2
    elif units == 'SI':
        colormap = np.power(10, colormap)
    elif units == 'cgs':
        colormap = np.power(10, colormap + 2)
    elif units == 'log_SI':
        pass
    return colormap


# todo: name does not give sense, you just calculated cosine of the angle between vectors
def cosine_similarity(a, b):
    """
    function calculates cosine of angle between vectors, use only in case that a, and b are not normalized, otherwise
    function calculate_cos_theta

    :param a: numpy array
    :param b: numpy array
    :return:
    """
    return np.inner(a, b) / (norm(a) * norm(b))


class IterableQueue(object):
    """ Transform standard python Queue instance to iterable one"""

    def __init__(self, source_queue):
        """
        :param source_queue: queue.Queue, (mandatory)
        """
        self.source_queue = source_queue

    def __iter__(self):
        while True:
            try:
                yield self.source_queue.get_nowait()
            except Empty:
                return






