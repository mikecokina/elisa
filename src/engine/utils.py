from queue import Empty

import numpy as np
import scipy as sp
import re
from copy import copy
from scipy.spatial import distance_matrix as dstm

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


def cylindrical_to_cartesian(radius, phi, z):
    """
    converts cylindrical coordinates to cartesian

    :param radius: : np.array
    :param phi: np.array
    :param z: np.array
    :return:
    """
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    return x, y, z


def arbitrary_rotation(theta, omega=None, vector=None, degrees=False):
    """
    function rotates `vector` around axis defined by `omega` vector by amount `theta`

    :param theta: float; radial vector of point of interest to ratate
    :param omega: 3d list of floats; arbitrary vector to rotate around
    :param vector: 3d list of floats;
    :param degrees: bool; units of incoming vector
    :return: np.array;
    """
    omega = np.array(omega) / np.linalg.norm(np.array(omega))
    theta = theta if not degrees else np.radians(theta)

    matrix = np.arange(9, dtype=np.float).reshape((3, 3))

    matrix[0, 0] = (np.cos(theta)) + (omega[0] ** 2 * (1. - np.cos(theta)))
    matrix[0, 1] = (omega[0] * omega[1] * (1. - np.cos(theta))) - (omega[2] * np.sin(theta))
    matrix[0, 2] = (omega[1] * np.sin(theta)) + (omega[0] * omega[2] * (1. - np.cos(theta)))

    matrix[1, 0] = (omega[2] * np.sin(theta)) + (omega[0] * omega[1] * (1. - np.cos(theta)))
    matrix[1, 1] = (np.cos(theta)) + (omega[1] ** 2 * (1. - np.cos(theta)))
    matrix[1, 2] = (- omega[0] * np.sin(theta)) + (omega[1] * omega[2] * (1. - np.cos(theta)))

    matrix[2, 0] = (- omega[1] * np.sin(theta)) + (omega[0] * omega[2] * (1. - np.cos(theta)))
    matrix[2, 1] = (omega[0] * np.sin(theta)) + (omega[1] * omega[2] * (1. - np.cos(theta)))
    matrix[2, 2] = (np.cos(theta)) + (omega[2] ** 2 * (1. - np.cos(theta)))

    return np.matmul(matrix, vector)


# def arbitrary_rotation(theta, omega=None, vector=None, degrees=False):
#     omega = np.array(omega) / np.linalg.norm(np.array(omega))
#     theta = theta if not degrees else np.radians(theta)
#     # if np.shape(theta)[0] != np.shape(vector)[0] and vector.ndim != 1:
#     #     raise ValueError('Number of angles theta does not correspond to number of vectors to rotate.')
#     #
#     if omega.ndim == 1:
#         omega = omega[np.newaxis, :]
#     # elif np.shape(vector) != np.shape(omega):
#     #     raise ValueError('Number of rotation axis is not `omega` and does not correspond to number of vectors to '
#     #                      'rotate.')
#
#     matrix = np.empty((np.shape(omega)[0], 3, 3), dtype=np.float)
#
#     matrix[:, 0, 0] = np.cos(theta) + (omega[:, 0] ** 2 * (1. - np.cos(theta)))
#     matrix[:, 0, 1] = (omega[:, 0] * omega[:, 1] * (1. - np.cos(theta))) - (omega[:, 2] * np.sin(theta))
#     matrix[:, 0, 2] = (omega[:, 1] * np.sin(theta)) + (omega[:, 0] * omega[:, 2] * (1. - np.cos(theta)))
#
#     matrix[:, 1, 0] = (omega[:, 2] * np.sin(theta)) + (omega[:, 0] * omega[:, 1] * (1. - np.cos(theta)))
#     matrix[:, 1, 1] = (np.cos(theta)) + (omega[:, 1] ** 2 * (1. - np.cos(theta)))
#     matrix[:, 1, 2] = (- omega[:, 0] * np.sin(theta)) + (omega[:, 1] * omega[:, 2] * (1. - np.cos(theta)))
#
#     matrix[:, 2, 0] = (- omega[:, 1] * np.sin(theta)) + (omega[:, 0] * omega[:, 2] * (1. - np.cos(theta)))
#     matrix[:, 2, 1] = (omega[:, 0] * np.sin(theta)) + (omega[:, 1] * omega[:, 2] * (1. - np.cos(theta)))
#     matrix[:, 2, 2] = (np.cos(theta)) + (omega[:, 2] ** 2 * (1. - np.cos(theta)))
#
#     return np.matmul(matrix, vector)



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


def find_nearest_value(array, value):
    array = np.array(array)
    value = array[(np.abs(array - value)).argmin()]
    index = np.where(array == value)[0][0]
    # index = array.tolist().index(value)
    return [value, index]


def find_surounded(array, value):
    # find surounded value in passed array
    arr, ret = np.array(array[:]), []
    f_nst = find_nearest_value(arr, value)
    ret.append(f_nst[0])

    new_arr = []
    if f_nst[0] > value:
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
    ret.append(find_nearest_value(arr, value)[0])
    ret = sorted(ret)
    # test
    return ret if ret[0] < value < ret[1] else [value]


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
