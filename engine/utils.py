import numpy as np
import re

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
    points = data[:]
    test_points, distances = points[:], []

    for i in range(0, len(test_points) - 1):
        points.remove(test_points[i])
        tree = KDTree(points)
        distance, ndx = tree.query([test_points[i]], k=1)
        distances.append(distance[0])
    return min(distances)


def spherical_to_cartesian(radius, phi, theta):
    """
    converts spherical coordinates into cartesian

    :param radius: np.array
    :param phi: np.array
    :param theta: np.array
    :return:
    """
    x = radius * np.cos(phi) * np.sin(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(theta)
    return x, y, z


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

    :param theta: float; radial vector of point of interest to ratate
    :param omega: 3d list of floats; arbitrary vector to rotate around
    :param vector: 3d list of floats;
    :param degrees: bool; units of incoming vector
    :return: np.array;
    """
    omega = np.array(omega) / np.linalg.norm(np.array(omega))
    theta = theta if not degrees else np.radians(theta)

    matrix = np.arange(9, dtype=np.float).reshape((3, 3))

    matrix[0][0] = (np.cos(theta)) + (omega[0] ** 2 * (1. - np.cos(theta)))
    matrix[0][1] = (omega[0] * omega[1] * (1. - np.cos(theta))) - (omega[2] * np.sin(theta))
    matrix[0][2] = (omega[1] * np.sin(theta)) + (omega[0] * omega[2] * (1. - np.cos(theta)))

    matrix[1][0] = (omega[2] * np.sin(theta)) + (omega[0] * omega[1] * (1. - np.cos(theta)))
    matrix[1][1] = (np.cos(theta)) + (omega[1] ** 2 * (1. - np.cos(theta)))
    matrix[1][2] = (- omega[0] * np.sin(theta)) + (omega[1] * omega[2] * (1. - np.cos(theta)))

    matrix[2][0] = (- omega[1] * np.sin(theta)) + (omega[0] * omega[2] * (1. - np.cos(theta)))
    matrix[2][1] = (omega[0] * np.sin(theta)) + (omega[1] * omega[2] * (1. - np.cos(theta)))
    matrix[2][2] = (np.cos(theta)) + (omega[2] ** 2 * (1. - np.cos(theta)))

    return np.dot(matrix, vector)
