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

