import numpy as np

from copy import deepcopy
from elisa import utils, umpy as up


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
