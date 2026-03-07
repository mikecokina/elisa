from __future__ import annotations

import datetime
import re
from collections.abc import Iterable, Iterator, Sized
from copy import copy, deepcopy
from queue import Empty, Queue
from re import Match
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy as sp
from matplotlib.cbook import flatten
from numpy.linalg import norm

from elisa import const
from elisa import umpy as up
from elisa.base.types import FLOAT, INT
from elisa.numba_functions import operations

if TYPE_CHECKING:
    from _typeshed import SupportsDunderGT, SupportsDunderLT
    from astropy.units import Unit
    from numpy.typing import NDArray

    from elisa.const import Position
    from elisa.types import Float, HasMeshData, Int, Number, Points2DList, Points3DList

FEBRUARY_MONTH_NUMBER = 2


def polar_to_cartesian(radius: Float | NDArray, phi: Float | NDArray) -> tuple[Float, Float] | tuple[NDArray, NDArray]:
    """Transform polar coordinates to cartesian.

    :param radius: (numpy.)float, (numpy.)int;
    :param phi: (numpy.)float, (numpy.)int;
    :return: Tuple ((numpy.)float, (numpy.)float);
    """
    x = radius * up.cos(phi)
    y = radius * up.sin(phi)
    return x, y


def invalid_kwarg_checker(kwargs: dict, kwarglist: list | tuple, instance: type) -> None:
    """Check if `kwargs` for `instance` are in allowed kwargs presented in `kwarglist`.

    :param kwargs: dict; kwargs to evaluate if are in kwarg list.
    :param kwarglist: list | tuple; list of allowed kwargs for given instance.
    :param instance: Any; class/instance with attribute `__name__`
    """
    invalid_kwargs = [kwarg for kwarg in kwargs if kwarg not in kwarglist]
    if len(invalid_kwargs) > 0:
        msg = (
            f"Invalid keyword argument(s): {', '.join(invalid_kwargs)} "
            f"in class instance {instance.__name__}.\n "
            f"List of available parameters: {', '.join(kwarglist)}"
        )
        raise ValueError(msg)


def invalid_param_checker(kwargs: dict, kwarglist: list | tuple, message: str) -> None:
    """Invalid parameter checker for any class instance or function.

    :param kwargs: dict; kwargs to evaluate if are in kwarg list.
    :param kwarglist: list | tuple; list of allowed kwargs for given instance
    :param message: Any class
    """
    invalid_kwargs = [kwarg for kwarg in kwargs if kwarg not in kwarglist]
    if len(invalid_kwargs) > 0:
        msg = (
            f"Invalid keyword argument(s): {', '.join(invalid_kwargs)} "
            f"in {message}.\n "
            f"List of available parameters: {', '.join(kwarglist)}"
        )
        raise ValueError(msg)


def is_plane(given: str, expected: str) -> Match[str] | None:
    """Find out whether `given` plane definition of 2d plane is as `expected`.

    E.g. if `yx` is `yx` or `xy`

    :param given: str;
    :param expected: str;
    :return: bool;
    """
    pattern = rf"^({expected})|({expected[::-1]})$"
    return re.search(pattern, given)


def find_nearest_dist_3d(data: Iterable) -> Float:
    """Find the smallest distance between given set of 3D points.

    :param data: Iterable;
    :return: float; minimal distance of points in dataset
    """
    from scipy.spatial import KDTree  # noqa: PLC0415

    if isinstance(data, np.ndarray):
        data = data.tolist()

    points = copy(data)
    test_points, distances = copy(points), []

    for i in range(len(test_points) - 1):
        points.remove(test_points[i])
        tree = KDTree(points)

        if np.isnan(test_points[i]).any():
            distances.append(np.inf)
        else:
            distance, _ndx = tree.query([test_points[i]], k=1)
            # noinspection PyUnresolvedReferences
            distances.append(distance[0])
    return np.min(distances)


def cartesian_to_spherical(
    points: Points3DList,
    *,
    degrees: bool = False,
) -> Points3DList | NDArray:
    """Convert cartesian to spherical coordinates.

    If only 1 point is given input an output is only 1D vector

    :param points:

        shape::

                    numpy.array([[x1, y1, z1],
                                [x2, y2, z2],
                                 ...
                                [xn, yn, zn]])

    :param degrees: bool; whether return spherical angular coordinates in degrees
    :return: numpy.array;

    shhape::

        ([[r1, phi1, theta1],
          [r2, phi2, theta2],
           ...
          [rn, phin, thetan]])
    """
    points = np.array(points)
    points = np.expand_dims(points, axis=0) if len(np.shape(points)) == 1 else points
    r = np.linalg.norm(points, axis=1)

    old_settings = np.seterr(divide="ignore", invalid="ignore")
    phi = up.arcsin(points[:, 1] / (np.linalg.norm(points[:, :2], axis=1)))  # vypocet azimutalneho (rovinneho) uhla
    phi[up.isnan(phi)] = 0

    theta = up.arccos(points[:, 2] / r)  # vypocet polarneho (elevacneho) uhla
    theta[up.isnan(theta)] = 0
    np.seterr(**old_settings)

    signtest = points[:, 0] < 0
    phi[signtest] = const.PI - phi[signtest]

    return_val = (
        np.column_stack((r, phi, theta)) if not degrees else np.column_stack((r, up.degrees(phi), up.degrees(theta)))
    )
    return np.squeeze(return_val, axis=0) if np.shape(return_val)[0] == 1 else return_val


def cartesian_to_polar(
    points: Points3DList,
    *,
    degrees: bool = False,
) -> Points3DList:
    """Convert cartesian to polar coordinates."""
    points = np.insert(points, 2, np.zeros(len(points)), axis=1)
    transform = cartesian_to_spherical(points, degrees=degrees)
    return transform.T[:2].T


def spherical_to_cartesian(spherical_points: Points3DList) -> Points3DList:
    """Convert spherical coordinates into cartesian.

    If input is one point, output is 1D vector.
    Function assumes that supplied spherical angular coordinates are in radians.

    :param spherical_points: numpy.array;

    shape::

        numpy.array([[r1, phi1, theta1],
                     [r2, phi2, theta2],
                      ...
                     [rn, phin, thetan]])

    :return: numpy.array;

    shape::

        ([[x1, y1, z1],
          [x2, y2, z2],
          ...
          [xn, yn, zn]])
    """
    spherical_points = np.array(spherical_points)
    spherical_points = (
        np.expand_dims(spherical_points, axis=0) if len(np.shape(spherical_points)) == 1 else spherical_points
    )
    x = spherical_points[:, 0] * up.cos(spherical_points[:, 1]) * up.sin(spherical_points[:, 2])
    y = spherical_points[:, 0] * up.sin(spherical_points[:, 1]) * up.sin(spherical_points[:, 2])
    z = spherical_points[:, 0] * up.cos(spherical_points[:, 2])
    points = np.column_stack((x, y, z))
    return np.squeeze(points, axis=0) if np.shape(points)[0] == 1 else points


def cylindrical_to_cartesian(cylindrical_points: Points3DList) -> Points3DList:
    """Convert cylindrical coordinates into cartesian.

    If input is one point, output is 1D vector.
    Function assumes that supplied spherical angular coordinates are in radians.

    :param cylindrical_points: numpy.array;

    shape ::

        numpy.array([[r1, phi1, z1],
                     [r2, phi2, z2],
                      ...
                     [rn, phin, zn]])

    :return: numpy.array;

    shape::

        ([[x1, y1, z1],
          [x2, y2, z2],
          ...
          [xn, yn, zn]])
    """
    cylindrical_points = np.array(cylindrical_points)
    cylindrical_points = (
        np.expand_dims(cylindrical_points, axis=0) if len(np.shape(cylindrical_points)) == 1 else cylindrical_points
    )
    x = cylindrical_points[:, 0] * up.cos(cylindrical_points[:, 1])
    y = cylindrical_points[:, 0] * up.sin(cylindrical_points[:, 1])
    points = np.column_stack((x, y, cylindrical_points[:, 2]))
    return np.squeeze(points, axis=0) if np.shape(points)[0] == 1 else points


def arbitrary_rotation(
    theta: Float,
    omega: NDArray,
    vector: NDArray,
    *,
    degrees: bool = False,
    omega_normalized: bool = False,
) -> NDArray:
    """Rodrigues's Rotation Formula.

    Function rotates `vector` around axis defined by `omega` vector by amount `theta`.

    :param theta: float; radial vector of point of interest to rotate
    :param omega: Union[List, numpy.ndarray]; 3d list of floats; arbitrary vector to rotate around
    :param vector: Union[List, numpy.ndarray]; 3d list of floats;
    :param degrees: bool; if True, theta supplied on intput is in degrees otherwise in radians
    :param omega_normalized: bool; if True, then in-function normalization of omega is not performed
    :return: numpy.array;
    """
    # this action normalizes the same vector over and over again during spot calculation, which is unnecessary
    if not omega_normalized:
        omega = np.array(omega) / np.linalg.norm(np.array(omega))

    theta = theta if not degrees else up.radians(theta)

    matrix = up.arange(9, dtype=FLOAT).reshape((3, 3))

    matrix[0, 0] = (up.cos(theta)) + (omega[0] ** 2 * (1.0 - up.cos(theta)))
    matrix[1, 0] = (omega[0] * omega[1] * (1.0 - up.cos(theta))) - (omega[2] * up.sin(theta))
    matrix[2, 0] = (omega[1] * up.sin(theta)) + (omega[0] * omega[2] * (1.0 - up.cos(theta)))

    matrix[0, 1] = (omega[2] * up.sin(theta)) + (omega[0] * omega[1] * (1.0 - up.cos(theta)))
    matrix[1, 1] = (up.cos(theta)) + (omega[1] ** 2 * (1.0 - up.cos(theta)))
    matrix[2, 1] = (-omega[0] * up.sin(theta)) + (omega[1] * omega[2] * (1.0 - up.cos(theta)))

    matrix[0, 2] = (-omega[1] * up.sin(theta)) + (omega[0] * omega[2] * (1.0 - up.cos(theta)))
    matrix[1, 2] = (omega[0] * up.sin(theta)) + (omega[1] * omega[2] * (1.0 - up.cos(theta)))
    matrix[2, 2] = (up.cos(theta)) + (omega[2] ** 2 * (1.0 - up.cos(theta)))

    return up.matmul(vector, matrix)


def around_axis_rotation(
    theta: Float,
    vector: list | NDArray,
    axis: str,
    *,
    inverse: bool = False,
    degrees: bool = False,
) -> NDArray:
    """Rotation of `vector` around `axis` by an amount `theta`.

    :param theta: float; degree of rotation
    :param vector: numpy.array; vector to rotate around
    :param axis: str; axis of rotation `x`, `y`, or `z`
    :param inverse: bool; rotate to inverse direction than is math positive
    :param degrees: bool; if True value theta is assumed to be in degrees
    :return: numpy.array; rotated vector(s)
    """
    matrix = up.arange(9, dtype=FLOAT).reshape((3, 3))
    theta = theta if not degrees else up.radians(theta)
    vector = np.array(vector)

    if axis == "x":
        matrix[0][0], matrix[1][0], matrix[2][0] = 1, 0, 0
        matrix[0][1], matrix[1][1], matrix[2][1] = 0, up.cos(theta), -up.sin(theta)
        matrix[0][2], matrix[1][2], matrix[2][2] = 0, up.sin(theta), up.cos(theta)
        if inverse:
            matrix[2][1], matrix[1][2] = up.sin(theta), -up.sin(theta)
    if axis == "y":
        matrix[0][0], matrix[1][0], matrix[2][0] = up.cos(theta), 0, up.sin(theta)
        matrix[0][1], matrix[1][1], matrix[2][1] = 0, 1, 0
        matrix[0][2], matrix[1][2], matrix[2][2] = -up.sin(theta), 0, up.cos(theta)
        if inverse:
            matrix[0][2], matrix[2][0] = +up.sin(theta), -up.sin(theta)
    if axis == "z":
        matrix[0][0], matrix[1][0], matrix[2][0] = up.cos(theta), -up.sin(theta), 0
        matrix[0][1], matrix[1][1], matrix[2][1] = up.sin(theta), up.cos(theta), 0
        matrix[0][2], matrix[1][2], matrix[2][2] = 0, 0, 1
        if inverse:
            matrix[1][0], matrix[0][1] = +up.sin(theta), -up.sin(theta)
    return up.matmul(vector, matrix)


def rotate_item(
    vector: NDArray,
    position: Position,
    inclination: Float,
) -> NDArray:
    """Transfer vector(s) from corotating reference frame to observers frame.

    :param vector: numpy.ndarray;
    :param position: elisa.const.Position;
    :param inclination: float;
    :return: numpy.ndarray
    """
    correction = np.sign(const.LINE_OF_SIGHT[0]) * const.HALF_PI
    args = (position.azimuth - correction, vector, "z")
    kwargs = {"inverse": False, "degrees": False}
    vector = around_axis_rotation(*args, **kwargs)

    inverse = const.LINE_OF_SIGHT[0] != 1
    args = (const.HALF_PI - inclination, vector, "y")
    kwargs = {"inverse": inverse, "degrees": False}
    return around_axis_rotation(*args, **kwargs)


def average_spacing_cgal(
    data: NDArray,
    neighbours: int = 6,
) -> Float:
    """Average Spacing.

    Calculate average distance between points using average distances to `neighbours` number of
    points.

    Match w/ CGAL average spacing function (https://www.cgal.org/).

    :param data: List; 3-dimensinal dataset
    :param neighbours: int; nearest neighbours to average
    :return: float;
    """
    if not isinstance(data, type(np.array)):
        data = np.array(data)

    # noinspection PyUnresolvedReferences
    dist = sp.spatial.distance.cdist(data, data, "euclidean")
    total = 0
    for line in dist:
        total += np.sort(line)[1 : 1 + neighbours].sum() / (neighbours + 1)
    return total / dist.shape[0]


def average_spacing(
    data: Points3DList,
    mean_angular_distance: Float,
) -> Float:
    """Calculate mean distance between points.

    Calculates mean distance between points using mean radius of data points and mean angular distance between them
    :param data: numpy.array;

        shape::

            ([[x1 y1 z1],
              [x2 y2 z2],
               ...
              [xN yN zN]])

    :param mean_angular_distance: (numpy.)float; in radians
    :return: float;
    """
    average_radius = np.mean(np.linalg.norm(data, axis=1)) if not np.isscalar(data) else data
    return average_radius * mean_angular_distance


def remap(
    x: list[list[int]],
    mapper: list[int],
) -> list[list[int | list[int]]]:
    """Rearrange list of points indices according to indices in mapper.

    Maper contains on the nth place new address of the nth point:

        mapper = <class 'list'>: [7, 8, 9, 1, 2, 3, 4, 5, 6]
        x = <class 'list'>: [[3, 4, 5], [6, 7, 8], [0, 2, 1]]
        result = <class 'list'>: [[1, 2, 3], [4, 5, 6], [7, 9, 8]]

    Value 3 from `x` was replaced by 3rd (in index meaning) value from `mapper`,
    value 6 from `x` is replaced by 6th value from mapper, etc.

    :param x: numpy.array; faces-like matrix
    :param mapper: numpy.array; transformation map - numpy.array(): new_index_of_point = mapper[old_index_of_point]
    :return: List; faces-like matrix
    """
    return [[mapper[val[0]], mapper[val[1]], mapper[val[2]]] for val in x]


def poly_areas(
    polygons: NDArray,
) -> NDArray[Float]:
    """Calculate surface areas of triangles, where `triangles` coordinates of points are in `polygons` variable.

    :param polygons: numpy.array; 3d points
    :return: numpy.ndarray;
    """
    polygons = np.array(polygons)
    return 0.5 * np.linalg.norm(np.cross(polygons[:, 1] - polygons[:, 0], polygons[:, 2] - polygons[:, 0]), axis=1)


def triangle_areas(
    triangles: NDArray[Int],
    points: NDArray,
) -> NDArray[Float]:
    """Calculate areas of triangles, where `triangles` indexes of vertices which coordinates are stored in `points`.

    :param triangles: numpy.array; indices of triangulation
    :param points: numpy.array; 3d points
    :return: numpy.array;
    """
    return 0.5 * np.linalg.norm(
        np.cross(points[triangles[:, 1]] - points[triangles[:, 0]], points[triangles[:, 2]] - points[triangles[:, 0]]),
        axis=1,
    )


def calculate_distance_matrix(
    points1: NDArray,
    points2: NDArray,
    *,
    return_join_vector_matrix: bool = False,
) -> tuple[NDArray[Float], NDArray[Float] | None]:
    """Return distance matrix between two sets of points.

    Return matrix consist of distances in order like foloowing:

        [[D(p1[0], p2[0]), D(p1[0], p2[1]), ..., D(p1[0], p2[n])],
         [D(p1[1], p2[0]), D(p1[1], p2[1]), ..., D(p1[1], p2[n])],
         ...
         [D(p1[m], p2[0]), D(p1[m], p2[1]), ..., D(p1[m], p2[n])]]

    If join vector set to True, normalized join vecetors are defined as vectors in between points
    and positions in matrix are related to their distance in matrix above.

    :param points1: numpy.ndarray;
    :param points2: numpy.ndarray;
    :param return_join_vector_matrix: bool; if True, function also returns normalized distance
                                            vectors useful for dot product during calculation of cos
    :return: Tuple[numpy.ndarray, Union[numpy.ndarray, None]]
    """
    # pairwise distance vector matrix
    distance_vector_matrix = operations.create_distance_vector_matrix(np.array(points1), np.array(points2))

    distance_matrix = operations.calculate_lengths_in_3d_array(distance_vector_matrix)

    if return_join_vector_matrix:
        normalized_distance_vectors = operations.divide_points_in_array_by_constants(
            distance_vector_matrix,
            distance_matrix,
        )
        return distance_matrix, normalized_distance_vectors
    return distance_matrix, None


def find_face_centres(faces: Points3DList) -> NDArray[Float]:
    """Calculate centres (center of mass) of each supplied face.

    i-th coordinate of center of mas for one given triangle is computed as::

        x_i = (X_j[0] + X_j[1] + X_j[2]) / 3; j, i = [0, 1, 2],

    where X_j is coordinate if j-th corner of face.

    :param faces: numpy.array;

    shape::

        numpy.array([[[x11,y11,z11],
                      [x11,y11,z11],
                      [x12,y12,z12]]
                     [[x21,y21,z21],
                      [x21,y21,z21],
                      [x22,y22,z22]]
                      ...,
                      ])

    :return: numpy.ndarray;
    """
    return np.mean(faces, axis=1)


def check_missing_kwargs(
    mandatory_kwargs: list | tuple,
    supplied_kwargs: dict[str, Any],
    instance_of: type,
) -> None:
    """Check if all `kwargs` are all in `instance kwargs`.

    If missing raise ValuerError with missing `kwargs`.

    :param mandatory_kwargs: list[str] | tuple[str]
    :param supplied_kwargs: dict[str, Any]
    :param instance_of: class
    :return:
    """
    missing_kwargs = [f"`{kwarg}`" for kwarg in mandatory_kwargs if kwarg not in supplied_kwargs]
    if len(missing_kwargs) > 0:
        msg = f"Missing argument(s): {', '.join(missing_kwargs)} in class instance {instance_of.__name__}"
        raise ValueError(msg)


def check_missing_params(
    mandatory_kwargs: list,
    supplied_kwargs: list,
    instance: object,
) -> None:
    """Check if all `kwargs` are all in parameter `obj`.

    If missing raise ValuerError with missing `kwargs`.

    :param mandatory_kwargs: List[str]
    :param supplied_kwargs: List[str]
    :param instance: class instance
    :return:
    """
    missing_kwargs = [f"`{kwarg}`" for kwarg in mandatory_kwargs if kwarg not in supplied_kwargs]
    if len(missing_kwargs) > 0:
        msg = f"Missing argument(s): {', '.join(missing_kwargs)} in object {instance}"
        raise ValueError(msg)


def numeric_logg_to_string(logg: Float) -> str:
    """Convert numeric form of cgs logg to string used in LD tables.

    Convert numeric form of cgs logg to string used in castelli-kurucz 04,
    kurucz 93 and van-hamme limb darkening table names.

    :param logg: float;
    :return: str;
    """
    return f"g{int(logg * 10):02d}"


def numeric_metallicity_to_string(metallicity: Float) -> str:
    """Convert numeric form of cgs metallicity (M/H units) to string used in LD tables.

    Convert numeric form of cgs metallicity (M/H units) to string used in castelli-kurucz 04,
    kurucz 93 and van-hamme limb darkening table names.

    :param metallicity: float
    :return: str
    """
    sign = "p" if metallicity >= 0 else "m"
    leadzeronum = abs(int(metallicity * 10))
    return f"{sign}{leadzeronum:02d}"


def numeric_metallicity_from_string(n_metallicity: str) -> Float:
    """Return numeric metallicity from string used in van-hamme tables.

    :param n_metallicity: str;
    :return: float;
    """
    m = n_metallicity
    sign = 1 if str(m).startswith("p") else -1
    value = float(m[1:]) / 10.0
    return value * sign


def find_nearest_value_as_matrix(
    look_in: NDArray[Float],
    look_for: Float | NDArray[Float],
) -> tuple[NDArray[Float], NDArray[Int]]:
    """Find values and indices of elements in `look_in` that are the closest to the value in `values`.

    :param look_in: numpy.array; elemnts we are looking the closest point in
    :param look_for: numpy.array; elements according to which the closest element in `look_in` is searched for
    :return: Tuple[numpy.array, numpy.ndarray]
    """
    val = np.array([look_for]) if np.isscalar(look_for) else look_for
    dif = up.abs(val[:, np.newaxis] - look_in)
    argmins = dif.argmin(axis=1)
    val = look_in[argmins]
    return val, argmins


def find_nearest_value(
    look_in: NDArray[Float] | list[Float],
    look_for: Float,
) -> tuple[NDArray[Float], Int]:
    """Find nearest value in `look_in` array to value `look_for`.

    :param look_in: numpy.ndarray;
    :param look_for: float;
    :return: List[look_for: float, int (index from look_for)];
    """
    look_in = np.array(look_in)
    look_for = look_in[(up.abs(look_in - look_for)).argmin()]
    index = up.where(look_in == look_for)[0][0]
    return look_for, index


def find_surrounded_as_matrix(
    look_in: NDArray[Float],
    look_for: NDArray[Float],
) -> NDArray[Float]:
    """Find values from `look_in` which souround values from `look_for`.

    If any value of `look_in` array is exact same
    as value in `look_for` surounded value is same value from left and right side.

    :param look_in: numpy.ndarray;
    :param look_for: numpy.ndarray;
    :return: numpy.array;
    """
    if not (np.array(look_in.min() <= look_for).all() and np.array(look_for <= look_in.max()).all()):
        msg = "At least one value in `look_for` is out of bound of `look_in`"
        raise ValueError(msg)

    dif = look_for[:, np.newaxis] - look_in
    positive_mask = dif >= 0
    # for values on the left side of look_in array
    all_positive = np.all(positive_mask, axis=1)
    # add artificial sign change for right boundary value
    # switch 'fancy' indexing to integer index since in numpy, combined assigment can't be done by fancy indexing
    all_positive_inline = up.arange(0, len(look_for))[all_positive]
    positive_mask[all_positive_inline, -1] = False
    # find signs switching columns
    sign_swith_mask = up.logical_xor(positive_mask[:, :-1], positive_mask[:, 1:])
    idx_array = np.ones(np.shape(dif), dtype=INT) * up.arange(np.shape(look_in)[0])
    idx_array = idx_array[:, :-1][sign_swith_mask]
    ret_matrix = np.column_stack((look_in[idx_array], look_in[idx_array + 1]))
    # consider on place value as not surounded (surounded by itself)
    isin_look_in = np.isin(look_for, look_in)
    ret_matrix[isin_look_in] = np.array([look_for, look_for]).T[isin_look_in]
    return ret_matrix


def find_surrounded(
    look_in: NDArray[Float],
    look_for: Float,
) -> list[Any] | list[SupportsDunderLT[Any] | SupportsDunderGT[Any] | Any]:
    """Find values from `look_in` which surround `look_for` value.

    If exact `look_for` value is supplied as exists in `look_for` just this one value is returned
    in array as left and right border.

    :param look_in: numpy.ndarray;
    :param look_for: float;
    :return: List [float, float];
    """
    if look_for > max(look_in) or look_for < min(look_in):
        msg = "Any value in `look_for` is out of bound of `look_in`"
        raise ValueError(msg)

    if look_for == min(look_in):
        return [min(look_in), min(look_in)]
    if look_for == max(look_in):
        return [max(look_in), max(look_in)]

    arr, ret = np.array(look_in[:]), []
    f_nst = find_nearest_value(arr, look_for)
    ret.append(float(f_nst[0]))

    new_arr = []
    if f_nst[0] > look_for:
        for i in range(len(arr)):
            if arr[i] < f_nst[0]:
                new_arr.extend([float(arr[i])])
    else:
        for i in range(len(arr)):
            if arr[i] > f_nst[0]:
                new_arr.extend([float(arr[i])])

    arr = new_arr[:]
    del new_arr

    # arr = np.delete(arr, f_nst[1], 0)
    ret.append(float(find_nearest_value(np.asarray(arr), look_for)[0]))
    ret = sorted(ret)
    # test
    return ret if ret[0] < look_for < ret[1] else [look_for, look_for]


def calculate_cos_theta(
    normals: NDArray,
    line_of_sight_vector: NDArray[Float],
) -> NDArray[Float]:
    """Calculate cosine between two set of normalized vectors.

    Depending on the shape of `line_of_sight_vector` function returns:
        - matrix(N * 3), matrix(3) - cosine between each matrix(N * 3) and matrix(3)
        - matrix(N * 3), matrix(M * 3) - cosine between each combination of matrix(N * 3) and matrix(M * 3)

    Combinations are made like normals[0] with all of line_of_sight_vector,
    normals[1] with all from line_of_sight_vector, and so on, and so forth.

    :param normals: numpy.ndarray;
    :param line_of_sight_vector: numpy.ndarray;
    :return: numpy.ndarray;
    """
    line_of_sight_vector = np.array(line_of_sight_vector)
    return (
        np.sum(up.multiply(normals, line_of_sight_vector[None, :]), axis=1)
        if np.ndim(line_of_sight_vector) == 1
        else np.sum(up.multiply(normals[:, None, :], line_of_sight_vector[None, :, :]), axis=2)
    )


def calculate_cos_theta_los_x(
    normals: NDArray,
) -> NDArray:
    """Calculate cosine of an angle between normalized vectors and line of sight vector [1, 0, 0].

    :param normals: numpy.ndarray
    :return: numpy.ndarray
    """
    return const.LINE_OF_SIGHT[0] * normals[:, 0]


def get_line_of_sight_single_system(
    phase: NDArray,
    inclination: Float,
) -> Points3DList:
    """Line of sight vector for given phase, inclination of the system and period of the rotation of given system.

    :param phase: numpy.ndarray;
    :param inclination: float;
    :return: numpy.array;
    """
    line_of_sight_spherical = np.empty((len(phase), 3), dtype=FLOAT)
    line_of_sight_spherical[:, 0] = 1
    line_of_sight_spherical[:, 1] = const.FULL_ARC * phase
    line_of_sight_spherical[:, 2] = inclination
    return spherical_to_cartesian(line_of_sight_spherical)


def convert_gravity_acceleration_array(
    colormap: NDArray,
    units: str,
) -> NDArray:
    """Convert gravity acceleration array from log_g(SI) units to other units such as `log_cgs`, `SI`, `cgs`.

    :param colormap: numpy.ndarray;
    :param units: str; - `log_cgs`, `SI`, `cgs`, `log_SI`
    :return: numpy.array;
    """
    if units == "log_cgs":
        # keep it in this way to avoid mutable rewrite of origin colormap array
        # for more information read python docs about __add__ and __iadd__ class methods
        colormap = colormap + 2
    elif units == "SI":
        colormap = up.power(10, colormap)
    elif units == "cgs":
        colormap = up.power(10, colormap + 2)
    elif units == "log_SI":
        pass
    return colormap


def cosine_similarity(a: NDArray, b: NDArray) -> Float:
    """Calculate cosine of angle between vectors.

    :note: Use only in case that a and b are not normalized, otherwise
           use function calculate_cos_theta; it is way faster since it doesn't normalize vectors on fly.

    :param a: numpy.array;
    :param b: numpy.array;
    :return: float;
    """
    return up.inner(a, b) / (norm(a) * norm(b))


def is_empty(value: Any) -> bool:  # noqa: PLR0911
    """Look for empty value.

    Consider `value` as empty if fit following rules::

        - is None
        - is Sized with zero length
        - is pandas.DataFrame with zero length
        - is pandas.NaT
        - is numpy.NaN

    :param value: Any
    :return: bool
    """
    if value is None:
        return True

    # Put DataFrame before Sized, otherwise this branch is unreachable
    if isinstance(value, pd.DataFrame):
        # noinspection PyUnresolvedReferences
        return value.empty

    if isinstance(value, type(pd.NaT)):
        return True

    # Only Sized check for fast len() == 0
    if isinstance(value, Sized) and len(value) == 0:
        return True

    # Only call flatten when it's actually iterable
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return len(list(flatten(value))) == 0

    # NaN check (keep your up.isnan if you want)
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def find_idx_of_nearest(array: NDArray[Float], values: NDArray[Float]) -> Int:
    """Find indices of elements in `array` that are closest to elements in `values`.

    :param array: 1D array (M) - points to be searched for the closest
    :param values: 1D array (N) - values to which closest point in the `array` should be found
    :return: np.ndarray with shape (N) that points to the closest values in `array`
    """
    array = np.asarray(array)
    return (up.abs(array[np.newaxis, :] - values[:, np.newaxis])).argmin(axis=1)


def rotation_in_spherical(
    phi: NDArray,
    theta: NDArray,
    phi_rotation: Float,
    theta_rotation: Float,
) -> tuple[NDArray, NDArray]:
    """Rotation of spherical coordinates.

    Transformation of phi, theta spherical coordinates into new spherical coordinates produced by rotation around old
    z_axis (in positive direction) by `phi_rotation` and second rotation (in positive direction) around new y-axis by
    value `theta rotation`.

    :param phi: numpy.array; - in radians
    :param theta: numpy.array; - in radians
    :param phi_rotation: float; - rotation of old spherical system around z axis, in radians
    :param theta_rotation: float; - rotation of z axis along new y-axis by this value, in radians
    :return: Tuple; transformed angular coordinates
    """
    # rotation around Z axis
    phi_rot = (phi - phi_rotation) % const.FULL_ARC

    # rotation around Y axis by `theta_rotation` angle
    cos_phi = up.cos(phi_rot)
    sin_theta = up.sin(theta)
    sin_axis_theta = up.sin(theta_rotation)
    cos_theta = up.cos(theta)
    cos_axis_theta = up.cos(theta_rotation)
    theta_new = up.arccos(cos_phi * sin_theta * sin_axis_theta + cos_theta * cos_axis_theta)
    phi_new = up.arctan2(up.sin(phi_rot) * sin_theta, cos_phi * sin_theta * cos_axis_theta - cos_theta * sin_axis_theta)
    return phi_new, theta_new


def derotation_in_spherical(
    phi: NDArray,
    theta: NDArray,
    phi_rotation: Float,
    theta_rotation: Float,
) -> tuple[NDArray, NDArray]:
    """Backward transformation of spherical coordinates.

    Backward transformation of spherical coordinates  produced by rotation around old
    z_axis by `phi_rotation` and second rotation around new y-axis by value `theta rotation` into original coordinate
    system

    :param phi: numpy.array; - in radians
    :param theta: numpy.array; - in radians
    :param phi_rotation: float; - rotation of old spherical system around z axis, in radians
    :param theta_rotation: float; - rotation of z axis along new y-axis by this value, in radians
    :return:
    """
    cos_theta = up.cos(theta)
    sin_theta = up.sin(theta)
    cos_phi = up.cos(phi)
    sin_phi = up.sin(phi)

    sin_axis_theta = up.sin(theta_rotation)
    cos_axis_theta = up.cos(theta_rotation)

    theta_new = up.arccos(np.round(cos_theta * cos_axis_theta - cos_phi * sin_theta * sin_axis_theta, 10))
    phi_new = up.arctan2(sin_phi * sin_theta, cos_phi * sin_theta * cos_axis_theta + cos_theta * sin_axis_theta)
    return (phi_new + phi_rotation) % const.FULL_ARC, theta_new


def calculate_equiv_radius(volume: NDArray | Number) -> NDArray | Number:
    """Return equivalent radius of a sphere with given volume."""
    return up.power(3.0 * volume / (4.0 * const.PI), 1.0 / 3.0)


def calculate_ellipsoid_volume(
    _a: NDArray | Number,
    _b: NDArray | Number,
    _c: NDArray | Number,
) -> NDArray | Number:
    """Calculate volume of ellipsoid with semi-axis _a, _b and _c.

    :param _a: Float | NDarray;
    :param _b: Float | NDarray;
    :param _c: Float | NDarray;
    :return: Float | NDarray;
    """
    return 4.0 * const.PI * _a * _b * _c / 3.0


class IterableQueue:
    """Transform standard python Queue instance to iterable one."""

    def __init__(self, source_queue: Queue) -> None:
        """Construct IterableQueue class.

        :param source_queue: queue.Queue, (mandatory)
        """
        self.source_queue = source_queue

    def __iter__(self) -> Iterator:
        try:
            while True:
                yield self.source_queue.get_nowait()
        except Empty:
            return


def is_even(x: Int) -> bool:
    """Check if integer is even."""
    return x % 2 == 0


def convert_binary_orbital_motion_arr_to_positions(arr: NDArray) -> list[const.Position]:
    """Convert array of binary orbital motion to list of `const.Position` instances."""
    # noinspection PyArgumentList
    return [const.Position(*[int(p[0]) if not np.isnan(p[0]) else p[0], *list(p[1:])]) for p in arr]


def nested_dict_values(dictionary: dict) -> Iterable:
    """Recursively yield values from nested dictionaries."""
    for value in dictionary.values():
        if isinstance(value, dict):
            yield from nested_dict_values(value)
        else:
            yield value


def calculate_volume_ellipse_approx(
    equator_points: Points3DList | NDArray | None = None,
    meridian_points: Points3DList | NDArray | None = None,
) -> Float:
    """Calculate volume of the object.

    Calculate volume of the object where only equator and meridian points where provided usin elipsoidal
    approximation for the points with the same x-cordinates.

    :param equator_points: numpy array; (yzx) column-wise
    :param meridian_points: numpy array; (yzx) column-wise
    :return: float;
    """
    areas = up.abs(const.PI * equator_points[:, 1] * meridian_points[:, 0])
    return up.abs(up.trapz(areas, equator_points[:, 2]))


def plane_projection(
    points: Points3DList,
    plane: str,
    *,
    keep_3d: bool = False,
) -> Points2DList:
    """Projects 3D points into given plane.

    :param keep_3d: bool; if True, the dimensions of the array is kept the same, with given column equal to zero
    :param points: numpy.array;
    :param plane: str; one of 'xy', 'yz' or 'zx'
    :return: numpy.array;
    """
    rm_index = {"xy": 2, "yz": 0, "zx": 1}[plane]
    if not keep_3d:
        indices_to_keep = [0, 1, 2]
        del indices_to_keep[rm_index]
        return points[:, indices_to_keep]
    in_plane = deepcopy(points)
    in_plane[:, rm_index] = 0.0
    return in_plane


def get_visible_projection(obj: HasMeshData) -> NDArray[Float]:
    """Return yz projection of nearside points.

    :param obj: instance;
    :return: numpy.ndarray
    """
    return plane_projection(
        obj.points[np.unique(obj.faces[obj.indices])],
        "yz",
    )


def split_to_batches(array: NDArray, n_proc: Int) -> list[NDArray]:
    """Split array to batches with size `batch_size`.

    :param n_proc: int; number of processes
    :param array: Union[List, numpy.ndarray];
    :return: List;
    """
    indices = np.linspace(0, len(array), num=n_proc + 1, endpoint=True, dtype=INT)
    indices = [(indices[ii - 1], indices[ii]) for ii in range(1, n_proc + 1)]
    return [array[idx[0] : idx[1]] for idx in indices]


def renormalize_async_result(result: list[dict[str, NDArray]]) -> dict[str, NDArray]:
    """Renormalize multiprocessing output to native form.

    Multiprocessing will return several dicts with same passband (due to supplied batches), but continuous
    computaion require dict in form like::

        [{'passband': [all fluxes]}]

    instead::

        [[{'passband': [fluxes in batch]}], [{'passband': [fluxes in batch]}], ...]

    :param result: List;
    :return: dict[str; numpy.ndarray];
    """
    return {key: np.concatenate([record[key] for record in result], axis=0) for key in result[-1]}


def random_sign() -> Int:
    """Return random sign (-1 or 1)."""
    rng = np.random.default_rng()
    random = rng.integers(0, 2)
    return 1 if random else -1


def str_repalce(x: str, old: str | Iterable[str], new: str | Iterable[str]) -> str:
    """Replace old values with new in strin `x`.

    :param x: str;
    :param old: Union[str, Iterable[str]];
    :param new: Union[str, Iterable[str]];
    :return: str;
    """
    old = [old] if isinstance(old, str) else old
    new = [new] if isinstance(new, str) else new

    for _old, _new in zip(old, new, strict=True):
        x = str(x).replace(str(_old), str(_new))
    return x


def magnitude_to_flux(data: NDArray | Float, zero_point: NDArray | Float) -> NDArray | Float:
    """Calculate flux from magnitude using Pogson's formula."""
    return np.power(10, (zero_point - data) / 2.5)


def magnitude_error_to_flux_error(error: NDArray | Float) -> NDArray | Float:
    return np.power(10, error / 2.5) - 1.0


def flux_to_magnitude(data: NDArray | Float, zero_point: NDArray | Float) -> NDArray | Float:
    """Calculate magnitude from flux using Pogson's formula."""
    return -2.5 * np.log10(data) + zero_point


def flux_error_to_magnitude_error(data: NDArray | Float, error: NDArray | Float) -> NDArray | Float:
    """Calculate magnitude error from flux error using error propagation formula."""
    return 2.5 * np.log10(1 + (error / data))


def discretization_correction_factor(
    discretization_factor: Float,
    correction_factors: NDArray[Float],
) -> Float:
    """Correction factor for the surface due to underestimation of the surface by the triangles.

    :param correction_factors: numpy.array; (2*N) [discretization factor, correction factor],
                                            sorted according to discretization factor
    :param discretization_factor: numpy.float;
    :return: float;
    """
    # treating edge cases
    if discretization_factor <= correction_factors[0, 0]:
        correction_factor = correction_factors[1, 0]
    elif discretization_factor >= correction_factors[0, -1]:
        correction_factor = correction_factors[1, -1]
    else:
        correction_factor = np.interp(discretization_factor, correction_factors[0], correction_factors[1])
    # correction for non-equilateral triangles
    alpha = correction_factor * discretization_factor
    # correction for surface underestimation
    return np.sqrt(alpha / np.sin(alpha))


def transform_values(value: Float | NDArray, default_unit: Unit, unit: Unit) -> Float | NDArray[Float]:
    """Quick function for transformation to desired units.

    :param value: Float | NDarray; input values in default unit
    :param default_unit: astropy.units.Unit; base unit in which `value` is stored
    :param unit: astropy.units.Unit; target unit
    :return: Float | NDarray; transformed values
    """
    return value if unit == "default" else (value * default_unit).to(unit).value


def jd_to_phase(
    times: NDArray[Float] | Float,
    period: Float,
    t0: Float,
    centre: Float = 0.5,
) -> NDArray[Float]:
    """Convert JD to phase according to supplied ephemeris.

    Phases will be returned in range ('centre' - 0.5, 'centre' + 0.5).

    :param times: numpy.array;
    :param period: float;
    :param t0: float;
    :param centre: float;
    :return: numpy.array; converted phases
    """
    start_phase = centre - 0.5
    t0 += start_phase * period
    return ((times - t0) / period) % 1.0 + start_phase


def jd_from_datetime(dt: datetime.datetime) -> Float:
    """Convert a Python datetime object to the Julian Date.

    :param dt: (datetime): The datetime object to convert. Assumes the Gregorian calendar.
    :return: (float) The Julian Date corresponding to the input datetime.
    """
    dt_utc = dt.astimezone(datetime.timezone.utc)

    # Extract year, month, and day with the fractional part of the day.
    year = dt_utc.year
    month = dt_utc.month
    # Compute the fractional day: hours, minutes, seconds, and microseconds
    day_fraction = dt_utc.day + (
            dt_utc.hour / 24.0 +
            dt_utc.minute / 1440.0 +
            dt_utc.second / 86400.0 +
            dt_utc.microsecond / 86400.0 / 1e6
    )

    # For January and February, treat them as months 13 and 14 of the previous year
    if month <= FEBRUARY_MONTH_NUMBER:
        year -= 1
        month += 12

    # Compute the Gregorian calendar correction:
    a_ = np.floor(year / 100)
    b_ = 2 - a_ + np.floor(a_ / 4)

    # Now, apply the formula.
    return (
            np.floor(365.25 * (year + 4716)) +
            np.floor(30.6001 * (month + 1)) +
            day_fraction + b_ - 1524.5
    )
