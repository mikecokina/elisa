from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    from numpy.typing import NDArray


@jit(nopython=True, cache=True, fastmath=True)
def create_distance_vector_matrix(
    points1: NDArray,
    points2: NDArray,
) -> NDArray:
    """Calculate distances between every point couple in arrays points1 and points2.

    :param points1: Array of shape (a, 3) representing the first set of points.
    :type points1: NDArray
    :param points2: Array of shape (b, 3) representing the second set of points.
    :type points2: NDArray
    :returns: Array of shape (a, b, 3) containing distance vectors.
    :rtype: NDArray
    """
    result = np.empty((points1.shape[0], points2.shape[0], points1.shape[1]), dtype=np.float64)
    for ii in range(points1.shape[0]):
        for jj in range(points2.shape[0]):
            for kk in range(points2.shape[1]):
                result[ii, jj, kk] = points2[jj, kk] - points1[ii, kk]
    return result


@jit(nopython=True, cache=True, fastmath=True)
def calculate_lengths_in_3d_array(
    matrix: NDArray,
) -> NDArray:
    """Calculate lengths of each 3D vector stored in a (a, b, 3) array.

    :param matrix: Array of shape (a, b, 3) containing 3D vectors.
    :type matrix: NDArray
    :returns: Array of shape (a, b) containing vector lengths.
    :rtype: NDArray
    """
    result = np.empty(matrix.shape[:-1], dtype=np.float64)
    for ii in range(matrix.shape[0]):
        for jj in range(matrix.shape[1]):
            result[ii, jj] = np.sqrt(
                matrix[ii, jj, 0] ** 2 + matrix[ii, jj, 1] ** 2 + matrix[ii, jj, 2] ** 2,
            )
    return result


@jit(nopython=True, cache=True, fastmath=True)
def divide_points_in_array_by_constants(
    matrix: NDArray,
    coefficients: NDArray,
) -> NDArray:
    """Divide 3D points in matrix by a separate coefficient (e.g. their norm).

    :param matrix: Array of shape (a, b, 3) containing 3D points.
    :type matrix: NDArray
    :param coefficients: Array of shape (a, b) containing coefficients for division.
    :type coefficients: NDArray
    :returns: Array of shape (a, b, 3) containing divided points.
    :rtype: NDArray
    """
    result = np.empty(matrix.shape, dtype=np.float64)
    for ii in range(matrix.shape[0]):
        for jj in range(matrix.shape[1]):
            coeff = coefficients[ii, jj]
            for kk in range(matrix.shape[2]):
                result[ii, jj, kk] = matrix[ii, jj, kk] / coeff
    return result
