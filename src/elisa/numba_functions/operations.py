from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def create_distance_vector_matrix(points1, points2):
    """
    Calculates distances between every point couple in arrays points1 and points2
    :param points1: numpy.array; (a, 3)
    :param points2: numpy.array; (b, 3)
    :return: numpy.array; (a, b, 3)
    """
    result = np.empty((points1.shape[0], points2.shape[0], points1.shape[1]))
    for ii in range(points1.shape[0]):
        for jj in range(points2.shape[0]):
            for kk in range(points2.shape[1]):
                result[ii, jj, kk] = points2[jj, kk] - points1[ii, kk]
    return result


@jit(nopython=True, cache=True)
def calculate_lengths_in_3d_array(matrix):
    """
    Calculates lengths of each 3d vector stored in (a, b, 3) array.

    :param matrix: numpy.array; (a, b, 3)
    :return: numpy.array; (a, b)
    """
    result = np.empty(matrix.shape[:-1])
    for ii in range(matrix.shape[0]):
        for jj in range(matrix.shape[1]):
            result[ii, jj] = (matrix[ii, jj, 0] ** 2 + matrix[ii, jj, 1] ** 2 + matrix[ii, jj, 2] ** 2) ** 0.5

    return result


@jit(nopython=True, cache=True)
def divide_points_in_array_by_constants(matrix, coefficients):
    """
    Divide 3D points in matrix by a separate coefficient (e.g. their norm).

    :param matrix: numpy.array; (a, b, 3)
    :param coefficients: numpy.array; (a, b)
    :return: numpy.array; (a, b, 3)
    """
    result = np.empty(matrix.shape)
    for ii in range(matrix.shape[0]):
        for jj in range(matrix.shape[1]):
            for kk in range(matrix.shape[2]):
                result[ii, jj, kk] = matrix[ii, jj, kk] / coefficients[ii, jj]

    return result
