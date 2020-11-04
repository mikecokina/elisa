from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def gamma_primary(normals, join_vector):
    """
    Calculate cosine of angles between visible normals of primary surface elements and each counterpart element on the
    secondary.

    :param normals: numpy.array; (a, 3)
    :param join_vector: numpy.array; (a, b, 3)
    :return: numpy.array; (a, b)
    """
    result = np.empty(join_vector.shape[:-1])
    for ii in range(normals.shape[0]):
        for jj in range(join_vector.shape[1]):
            result[ii, jj] = normals[ii, 0] * join_vector[ii, jj, 0] + \
                             normals[ii, 1] * join_vector[ii, jj, 1] + \
                             normals[ii, 2] * join_vector[ii, jj, 2]

    return result


@jit(nopython=True, cache=True)
def gamma_secondary(normals, join_vector):
    """
    Calculate cosine of angles between visible normals of primary surface elements and each counterpart element on the
    secondary.

    :param normals: numpy.array; (b, 3)
    :param join_vector: numpy.array; (a, b, 3)
    :return: numpy.array; (a, b)
    """
    result = np.empty(join_vector.shape[:-1])
    for ii in range(join_vector.shape[0]):
        for jj in range(join_vector.shape[1]):
            result[ii, jj] = - normals[jj, 0] * join_vector[ii, jj, 0] - \
                             normals[jj, 1] * join_vector[ii, jj, 1] - \
                             normals[jj, 2] * join_vector[ii, jj, 2]

    return result

