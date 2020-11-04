from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def write_distances_to_common_array(dist1, dist2, join1, join2, shape, shape_reduced):
    """
    Writing partial distance matrices and join vector.

    :param dist1: numpy.array; (a, b)
    :param dist2: numpy.array; (c, d)
    :param join1: numpy.array; (a, b, 3)
    :param join2: numpy.array; (c, d, 3)
    :param shape: tuple; (a+c, b, 3)
    :param shape_reduced: tuple; (a, d)
    :return: tuple; (a+c, b), (a+c, b, 3)
    """
    distance = np.empty(shape=shape[:-1])
    join_vector = np.empty(shape=shape)
    # top part of the symmetric array
    for ii in range(shape_reduced[0]):
        for jj in range(shape[1]):
            distance[ii, jj] = dist1[ii, jj]
            for kk in range(shape[2]):
                join_vector[ii, jj, kk] = join1[ii, jj, kk]

    # filling bottom left part
    for ii in range(shape_reduced[0], shape[0]):
        for jj in range(shape_reduced[1]):
            distance[ii, jj] = dist2[ii - shape_reduced[0], jj]
            for kk in range(shape[2]):
                join_vector[ii, jj, kk] = join2[ii - shape_reduced[0], jj, kk]

    return distance, join_vector
