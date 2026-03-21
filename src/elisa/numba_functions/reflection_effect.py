from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    from numpy.typing import NDArray


@jit(nopython=True, cache=True, fastmath=True)
def gamma_primary(
        normals: NDArray,
        join_vector: NDArray,
) -> NDArray:
    """Calculate cosine of angles between visible normals of primary surface elements.

    Calculates the cosine of angles between visible normals of primary surface elements
    and each counterpart element on the secondary.

    :param normals: Array of shape (a, 3) representing visible normals of primary surface elements.
    :type normals: NDArray
    :param join_vector: Array of shape (a, b, 3) representing vectors joining primary and secondary elements.
    :type join_vector: NDArray
    :returns: Array of shape (a, b) containing cosines of angles between normals and join vectors.
    :rtype: NDArray
    """
    result = np.empty(join_vector.shape[:-1], dtype=np.float64)
    for ii in range(normals.shape[0]):
        for jj in range(join_vector.shape[1]):
            result[ii, jj] = (
                    normals[ii, 0] * join_vector[ii, jj, 0]
                    + normals[ii, 1] * join_vector[ii, jj, 1]
                    + normals[ii, 2] * join_vector[ii, jj, 2]
            )
    return result


@jit(nopython=True, cache=True, fastmath=True)
def gamma_secondary(
        normals: NDArray,
        join_vector: NDArray,
) -> NDArray:
    """Calculate cosine of angles between visible normals of secondary surface elements.

    Calculates the cosine of angles between visible normals of secondary surface elements
    and each counterpart element on the primary.

    :param normals: Array of shape (b, 3) representing visible normals of secondary surface elements.
    :type normals: NDArray
    :param join_vector: Array of shape (a, b, 3) representing vectors joining primary and secondary elements.
    :type join_vector: NDArray
    :returns: Array of shape (a, b) containing cosines of angles between normals and join vectors.
    :rtype: NDArray
    """
    result = np.empty(join_vector.shape[:-1], dtype=np.float64)
    for ii in range(join_vector.shape[0]):
        for jj in range(join_vector.shape[1]):
            result[ii, jj] = (
                    - normals[jj, 0] * join_vector[ii, jj, 0]
                    - normals[jj, 1] * join_vector[ii, jj, 1]
                    - normals[jj, 2] * join_vector[ii, jj, 2]
            )
    return result
