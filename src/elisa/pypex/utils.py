from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from elisa.base.types import FLOAT

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike


def md5_content(content: str) -> str:
    """Calculate the MD5 hash of the given content.

    :param content: Content to hash
    :type content: str
    :return: MD5 hash as a hexadecimal string
    :rtype: str
    """
    md5 = hashlib.md5()  # noqa: S324
    content = content.encode("utf-8") if isinstance(content, str) else content
    md5.update(content)
    return md5.hexdigest()


def sha256_content(content: str) -> str:
    """Calculate the SHA256 hash of the given content.

    :param content: Content to hash
    :type content: str
    :return: SHA256 hash as a hexadecimal string
    :rtype: str
    """
    sha256 = hashlib.sha256()
    content = content.encode("utf-8") if isinstance(content, str) else content
    sha256.update(content)
    return sha256.hexdigest()


def det_2d(matrix: ArrayLike) -> float:
    """Calculate the 2D determinant of a 2x2 matrix.

    :param matrix: 2x2 matrix as array-like (list, tuple, or numpy.ndarray)
    :type matrix: ArrayLike
    :return: Determinant value
    :rtype: float
    """
    return FLOAT(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])


def multiple_determinants(matrix: np.ndarray) -> np.ndarray:
    """Calculate 2D determinant on every level of given 3D matrix.

    :param matrix: np.array (Nx2x2), where i-th slice looks like:
        [[xi1, yi1], [xi2, yi2]]
    :type matrix: numpy.ndarray
    :return: N-dim vector where each element is 2D determinant of two 2D vectors stored on given level in `matrix`
    :rtype: numpy.ndarray
    """
    return matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]
