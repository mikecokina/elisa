import hashlib


def md5_content(content):
    md5 = hashlib.md5()
    content = content.encode('utf-8') if isinstance(content, str) else content
    md5.update(content)
    return md5.hexdigest()


def sha256_content(content):
    sha256 = hashlib.sha256()
    content = content.encode('utf-8') if isinstance(content, str) else content
    sha256.update(content)
    return sha256.hexdigest()


def det_2d(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def multiple_determinants(matrix):
    """
    calculates 2D determinant on every level of given 3D matrix

    :param matrix: np.array (Nx2x2), where i-th slice looks like:
                                    [[xi1, yi1],
                                     [xi2, yi2]]
    :return: np.array - N-dim vector where each element is 2D determinant of 2 2D vectors stored on given level in
                        `matrix`
    """
    return matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]
