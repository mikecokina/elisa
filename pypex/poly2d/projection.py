import numpy as np


def projection(vector_a, vector_d):
    """
    Projection vector ``p`` (scalar) of vector ``A`` in direction defined by unit vector ``E`` is computed as dot
    product of following::

        (1)   p = (A . E) * E = |E| * |A| * cos(t) * E

    where ``t`` is angle between ``A`` and ``E``.
    Instead of unit vector ``E`` we can use arbitrary vector in direction of ``E``, e.g. ``B`` and compute this
    unit vector from vector ``B`` as following::

        E = B / |B|

    After substitution we get::

        p = (A . E) * E = ((A . B) / |B|) * (B / |B|) = ((A . B) / (|B|^2)) * B = ((A . B) / (B . B)) * B

    :param vector_a: numpy.array; vector to project
    :param vector_d: numpy.array; direction vector
    :return: numpy.array
    """

    vector_d, vector_a = np.array(vector_d), np.array(vector_a)
    return (np.dot(vector_a, vector_d) / np.dot(vector_d, vector_d)) * vector_d


def cartesian_to_vectors_defined(tn, nn, vector):
    """
    Transform coordinate from standard cartesian 2d defined by unit vector e1 and e2 to coordinate system defined
    by perpendicular vectors 'tn' and 'nn' (tangential and normal vector).
    'tn' represents like new 'x' axis::

        |c1|   | e1 . t    e2 . n |   | v1 |
        |  | = |                  | x |    |
        |c2|   | e1 . n    e2 . t |   | v2 |

    :param tn: numpy.array
    :param nn: numpy.array
    :param vector: numpy.array
    :return: numpy.array
    """

    m = [[np.dot(np.array([1.0, 0.0]), tn), np.dot(np.array([0.0, 1.0]), tn)],
         [np.dot(np.array([1.0, 0.0]), nn), np.dot(np.array([0.0, 1.0]), nn)]]
    return np.dot(m, vector)
