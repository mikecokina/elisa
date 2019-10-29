import numpy as np
from elisa import utils, umpy as up


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
