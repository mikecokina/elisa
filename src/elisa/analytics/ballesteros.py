import numpy as np


def ballesteros_formula(color_bv):
    """
    Ballesteros formula.
    Estimate start temperature from B-V index.

    color_bv: B - V = -2.5 * log(F_b / F_v)

    :param color_bv: float; B-V index
    :return: float; temperature stimation
    """
    return 4600.0 * ((1.0 / (0.92 * color_bv + 1.7)) + (1.0 / (0.92 * color_bv + 0.62)))


def pogsons_formula(f1, f2):
    """
    return -2.5 * log(f1 / f2)

    :param f1: Union[float, numpy.array];
    :param f2: Union[float, numpy.array];
    :return: Union[float, numpy.array];
    """

    return -2.5 * np.log(f1 / f2)
