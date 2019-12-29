import numpy as np


def ballesteros_formula(color_bv):
    """
    Ballesteros formula.
    Estimate start temperature from B-V index.

    color_bv: B - V = -2.5 * log(F_b / F_v)

    :param color_bv: float; B-V index
    :return: float; temperature stimation
    """
    return 4600.0 * ((1.0 / ((0.92 * color_bv) + 1.7)) + (1.0 / ((0.92 * color_bv) + 0.62)))


def pogsons_formula(f1, f2):
    """
    return -2.5 * log(f1 / f2)

    :param f1: Union[float, numpy.array];
    :param f2: Union[float, numpy.array];
    :return: Union[float, numpy.array];
    """

    return -2.5 * np.log10(f1 / f2)


def _overcontact_temperature_estimation(b_v):
    a, b, c = 1270.92384801, 1.73588834, 1290.17377233
    return (a / np.log10((b + b_v))) + c


def _solar_like_temperature_estimation(b_v):
    a, b, c = 1768.60111726, 1.78240258, -264.71844987
    return (a / np.log10((b + b_v))) + c


def elisa_bv_temperature(b_v, morphology="detached"):
    """
    Temperature estimation for elisa based on B-V index.

    :param b_v: Union[float, numpy.array]; B-V index/indices
    :param morphology: str;
    :return: Union[float, numpy.array]; temperature/s estimation
    """

    if morphology in ['overcontact']:
        return _overcontact_temperature_estimation(b_v)
    return _solar_like_temperature_estimation(b_v)
