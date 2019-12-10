import numpy as np

from elisa.base import error


def jd_to_phase(t0, period, jd):
    """
    Convert JD time to phase from -1 to 1

    :param t0: float; reference primary minimum time
    :param period: float; period of binary system
    :param jd: Union[float, numpy.array]; measurement JD times
    :return: Union[float, numpy.array]; phases
    """
    if period <= 0:
        raise error.ValidationError("Period has to be > 0.")
    if t0 <= 0:
        raise error.ValidationError("Primary minimum time has to be > 0.")
    if isinstance(jd, list):
        jd = np.array(jd)
    return ((jd - t0) / period) % 1.0
