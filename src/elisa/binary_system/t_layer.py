import numpy as np

from elisa.base import error


def validate_period(period):
    if period <= 0:
        raise error.ValidationError("Period has to be > 0.")


def validate_pirmary_minimum_time(t0):
    if t0 <= 0:
        raise error.ValidationError("Primary minimum time has to be > 0.")


def jd_to_phase(t0, period, jd):
    """
    Convert JD time to phase from -1 to 1

    :param t0: float; reference primary minimum time
    :param period: float; period of binary system
    :param jd: Union[float, numpy.array]; measurement JD times
    :return: Union[float, numpy.array]; phases
    """
    validate_period(period)
    validate_pirmary_minimum_time(t0)

    if isinstance(jd, list):
        jd = np.array(jd)
    return ((jd - t0) / period) % 1.0


def phase_to_jd(t0, period, phases):
    """
    Convert phase to JD time

    :param t0: float; reference primary minimum time
    :param period: float; period of binary system
    :param phases: Union[float, numpy.array]; phases
    :return: Union[float, numpy.array]; phases
    """
    validate_period(period)
    validate_pirmary_minimum_time(t0)

    if isinstance(phases, list):
        phases = np.array(phases)
    return (period * phases) + t0
