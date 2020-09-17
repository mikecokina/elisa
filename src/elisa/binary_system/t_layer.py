import numpy as np
from .. base import error


def validate_period(period):
    if period <= 0:
        raise error.ValidationError("Period has to be > 0.")


def validate_pirmary_minimum_time(t0):
    if t0 <= 0:
        raise error.ValidationError("Primary minimum time has to be > 0.")


def adjust_phases(phases, centre=0.5):
    """
    shift phases to centre them on given value

    :param phases: Union[float, numpy.array];
    :param centre: float; centre around which phases will be calculated (+-0.5 around `centre` value)
    :return:
    """
    if isinstance(phases, list):
        phases = np.array(phases)
    shift = centre - 0.5
    return (phases - shift) % 1.0 + shift


def jd_to_phase(t0, period, jd, centre=0.5):
    """
    Convert JD time to phase

    :param centre: float; centre around which phases will be calculated (+-0.5 around `centre` value)
    :param t0: float; reference primary minimum time
    :param period: float; period of binary system
    :param jd: Union[float, numpy.array]; measurement JD times
    :return: Union[float, numpy.array]; phases
    """
    validate_period(period)
    validate_pirmary_minimum_time(t0)

    if isinstance(jd, list):
        jd = np.array(jd)
    shift = centre - 0.5
    return (((jd - t0) / period) - shift) % 1.0 + shift


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
