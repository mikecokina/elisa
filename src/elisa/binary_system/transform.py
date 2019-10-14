import numpy as np
import sys

from astropy import units as au
from elisa import units, const

WHEN_FLOAT64 = (int, np.int, float, np.float)
MODULE = sys.modules[__name__]


def quantity_transform(value, unit, when_float64):
    if isinstance(value, au.quantity.Quantity):
        value = np.float64(value.to(unit))
    elif isinstance(value, when_float64):
        value = np.float64(value)
    else:
        raise TypeError('Input of variable is not (numpy.)int or (numpy.)float '
                        'nor astropy.unit.quantity.Quantity instance.')
    return value


def period(value):
    """
    Transform and validate orbital period of binary star system.
    If unit is not specified, default period unit is assumed.

    :param value: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
    :return: float
    """
    return quantity_transform(value, units.PERIOD_UNIT, WHEN_FLOAT64)


def eccentricity(value):
    """
    Transform and validate eccentricity.

    :param value: (numpy.)int, (numpy.)float
    :return:
    """
    if not isinstance(value, (int, np.int, float, np.float)):
        raise TypeError('Input of variable `eccentricity` is not (numpy.)int or (numpy.)float.')
    if value < 0 or value >= 1:
        raise ValueError('Input of variable `eccentricity` is  or it is out of boundaries.')
    return np.float64(value)


def argument_of_periastron(value):
    """
    Transform and validate argument of periastron, if unit is not supplied, value in degrees is assumed.

    :param value: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
    :return:
    """
    if isinstance(value, au.quantity.Quantity):
        value = np.float64(value.to(units.ARC_UNIT))
    elif isinstance(value, WHEN_FLOAT64):
        value = np.float64((value * au.deg).to(units.ARC_UNIT))
    else:
        raise TypeError('Input of variable `argument_of_periastron` is not (numpy.)int or (numpy.)float '
                        'nor astropy.unit.quantity.Quantity instance.')
    if not 0 <= value <= const.FULL_ARC:
        value %= const.FULL_ARC
    return value


def primary_minimum_time(value):
    """
    Transform and validity check for time of primary minima.

    :param value: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
    :return: float
    """
    return quantity_transform(value, units.PERIOD_UNIT, WHEN_FLOAT64)


def transform_binary_input(**kwargs):
    """
    Transform BinarySystem input kwargs to internal representations.

    :param kwargs: Dict
    :return: Dict
    """
    return {key: getattr(MODULE, key)(val) if hasattr(MODULE, key) else val for key, val in kwargs.items()}



