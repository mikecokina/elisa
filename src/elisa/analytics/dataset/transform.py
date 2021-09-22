import numpy as np

from ... import units as u
from ... base.transform import (
    TransformProperties,
    WHEN_FLOAT64,
    WHEN_ARRAY
)
from elisa.utils import is_empty


def array_transform(value, when_array):
    """
    Check whether `value` is array-like and then transforms it to numpy.array.

    :param value: Union[(numpy.)array, list, tuple];
    :param when_array: Tuple(Types);
    :return: numpy.array;
    """
    if isinstance(value, when_array):
        return np.array(value, dtype=np.float)
    elif not is_empty(value):
        raise TypeError('Input of variable is not array-like.')


def unit_check(value, base_units):
    """
    Checking if the supplied unit is equivalent to the base unit for given variable.

    :param value: Union[str, astropy.unit.Unit]; unit which compatibility will be checked
    :param base_units: List[astropy.unit.Unit]; base units of given parameter
    :return: astropy.unit.Unit;
    """
    if value is None or value.to_string() == '':
        value = u.dimensionless_unscaled

    if not value.is_equivalent(base_units):
        raise ValueError(f'Input {value} is not NoneType or `astropy.Unit` not convertible into desired base units.')

    return value


class DatasetProperties(TransformProperties):
    """
    Transforming various input time series x,y and y_err to numpy.array format.
    """
    @staticmethod
    def x_data(value):
        return array_transform(value, WHEN_ARRAY)

    @staticmethod
    def y_data(value):
        return array_transform(value, WHEN_ARRAY)

    @staticmethod
    def y_err(value):
        return array_transform(value, WHEN_ARRAY)


class RVDataProperties(DatasetProperties):
    """
    Making sure that time and RV units are convertible to the ELISa's base units.
    """
    @staticmethod
    def x_unit(value):
        return unit_check(value, (u.dimensionless_unscaled, u.TIME_UNIT))

    @staticmethod
    def y_unit(value):
        return unit_check(value, (u.VELOCITY_UNIT,))


class LCDataProperties(DatasetProperties):
    """
    Making sure that time and LC units are convertible to the ELISa's base units.
    """
    @staticmethod
    def x_unit(value):
        return unit_check(value, (u.dimensionless_unscaled, u.TIME_UNIT))

    @staticmethod
    def y_unit(value):
        return unit_check(value, (u.dimensionless_unscaled, u.mag))

    @staticmethod
    def zero_magnitude(value):
        if isinstance(value, u.Quantity):
            value.is_equivalent(u.mag)
            value = np.float64(value.to(u.mag))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64(value)
        else:
            raise TypeError('Input of variable is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        return value
