import numpy as np

from ... import units as u
from ... base.transform import (
    TransformProperties,
    WHEN_FLOAT64,
    WHEN_ARRAY
)


def array_transform(value, when_array):
    """
    Check whether `value` is array like and then transforms it to numpy.array.

    :param value: Union[(numpy.)array, list, tuple];
    :param when_array: Tuple(Types);
    :return: numpy.array;
    """
    if isinstance(value, when_array):
        return np.array(value, dtype=np.float)
    else:
        raise TypeError('Input of variable is not array-like.')


def unit_transform(value, base_units):
    if value is None or value.to_string() == '':
        value = u.dimensionless_unscaled

    if value.is_equivalent(base_units):
        return value
    else:
        raise ValueError(f'Input {value} is not NoneType or `astropy.Unit` not convertible into desired base units.')


class DatasetProperties(TransformProperties):
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
    @staticmethod
    def x_unit(value):
        return unit_transform(value, (u.dimensionless_unscaled, u.TIME_UNIT))

    @staticmethod
    def y_unit(value):
        return unit_transform(value, (u.VELOCITY_UNIT,))


class LCDataProperties(DatasetProperties):
    @staticmethod
    def x_unit(value):
        return unit_transform(value, (u.dimensionless_unscaled, u.TIME_UNIT))

    @staticmethod
    def y_unit(value):
        return unit_transform(value, (u.dimensionless_unscaled, u.mag))

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
