import numpy as np

from astropy import units as u

from elisa.base.transform import TransformProperties
from elisa import units


WHEN_ARRAY = (list, np.ndarray, tuple)


def array_transform(value, when_array):
    """
    check whether `value` is array like and then transforms it to numpy array
    :param value: Union[(numpy.)array, list, tuple];
    :param when_array: Tuple(Types);
    :return:
    """
    if isinstance(value, when_array):
        return np.array(value, dtype=np.float)
    else:
        raise TypeError('Input of variable is not array-like.')


def unit_transform(value, base_units):
    if value is None:
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
    def yerr(value):
        return array_transform(value, WHEN_ARRAY)


class RVdataProperties(DatasetProperties):
    @staticmethod
    def x_unit(value):
        return unit_transform(value, (u.dimensionless_unscaled, units.TIME_UNIT))

    @staticmethod
    def y_unit(value):
        return unit_transform(value, (units.VELOCITY_UNIT,))
