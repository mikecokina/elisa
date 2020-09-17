import numpy as np

from ... import units as u
from ... base.transform import (
    TransformProperties,
    quantity_transform,
    WHEN_FLOAT64
)


class OrbitProperties(TransformProperties):
    @staticmethod
    def eccentricity(value):
        """
        Transform and validate eccentricity.

        :param value: Union[(numpy.)int, (numpy.)float]
        :return: float
        """
        if not isinstance(value, (int, np.int, float, np.float)):
            raise TypeError('Input of variable `eccentricity` is not (numpy.)int or (numpy.)float.')
        if value < 0 or value >= 1:
            raise ValueError('Input of variable `eccentricity` is  or it is out of boundaries.')
        return np.float64(value)

    @staticmethod
    def period(value):
        """
        Transform and validate orbital period of binary star system.
        If unit is not specified, default period unit is assumed.

        :param value: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]
        :return: float
        """
        return quantity_transform(value, u.PERIOD_UNIT, WHEN_FLOAT64)

    @staticmethod
    def argument_of_periastron(value):
        """
        If unit is not supplied, value in radians is assumed.

        :param value: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]
        :return: float
        """
        return quantity_transform(value, u.ARC_UNIT, WHEN_FLOAT64)

    @staticmethod
    def inclination(value):
        """
        If unitless values is supplied, default units suppose to be radians.

        :param value: Union[float, astropy.units.Quantity]
        :return: float
        """
        return quantity_transform(value, u.ARC_UNIT, WHEN_FLOAT64)
