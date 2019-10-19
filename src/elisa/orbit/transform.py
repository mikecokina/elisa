import numpy as np

from elisa import units
from elisa.base.transform import TransformProperties, WHEN_FLOAT64, quantity_transform


class OrbitProperties(TransformProperties):
    #
    @staticmethod
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

    @staticmethod
    def period(value):
        """
        Transform and validate orbital period of binary star system.
        If unit is not specified, default period unit is assumed.
        :param value: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
        :return: float
        """
        return quantity_transform(value, units.PERIOD_UNIT, WHEN_FLOAT64)

    @staticmethod
    def argument_of_periastron(value):
        """
        If unit is not supplied, value in radians is assumed.
        :param value: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
        :return:
        """
        return quantity_transform(value, units.ARC_UNIT, WHEN_FLOAT64)

    @staticmethod
    def inclination(value):
        """
        If unitless values is supplied, default units suppose to be radians.
        :param value: float or astropy.units.Quantity
        :return: float
        """
        return quantity_transform(value, units.ARC_UNIT, WHEN_FLOAT64)
