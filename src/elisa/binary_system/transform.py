import numpy as np

from astropy import units as au
from elisa import units, const
from elisa.base.transform import SystemParameters, WHEN_FLOAT64, quantity_transform


class BinarySystemParameters(SystemParameters):
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
    def argument_of_periastron(value):
        """
        Transform and validate argument of periastron, if unit is not supplied, value in degrees is assumed.

        :param value: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(value, au.quantity.Quantity):
            value = np.float64(value.to(units.ARC_UNIT))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64((value * units.deg).to(units.ARC_UNIT))
        else:
            raise TypeError('Input of variable `argument_of_periastron` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        if not 0 <= value <= const.FULL_ARC:
            value %= const.FULL_ARC
        return value

    @staticmethod
    def phase_shift(value):
        """
        Returns phase shift of the primary eclipse minimum with respect to ephemeris
        true_phase is used during calculations, where: true_phase = phase + phase_shift.

        :param value: float
        :return: float
        """
        return np.float64(value)

    @staticmethod
    def primary_minimum_time(value):
        """
        Transform and validity check for time of primary minima.

        :param value: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
        :return: float
        """
        return quantity_transform(value, units.PERIOD_UNIT, WHEN_FLOAT64)
