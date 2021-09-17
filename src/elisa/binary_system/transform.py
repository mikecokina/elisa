import numpy as np

from .. import units, const
from .. units import DefaultBinarySystemInputUnits
from .. base.transform import SystemProperties, WHEN_FLOAT64, quantity_transform


class BinarySystemProperties(SystemProperties):
    @staticmethod
    def eccentricity(value):
        """
        Transform and validate eccentricity.

        :param value: Union[(numpy.)int, (numpy.)float]
        :return: float;
        """
        if not isinstance(value, (int, np.int, float, np.float)):
            raise TypeError('Input of variable `eccentricity` is not (numpy.)int or (numpy.)float.')
        if value < 0 or value >= 1:
            raise ValueError('Input of variable `eccentricity` is out of boundaries [0, 1)')
        return np.float64(value)

    @staticmethod
    def argument_of_periastron(value):
        """
        Transform and validate argument of periastron, if unit is not supplied, value in degrees is assumed.

        :param value: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]
        :return: float;
        """
        if isinstance(value, (units.Quantity, str)):
            value = units.Quantity(value) if isinstance(value, str) else value
            value = np.float64(value.to(units.ARC_UNIT))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64((value * DefaultBinarySystemInputUnits.system.argument_of_periastron).to(units.ARC_UNIT))
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

        :param value: float;
        :return: float;
        """
        return np.float64(value)

    @staticmethod
    def primary_minimum_time(value):
        """
        Transform and validity check for time of primary minima.

        :param value: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]
        :return: float;
        """
        return quantity_transform(value, units.PERIOD_UNIT, WHEN_FLOAT64)

    @classmethod
    def t0(cls, value):
        return cls.primary_minimum_time(value)


class RadialVelocityObserverProperties(SystemProperties):
    eccentricity = BinarySystemProperties.eccentricity
    argument_of_periastron = BinarySystemProperties.argument_of_periastron
    period = BinarySystemProperties.period
    gamma = SystemProperties.gamma

    @staticmethod
    def mass_ratio(value):
        """
        Validate mass ratio.

        :param value: float;
        :return: float;
        """
        if not value > 0:
            raise ValueError(f"Invalid value of property `mass_ratio`. Expected > 0, given {value}")
        return np.float(value)

    @staticmethod
    def asini(value):
        """
        Transform and validate asini. If value is supplied without unit then default unit is assumed to be solar radii.

        :param value: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]
        :return: float;
        """
        if isinstance(value, (units.Quantity, str)):
            value = units.Quantity(value) if isinstance(value, str) else value
            value = np.float64(value.to(units.solRad))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64(value)
        else:
            raise TypeError('Input of variable `asini` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        if value < 0:
            raise ValueError('Value of `asini` cannot be negative.')
        return value
