import numpy as np

from .. import const as c, units as u
from .. base.transform import SystemProperties, quantity_transform, deg_transform, WHEN_FLOAT64


class PulsationModeProperties(SystemProperties):
    @staticmethod
    def l(value):
        """
        Returns angular degree of the pulsation mode.

        :param value: int;
        :return: int; validated value of angular degree
        """
        if int(value) - value == 0:
            value = int(value)
        if not isinstance(value, (int, np.int)):
            raise TypeError('Angular degree `l` is not (numpy.)int ')
        return value

    @staticmethod
    def m(value):
        """
        Returns azimuthal order of the pulsation mode.

        :param value: int;
        :return: int; validated value of azimuthal order
        """
        if int(value) - value == 0:
            value = int(value)
        if not isinstance(value, (int, np.int)):
            raise TypeError('Angular degree `m` is not (numpy.)int ')
        return value

    @staticmethod
    def amplitude(value):
        """
        Returns evaluated radial velocity amplitude of pulsation mode.

        :param value: Union[float, astropy.unit.quantity.units.Quantity]
        :return: float;
        """
        if isinstance(value, (u.Quantity, str)):
            retval = u.Quantity(value) if isinstance(value, str) else value
            retval = np.float64(retval.to(u.VELOCITY_UNIT))
        elif isinstance(value, (int, np.int, float, np.float)):
            retval = np.float(value)
        else:
            raise TypeError('Value of `amplitude` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        if retval < 0:
            raise ValueError('Temperature amplitude of mode has to be non-negative number.')

        return retval

    @staticmethod
    def frequency(value):
        """
        Returns evaluated frequency of pulsation mode.

        :param value: Union[float, astropy.unit.quantity.units.Quantity]
        :return: float;
        """
        if isinstance(value, (u.Quantity, str)):
            retval = u.Quantity(value) if isinstance(value, str) else value
            retval = np.float64(retval.to(u.FREQUENCY_UNIT))
        elif isinstance(value, (int, np.int, float, np.float)):
            retval = (np.float(value) * u.DefaultPulsationsInputUnits['frequency']).to(u.FREQUENCY_UNIT).value
        else:
            raise TypeError('Value of `frequency` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        if retval < 0:
            raise ValueError('Frequency of the mode has to be non-negative number.')

        return retval

    @staticmethod
    def start_phase(value):
        """
        Returns phase shift of the given pulsation mode.

        :param value: float;
        :return: float;
        """
        return quantity_transform(value, u.ARC_UNIT, WHEN_FLOAT64)

    @staticmethod
    def mode_axis_theta(value):
        """
        Evaluates value for latitude of pulsation mode axis.
        If unit is not supplied, degrees are assumed.

        :param value: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.units.units.Quantity]
        :return: float;
        """
        retval = deg_transform(value, u.ARC_UNIT, WHEN_FLOAT64)
        if not 0 <= retval < c.PI:
            raise ValueError(f'Value of `mode_axis_theta`: {retval} is outside bounds (0, pi).')

        return retval

    @staticmethod
    def mode_axis_phi(value):
        """
        Evaluates longitude of pulsation mode axis.
        If unit is not supplied, degrees are assumed.

        :param value: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]
        :return: float;
        """
        return deg_transform(value, u.ARC_UNIT, WHEN_FLOAT64)

    @staticmethod
    def temperature_perturbation_phase_shift(value):
        """
        evaluates a phase shift between surface geometry perturbation and temperature perturbations

        :param value: float; phase shift in radians
        :return:
        """
        return deg_transform(value, u.ARC_UNIT, WHEN_FLOAT64)

    @staticmethod
    def horizontal_to_radial_amplitude_ratio(value):
        """
        Defines amplitude ratio between radial and horizontal displacement.

        :param value: float; phase shift in radians
        :return:
        """
        if not isinstance(value, (int, np.int, float, np.float)):
            raise TypeError('Parameter is not is not (numpy.)int or (numpy.)float')
        return value

    @staticmethod
    def tidally_locked(value):
        """
        Defines whether the pulsation mode is fixed with respect to the tidal axis.
        If false, the mode axis will drift with the stellar surface.

        :param value: bool;
        :return: bool
        """
        if not isinstance(value, bool):
            raise TypeError('Parameter `tidally_locked` can contain only boolean')
        return value
