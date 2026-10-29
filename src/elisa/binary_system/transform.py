from __future__ import annotations

from typing import Any

import numpy as np

from elisa import const, units
from elisa.base.error import ElisaParametersValidationError
from elisa.base.transform import WHEN_FLOAT64, SystemProperties, quantity_transform
from elisa.base.types import FLOAT, INT
from elisa.units import DefaultBinarySystemInputUnits


class BinarySystemProperties(SystemProperties):
    @staticmethod
    def eccentricity(value: Any) -> float:
        """Validate and transform eccentricity.

        :param value: Numeric eccentricity value.
        :returns: Eccentricity as float.
        """
        if not isinstance(value, (int, INT, float, FLOAT)):
            msg = "Input of variable `eccentricity` is not (numpy.)int or (numpy.)float."
            raise ElisaParametersValidationError(msg)
        if not np.isfinite(value):
            msg = "Nonfinite eccentricity."
            raise ElisaParametersValidationError(msg)
        if value < 0 or value >= 1:
            msg = "Input of variable `eccentricity` is out of boundaries [0, 1)."
            raise ElisaParametersValidationError(msg)
        return float(np.float64(value))

    @staticmethod
    def argument_of_periastron(value: Any) -> float:
        """Validate and transform argument of periastron.

        If no unit is supplied, degrees are assumed.

        :param value: Numeric value or astropy Quantity.
        :returns: Argument of periastron as float.
        """
        if isinstance(value, (units.Quantity, str)):
            value = units.Quantity(value) if isinstance(value, str) else value
            value = np.float64(value.to(units.DefaultBinarySystemUnits.system.argument_of_periastron))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64(
                (value * DefaultBinarySystemInputUnits.system.argument_of_periastron).to(
                    units.DefaultBinarySystemUnits.system.argument_of_periastron,
                ),
            )
        else:
            msg = (
                "Input of variable `argument_of_periastron` is not (numpy.)int or (numpy.)float "
                "nor astropy.unit.quantity.Quantity instance."
            )
            raise ElisaParametersValidationError(msg)
        if not np.isfinite(value):
            msg = "Nonfinite argument_of_periastron."
            raise ElisaParametersValidationError(msg)
        if not 0 <= value <= const.FULL_ARC:
            value %= const.FULL_ARC
        return float(value)

    @staticmethod
    def phase_shift(value: Any) -> float:
        """Return phase shift of the primary eclipse minimum.

        The phase shift is used during calculations where:
        true_phase = phase + phase_shift.

        :param value: Phase shift value.
        :returns: Phase shift as float.
        """
        value = float(np.float64(value))
        if not np.isfinite(value):
            msg = "Nonfinite phase_shift."
            raise ElisaParametersValidationError(msg)
        return value

    @staticmethod
    def primary_minimum_time(value: Any) -> float:
        """Transform and validate time of primary minimum.

        :param value: Numeric value or astropy Quantity.
        :returns: Time of primary minimum as float.
        """
        value = quantity_transform(
            value,
            units.DefaultBinarySystemUnits.system.primary_minimum_time,
            WHEN_FLOAT64,
            units.DefaultBinarySystemInputUnits.system.primary_minimum_time,
        )
        if not np.isfinite(value):
            msg = "Nonfinite primary_minimum_time."
            raise ElisaParametersValidationError(msg)
        return value

    @classmethod
    def t0(cls, value: Any) -> float:
        """Alias for primary_minimum_time.

        :param value: Time value.
        :returns: Time of primary minimum as float.
        """
        return cls.primary_minimum_time(value)


class RadialVelocityObserverProperties(SystemProperties):
    """Properties for radial velocity observer calculations."""

    eccentricity = BinarySystemProperties.eccentricity
    argument_of_periastron = BinarySystemProperties.argument_of_periastron
    period = BinarySystemProperties.period
    gamma = SystemProperties.gamma

    @staticmethod
    def mass_ratio(value: Any) -> float:
        """Validate mass ratio.

        :param value: Mass ratio value (must be > 0).
        :returns: Mass ratio as float.
        """
        if not np.isfinite(value):
            msg = "Nonfinite mass_ratio."
            raise ElisaParametersValidationError(msg)
        if not value > 0:
            msg = f"Invalid value of property `mass_ratio`. Expected > 0, given {value}."
            raise ElisaParametersValidationError(msg)
        return float(np.float64(value))

    @staticmethod
    def asini(value: Any) -> float:
        """Transform and validate asini parameter.

        If no unit is supplied, solar radii are assumed as default.

        :param value: Numeric value or astropy Quantity.
        :returns: asini value as float.
        """
        if isinstance(value, (units.Quantity, str)):
            value = units.Quantity(value) if isinstance(value, str) else value
            value = np.float64(value.to(units.solRad))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64(value)
        else:
            msg = (
                "Input of variable `asini` is not (numpy.)int or (numpy.)float "
                "nor astropy.unit.quantity.Quantity instance."
            )
            raise ElisaParametersValidationError(msg)
        if not np.isfinite(value):
            msg = "Nonfinite asini."
            raise ElisaParametersValidationError(msg)
        if value < 0:
            msg = "Value of `asini` cannot be negative."
            raise ElisaParametersValidationError(msg)
        return float(value)
