from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import const
from elisa import units as u
from elisa.base.error import ElisaParametersValidationError
from elisa.base.transform import (
    WHEN_FLOAT64,
    SystemProperties,
    TransformProperties,
    quantity_transform,
)
from elisa.base.types import FLOAT, INT

if TYPE_CHECKING:
    from elisa.types import Float


class OrbitProperties(TransformProperties):
    """Transformation and validation helpers for orbit-related properties."""

    @staticmethod
    def eccentricity(value: int | INT | Float | FLOAT) -> Float:
        """Transform and validate orbital eccentricity.

        The eccentricity must be a numeric scalar in the interval
        ``[0, 1)``.

        :param value: Orbital eccentricity value.
        :type value: int | INT | Float | FLOAT
        :return: Validated eccentricity converted to ``Float``.
        :rtype: Float
        :raises ElisaParametersValidationError: If ``value`` is not a supported numeric scalar.
        :raises ElisaParametersValidationError: If ``value`` is outside the allowed interval
            ``[0, 1)``.
        :raises ElisaParametersValidationError: If ``value`` is nonfinite.
        """
        if not isinstance(value, (int, INT, float, FLOAT)):
            message = (
                "Input of variable `eccentricity` is not "
                "(numpy.)int or (numpy.)float."
            )
            raise ElisaParametersValidationError(message)

        if not np.isfinite(value):
            message = "Nonfinite eccentricity."
            raise ElisaParametersValidationError(message)
        if value < 0 or value >= 1:
            message = (
                "Input of variable `eccentricity` is invalid or out of "
                "boundaries."
            )
            raise ElisaParametersValidationError(message)

        return np.float64(value)

    @staticmethod
    def period(value: Float) -> Float:
        """Transform and validate orbital period of a binary star system.

        If the unit is not specified, the default binary-system period unit is
        assumed.

        :param value: Orbital period value or quantity.
        :type value: Float
        :return: Period converted to the internal ``Float`` form.
        :rtype: Float
        """
        return SystemProperties.period(value)

    @staticmethod
    def argument_of_periastron(value: Float) -> Float:
        """Transform and validate the argument of periastron.

        If the unit is not supplied, the value is assumed to be in radians.

        :param value: Argument of periastron value or quantity.
        :type value: Float
        :return: Argument of periastron converted to internal
            ``Float`` form.
        :rtype: Float
        """
        value = quantity_transform(
            value,
            u.DefaultBinarySystemUnits.system.argument_of_periastron,
            WHEN_FLOAT64,
        )
        if not np.isfinite(value):
            message = "Nonfinite argument_of_periastron."
            raise ElisaParametersValidationError(message)
        return value

    @staticmethod
    def inclination(value: Float) -> Float:
        """Transform and validate orbital inclination.

        If a unitless value is supplied, the default unit is assumed to be
        radians.

        :param value: Inclination value or quantity.
        :type value: Float
        :return: Inclination converted to internal ``Float`` form.
        :rtype: Float
        """
        value = quantity_transform(
            value,
            u.DefaultBinarySystemUnits.system.inclination,
            WHEN_FLOAT64,
        )
        if not np.isfinite(value):
            message = "Nonfinite inclination."
            raise ElisaParametersValidationError(message)
        if not 0 <= value <= const.PI:
            message = f"Inclination value of {value} is out of bounds (0, pi)."
            raise ElisaParametersValidationError(message)
        return value
