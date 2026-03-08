from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elisa import units as u
from elisa.base.transform import (
    WHEN_FLOAT64,
    SystemProperties,
    quantity_transform,
)

if TYPE_CHECKING:
    from elisa.types import Float


class SingleSystemProperties(SystemProperties):
    """Transformation helpers for single-system properties."""

    @staticmethod
    def rotation_period(value: Any) -> Float:
        """Transform and validate the rotational period of a star in a single-star system.

        If the input unit is not specified, the default input unit for the
        single-system rotation period is assumed.

        :param value: Rotation period as a numeric value or quantity.
        :type value: Float | int | astropy.units.quantity.Quantity
        :return: Transformed rotation period in the default internal unit.
        :rtype: Float
        """
        return quantity_transform(
            value,
            u.DefaultSingleSystemUnits.system.rotation_period,
            WHEN_FLOAT64,
            u.DefaultSingleSystemInputUnits.system.rotation_period,
        )

    @staticmethod
    def reference_time(value: Any) -> Float:
        """Transform and validate the reference time.

        If the input unit is not specified, the default input unit for the
        single-system reference time is assumed.

        :param value: Reference time as a numeric value or quantity.
        :type value: Float | int | astropy.units.quantity.Quantity
        :return: Transformed reference time in the default internal unit.
        :rtype: Float
        """
        return quantity_transform(
            value,
            u.DefaultSingleSystemUnits.system.reference_time,
            WHEN_FLOAT64,
            u.DefaultSingleSystemInputUnits.system.reference_time,
        )
