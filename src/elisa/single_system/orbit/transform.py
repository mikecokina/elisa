from __future__ import annotations

from typing import TYPE_CHECKING

from elisa import units as u
from elisa.base.transform import WHEN_FLOAT64, TransformProperties, quantity_transform

if TYPE_CHECKING:
    from elisa.types import AstropyQuantity as Quantity
    from elisa.types import Float, Int


class OrbitProperties(TransformProperties):
    """Transform helpers for orbital properties of a single-star system."""

    @staticmethod
    def rotational_period(value: Quantity | Float | Int) -> Float:
        """Transform and validate rotational period of single star system.

        If unit is not specified, the default period unit from
        :attr:`elisa.units.DefaultSingleSystemUnits.system.rotation_period` is assumed.

        :param value: Rotational period as an :class:`astropy.units.Quantity` or a numeric value.
        :type value: Quantity | Float | Int
        :return: Rotational period converted to the default single system units.
        :rtype: Float
        """
        return quantity_transform(value, u.DefaultSingleSystemUnits.system.rotation_period, WHEN_FLOAT64)

    @staticmethod
    def inclination(value: Quantity | Float | Int) -> Float:
        """Transform and validate system inclination.

        If a unitless value is supplied, radians are assumed by default.

        :param value: Inclination as an :class:`astropy.units.Quantity` or a numeric value.
        :type value: Quantity | Float | Int
        :return: Inclination converted to the default single system units (radians).
        :rtype: Float
        """
        return quantity_transform(value, u.DefaultSingleSystemUnits.system.inclination, WHEN_FLOAT64)
