from ... import units as u
from elisa.base.transform import (
    TransformProperties,
    quantity_transform,
    WHEN_FLOAT64
)


class OrbitProperties(TransformProperties):
    @staticmethod
    def rotational_period(value):
        """
        Transform and validate rotational period of single star system.
        If unit is not specified, default period unit is assumed.

        :param value: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]
        :return: float
        """
        return quantity_transform(value, u.PERIOD_UNIT, WHEN_FLOAT64)

    @staticmethod
    def inclination(value):
        """
        If unitless values is supplied, default units suppose to be radians.

        :param value: Union[float, astropy.units.Quantity]
        :return: float
        """
        return quantity_transform(value, u.ARC_UNIT, WHEN_FLOAT64)
