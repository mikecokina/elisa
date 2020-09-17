from .. import units as u
from .. base.transform import SystemProperties, WHEN_FLOAT64, quantity_transform


class SingleSystemProperties(SystemProperties):
    @staticmethod
    def rotation_period(value):
        """
        Transform and validate rotational period of star in single star system, if unit is not specified, default period
        unit is assumed
        :param value: quantity or float; rotation period
        :return: float
        """
        return quantity_transform(value, u.PERIOD_UNIT, WHEN_FLOAT64)

    @staticmethod
    def reference_time(value):
        """
        Transform and validity check for reference time.

        :param value: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]
        :return: float
        """
        return quantity_transform(value, u.PERIOD_UNIT, WHEN_FLOAT64)
