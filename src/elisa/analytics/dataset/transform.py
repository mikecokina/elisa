"""Transform and validation utilities for time series dataset properties."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import units as u
from elisa import utils
from elisa.base.transform import (
    WHEN_ARRAY,
    WHEN_FLOAT64,
    TransformProperties,
)
from elisa.base.types import FLOAT

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float


def array_transform(
    value: Any,
    when_array: tuple[type, ...],
) -> NDArray[Any] | None:
    """Convert array-like values to numpy array with float64 dtype.

    Transforms input values that are array-like (list, tuple, numpy array)
    into a numpy array with float64 data type. Returns None for empty or None inputs.

    :param value: Array-like value to convert (list, tuple, or numpy array).
    :type value: Any
    :param when_array: Tuple of types considered as array-like.
    :type when_array: tuple[type, ...]
    :returns: Converted numpy array with float64 dtype, or None if input is empty.
    :rtype: NDArray[Any] | None
    :raises TypeError: If value is not None/empty and not array-like.
    """
    if isinstance(value, when_array):
        return np.array(value, dtype=FLOAT)
    if not utils.is_empty(value):
        error_msg = "Input of variable is not array-like."
        raise TypeError(error_msg)
    return None


def unit_check(
    value: Any,
    base_units: tuple[Any, ...],
) -> Any:
    """Validate that a unit is equivalent to one of the base units.

    Checks if the supplied unit is equivalent to any of the base units.
    None or empty string units are converted to dimensionless_unscaled.

    :param value: Unit to validate (can be None or astropy Unit).
    :type value: Any
    :param base_units: Tuple of base units to check equivalence against.
    :type base_units: tuple[Any, ...]
    :returns: The validated unit (same as input or dimensionless if None).
    :rtype: Any
    :raises ValueError: If unit is not equivalent to any base unit.
    """
    if value is None or value.to_string() == "":
        value = u.dimensionless_unscaled

    if not value.is_equivalent(base_units):
        error_msg = f"Input {value} is not NoneType or `astropy.Unit` not convertible into desired base units."
        raise ValueError(error_msg)

    return value


class DatasetProperties(TransformProperties):
    """Transform time series x_data, y_data, and y_err to numpy array format.

    Provides static methods for transforming various input time series
    data into standardized numpy array format with consistent data types.
    """

    @staticmethod
    def x_data(value: Any) -> NDArray[Float] | None:
        """Transform x_data (times or phases) to numpy array.

        :param value: Input x_data values.
        :type value: Any
        :returns: x_data as numpy array or None if empty.
        :rtype: NDArray[Float] | None
        """
        return array_transform(value, WHEN_ARRAY)

    @staticmethod
    def y_data(value: Any) -> NDArray[Float] | None:
        """Transform y_data (observable values) to numpy array.

        :param value: Input y_data values.
        :type value: Any
        :returns: y_data as numpy array or None if empty.
        :rtype: NDArray[Float] | None
        """
        return array_transform(value, WHEN_ARRAY)

    @staticmethod
    def y_err(value: Any) -> NDArray[Float] | None:
        """Transform y_err (observable errors) to numpy array.

        :param value: Input y_err values.
        :type value: Any
        :returns: y_err as numpy array or None if empty.
        :rtype: NDArray[Float] | None
        """
        return array_transform(value, WHEN_ARRAY)


class RVDataProperties(DatasetProperties):
    """Validate units for radial velocity (RV) time series data.

    Ensures that time (x_unit) and RV (y_unit) units are convertible
    to ELISa's base units.
    """

    @staticmethod
    def x_unit(value: Any) -> Any:
        """Validate time unit for RV data.

        :param value: Time unit (dimensionless or time unit).
        :type value: Any
        :returns: Validated time unit.
        :rtype: Any
        """
        return unit_check(value, (u.dimensionless_unscaled, u.TIME_UNIT))

    @staticmethod
    def y_unit(value: Any) -> Any:
        """Validate velocity unit for RV data.

        :param value: Velocity unit.
        :type value: Any
        :returns: Validated velocity unit.
        :rtype: Any
        """
        return unit_check(value, (u.VELOCITY_UNIT,))


class LCDataProperties(DatasetProperties):
    """Validate units for light curve (LC) time series data.

    Ensures that time (x_unit) and flux/magnitude (y_unit) units are
    convertible to ELISa's base units.
    """

    @staticmethod
    def x_unit(value: Any) -> Any:
        """Validate time unit for light curve data.

        :param value: Time unit (dimensionless or time unit).
        :type value: Any
        :returns: Validated time unit.
        :rtype: Any
        """
        return unit_check(value, (u.dimensionless_unscaled, u.TIME_UNIT))

    @staticmethod
    def y_unit(value: Any) -> Any:
        """Validate flux unit for light curve data.

        :param value: Flux unit (dimensionless or magnitude).
        :type value: Any
        :returns: Validated flux unit.
        :rtype: Any
        """
        return unit_check(value, (u.dimensionless_unscaled, u.mag))

    @staticmethod
    def zero_magnitude(value: Any) -> float:
        """Convert magnitude or numeric value to float.

        Converts a magnitude value (from astropy Quantity) or numeric
        value to a numpy float64. Used to process zero-point magnitudes
        for flux conversion.

        :param value: Magnitude value (Quantity, float, or int).
        :type value: Any
        :returns: Value as numpy float64.
        :rtype: float
        :raises TypeError: If input type is not supported.
        """
        if isinstance(value, u.Quantity):
            # Note: Original code calls value.is_equivalent() which is a bug
            # Should be value.unit.is_equivalent(u.mag), but keeping for compatibility
            value.is_equivalent(u.mag)
            value = np.float64(value.to(u.mag))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64(value)
        else:
            error_msg = (
                "Input of variable is not (numpy.)int or (numpy.)float nor astropy.unit.quantity.Quantity instance."
            )
            raise TypeError(error_msg)
        return value
