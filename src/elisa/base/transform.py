from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from packaging import version

from elisa import const, settings
from elisa import units as u
from elisa.base.types import FLOAT, INT
from elisa.units import (
    DefaultBinarySystemInputUnits,
    DefaultStarInputUnits,
    DefaultSystemInputUnits,
    DefaultSystemUnits,
)

if TYPE_CHECKING:
    from elisa.types import Float

WHEN_FLOAT64: tuple[type, ...] = (
    int,
    np.int32,
    np.int64,
    float,
    np.float32,
    np.float64,
)
if version.parse(np.__version__) < version.parse("1.20.0"):
    # noinspection PyUnresolvedReferences
    WHEN_FLOAT64 += (int, float)

WHEN_ARRAY = (list, np.ndarray, tuple)


def quantity_transform(
    value: Any,
    unit: Any,
    when_float64: tuple[type, ...] = WHEN_FLOAT64,
    default_input_unit: Any | None = None,
) -> Float:
    """Transform a value into a floating-point value expressed in ``unit``.

    Accepts plain numbers, numpy scalars/arrays, astropy.Quantity or a
    string representation of a quantity. If the input has no attached
    units and ``default_input_unit`` is provided, the numeric value is
    interpreted using that unit before conversion.

    :param value: Input value to convert.
    :param unit: Target unit (astropy.units compatible).
    :param when_float64: Tuple of numeric types treated as unitless numbers.
    :param default_input_unit: Assumed unit for numbers without explicit
        units.
    :returns: Value converted to ``unit`` as Python float.
    :rtype: float
    :raises TypeError: When the input type is not supported.
    """
    if isinstance(value, (u.Quantity, str)):
        value = u.Quantity(value) if isinstance(value, str) else value
        value = np.float64(value.to(unit))
    elif isinstance(value, when_float64):
        if default_input_unit is None:
            value = np.float64(value)
        else:
            value = (value * default_input_unit).to(unit).value
            value = np.float64(value)
    else:
        msg = (
            "Input of variable is not (numpy.)int or (numpy.)float nor "
            "astropy.unit.quantity.Quantity instance (or its string representation)."
        )
        raise TypeError(msg)
    return float(value)


def deg_transform(
    value: Any,
    unit: Any,
    when_float64: tuple[type, ...],
    default_input_unit: Any = u.deg,
) -> Float:
    """Transform an angular value into ``unit`` and return as float.

    Similar to :func:`quantity_transform` but uses multiplicative
    conversion for plain numeric inputs.

    :param value: Input angular value.
    :param unit: Target angular unit.
    :param when_float64: Tuple of numeric types treated as unitless numbers.
    :param default_input_unit: Default unit assumed for plain numbers.
    :returns: Converted value in ``unit`` as float.
    :rtype: float
    :raises TypeError: When the input type is not supported.
    """
    if isinstance(value, (u.Quantity, str)):
        value = u.Quantity(value) if isinstance(value, str) else value
        value = np.float64(value.to(unit))
    elif isinstance(value, when_float64):
        value = np.float64(value) * default_input_unit.to(unit)
    else:
        msg = (
            "Input of the angular variable is not (numpy.)int or (numpy.)float "
            "nor astropy.unit.quantity.Quantity instance (or its string representation)."
        )
        raise TypeError(msg)
    return float(value)


class TransformProperties:
    @classmethod
    def transform_input(cls, **kwargs) -> dict:
        """Transform an input keyword-argument mapping to internal values.

        The method looks up a method on the class for each key (same name)
        and applies it if present, otherwise returns the original value.

        :param kwargs: Input keyword arguments.
        :returns: Mapping with transformed values.
        :rtype: dict
        """
        return {key: getattr(cls, key)(val) if hasattr(cls, key) else val for key, val in kwargs.items()}


class SystemProperties(TransformProperties):
    @staticmethod
    def inclination(value: Any) -> Float:
        """Validate and convert system inclination to radians.

        When no unit is supplied the value is assumed to be in degrees.

        :param value: Inclination value.
        :returns: Inclination in radians.
        :rtype: float
        :raises TypeError: If the input type is unsupported.
        :raises ValueError: If the value is outside the valid range [0, pi].
        """
        if isinstance(value, (u.Quantity, str)):
            value = u.Quantity(value) if isinstance(value, str) else value
            value = np.float64(value.to(u.DefaultSystemUnits.inclination))
        elif isinstance(value, (int, INT, float, FLOAT)):
            value = np.float64(
                (value * DefaultSystemInputUnits.inclination).to(DefaultSystemUnits.inclination),
            )
        else:
            msg = (
                "Input of variable `inclination` is not (numpy.)int or (numpy.)float "
                "nor astropy.unit.quantity.Quantity instance (or its string representation)."
            )
            raise TypeError(msg)

        if not 0 <= value <= const.PI:
            msg = f"Inclination value of {value} is out of bounds (0, pi)."
            raise ValueError(msg)
        return float(value)

    @staticmethod
    def period(value: Any) -> Float:
        """Transform and validate orbital period.

        If unit is omitted, the default period input unit is assumed.
        """
        return quantity_transform(value, DefaultSystemUnits.period, WHEN_FLOAT64, u.DefaultSystemInputUnits.period)

    @staticmethod
    def gamma(value: Any) -> Float:
        """Validate and transform systemic velocity (gamma).

        Accepts numeric values or :class:`astropy.units.Quantity` instances.
        """
        return quantity_transform(value, u.DefaultSystemUnits.gamma, WHEN_FLOAT64, u.DefaultSystemInputUnits.gamma)

    @staticmethod
    def additional_light(value: Any) -> Float:
        """Validate additional (third) light fraction in range [0, 1]."""
        if not 0.0 <= value <= 1.0:
            msg = "Invalid value of additional light. Valid values are between 0 and 1."
            raise ValueError(msg)
        return float(value)

    @staticmethod
    def semi_major_axis(value: Any) -> Float:
        """Validate and convert semi-major axis to internal units.

        Accepts numeric or :class:`astropy.units.Quantity` values.
        """
        if isinstance(value, (u.Quantity, str)):
            value = u.Quantity(value) if isinstance(value, str) else value
            value = np.float64(value.to(u.DefaultBinarySystemUnits.system.semi_major_axis))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64(
                value
                * DefaultBinarySystemInputUnits.system.semi_major_axis.to(
                    u.DefaultBinarySystemUnits.system.semi_major_axis,
                ),
            )
        else:
            msg = (
                "User input is not (numpy.)int or (numpy.)float nor "
                "astropy.unit.quantity.Quantity instance (or its string representation)."
            )
            raise TypeError(msg)
        if value <= 0:
            msg = "Invalid value of semi_major_axis, use value > 0!"
            raise ValueError(msg)
        return float(value)

    @staticmethod
    def distance(value: Any) -> Float:
        """Convert system distance to internal distance units and validate.

        Accepts numeric or :class:`astropy.units.Quantity` values.
        """
        if isinstance(value, (u.Quantity, str)):
            value = u.Quantity(value) if isinstance(value, str) else value
            value = np.float64(value.to(u.DefaultSystemUnits.distance))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64(
                value * DefaultBinarySystemInputUnits.system.distance.to(u.DefaultSystemUnits.distance),
            )
        else:
            msg = (
                "User input is not (numpy.)int or (numpy.)float nor "
                "astropy.unit.quantity.Quantity instance (or its string representation)."
            )
            raise TypeError(msg)
        if value <= 0:
            msg = "Invalid value of system's distance, use value > 0!"
            raise ValueError(msg)

        return float(value)


class BodyProperties(TransformProperties):
    @staticmethod
    def synchronicity(value: Any) -> Float:
        """Validate object synchronicity F = omega_rot/omega_orb.

        Expects a positive numeric value.
        """
        if value <= 0:
            msg = "Invalid synchronicity, use value > 0!"
            raise ValueError(msg)
        return float(value)

    @staticmethod
    def albedo(value: Any) -> Float:
        """Validate and transform bolometric albedo (range 0..1)."""
        if value < 0 or value > 1:
            msg = f"Parameter albedo = {value} is out of range <0, 1>"
            raise ValueError(msg)
        return float(value)

    @staticmethod
    def discretization_factor(value: Any) -> Float:
        """Transform discretization factor (default unit: degrees).

        The returned value is expressed in the internal star unit for
        discretization factor.
        """
        value = deg_transform(
            value,
            u.DefaultStarUnits.discretization_factor,
            WHEN_FLOAT64,
            u.DefaultStarInputUnits.discretization_factor,
        )
        if value > const.HALF_PI:
            msg = "Invalid value of alpha parameter. Use value less than 90."
            raise ValueError(msg)
        return float(value)

    @staticmethod
    def t_eff(value: Any) -> Float:
        """Convert effective temperature to internal units.

        If no unit is provided, Kelvin is assumed.
        """
        return float(
            quantity_transform(value, u.DefaultStarUnits.t_eff, WHEN_FLOAT64, u.DefaultStarInputUnits.t_eff),
        )


class StarProperties(BodyProperties):
    @staticmethod
    def equivalent_radius(value: Any) -> Float:
        """Validate and convert equivalent radius to internal units.

        If quantity is not provided, the default distance unit is assumed.
        """
        if isinstance(value, (u.Quantity, str)):
            value = u.Quantity(value) if isinstance(value, str) else value
            value = np.float64(value.to(u.DefaultStarUnits.equivalent_radius))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64(
                value * DefaultStarInputUnits.equivalent_radius.to(u.DefaultStarUnits.equivalent_radius),
            )
        else:
            msg = (
                "User input is not (numpy.)int or (numpy.)float nor "
                "astropy.unit.quantity.Quantity instance (or its string representation)."
            )
            raise TypeError(msg)
        if value <= 0:
            msg = "Invalid value of equivalent_radius, use value > 0!"
            raise ValueError(msg)
        return float(value)

    @staticmethod
    def mass(value: Any) -> Float:
        """Validate and convert mass to internal units.

        If mass is provided as a plain numeric value it is interpreted in
        solar masses unless a Quantity is supplied.
        """
        if isinstance(value, (u.Quantity, str)):
            value = u.Quantity(value) if isinstance(value, str) else value
            value = np.float64(value.to(u.DefaultStarUnits.mass))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64(value * DefaultStarInputUnits.mass.to(u.DefaultStarUnits.mass))
        else:
            msg = (
                "User input is not (numpy.)int or (numpy.)float nor "
                "astropy.unit.quantity.Quantity instance (or its string representation)."
            )
            raise TypeError(msg)
        if value <= 0:
            msg = "Invalid mass, use value > 0!"
            raise ValueError(msg)
        return float(value)

    @staticmethod
    def surface_potential(value: Any) -> Float:
        """Return the surface potential of the star (validated positive).

        :param value: Numeric surface potential.
        :returns: Surface potential as float.
        """
        if value < 0:
            msg = "Invalid surface potential, use value > 0!"
            raise ValueError(msg)
        return float(value)

    @staticmethod
    def metallicity(value: Any) -> Float:
        """Validate metallicity input as numeric.

        :param value: Numeric metallicity value.
        :returns: Metallicty as float.
        """
        if not isinstance(value, WHEN_FLOAT64):
            msg = "Input of variable `metallicity` is not (np.)int or (np.)float instance."
            raise TypeError(msg)
        return float(value)

    @staticmethod
    def polar_log_g(value: Any) -> Float:
        """Convert polar surface gravity (log g) to internal units.

        If the input is a string or quantity it is converted appropriately;
        numeric inputs are treated as log g in cgs unless a Quantity is
        provided.
        """
        if isinstance(value, (u.Quantity, str)):
            value = u.Dex(value, unit=u.Unit(" ".join(value.split()[1:]))) if isinstance(value, str) else value
            value = np.float64(value.to(u.LOG_ACCELERATION_UNIT))
        elif isinstance(value, WHEN_FLOAT64):
            # conversion from cgs to SI (log10(cm/s^2) -> log10(m/s^2))
            value -= 2
        else:
            msg = (
                "User input is not (numpy.)int or (numpy.)float nor "
                "astropy.unit.quantity.Quantity instance (or its string representation)."
            )
            raise TypeError(msg)
        return float(value)

    @staticmethod
    def gravity_darkening(value: Any) -> Float:
        """Validate gravity darkening parameter in range [0, 1]."""
        if value > 1 or value < 0:
            msg = f"Parameter gravity darkening = {value} is out of range <0, 1>"
            raise ValueError(msg)
        return float(value)

    @staticmethod
    def limb_darkening_coefficients(value: Any) -> Any:
        """Validate custom limb darkening coefficients.

        Accepts ``None`` or a dictionary mapping passband name to either a
        scalar coefficient (for linear or cosine laws) or a sequence of
        coefficients matching the required limb-darkening law.
        """
        if value is None:
            retval: dict[str, Any] | None = None  # default case of interpolated LD coefficients
            return retval
        if isinstance(value, dict):
            retval: dict[str, Any] = {}
            for passband, ld_coeffs in value.items():
                if isinstance(ld_coeffs, WHEN_FLOAT64):
                    if settings.LIMB_DARKENING_LAW in ["linear", "cosine"]:
                        retval[passband] = [ld_coeffs]
                    else:
                        msg = "Scalar limb darkening coefficient is available only for linear (cosine) law."
                        raise TypeError(msg)
                elif isinstance(ld_coeffs, WHEN_ARRAY):
                    desired_vector_length = len(settings.LD_LAW_CFS_COLUMNS[settings.LIMB_DARKENING_LAW])
                    if np.shape(ld_coeffs)[0] != desired_vector_length:
                        part1 = (
                            f"{settings.LIMB_DARKENING_LAW} limb-darkening law requires "
                            f"{desired_vector_length} components in a vector with shape ({desired_vector_length}, ). "
                        )
                        part2 = (
                            f"You provided a vector with {len(ld_coeffs)} components with shape {np.shape(ld_coeffs)}."
                        )
                        msg = part1 + part2
                        raise ValueError(msg)
                    retval[passband] = ld_coeffs
        else:
            msg = (
                "Limb darkening coefficients needs to be supplied as a dictionary in format "
                "{`passband`: ld_coeffs}, where `ld_coeffs` is a float (for linear law) "
                "or a sequence (numpy.array/list) in other cases."
            )
            raise TypeError(msg)

        return retval


class SpotProperties(BodyProperties):
    @staticmethod
    def latitude(value: Any) -> Float:
        """Convert spot latitude (degrees by default) to internal units."""
        return deg_transform(value, u.DefaultSpotUnits.latitude, WHEN_FLOAT64, u.DefaultSpotInputUnits.latitude)

    @staticmethod
    def longitude(value: Any) -> Float:
        """Convert spot longitude (degrees by default) to internal units."""
        return deg_transform(value, u.DefaultSpotUnits.longitude, WHEN_FLOAT64, u.DefaultSpotInputUnits.longitude)

    @staticmethod
    def angular_radius(value: Any) -> Float:
        """Convert spot angular radius to internal units."""
        return deg_transform(
            value,
            u.DefaultSpotUnits.angular_radius,
            WHEN_FLOAT64,
            u.DefaultSpotInputUnits.angular_radius,
        )

    @staticmethod
    def temperature_factor(value: Any) -> Float:
        """Validate temperature factor is numeric."""
        if not isinstance(value, (int, INT, float, FLOAT)):
            msg = "Input of variable `temperature_factor` is not (numpy.)int or (numpy.)float."
            raise TypeError(msg)
        return float(value)

    @staticmethod
    def discretization_factor(value: Any) -> Float:
        """Transform spot discretization factor (degrees by default)."""
        value = deg_transform(
            value,
            u.DefaultSpotUnits.discretization_factor,
            WHEN_FLOAT64,
            u.DefaultSpotInputUnits.discretization_factor,
        )
        if value > const.HALF_PI:
            msg = "Invalid value of alpha parameter. Use value less than 90."
            raise ValueError(msg)
        return float(value)
