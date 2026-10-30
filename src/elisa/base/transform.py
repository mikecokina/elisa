from __future__ import annotations

from math import isfinite
from numbers import Number
from typing import TYPE_CHECKING, Any

import numpy as np
from packaging import version

from elisa import const, settings
from elisa import units as u
from elisa.atm.const import normalize_atlas
from elisa.base.error import raise_elisa_parameters_error
from elisa.base.types import FLOAT, INT
from elisa.ld import CustomLimbDarkeningCoefficients
from elisa.units import (
    DefaultBinarySystemInputUnits,
    DefaultStarInputUnits,
    DefaultSystemInputUnits,
    DefaultSystemUnits,
)

if TYPE_CHECKING:
    from elisa.types import Float, Int, NumpyBool

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


def scalar_transform(
    value: Any,
    *,
    parameter: str,
    accepted_types: tuple[type, ...] = (
        int, float, np.integer, np.floating, np.bool_, np.ndarray, u.Quantity,
    ),
) -> Float | Int | NumpyBool:
    """Validate a real scalar and its finite float representation before range checks.

    Numeric-only validators share this boundary; their physical limits remain
    with the owning property. Zero-dimensional numeric arrays and dimensionless
    scalar quantities are supported by default. A property with a narrower input
    contract can supply its existing accepted types. Strings are accepted only
    when explicitly listed, as required by phase_shift's existing API. Numeric
    input precision is retained for the property's range checks; casting to its
    output type remains the property's responsibility.

    :raises ElisaParametersError: If the type, shape or finite representation is invalid.
    """
    # Keep each property's existing input contract: being convertible with float()
    # does not automatically make an input valid (e.g. numeric strings).
    if not isinstance(value, accepted_types):
        message = f"Input of variable `{parameter}` is not a supported real scalar."
        raise_elisa_parameters_error(message)
    try:
        # These properties represent dimensionless numbers. Convert compatible
        # quantities to a plain value, including their scale (e.g. 50% -> 0.5).
        # A physical unit such as metres fails here and keeps its original cause.
        if isinstance(value, u.Quantity):
            value = value.to_value(u.dimensionless_unscaled)

        # Quantity conversion may also produce an array, so this is a separate
        # check, not an elif. Accept only a numeric 0-D array such as np.array(0.5);
        # even a one-element vector is not a scalar. The dtype kinds are boolean,
        # signed integer, unsigned integer and floating point (b, i, u, f).
        if isinstance(value, np.ndarray):
            if value.ndim != 0 or value.dtype.kind not in "biuf":
                message = f"Parameter `{parameter}` must be a real scalar."
                raise_elisa_parameters_error(message)
            # Extract the NumPy scalar without first converting it to float.
            value = value[()]

        # A caller can allow broader types (e.g. Number for integer validation).
        # Reject complex numbers explicitly; some NumPy conversions would merely
        # warn and discard their imaginary part.
        if isinstance(value, (complex, np.complexfloating)):
            message = f"Parameter `{parameter}` must be a real scalar."
            raise_elisa_parameters_error(message)

        # Probe float representability before the caller performs range checks.
        # This catches failed parsing and overflowing integers such as 10**400.
        # The temporary float is not necessarily the value we return below.
        converted = float(value)
    except (TypeError, ValueError, ArithmeticError) as exc:
        # Translate conversion failures at their source and retain their cause.
        message = f"Cannot convert parameter `{parameter}` to a real scalar."
        raise_elisa_parameters_error(message, exc)

    # Successful float conversion still permits NaN and infinity. Reject both,
    # including finite inputs too large to have a finite Python float value.
    if not isfinite(converted):
        message = f"Nonfinite {parameter}."
        raise_elisa_parameters_error(message)

    # Preserve numeric precision for the property's own range checks. Returning
    # the temporary float could round a longdouble just above 1 down to 1 and
    # incorrectly accept it. The property applies its final output cast later.
    if isinstance(value, (int, float, np.integer, np.floating, np.bool_)):
        return value

    # Other explicitly allowed inputs (e.g. phase_shift's strings) have passed
    # conversion and finiteness checks; return their parsed numeric value.
    return converted


def integer_transform(value: Any, *, parameter: str) -> int:
    """Validate an integral numeric scalar without rounding its original value."""
    scalar_transform(value, parameter=parameter, accepted_types=(Number,))
    try:
        converted = int(value)
        integral = converted - value == 0
    except (TypeError, ValueError, ArithmeticError) as exc:
        message = f"Cannot convert parameter `{parameter}` to an integer."
        raise_elisa_parameters_error(message, exc)
    if not integral:
        message = f"Parameter `{parameter}` must be an integer."
        raise_elisa_parameters_error(message)
    return converted


def normalize_morphology(morphology: str) -> str:
    """Normalize the legacy overcontact alias to the internal morphology name.

    Other values are preserved for validation by the calling layer.
    """
    return "over-contact" if morphology == "overcontact" else morphology


def quantity_transform(
    value: Any,
    unit: Any,
    when_float64: tuple[type, ...] = WHEN_FLOAT64,
    default_input_unit: Any | None = None,
    *,
    parameter: str = "value",
) -> Float:
    """Transform a scalar into a finite floating-point value expressed in ``unit``.

    Accepts plain numbers, supported NumPy scalars, scalar astropy.Quantity or a
    string representation of a quantity. If the input has no attached
    units and ``default_input_unit`` is provided, the numeric value is
    interpreted using that unit before conversion.

    :param value: Input value to convert.
    :param unit: Target unit (astropy.units compatible).
    :param when_float64: Tuple of numeric types treated as unitless numbers.
    :param default_input_unit: Assumed unit for numbers without explicit
        units.
    :param parameter: Parameter name included in conversion errors.
    :returns: Value converted to ``unit`` as Python float.
    :rtype: float
    :raises ElisaParametersError: If the shape, parsing, unit conversion or finite representation is invalid.
    """
    try:
        if isinstance(value, (u.Quantity, str)):
            value = u.Quantity(value) if isinstance(value, str) else value
            if not value.isscalar:
                message = f"Parameter `{parameter}` must be a scalar quantity."
                raise_elisa_parameters_error(message)
            value = value.to_value(unit)
        elif isinstance(value, when_float64):
            if default_input_unit is None:
                value = np.float64(value)
            else:
                value = (value * default_input_unit).to(unit).value
                value = np.float64(value)
        else:
            msg = (
                f"Input of parameter `{parameter}` is not (numpy.)int or (numpy.)float nor "
                "astropy.unit.quantity.Quantity instance (or its string representation)."
            )
            raise_elisa_parameters_error(msg)
        value = float(value)
    except (TypeError, ValueError, ArithmeticError) as exc:
        message = f"Cannot convert parameter `{parameter}` to unit `{unit}`."
        raise_elisa_parameters_error(message, exc)
    # Every caller receives a finite scalar; physical ranges stay in properties.
    if not isfinite(value):
        message = f"Nonfinite {parameter} after conversion to unit `{unit}`."
        raise_elisa_parameters_error(message)
    return value


def deg_transform(
    value: Any,
    unit: Any,
    when_float64: tuple[type, ...],
    default_input_unit: Any = u.deg,
    *,
    parameter: str = "value",
) -> Float:
    """Transform a scalar angle into ``unit`` and return a finite float.

    Similar to :func:`quantity_transform` but uses multiplicative
    conversion for plain numeric inputs.

    :param value: Input angular value.
    :param unit: Target angular unit.
    :param when_float64: Tuple of numeric types treated as unitless numbers.
    :param default_input_unit: Default unit assumed for plain numbers.
    :param parameter: Parameter name included in conversion errors.
    :returns: Converted value in ``unit`` as float.
    :rtype: float
    :raises ElisaParametersError: If the shape, parsing, unit conversion or finite representation is invalid.
    """
    try:
        if isinstance(value, (u.Quantity, str)):
            value = u.Quantity(value) if isinstance(value, str) else value
            if not value.isscalar:
                message = f"Parameter `{parameter}` must be a scalar quantity."
                raise_elisa_parameters_error(message)
            value = value.to_value(unit)
        elif isinstance(value, when_float64):
            value = np.float64(value) * default_input_unit.to(unit)
        else:
            msg = (
                f"Input of angular parameter `{parameter}` is not (numpy.)int or (numpy.)float "
                "nor astropy.unit.quantity.Quantity instance (or its string representation)."
            )
            raise_elisa_parameters_error(msg)
        value = float(value)
    except (TypeError, ValueError, ArithmeticError) as exc:
        message = f"Cannot convert parameter `{parameter}` to unit `{unit}`."
        raise_elisa_parameters_error(message, exc)
    # Every caller receives a finite scalar; physical ranges stay in properties.
    if not isfinite(value):
        message = f"Nonfinite {parameter} after conversion to unit `{unit}`."
        raise_elisa_parameters_error(message)
    return value


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
        :raises ElisaParametersError: If the input type is unsupported.
        :raises ElisaParametersError: If the value is outside the valid range [0, pi].
        :raises ElisaParametersError: If the converted value is nonfinite.
        """
        value = quantity_transform(
            value,
            u.DefaultSystemUnits.inclination,
            (int, INT, float, FLOAT),
            DefaultSystemInputUnits.inclination,
            parameter="inclination",
        )

        if not 0 <= value <= const.PI:
            msg = f"Inclination value of {value} is out of bounds (0, pi)."
            raise_elisa_parameters_error(msg)
        return float(value)

    @staticmethod
    def period(value: Any) -> Float:
        """Transform and validate orbital period.

        If unit is omitted, the default period input unit is assumed.

        :raises ElisaParametersError: If the converted period is not strictly positive.
        :raises ElisaParametersError: If the converted period is nonfinite.
        """
        value = quantity_transform(
            value,
            DefaultSystemUnits.period,
            WHEN_FLOAT64,
            u.DefaultSystemInputUnits.period,
            parameter="period",
        )
        if value <= 0:
            msg = "Invalid orbital period, use value > 0!"
            raise_elisa_parameters_error(msg)
        return value

    @staticmethod
    def gamma(value: Any) -> Float:
        """Validate and transform systemic velocity (gamma).

        Accepts numeric values or :class:`astropy.units.Quantity` instances.
        """
        return quantity_transform(
            value,
            u.DefaultSystemUnits.gamma,
            WHEN_FLOAT64,
            u.DefaultSystemInputUnits.gamma,
            parameter="gamma",
        )

    @staticmethod
    def additional_light(value: Any) -> Float:
        """Validate additional (third) light fraction in range [0, 1]."""
        value = scalar_transform(value, parameter="additional_light")
        if not 0.0 <= value <= 1.0:
            msg = "Invalid value of additional light. Valid values are between 0 and 1."
            raise_elisa_parameters_error(msg)
        return float(value)

    @staticmethod
    def semi_major_axis(value: Any) -> Float:
        """Validate and convert semi-major axis to internal units.

        Accepts numeric or :class:`astropy.units.Quantity` values.
        """
        value = quantity_transform(
            value,
            u.DefaultBinarySystemUnits.system.semi_major_axis,
            WHEN_FLOAT64,
            DefaultBinarySystemInputUnits.system.semi_major_axis,
            parameter="semi_major_axis",
        )
        if value <= 0:
            msg = "Invalid value of semi_major_axis, use value > 0!"
            raise_elisa_parameters_error(msg)
        return float(value)

    @staticmethod
    def distance(value: Any) -> Float:
        """Convert system distance to internal distance units and validate.

        Accepts numeric or :class:`astropy.units.Quantity` values.
        """
        value = quantity_transform(
            value,
            u.DefaultSystemUnits.distance,
            WHEN_FLOAT64,
            DefaultBinarySystemInputUnits.system.distance,
            parameter="distance",
        )
        if value <= 0:
            msg = "Invalid value of system's distance, use value > 0!"
            raise_elisa_parameters_error(msg)

        return float(value)


class BodyProperties(TransformProperties):
    @staticmethod
    def synchronicity(value: Any) -> Float:
        """Validate object synchronicity F = omega_rot/omega_orb.

        Expects a positive numeric value.
        """
        value = scalar_transform(value, parameter="synchronicity")
        if value <= 0:
            msg = "Invalid synchronicity, use value > 0!"
            raise_elisa_parameters_error(msg)
        return float(value)

    @staticmethod
    def albedo(value: Any) -> Float:
        """Validate and transform bolometric albedo (range 0..1)."""
        value = scalar_transform(value, parameter="albedo")
        if value < 0 or value > 1:
            msg = f"Parameter albedo = {value} is out of range <0, 1>"
            raise_elisa_parameters_error(msg)
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
            parameter="discretization_factor",
        )
        if value < 0 or value > const.HALF_PI:
            msg = "Invalid value of alpha parameter. Use value between 0 and 90 degrees."
            raise_elisa_parameters_error(msg)
        return float(value)

    @staticmethod
    def t_eff(value: Any) -> Float:
        """Convert and validate a strictly positive effective temperature.

        If no unit is provided, Kelvin is assumed.
        """
        value = quantity_transform(
            value,
            u.DefaultStarUnits.t_eff,
            WHEN_FLOAT64,
            u.DefaultStarInputUnits.t_eff,
            parameter="t_eff",
        )
        if value <= 0:
            msg = "Invalid effective temperature, use value > 0!"
            raise_elisa_parameters_error(msg)
        return value


class StarProperties(BodyProperties):
    @staticmethod
    def atmosphere(value: Any) -> str | None:
        """Resolve fixed atlas configuration when a star is created or reinitialized."""
        # Despite living among model property validators, invalid atlas names raise
        # ElisaInitialParametersError: the atlas is fixed configuration, not a value
        # that drifts during fitting. Abort the fit rather than penalizing proposals.
        # Empty/None retain the existing request to use settings.ATM_ATLAS.
        if value is None or (isinstance(value, str) and value == ""):
            return value
        return normalize_atlas(value)

    @staticmethod
    def equivalent_radius(value: Any) -> Float:
        """Validate and convert equivalent radius to internal units.

        If quantity is not provided, the default distance unit is assumed.
        """
        value = quantity_transform(
            value,
            u.DefaultStarUnits.equivalent_radius,
            WHEN_FLOAT64,
            DefaultStarInputUnits.equivalent_radius,
            parameter="equivalent_radius",
        )
        if value <= 0:
            msg = "Invalid value of equivalent_radius, use value > 0!"
            raise_elisa_parameters_error(msg)
        return float(value)

    @staticmethod
    def mass(value: Any) -> Float:
        """Validate and convert mass to internal units.

        If mass is provided as a plain numeric value it is interpreted in
        solar masses unless a Quantity is supplied.
        """
        value = quantity_transform(
            value,
            u.DefaultStarUnits.mass,
            WHEN_FLOAT64,
            DefaultStarInputUnits.mass,
            parameter="mass",
        )
        if value <= 0:
            msg = "Invalid mass, use value > 0!"
            raise_elisa_parameters_error(msg)
        return float(value)

    @staticmethod
    def surface_potential(value: Any) -> Float:
        """Return the surface potential of the star (validated non-negative).

        :param value: Numeric surface potential.
        :returns: Surface potential as float.
        """
        value = scalar_transform(value, parameter="surface_potential")
        if value < 0:
            msg = "Invalid surface potential, use value >= 0!"
            raise_elisa_parameters_error(msg)
        return float(value)

    @staticmethod
    def metallicity(value: Any) -> Float:
        """Validate metallicity input as numeric.

        :param value: Numeric metallicity value.
        :returns: Metallicty as float.
        """
        value = scalar_transform(value, parameter="metallicity", accepted_types=WHEN_FLOAT64)
        return float(value)

    @staticmethod
    def polar_log_g(value: Any) -> Float:
        """Convert polar surface gravity (log g) to internal units.

        If the input is a string or quantity it is converted appropriately;
        numeric inputs are treated as log g in cgs unless a Quantity is
        provided.
        """
        try:
            if isinstance(value, (u.Quantity, str)):
                value = u.Dex(value, unit=u.Unit(" ".join(value.split()[1:]))) if isinstance(value, str) else value
                value = np.float64(value.to(u.LOG_ACCELERATION_UNIT))
            elif isinstance(value, WHEN_FLOAT64):
                # conversion from cgs to SI (log10(cm/s^2) -> log10(m/s^2))
                value = np.float64(value - 2)
            else:
                msg = (
                    "User input is not (numpy.)int or (numpy.)float nor "
                    "astropy.unit.quantity.Quantity instance (or its string representation)."
                )
                raise_elisa_parameters_error(msg)
            value = float(value)
        except (TypeError, ValueError, ArithmeticError) as exc:
            message = f"Cannot convert parameter `polar_log_g` to unit `{u.LOG_ACCELERATION_UNIT}`."
            raise_elisa_parameters_error(message, exc)
        if not np.isfinite(value):
            msg = "Nonfinite polar_log_g."
            raise_elisa_parameters_error(msg)
        return float(value)

    @staticmethod
    def gravity_darkening(value: Any) -> Float:
        """Validate gravity darkening parameter in range [0, 1]."""
        value = scalar_transform(value, parameter="gravity_darkening")
        if value > 1 or value < 0:
            msg = f"Parameter gravity darkening = {value} is out of range <0, 1>"
            raise_elisa_parameters_error(msg)
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
            if isinstance(value, CustomLimbDarkeningCoefficients):
                value.validate_law()
            retval: dict[str, Any] = {}
            for passband, ld_coeffs in value.items():
                if isinstance(ld_coeffs, WHEN_FLOAT64):
                    if settings.LIMB_DARKENING_LAW in ["linear", "cosine"]:
                        retval[passband] = [ld_coeffs]
                    else:
                        msg = "Scalar limb darkening coefficient is available only for linear (cosine) law."
                        raise_elisa_parameters_error(msg)
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
                        raise_elisa_parameters_error(msg)
                    retval[passband] = ld_coeffs
        else:
            msg = (
                "Limb darkening coefficients needs to be supplied as a dictionary in format "
                "{`passband`: ld_coeffs}, where `ld_coeffs` is a float (for linear law) "
                "or a sequence (numpy.array/list) in other cases."
            )
            raise_elisa_parameters_error(msg)

        law = value.law if isinstance(value, CustomLimbDarkeningCoefficients) else settings.LIMB_DARKENING_LAW
        return CustomLimbDarkeningCoefficients(retval, law)


class SpotProperties(BodyProperties):
    @staticmethod
    def latitude(value: Any) -> Float:
        """Convert spot latitude (degrees by default) to internal units."""
        return deg_transform(
            value,
            u.DefaultSpotUnits.latitude,
            WHEN_FLOAT64,
            u.DefaultSpotInputUnits.latitude,
            parameter="latitude",
        )

    @staticmethod
    def longitude(value: Any) -> Float:
        """Convert spot longitude (degrees by default) to internal units."""
        return deg_transform(
            value,
            u.DefaultSpotUnits.longitude,
            WHEN_FLOAT64,
            u.DefaultSpotInputUnits.longitude,
            parameter="longitude",
        )

    @staticmethod
    def angular_radius(value: Any) -> Float:
        """Convert spot angular radius to internal units."""
        value = deg_transform(
            value,
            u.DefaultSpotUnits.angular_radius,
            WHEN_FLOAT64,
            u.DefaultSpotInputUnits.angular_radius,
            parameter="angular_radius",
        )
        if value <= 0:
            msg = "Spot angular radius must be strictly positive."
            raise_elisa_parameters_error(msg)
        return value

    @staticmethod
    def temperature_factor(value: Any) -> Float:
        """Validate temperature factor is a finite numeric scalar."""
        value = scalar_transform(value, parameter="temperature_factor", accepted_types=(int, INT, float, FLOAT))
        return float(value)

    @staticmethod
    def discretization_factor(value: Any) -> Float:
        """Transform spot discretization factor (degrees by default)."""
        value = deg_transform(
            value,
            u.DefaultSpotUnits.discretization_factor,
            WHEN_FLOAT64,
            u.DefaultSpotInputUnits.discretization_factor,
            parameter="discretization_factor",
        )
        if value <= 0:
            msg = "Spot discretization factor must be strictly positive."
            raise_elisa_parameters_error(msg)
        if value > const.HALF_PI:
            msg = "Invalid value of alpha parameter. Use value less than 90."
            raise_elisa_parameters_error(msg)
        return float(value)
