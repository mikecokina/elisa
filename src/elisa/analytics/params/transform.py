"""Convert input parameters to default internal units."""

from __future__ import annotations

from functools import partial
from math import isfinite
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

import elisa.units as u
from elisa.analytics.params import conf
from elisa.base.error import (
    ElisaParametersError,
    raise_elisa_initial_parameters_error,
)
from elisa.base.transform import (
    WHEN_FLOAT64,
    SpotProperties,
    StarProperties,
    TransformProperties,
)
from elisa.binary_system.transform import (
    BinarySystemProperties,
    RadialVelocityObserverProperties,
)
from elisa.pulse.transform import PulsationModeProperties

if TYPE_CHECKING:
    from collections.abc import Callable

    from elisa.types import AstropyQuantity as Quantity
    from elisa.types import Float


_ValueT = TypeVar("_ValueT")


def _convert_initial_parameter_value(value: Any, unit: Any, parameter: str) -> Float:
    """Convert user-supplied values while preparing initial fitting parameters.

    Quantities and quantity strings are converted to ``unit``; bare numbers
    are already expressed in the initial-parameter unit and are only cast to
    finite float64. Conversion failures raise ElisaInitialParametersError with the
    original cause. Candidate model validation uses model validators.
    """
    if not isinstance(value, (u.Quantity, str, *WHEN_FLOAT64)):
        message = (
            f"Input of initial parameter `{parameter}` is not (numpy.)int or (numpy.)float "
            "nor astropy.unit.quantity.Quantity instance (or its string representation)."
        )
        raise_elisa_initial_parameters_error(message)
    try:
        if isinstance(value, (u.Quantity, str)):
            quantity = u.Quantity(value) if isinstance(value, str) else value
            if not quantity.isscalar:
                message = f"Initial parameter `{parameter}` must be a scalar quantity."
                raise_elisa_initial_parameters_error(message)
            converted = quantity.to_value(unit)
        else:
            converted = np.float64(value)
        converted = np.float64(float(converted))
    except (TypeError, ValueError, ArithmeticError) as exc:
        message = f"Cannot convert initial parameter `{parameter}` to unit `{unit}`."
        raise_elisa_initial_parameters_error(message, exc)
    if not isfinite(converted):
        message = f"Nonfinite initial parameter `{parameter}` after conversion to unit `{unit}`."
        raise_elisa_initial_parameters_error(message)
    return converted


def angular(value: Float | Quantity | str, *, parameter: str = "angle") -> Float:
    """Convert fitting angles to their input unit, preserving their fitting ranges.

    :param value: Numeric angle, Quantity or quantity string; bare numbers keep their unit.
    :param parameter: Parameter name included in conversion errors.
    :returns: Angle in the default fitting angular unit.
    :raises ElisaInitialParametersError: If the type or conversion is invalid.
    """
    return _convert_initial_parameter_value(value, conf.DEFAULT_FLOAT_ANGULAR_UNIT, parameter)


def _initial_parameter(validator: Callable[[Any], _ValueT]) -> staticmethod:
    """Adapt a model validator to report initial-parameter validation errors."""
    def wrapped(value: Any) -> _ValueT:
        try:
            return validator(value)
        except ElisaParametersError as exc:
            raise_elisa_initial_parameters_error(str(exc), exc)

    return staticmethod(wrapped)


class BinaryInitialProperties(TransformProperties):
    """Handler for conversion of binary system parameters units."""

    @staticmethod
    def semi_major_axis(value: Float | Quantity | str) -> float:
        """Transform semi-major axis value to internal units.

        :param value: Semi-major axis value to transform.
        :type value: float | Quantity | str
        :returns: Transformed semi-major axis value in solar radii.
        :rtype: float
        :raises ElisaInitialParametersError: If the type, quantity string or unit conversion is invalid.
        :raises ElisaInitialParametersError: If value is not strictly positive.
        :raises ElisaInitialParametersError: If the converted value is nonfinite.
        """
        value = _convert_initial_parameter_value(value, u.solRad, "semi_major_axis")
        if value <= 0:
            error_msg = "Invalid value of semi_major_axis, use value > 0!"
            raise_elisa_initial_parameters_error(error_msg)
        return value

    @staticmethod
    def mass(value: Float | Quantity | str) -> Float:
        """Transform mass value to internal units.

        :param value: Mass value to transform.
        :type value: float | Quantity | str
        :returns: Transformed mass value in default mass unit.
        :rtype: float
        :raises ElisaInitialParametersError: If the type, quantity string or unit conversion is invalid.
        :raises ElisaInitialParametersError: If value is not positive.
        :raises ElisaInitialParametersError: If the converted value is nonfinite.
        """
        value = _convert_initial_parameter_value(value, conf.DEFAULT_FLOAT_MASS_UNIT, "mass")
        if value <= 0:
            error_msg = "Invalid mass, use value > 0!"
            raise_elisa_initial_parameters_error(error_msg)
        return value

    eccentricity = _initial_parameter(BinarySystemProperties.eccentricity)
    argument_of_periastron = staticmethod(partial(angular, parameter="argument_of_periastron"))
    inclination = staticmethod(partial(angular, parameter="inclination"))
    gamma = _initial_parameter(BinarySystemProperties.gamma)
    period = _initial_parameter(BinarySystemProperties.period)
    mass_ratio = _initial_parameter(RadialVelocityObserverProperties.mass_ratio)
    asini = _initial_parameter(RadialVelocityObserverProperties.asini)
    additional_light = _initial_parameter(BinarySystemProperties.additional_light)
    phase_shift = _initial_parameter(BinarySystemProperties.phase_shift)
    primary_minimum_time = _initial_parameter(BinarySystemProperties.primary_minimum_time)


class StarInitialProperties(StarProperties):
    """Handler for conversion of star component parameters units."""

    mass = staticmethod(BinaryInitialProperties.mass)
    t_eff = _initial_parameter(StarProperties.t_eff)
    albedo = _initial_parameter(StarProperties.albedo)
    gravity_darkening = _initial_parameter(StarProperties.gravity_darkening)
    metallicity = _initial_parameter(StarProperties.metallicity)
    surface_potential = _initial_parameter(StarProperties.surface_potential)
    synchronicity = _initial_parameter(StarProperties.synchronicity)
    discretization_factor = _initial_parameter(StarProperties.discretization_factor)
    equivalent_radius = _initial_parameter(StarProperties.equivalent_radius)
    polar_log_g = _initial_parameter(StarProperties.polar_log_g)
    limb_darkening_coefficients = _initial_parameter(StarProperties.limb_darkening_coefficients)


class SpotInitialProperties(SpotProperties):
    """Handler for conversion of spot parameters units."""

    latitude = staticmethod(partial(angular, parameter="latitude"))
    longitude = staticmethod(partial(angular, parameter="longitude"))
    angular_radius = staticmethod(partial(angular, parameter="angular_radius"))
    temperature_factor = _initial_parameter(SpotProperties.temperature_factor)
    albedo = _initial_parameter(SpotProperties.albedo)
    discretization_factor = _initial_parameter(SpotProperties.discretization_factor)
    synchronicity = _initial_parameter(SpotProperties.synchronicity)
    t_eff = _initial_parameter(SpotProperties.t_eff)


class NuisanceInitialProperties(TransformProperties):
    """Handler for unit conversion of nuisance fit parameters."""

    @staticmethod
    def ln_f(value: Float | Quantity | str) -> Float:
        """Validate a finite dimensionless noise parameter without imposing bounds."""
        return _convert_initial_parameter_value(value, u.dimensionless_unscaled, "ln_f")


class PulsationModeInitialProperties(PulsationModeProperties):
    """Handler for unit conversion of pulsation mode parameters."""

    l = _initial_parameter(PulsationModeProperties.l)  # noqa: E741
    m = _initial_parameter(PulsationModeProperties.m)
    amplitude = _initial_parameter(PulsationModeProperties.amplitude)
    start_phase = _initial_parameter(PulsationModeProperties.start_phase)
    additional_light = _initial_parameter(PulsationModeProperties.additional_light)
    distance = _initial_parameter(PulsationModeProperties.distance)
    gamma = _initial_parameter(PulsationModeProperties.gamma)
    horizontal_to_radial_amplitude_ratio = _initial_parameter(
        PulsationModeProperties.horizontal_to_radial_amplitude_ratio,
    )
    inclination = _initial_parameter(PulsationModeProperties.inclination)
    period = _initial_parameter(PulsationModeProperties.period)
    semi_major_axis = _initial_parameter(PulsationModeProperties.semi_major_axis)
    temperature_perturbation_phase_shift = _initial_parameter(
        PulsationModeProperties.temperature_perturbation_phase_shift,
    )
    tidally_locked = _initial_parameter(PulsationModeProperties.tidally_locked)
    mode_axis_theta = staticmethod(partial(angular, parameter="mode_axis_theta"))
    mode_axis_phi = staticmethod(partial(angular, parameter="mode_axis_phi"))

    @staticmethod
    def frequency(value: Float | Quantity | str) -> Float:
        """Store fitting frequencies in cycles/day for model reconstruction.

        Keep the mode's input validation, but retain the bare-number input unit
        expected by the next PulsationMode construction rather than storing Hz.
        """
        try:
            PulsationModeProperties.frequency(value)
        except ElisaParametersError as exc:
            raise_elisa_initial_parameters_error(str(exc), exc)
        try:
            if isinstance(value, (u.Quantity, str)):
                return u.Quantity(value).to_value(u.DefaultPulsationsInputUnits.frequency)
            return np.float64(value)
        except (TypeError, ValueError, ArithmeticError) as exc:
            message = (
                "Cannot convert initial parameter `frequency` to unit "
                f"`{u.DefaultPulsationsInputUnits.frequency}`."
            )
            raise_elisa_initial_parameters_error(message, exc)
