"""Convert input parameters to default internal units."""

from __future__ import annotations

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


def angular(value: Float | Quantity) -> Float:
    """Transform all angular units to ELISa's default internal angular unit.

    :param value: Numeric value to transform, can be a float or astropy Quantity.
    :type value: float | Quantity
    :returns: Transformed angular value in default internal angular unit.
    :rtype: float
    :raises ElisaInitialParametersError: If input is not a numeric type or astropy Quantity.
    """
    if isinstance(value, u.Quantity):
        value = np.float64(value.to(conf.DEFAULT_FLOAT_ANGULAR_UNIT))
    elif isinstance(value, WHEN_FLOAT64):
        value = np.float64(value)
    else:
        error_msg = (
            "Input of variable is not (numpy.)int or (numpy.)float "
            "nor astropy.unit.quantity.Quantity instance."
        )
        raise_elisa_initial_parameters_error(error_msg)
    return value


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
    def semi_major_axis(value: Float | Quantity) -> float:
        """Transform semi-major axis value to internal units.

        :param value: Semi-major axis value to transform.
        :type value: float | Quantity
        :returns: Transformed semi-major axis value in solar radii.
        :rtype: float
        :raises ElisaInitialParametersError: If input is not a numeric type or astropy Quantity.
        :raises ElisaInitialParametersError: If value is not strictly positive.
        :raises ElisaInitialParametersError: If the converted value is nonfinite.
        """
        if isinstance(value, u.Quantity):
            value = np.float64(value.to(u.solRad))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64(value)
        else:
            error_msg = (
                "Input of variable `semi_major_axis` is not (numpy.)int or (numpy.)float "
                "nor astropy.unit.quantity.Quantity instance."
            )
            raise_elisa_initial_parameters_error(error_msg)
        if not np.isfinite(value):
            error_msg = "Nonfinite semi_major_axis."
            raise_elisa_initial_parameters_error(error_msg)
        if value <= 0:
            error_msg = "Invalid value of semi_major_axis, use value > 0!"
            raise_elisa_initial_parameters_error(error_msg)
        return value

    @staticmethod
    def mass(value: Float | Quantity) -> Float:
        """Transform mass value to internal units.

        :param value: Mass value to transform.
        :type value: float | Quantity
        :returns: Transformed mass value in default mass unit.
        :rtype: float
        :raises ElisaInitialParametersError: If input is not a numeric type or astropy Quantity.
        :raises ElisaInitialParametersError: If value is not positive.
        :raises ElisaInitialParametersError: If the converted value is nonfinite.
        """
        if isinstance(value, u.Quantity):
            value = np.float64(value.to(conf.DEFAULT_FLOAT_MASS_UNIT))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64(value)
        else:
            error_msg = (
                "User input is not (numpy.)int or (numpy.)float "
                "nor astropy.unit.quantity.Quantity instance."
            )
            raise_elisa_initial_parameters_error(error_msg)
        if not np.isfinite(value):
            error_msg = "Nonfinite mass."
            raise_elisa_initial_parameters_error(error_msg)
        if value <= 0:
            error_msg = "Invalid mass, use value > 0!"
            raise_elisa_initial_parameters_error(error_msg)
        return value

    eccentricity = _initial_parameter(BinarySystemProperties.eccentricity)
    argument_of_periastron = staticmethod(angular)
    inclination = staticmethod(angular)
    gamma = _initial_parameter(BinarySystemProperties.gamma)
    period = _initial_parameter(BinarySystemProperties.period)
    mass_ratio = _initial_parameter(RadialVelocityObserverProperties.mass_ratio)
    asini = _initial_parameter(RadialVelocityObserverProperties.asini)
    additional_light = _initial_parameter(BinarySystemProperties.additional_light)
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

    latitude = staticmethod(angular)
    longitude = staticmethod(angular)
    angular_radius = staticmethod(angular)
    temperature_factor = _initial_parameter(SpotProperties.temperature_factor)
    albedo = _initial_parameter(SpotProperties.albedo)
    discretization_factor = _initial_parameter(SpotProperties.discretization_factor)
    synchronicity = _initial_parameter(SpotProperties.synchronicity)
    t_eff = _initial_parameter(SpotProperties.t_eff)


class NuisanceInitialProperties(TransformProperties):
    """Handler for unit conversion of nuisance fit parameters."""

    @staticmethod
    def ln_f(value: Float) -> Float:
        """Identity transformation for ln_f nuisance parameter.

        :param value: Nuisance parameter value.
        :type value: float
        :returns: Unchanged parameter value.
        :rtype: float
        """
        return value


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
    mode_axis_theta = staticmethod(angular)
    mode_axis_phi = staticmethod(angular)

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
        if isinstance(value, (u.Quantity, str)):
            return u.Quantity(value).to_value(u.DefaultPulsationsInputUnits.frequency)
        return np.float64(value)
