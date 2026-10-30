"""Finite scalar contracts at model and initial-definition conversion boundaries."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import units as u
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.analytics.params.transform import (
    BinaryInitialProperties, NuisanceInitialProperties, PulsationModeInitialProperties, SpotInitialProperties,
)
from elisa.base.error import ElisaInitialParametersError, ElisaParametersError
from elisa.base.transform import SpotProperties, WHEN_FLOAT64, deg_transform, quantity_transform
from elisa.pulse.transform import PulsationModeProperties
from elisa.single_system.orbit.transform import OrbitProperties as SingleOrbitProperties
from elisa.single_system.transform import SingleSystemProperties


MODEL_ANGLES = [
    (SpotProperties.latitude, u.deg, "latitude"),
    (SpotProperties.longitude, u.deg, "longitude"),
    (SpotProperties.angular_radius, u.deg, "angular_radius"),
    (SpotProperties.discretization_factor, u.deg, "discretization_factor"),
    (PulsationModeProperties.start_phase, u.rad, "start_phase"),
    (PulsationModeProperties.mode_axis_theta, u.deg, "mode_axis_theta"),
    (PulsationModeProperties.mode_axis_phi, u.deg, "mode_axis_phi"),
    (PulsationModeProperties.temperature_perturbation_phase_shift, u.rad, "temperature_perturbation_phase_shift"),
    (SingleOrbitProperties.inclination, u.rad, "inclination"),
    (SingleSystemProperties.rotation_period, u.d, "rotation_period"),
    (SingleSystemProperties.reference_time, u.d, "reference_time"),
]

INITIAL_ANGLES = [
    (BinaryInitialProperties.inclination, "inclination"),
    (BinaryInitialProperties.argument_of_periastron, "argument_of_periastron"),
    (SpotInitialProperties.latitude, "latitude"),
    (SpotInitialProperties.longitude, "longitude"),
    (SpotInitialProperties.angular_radius, "angular_radius"),
    (PulsationModeInitialProperties.mode_axis_theta, "mode_axis_theta"),
    (PulsationModeInitialProperties.mode_axis_phi, "mode_axis_phi"),
]


@pytest.mark.parametrize("validator,unit,name", MODEL_ANGLES)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("quantity", [False, True])
def test_model_conversions_always_reject_nonfinite_values(validator, unit, name, value, quantity):
    supplied = value * unit if quantity else value
    with np.errstate(all="raise"), pytest.raises(ElisaParametersError, match=name):
        validator(supplied)


@pytest.mark.parametrize("validator,name", INITIAL_ANGLES)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("quantity", [False, True])
def test_initial_angles_reject_nonfinite_values_in_the_initial_layer(validator, name, value, quantity):
    with pytest.raises(ElisaInitialParametersError, match=name):
        validator(value * u.deg if quantity else value)


@pytest.mark.parametrize("validator,name", INITIAL_ANGLES)
def test_initial_angle_conversion_preserves_units_and_does_not_copy_model_ranges(validator, name):
    # Initial definitions also transform bounds. Their angular conversion must
    # keep the existing fitting contract rather than wrap or constrain angles.
    assert validator(-720.0) == -720.0
    assert validator(720.0) == 720.0
    assert_allclose(validator(np.pi * u.rad), 180.0)
    assert validator("180 deg") == 180.0


@pytest.mark.parametrize("value", [None, "abc", [0.5], np.array([0.5]), np.nan, np.inf, -np.inf, 10**400, 1j, 1 * u.m])
def test_initial_noise_parameter_is_a_finite_dimensionless_scalar(value):
    with pytest.raises(ElisaInitialParametersError, match="ln_f"):
        NuisanceInitialProperties.ln_f(value)


@pytest.mark.parametrize("value", [-1000.0, -20, 0.0, 1000.0, "-2", -2 * u.dimensionless_unscaled])
def test_noise_parameter_keeps_finite_values_without_imposing_a_range(value):
    assert NuisanceInitialProperties.ln_f(value) == float(value)


@pytest.mark.parametrize("field", ["value", "min", "max", "sigma"])
def test_noise_metadata_is_validated_when_defining_the_fit(field):
    metadata = {"value": -2.0, "min": -5.0, "max": 1.0, "sigma": 0.1, field: np.nan}
    with pytest.raises(ElisaInitialParametersError, match="ln_f"):
        BinaryInitialParameters(system={"period": {"value": 1.0, "fixed": True}}, nuisance={"ln_f": metadata})


@pytest.mark.parametrize("unit", [u.rad, u.d])
@pytest.mark.parametrize("value", [np.array([1.0]), np.array([1.0, 2.0]), np.empty(0)])
def test_quantity_helpers_reject_vectors_even_with_one_element(value, unit):
    for helper in (quantity_transform, deg_transform):
        with pytest.raises(ElisaParametersError, match="sample"):
            helper(value * unit, unit, WHEN_FLOAT64, parameter="sample")


@pytest.mark.parametrize("validator,name", INITIAL_ANGLES)
def test_initial_angles_reject_vector_quantities(validator, name):
    with pytest.raises(ElisaInitialParametersError, match=name):
        validator(np.array([90.0]) * u.deg)


@pytest.mark.parametrize("value", [-1.0, 0.0])
def test_nonpositive_rotation_period_uses_the_model_layer(value):
    with pytest.raises(ElisaParametersError, match="rotation period"):
        SingleSystemProperties.rotation_period(value)


def test_quantity_and_angle_helpers_preserve_scalar_arrays_and_conversions():
    assert quantity_transform(np.array(86400.0) * u.s, u.d, parameter="period") == 1.0
    assert_allclose(deg_transform(np.array(180.0) * u.deg, u.rad, WHEN_FLOAT64, parameter="angle"), np.pi)
