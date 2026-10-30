"""Numeric model boundaries reject malformed scalars and retain input precision."""

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from astropy.units import UnitConversionError

from elisa import units as u
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.analytics.params.transform import (
    BinaryInitialProperties,
    PulsationModeInitialProperties,
    SpotInitialProperties,
    StarInitialProperties,
)
from elisa.base.error import ElisaInitialParametersError, ElisaParametersError
from elisa.base.transform import BodyProperties, SpotProperties, StarProperties, SystemProperties
from elisa.binary_system.orbit.transform import OrbitProperties
from elisa.binary_system.transform import BinarySystemProperties, RadialVelocityObserverProperties
from elisa.pulse.transform import PulsationModeProperties


SCALAR_CASES = [
    (SystemProperties.additional_light, BinaryInitialProperties.additional_light, "additional_light"),
    (BodyProperties.synchronicity, StarInitialProperties.synchronicity, "synchronicity"),
    (BodyProperties.albedo, StarInitialProperties.albedo, "albedo"),
    (StarProperties.surface_potential, StarInitialProperties.surface_potential, "surface_potential"),
    (StarProperties.gravity_darkening, StarInitialProperties.gravity_darkening, "gravity_darkening"),
    (RadialVelocityObserverProperties.mass_ratio, BinaryInitialProperties.mass_ratio, "mass_ratio"),
    (BinarySystemProperties.eccentricity, BinaryInitialProperties.eccentricity, "eccentricity"),
    (OrbitProperties.eccentricity, BinaryInitialProperties.eccentricity, "eccentricity"),
    (StarProperties.metallicity, StarInitialProperties.metallicity, "metallicity"),
    (SpotProperties.temperature_factor, SpotInitialProperties.temperature_factor, "temperature_factor"),
    (BinarySystemProperties.phase_shift, BinaryInitialProperties.phase_shift, "phase_shift"),
    (PulsationModeProperties.horizontal_to_radial_amplitude_ratio,
     PulsationModeInitialProperties.horizontal_to_radial_amplitude_ratio, "horizontal_to_radial_amplitude_ratio"),
]

INVALID_SCALARS = [
    None, object(), {}, "abc", [0.5], [0.1, 0.2],
    np.array([0.5]), np.array([[0.5]]), np.array([]),
    np.array("0.5"), np.array(object(), dtype=object),
    0.5 + 0j, np.complex64(0.5), np.complex128(0.5 + 1j),
    np.nan, np.inf, -np.inf, 10 ** 400,
]


@pytest.mark.parametrize("model,initial,parameter", SCALAR_CASES)
@pytest.mark.parametrize("value", INVALID_SCALARS)
def test_invalid_scalar_uses_the_owning_validation_layer(model, initial, parameter, value):
    with np.errstate(all="raise"):
        with pytest.raises(ElisaParametersError, match=parameter):
            model(value)
        with pytest.raises(ElisaInitialParametersError, match=parameter) as caught:
            initial(value)
    assert type(caught.value.__cause__) is ElisaParametersError


@pytest.mark.parametrize("model,initial,parameter", SCALAR_CASES)
def test_unrepresentable_python_integer_preserves_the_overflow_cause(model, initial, parameter):
    with pytest.raises(ElisaParametersError, match=parameter) as direct:
        model(10 ** 400)
    assert isinstance(direct.value.__cause__, OverflowError)
    with pytest.raises(ElisaInitialParametersError, match=parameter) as delegated:
        initial(10 ** 400)
    assert type(delegated.value.__cause__) is ElisaParametersError
    assert isinstance(delegated.value.__cause__.__cause__, OverflowError)


@pytest.mark.parametrize("model,initial,parameter", SCALAR_CASES[:6])
@pytest.mark.parametrize("value", [
    np.float16(0.5), np.float32(0.5), np.float64(0.5), np.longdouble(0.5),
    np.int32(1), np.int64(1), np.uint64(1), np.array(0.5),
    0.5 * u.dimensionless_unscaled, True, np.bool_(True),
])
def test_existing_numeric_scalar_representations_remain_supported(model, initial, parameter, value):
    expected = float(value)
    assert model(value) == expected
    assert initial(value) == expected


@pytest.mark.parametrize("model,initial,parameter", SCALAR_CASES[:6])
def test_numeric_properties_do_not_silently_strip_physical_units(model, initial, parameter):
    with pytest.raises(ElisaParametersError, match=parameter) as caught:
        model(1 * u.m)
    assert isinstance(caught.value.__cause__, UnitConversionError)
    with pytest.raises(ElisaInitialParametersError, match=parameter) as caught:
        initial(1 * u.m)
    assert isinstance(caught.value.__cause__.__cause__, UnitConversionError)


@pytest.mark.parametrize("value", ["0.25", b"0.25", Decimal("0.25"), Fraction(1, 4)])
def test_phase_shift_preserves_its_existing_numeric_conversion_inputs(value):
    assert BinarySystemProperties.phase_shift(value) == 0.25
    assert BinaryInitialProperties.phase_shift(value) == 0.25


def test_initial_phase_shift_is_validated_during_parameter_construction():
    with pytest.raises(ElisaInitialParametersError, match="phase_shift") as caught:
        BinaryInitialParameters(system={
            "period": {"value": 1, "min": 0.5, "max": 2},
            "phase_shift": {"value": "abc", "fixed": True},
        })
    assert type(caught.value.__cause__) is ElisaParametersError
    assert isinstance(caught.value.__cause__.__cause__, ValueError)


@pytest.mark.parametrize("validator", [
    SystemProperties.additional_light, BodyProperties.albedo, StarProperties.gravity_darkening,
])
def test_range_checks_retain_numpy_precision_before_output_conversion(validator):
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("This platform has no extended float precision")
    just_above_one = np.nextafter(np.longdouble(1), np.longdouble(2))
    assert float(just_above_one) == 1.0
    with pytest.raises(ElisaParametersError):
        validator(just_above_one)


@pytest.mark.parametrize("model,initial,name", [
    (PulsationModeProperties.l, PulsationModeInitialProperties.l, "l"),
    (PulsationModeProperties.m, PulsationModeInitialProperties.m, "m"),
])
@pytest.mark.parametrize("value", [
    None, "1", [], np.array(1), np.array([1]), 1 + 0j, np.complex64(1),
    np.nan, np.inf, -np.inf, 10 ** 400, 1.5,
    Decimal("1.0000000000000000000001"), Fraction(2 ** 60 + 1, 2 ** 60),
])
def test_integer_validation_rejects_bad_or_nonintegral_values_without_rounding(model, initial, name, value):
    with pytest.raises(ElisaParametersError, match=name):
        model(value)
    with pytest.raises(ElisaInitialParametersError, match=name) as caught:
        initial(value)
    assert type(caught.value.__cause__) is ElisaParametersError


@pytest.mark.parametrize("value", [1, 1.0, np.int32(1), np.int64(1), Decimal("1"), Fraction(2, 2)])
def test_integer_validation_preserves_existing_exact_numeric_inputs(value):
    assert PulsationModeProperties.l(value) == 1
    assert PulsationModeProperties.m(-value) == -1


def test_integer_validation_does_not_round_large_exact_integers():
    value = 2 ** 53 + 1
    assert PulsationModeProperties.l(value) == value
    assert PulsationModeProperties.m(-value) == -value


def test_unexpected_errors_in_numeric_objects_are_not_reclassified():
    failure = RuntimeError("broken numeric implementation")

    class BrokenFloat(float):
        def __float__(self):
            raise failure

    for validator in (SystemProperties.additional_light, BinaryInitialProperties.additional_light):
        with pytest.raises(RuntimeError) as caught:
            validator(BrokenFloat(0.5))
        assert caught.value is failure
