"""Explicit validation raises and initial-parameter adapters keep their categories."""
import numpy as np
import pytest
from astropy.units import UnitConversionError

from elisa import units as u
from elisa.analytics.dataset.base import LCData
from elisa.analytics.dataset.transform import RVDataProperties
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.analytics.params.transform import (
    BinaryInitialProperties, PulsationModeInitialProperties, SpotInitialProperties,
    StarInitialProperties, _initial_parameter,
)
from elisa.analytics.tasks import LCBinaryAnalyticsTask
from elisa.analytics.transform import LCBinaryAnalyticsProperties
from elisa.base.error import (
    ElisaAnalyticsDataValidationError,
    ElisaAnalyticsTaskValidationError,
    ElisaError,
    ElisaInitialParametersValidationError,
    ElisaParametersValidationError,
)
from elisa.base.transform import StarProperties
from elisa.binary_system.transform import BinarySystemProperties


@pytest.mark.parametrize("error_type", [
    ElisaAnalyticsTaskValidationError, ElisaAnalyticsDataValidationError,
    ElisaInitialParametersValidationError, ElisaParametersValidationError,
])
def test_validation_errors_inherit_directly_from_elisa_error(error_type):
    assert error_type.__bases__ == (ElisaError,)


@pytest.mark.parametrize("value", [-1.0, 1.0, np.nan, np.inf, -np.inf, "invalid"])
def test_eccentricity_adapter_translates_only_the_model_validation_error(value):
    with pytest.raises(ElisaParametersValidationError) as physical:
        BinarySystemProperties.eccentricity(value)
    with pytest.raises(ElisaInitialParametersValidationError) as initial:
        BinaryInitialProperties.eccentricity(value)
    assert type(initial.value.__cause__) is ElisaParametersValidationError
    assert str(initial.value) == str(physical.value)


@pytest.mark.parametrize("method,value", [
    ("gamma", np.inf), ("period", 0.0), ("mass_ratio", -1.0),
    ("asini", -1.0), ("additional_light", 2.0), ("primary_minimum_time", np.inf),
])
def test_shared_initial_adapters_keep_the_original_validation_cause(method, value):
    with pytest.raises(ElisaInitialParametersValidationError) as error:
        getattr(BinaryInitialProperties, method)(value)
    assert type(error.value.__cause__) is ElisaParametersValidationError


@pytest.mark.parametrize("validator,value,error_type", [
    (LCBinaryAnalyticsProperties.data, "invalid", ElisaAnalyticsTaskValidationError),
    (RVDataProperties.y_unit, u.kg, ElisaAnalyticsDataValidationError),
    (BinaryInitialProperties.mass, 0.0, ElisaInitialParametersValidationError),
    (BinaryInitialProperties.mass, np.inf, ElisaInitialParametersValidationError),
    (StarProperties.mass, object(), ElisaParametersValidationError),
    (StarProperties.t_eff, -1.0, ElisaParametersValidationError),
    (StarProperties.t_eff, np.inf, ElisaParametersValidationError),
    (PulsationModeInitialProperties.frequency, -1.0, ElisaInitialParametersValidationError),
])
def test_explicit_raises_use_their_validation_layer(validator, value, error_type):
    with pytest.raises(error_type):
        validator(value)


def test_existing_data_and_parameter_definition_checks_use_new_types():
    with pytest.raises(ElisaAnalyticsDataValidationError, match="same shape"):
        LCData.check_data_validity(x_data=[0.0, 1.0], y_data=[1.0])
    with pytest.raises(ElisaInitialParametersValidationError, match="no variable"):
        BinaryInitialParameters(system={})
    with pytest.raises(ElisaAnalyticsTaskValidationError, match="Invalid fitting method"):
        LCBinaryAnalyticsTask.validate_method("invalid")


def test_adapter_does_not_translate_unexpected_errors():
    failure = RuntimeError("unexpected computation failure")
    def validator(value):
        raise failure

    class Properties:
        value = _initial_parameter(validator)

    with pytest.raises(RuntimeError) as error:
        Properties.value(0.5)
    assert error.value is failure


def test_implicit_library_conversion_errors_are_not_globally_wrapped():
    with pytest.raises(UnitConversionError):
        BinaryInitialProperties.period(1.0 * u.m)


def test_valid_conversions_are_unchanged():
    assert BinaryInitialProperties.period(86400 * u.s) == 1.0
    assert BinaryInitialProperties.eccentricity(0.5) == 0.5
    assert BinaryInitialProperties.inclination(90.0) == 90.0
    assert BinaryInitialProperties.mass(1.0) == 1.0
    assert StarProperties.mass(1.0) == u.solMass.to(u.kg)


@pytest.mark.parametrize("properties,name,value", [
    (BinaryInitialProperties, "eccentricity", 0.5),
    (BinaryInitialProperties, "inclination", 90.0),
    (StarInitialProperties, "mass", 1.0),
    (SpotInitialProperties, "latitude", 45.0),
    (PulsationModeInitialProperties, "mode_axis_theta", 45.0),
])
def test_aliases_keep_static_method_behavior(properties, name, value):
    assert getattr(properties(), name)(value) == getattr(properties, name)(value)
