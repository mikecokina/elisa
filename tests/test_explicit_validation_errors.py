"""Explicit validation raises and initial-parameter adapters keep their categories."""
import numpy as np
import pytest
from astropy.units import UnitConversionError

from elisa import units as u
from elisa.analytics.dataset.base import LCData
from elisa.analytics.dataset.transform import RVDataProperties
from elisa.analytics.params.parameters import (
    BinaryInitialParameters, PulsationInitialParameters, SpotInitialParameters, StarInitialParameters,
)
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
    ElisaMissingKwargsError,
    ElisaParametersValidationError,
)
from elisa.base.transform import StarProperties
from elisa.base.spot import Spot
from elisa.base.star import Star
from elisa.binary_system.transform import BinarySystemProperties
from elisa.pulse.mode import PulsationMode


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


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("phenomenon,values,message", [
    ("pulsations", dict(l=1, m=2, amplitude=1.0, frequency=1.0), "azimuthal order"),
    ("pulsations", dict(l=1, m=-2, amplitude=1.0, frequency=1.0), "azimuthal order"),
    ("spots", dict(longitude=0, latitude=60, angular_radius=0, temperature_factor=0.9), "angular radius"),
    ("spots", dict(longitude=0, latitude=60, angular_radius=20, temperature_factor="invalid"), "temperature_factor"),
    ("spots", dict(longitude=0, latitude=60, angular_radius=20, temperature_factor=0.9,
                   discretization_factor=0), "discretization factor"),
])
def test_phenomenon_validation_uses_model_error_through_star_construction(nested, phenomenon, values, message):
    with pytest.raises(ElisaParametersValidationError, match=message):
        if nested:
            Star(mass=1.0, t_eff=6000, **{phenomenon: [values]})
        else:
            constructor = PulsationMode if phenomenon == "pulsations" else Spot
            constructor(**values)


@pytest.mark.parametrize("phenomenon,properties,parameter,value", [
    ("pulsations", PulsationModeInitialProperties, "l", -1),
    ("pulsations", PulsationModeInitialProperties, "m", 1.5),
    ("pulsations", PulsationModeInitialProperties, "amplitude", -1.0),
    ("pulsations", PulsationModeInitialProperties, "start_phase", object()),
    ("spots", SpotInitialProperties, "temperature_factor", "invalid"),
])
def test_initial_phenomenon_validation_preserves_cause_through_star_parameters(
    phenomenon, properties, parameter, value,
):
    with pytest.raises(ElisaInitialParametersValidationError) as direct:
        getattr(properties, parameter)(value)
    assert type(direct.value.__cause__) is ElisaParametersValidationError

    definition = {"label": "feature", parameter: {"value": value, "fixed": True}}
    with pytest.raises(ElisaInitialParametersValidationError) as nested:
        StarInitialParameters(**{phenomenon: [definition]})
    assert type(nested.value.__cause__) is ElisaParametersValidationError
    assert str(nested.value) == str(direct.value)


def test_valid_spots_and_pulsations_keep_their_values_in_both_layers():
    spot = dict(longitude=0.0, latitude=60.0, angular_radius=20.0, temperature_factor=0.9)
    mode = dict(l=1, m=-1, amplitude=1.0, frequency=1.0, start_phase=0.0)
    star = Star(mass=1.0, t_eff=6000, spots=[spot], pulsations=[mode])
    initial = StarInitialParameters(**{
        phenomenon: [{"label": "feature", **{
            name: {"value": value, "fixed": True} for name, value in values.items()
        }}]
        for phenomenon, values in (("spots", spot), ("pulsations", mode))
    })

    assert star.spots[0].temperature_factor == initial.spots[0].temperature_factor.value == 0.9
    assert star.pulsations[0].m == initial.pulsations[0].m.value == -1
    assert star.pulsations[0].frequency == pytest.approx(1.0 / 86400)
    assert initial.pulsations[0].frequency.value == 1.0


@pytest.mark.parametrize("constructor,kwargs,missing", [
    (BinaryInitialParameters, {}, "system"),
    (BinaryInitialParameters, {"primary": {}}, "system"),
    (SpotInitialParameters, {}, "label"),
    (SpotInitialParameters, {"longitude": {"value": 0.0, "fixed": True}}, "label"),
    (PulsationInitialParameters, {}, "label"),
    (PulsationInitialParameters, {"l": {"value": 1, "fixed": True}}, "label"),
])
def test_missing_initial_structure_keys_report_argument_and_constructor(constructor, kwargs, missing):
    with pytest.raises(ElisaMissingKwargsError) as caught:
        constructor(**kwargs)
    assert str(caught.value) == f"Missing argument(s): `{missing}` in class instance {constructor.__name__}"


@pytest.mark.parametrize("phenomenon,constructor,definition", [
    ("spots", SpotInitialParameters, {"longitude": {"value": 0.0, "fixed": True}}),
    ("pulsations", PulsationInitialParameters, {"l": {"value": 1, "fixed": True}}),
])
def test_missing_nested_phenomenon_label_keeps_its_own_constructor_context(phenomenon, constructor, definition):
    with pytest.raises(ElisaMissingKwargsError) as caught:
        StarInitialParameters(**{phenomenon: [definition]})
    assert str(caught.value) == f"Missing argument(s): `label` in class instance {constructor.__name__}"
