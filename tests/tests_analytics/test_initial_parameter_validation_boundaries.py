"""Initial definition structure, metadata names and local unit-conversion failures."""

from copy import deepcopy
from types import MappingProxyType

import numpy as np
import pytest
from astropy.units import UnitConversionError

from elisa import units as u
from elisa.analytics.params.parameters import (
    BinaryInitialParameters,
    InitialParameter,
    NuisanceInitialPrameters,
    PulsationInitialParameters,
    SpotInitialParameters,
    StarInitialParameters,
)
from elisa.analytics.params.transform import (
    BinaryInitialProperties,
    PulsationModeInitialProperties,
    SpotInitialProperties,
    StarInitialProperties,
)
from elisa.base.error import (
    ElisaInitialParametersError,
    ElisaInvalidKwargsError,
    ElisaMissingKwargsError,
    ElisaParametersError,
)
from elisa.pulse.transform import PulsationModeProperties


SECTION_CASES = [
    (BinaryInitialParameters, "period"),
    (StarInitialParameters, "t_eff"),
    (SpotInitialParameters, "longitude"),
    (PulsationInitialParameters, "frequency"),
    (NuisanceInitialPrameters, "ln_f"),
]


def section(constructor, definitions):
    if constructor is BinaryInitialParameters:
        return constructor(system=definitions)
    if constructor in (SpotInitialParameters, PulsationInitialParameters):
        return constructor(label="feature", **definitions)
    return constructor(**definitions)


def binary_definitions():
    return {"system": {"period": {"value": 1.0, "min": 0.5, "max": 2.0}}}


@pytest.mark.parametrize("constructor,parameter", SECTION_CASES)
@pytest.mark.parametrize("definition", [None, 1, "invalid", [], [1, 2]])
def test_scalar_parameter_definition_must_be_a_mapping(constructor, parameter, definition):
    with pytest.raises(ElisaInitialParametersError, match=parameter):
        section(constructor, {parameter: definition})


@pytest.mark.parametrize("constructor,parameter", SECTION_CASES)
def test_unknown_parameter_name_is_rejected_before_default_lookup(constructor, parameter):
    with pytest.raises(ElisaInvalidKwargsError, match="unknown"):
        section(constructor, {"unknown": {"value": 1.0}})


@pytest.mark.parametrize("definition", [None, 1, "invalid", [], [("period", {"value": 1})]])
def test_required_system_section_rejects_non_mappings(definition):
    with pytest.raises(ElisaInitialParametersError, match="system"):
        BinaryInitialParameters(system=definition)


@pytest.mark.parametrize("name", ["primary", "secondary", "nuisance"])
@pytest.mark.parametrize("definition", [1, "invalid", [], [1, 2]])
def test_optional_section_rejects_wrong_structure_when_supplied(name, definition):
    definitions = binary_definitions()
    definitions[name] = definition
    with pytest.raises(ElisaInitialParametersError, match=name):
        BinaryInitialParameters(**definitions)


def test_unknown_top_level_section_is_not_silently_ignored():
    with pytest.raises(ElisaInvalidKwargsError, match="tertiary"):
        BinaryInitialParameters(**binary_definitions(), tertiary={})


@pytest.mark.parametrize("name", ["spots", "pulsations"])
@pytest.mark.parametrize("definitions", [1, "", "invalid", {}, {"label": "feature"}, np.array(1)])
def test_phenomena_require_a_collection_of_definitions(name, definitions):
    with pytest.raises(ElisaInitialParametersError, match=name):
        StarInitialParameters(**{name: definitions})


@pytest.mark.parametrize("name", ["spots", "pulsations"])
@pytest.mark.parametrize("definition", [None, 1, "invalid", []])
def test_invalid_phenomenon_item_reports_its_collection_index(name, definition):
    with pytest.raises(ElisaInitialParametersError, match=rf"{name}\[1\]"):
        StarInitialParameters(**{name: [{"label": "valid"}, definition]})


@pytest.mark.parametrize("name", ["spots", "pulsations"])
def test_missing_nested_label_keeps_missing_argument_category(name):
    with pytest.raises(ElisaMissingKwargsError, match="label"):
        StarInitialParameters(**{name: [{}]})


@pytest.mark.parametrize("constructor,parameter", SECTION_CASES)
def test_unknown_metadata_is_rejected_for_every_parameter_section(constructor, parameter):
    with pytest.raises(ElisaInvalidKwargsError, match="unknown"):
        section(constructor, {parameter: {"value": 1.0, "fixed": True, "unknown": 2}})


@pytest.mark.parametrize("metadata", [
    {"value": 1.0, "fixed": True},
    {"value": 1.0, "min": 0.5, "max": 2.0},
    {"constraint": "2 * system@period"},
])
def test_direct_initial_parameter_rejects_unknown_metadata_in_every_kind(metadata):
    with pytest.raises(ElisaInvalidKwargsError, match="unknown"):
        InitialParameter(BinaryInitialProperties, param="period", unknown=2, **metadata)


def test_unknown_metadata_is_reported_before_invalid_unit_conversion():
    with pytest.raises(ElisaInvalidKwargsError, match="unknown"):
        InitialParameter(BinaryInitialProperties, param="period", value=1, unit="nonsense", unknown=2)


@pytest.mark.parametrize("constructor,parameter", SECTION_CASES)
def test_parameter_name_cannot_be_overridden_by_nested_metadata(constructor, parameter):
    with pytest.raises(ElisaInvalidKwargsError, match="param"):
        section(constructor, {parameter: {"param": "other", "value": 1.0}})


@pytest.mark.parametrize("constructor,definition", [
    (StarInitialParameters, {1: {"value": 1}}),
    (SpotInitialParameters, {"label": "feature", 1: {"value": 1}}),
    (PulsationInitialParameters, {"label": "feature", 1: {"value": 1}}),
])
def test_nested_mapping_keys_must_be_argument_names(constructor, definition):
    with pytest.raises(ElisaInvalidKwargsError, match="non-string"):
        constructor.from_definition(definition, context="definition")


def test_parameter_metadata_keys_must_be_argument_names():
    with pytest.raises(ElisaInvalidKwargsError, match="non-string"):
        BinaryInitialParameters(system={"period": {1: 1}})


def test_valid_nested_definitions_preserve_values_and_caller_data():
    definitions = binary_definitions()
    definitions.update(
        primary={
            "mass": {"value": 1.0, "fixed": True},
            "spots": [{"label": "feature", "longitude": {"value": 30, "fixed": True}}],
            "pulsations": [{"label": "mode", "frequency": {"value": 2, "fixed": True}}],
        },
        secondary=None,
        nuisance=None,
    )
    original = deepcopy(definitions)
    initial = BinaryInitialParameters.from_definition(MappingProxyType(definitions), context="binary")
    assert isinstance(initial, BinaryInitialParameters)
    assert isinstance(initial.primary, StarInitialParameters)
    assert isinstance(initial.primary.spots[0], SpotInitialParameters)
    assert isinstance(initial.primary.pulsations[0], PulsationInitialParameters)
    assert initial.primary.mass.value == 1.0
    assert initial.primary.spots[0].longitude.value == 30.0
    assert initial.primary.pulsations[0].frequency.value == 2.0
    assert initial.secondary is None
    assert initial.nuisance.ln_f.value == -20.0
    assert definitions == original


@pytest.mark.parametrize("optional", [None, {}])
def test_optional_component_defaults_still_work(optional):
    definitions = binary_definitions()
    definitions.update(primary=optional, secondary=optional)
    initial = BinaryInitialParameters(**definitions)
    assert initial.primary is None
    assert initial.secondary is None


@pytest.mark.parametrize("name", ["spots", "pulsations"])
@pytest.mark.parametrize("definitions", [None, [], ()])
def test_optional_phenomena_defaults_still_work(name, definitions):
    initial = StarInitialParameters(**{name: definitions})
    assert not getattr(initial, name, [])


def test_nested_factory_accepts_iterable_definitions_without_consuming_them_twice():
    definitions = ({"label": label} for label in ("first", "second"))
    spots = SpotInitialParameters.from_definitions(definitions, context="spots")
    assert [spot.label for spot in spots] == ["spot@first", "spot@second"]


@pytest.mark.parametrize("field", ["value", "min", "max", "sigma"])
def test_unit_attachment_failure_reports_parameter_field_and_original_cause(field):
    metadata = dict(value=1.0, min=0.5, max=2.0, sigma=0.1, unit="d")
    metadata[field] = "invalid"
    with pytest.raises(ElisaInitialParametersError, match=field) as caught:
        InitialParameter(BinaryInitialProperties, param="period", **metadata)
    assert "period" in str(caught.value)
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))


def test_unit_parser_failure_reports_parameter_and_original_cause():
    with pytest.raises(ElisaInitialParametersError, match="unit") as caught:
        InitialParameter(BinaryInitialProperties, param="period", value=1, min=0.5, max=2, unit="nonsense")
    assert "period" in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)


def test_known_metadata_still_converts_units_and_preserves_the_input():
    metadata = dict(value=24, min=12, max=48, sigma=2.4, unit="h", fixed=False, constraint=None)
    original = deepcopy(metadata)
    initial = InitialParameter(BinaryInitialProperties, param="period", **metadata)
    assert initial.value == 1.0
    assert initial.min == 0.5
    assert initial.max == 2.0
    assert initial.sigma == pytest.approx(0.1)
    assert initial.unit == u.d
    assert metadata == original


@pytest.mark.parametrize("validator,unit,value", [
    (BinaryInitialProperties.inclination, u.deg, 90),
    (BinaryInitialProperties.argument_of_periastron, u.deg, 30),
    (SpotInitialProperties.longitude, u.deg, 45),
    (SpotInitialProperties.latitude, u.deg, 60),
    (SpotInitialProperties.angular_radius, u.deg, 20),
    (PulsationModeInitialProperties.mode_axis_theta, u.deg, 30),
    (PulsationModeInitialProperties.mode_axis_phi, u.deg, 45),
    (BinaryInitialProperties.mass, u.solMass, 1),
    (StarInitialProperties.mass, u.solMass, 2),
    (BinaryInitialProperties.semi_major_axis, u.solRad, 3),
])
def test_initial_values_accept_numbers_quantities_and_quantity_strings(validator, unit, value):
    for supplied in (value, value * unit, f"{value} {unit}"):
        assert validator(supplied) == pytest.approx(value)


def test_initial_angles_stay_in_degrees_when_supplied_in_radians():
    for supplied in (np.pi / 2 * u.rad, f"{np.pi / 2} rad"):
        assert BinaryInitialProperties.inclination(supplied) == pytest.approx(90)


@pytest.mark.parametrize("validator,parameter", [
    (BinaryInitialProperties.mass, "mass"),
    (BinaryInitialProperties.semi_major_axis, "semi_major_axis"),
    (BinaryInitialProperties.inclination, "inclination"),
    (SpotInitialProperties.longitude, "longitude"),
])
@pytest.mark.parametrize("supplied", ["nonsense", 1 * u.s, 10 ** 400])
def test_local_initial_conversions_keep_parameter_name_and_native_cause(validator, parameter, supplied):
    with pytest.raises(ElisaInitialParametersError, match=parameter) as caught:
        validator(supplied)
    assert isinstance(caught.value.__cause__, (TypeError, ValueError, ArithmeticError))


@pytest.mark.parametrize("value,expected_initial,expected_model", [
    (1.0, 1.0, 1 / 86400),
    (1 / u.d, 1.0, 1 / 86400),
    ("1 / d", 1.0, 1 / 86400),
    (1 * u.FREQUENCY_UNIT, 86400.0, 1.0),
    ("1 Hz", 86400.0, 1.0),
])
def test_frequency_reconstruction_keeps_cycles_per_day_and_hz_conventions(value, expected_initial, expected_model):
    initial = PulsationModeInitialProperties.frequency(value)
    assert initial == pytest.approx(expected_initial)
    assert PulsationModeProperties.frequency(initial) == pytest.approx(expected_model)


def test_delegated_initial_frequency_preserves_both_model_and_unit_causes():
    with pytest.raises(ElisaInitialParametersError, match="frequency") as caught:
        PulsationModeInitialProperties.frequency(1 * u.m)
    assert type(caught.value.__cause__) is ElisaParametersError
    assert isinstance(caught.value.__cause__.__cause__, UnitConversionError)
