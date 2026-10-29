"""Explicit validation raises and initial-parameter adapters keep their categories."""
from copy import deepcopy

import numpy as np
import pytest
from astropy.units import UnitConversionError

from elisa import settings
from elisa import units as u
from elisa.analytics.dataset.base import LCData
from elisa.analytics.dataset.transform import RVDataProperties
from elisa.analytics.binary_fit.lc_fit import LCFitLeastSquares, LCFitMCMC
from elisa.analytics.binary_fit.shared import check_for_boundary_surface_potentials
from elisa.analytics.params.parameters import (
    BinaryInitialParameters, PulsationInitialParameters, SpotInitialParameters, StarInitialParameters,
)
from elisa.analytics.params.transform import (
    BinaryInitialProperties, PulsationModeInitialProperties, SpotInitialProperties,
    StarInitialProperties, _initial_parameter,
)
from elisa.analytics.tasks import LCBinaryAnalyticsTask
from elisa.analytics.tools.bvi import elisa_bv_temperature
from elisa.analytics.transform import LCBinaryAnalyticsProperties
from elisa.base.error import (
    ElisaAnalyticsDataValidationError,
    ElisaAnalyticsTaskValidationError,
    ElisaError,
    ElisaInitialParametersValidationError,
    ElisaMissingKwargsError,
    ElisaParametersValidationError,
)
from elisa.base.transform import SpotProperties, StarProperties
from elisa.base.spot import Spot
from elisa.base.star import Star
from elisa.binary_system.transform import BinarySystemProperties
from elisa.pulse.mode import PulsationMode
from elisa.pulse.transform import PulsationModeProperties


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


@pytest.mark.parametrize("initial,model,parameter,invalid,valid", [
    (StarInitialProperties, StarProperties, "albedo", 2.0, 0.5),
    (StarInitialProperties, StarProperties, "discretization_factor", -1.0, 5.0),
    (StarInitialProperties, StarProperties, "equivalent_radius", 0.0, 1.0),
    (StarInitialProperties, StarProperties, "gravity_darkening", 2.0, 0.3),
    (StarInitialProperties, StarProperties, "limb_darkening_coefficients", "invalid", None),
    (StarInitialProperties, StarProperties, "metallicity", np.nan, 0.0),
    (StarInitialProperties, StarProperties, "polar_log_g", object(), 4.2),
    (StarInitialProperties, StarProperties, "surface_potential", -1.0, 3.0),
    (StarInitialProperties, StarProperties, "synchronicity", 0.0, 1.0),
    (StarInitialProperties, StarProperties, "t_eff", -1.0, 6000.0),
    (SpotInitialProperties, SpotProperties, "albedo", 2.0, 0.5),
    (SpotInitialProperties, SpotProperties, "discretization_factor", -1.0, 5.0),
    (SpotInitialProperties, SpotProperties, "synchronicity", 0.0, 1.0),
    (SpotInitialProperties, SpotProperties, "t_eff", -1.0, 6000.0),
    (PulsationModeInitialProperties, PulsationModeProperties, "additional_light", 2.0, 0.1),
    (PulsationModeInitialProperties, PulsationModeProperties, "distance", 0.0, 10.0),
    (PulsationModeInitialProperties, PulsationModeProperties, "gamma", np.inf, 0.0),
    (PulsationModeInitialProperties, PulsationModeProperties, "horizontal_to_radial_amplitude_ratio", object(), 0.5),
    (PulsationModeInitialProperties, PulsationModeProperties, "inclination", object(), 90.0),
    (PulsationModeInitialProperties, PulsationModeProperties, "period", 0.0, 1.0),
    (PulsationModeInitialProperties, PulsationModeProperties, "semi_major_axis", 0.0, 1.0),
    (PulsationModeInitialProperties, PulsationModeProperties, "temperature_perturbation_phase_shift", object(), 90.0),
    (PulsationModeInitialProperties, PulsationModeProperties, "tidally_locked", "yes", True),
])
def test_delegated_initial_validators_keep_their_layer_and_valid_conversions(initial, model, parameter, invalid, valid):
    with pytest.raises(ElisaParametersValidationError) as original:
        getattr(model, parameter)(invalid)

    if parameter == "limb_darkening_coefficients":
        valid = {"bolometric": [0.1] * len(settings.LD_LAW_CFS_COLUMNS[settings.LIMB_DARKENING_LAW])}
    expected = getattr(model, parameter)(valid)

    def transform(value):
        return initial.transform_input(**{parameter: value})[parameter]

    for validate in (getattr(initial(), parameter), transform):
        with pytest.raises(ElisaInitialParametersValidationError) as caught:
            validate(invalid)
        assert type(caught.value.__cause__) is ElisaParametersValidationError
        assert str(caught.value) == str(caught.value.__cause__) == str(original.value)
        converted = validate(valid)
        assert converted == expected
        assert type(converted) is type(expected)


@pytest.mark.parametrize("parameter,value", [
    ("t_eff", -1.0), ("albedo", 2.0), ("gravity_darkening", 2.0),
    ("metallicity", np.nan), ("surface_potential", -1.0), ("synchronicity", 0.0),
])
def test_initial_binary_star_parameters_preserve_the_model_validation_cause(parameter, value):
    with pytest.raises(ElisaInitialParametersValidationError) as caught:
        BinaryInitialParameters(
            system={"period": {"value": 1.0, "min": 0.5, "max": 2.0}},
            primary={parameter: {"value": value, "fixed": True}},
        )
    assert type(caught.value.__cause__) is ElisaParametersValidationError
    assert str(caught.value) == str(caught.value.__cause__)


@pytest.fixture
def lc_task_data():
    return {"bolometric": LCData(x_data=[0.0, 0.5], y_data=[1.0, 0.9], x_unit=None, y_unit=None)}


@pytest.mark.parametrize("method", ["least_squares", "mcmc"])
@pytest.mark.parametrize("morphology", [
    "invalid", "", "semi-detached", "Detached", None, 0, [], {}, np.array(["detached", "overcontact"]),
])
def test_lc_task_rejects_invalid_morphology_with_task_validation_error(method, morphology, lc_task_data):
    with pytest.raises(ElisaAnalyticsTaskValidationError) as caught:
        LCBinaryAnalyticsTask(method=method, expected_morphology=morphology, data=lc_task_data)
    message = str(caught.value)
    assert "expected_morphology" in message
    assert repr(morphology) in message
    for supported in ("detached", "over-contact", "overcontact"):
        assert supported in message


@pytest.mark.parametrize("method", ["least_squares", "mcmc"])
@pytest.mark.parametrize("morphology", ["detached", "over-contact", "overcontact"])
def test_lc_task_accepts_supported_morphologies_and_alias(method, morphology, lc_task_data):
    task = LCBinaryAnalyticsTask(method=method, expected_morphology=morphology, data=lc_task_data)
    expected = "detached" if morphology == "detached" else "over-contact"
    assert task.method == method
    assert task.fit_cls.morphology == expected
    assert task.fit_cls.fit_method_instance.MORPHOLOGY == expected
    assert BinaryInitialParameters.is_overcontact(morphology) is (expected == "over-contact")


@pytest.mark.parametrize("method", ["least_squares", "mcmc"])
def test_lc_task_preserves_default_detached_morphology(method, lc_task_data):
    task = LCBinaryAnalyticsTask(method=method, data=lc_task_data)
    assert task.fit_cls.morphology == "detached"


@pytest.mark.parametrize("constructor", [LCFitLeastSquares, LCFitMCMC])
@pytest.mark.parametrize("morphology", ["detached", "over-contact", "overcontact"])
def test_direct_lc_fit_construction_normalizes_morphology(constructor, morphology):
    fit = constructor(morphology, None, None)
    expected = "detached" if morphology == "detached" else "over-contact"
    assert fit.morphology == fit.fit_method_instance.MORPHOLOGY == expected


@pytest.mark.parametrize("method", ["least_squares", "mcmc"])
@pytest.mark.parametrize("morphology", ["over-contact", "overcontact"])
def test_lc_task_morphology_preserves_overcontact_potential_correction(method, morphology, lc_task_data):
    task = LCBinaryAnalyticsTask(method=method, expected_morphology=morphology, data=lc_task_data)
    results = {
        "primary@surface_potential": {"value": 3.0, "fixed": False},
        "secondary@surface_potential": {"value": 3.01, "fixed": False},
        "system@mass_ratio": {"value": 1.0},
        "system@eccentricity": {"value": 0.0},
    }
    corrected = check_for_boundary_surface_potentials(deepcopy(results), task.fit_cls.morphology)
    assert corrected["primary@surface_potential"]["value"] == corrected["secondary@surface_potential"]["value"] == 3.0


def test_bv_temperature_accepts_canonical_and_legacy_overcontact_names():
    legacy = elisa_bv_temperature(0.3, morphology="overcontact")
    canonical = elisa_bv_temperature(0.3, morphology="over-contact")
    assert canonical == pytest.approx(legacy)
    assert canonical != pytest.approx(elisa_bv_temperature(0.3, morphology="detached"))
    assert elisa_bv_temperature(0.3, morphology="unknown") == elisa_bv_temperature(0.3, morphology="detached")
