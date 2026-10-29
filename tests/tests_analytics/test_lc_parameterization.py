"""LC fitting accepts complete physical parameterizations without mixed masses."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from unittest.mock import patch

import pytest
from jsonschema import ValidationError

from elisa.analytics.binary_fit.lc_fit import LCFit, LCFitLeastSquares, LCFitMCMC
from elisa.analytics.params.parameters import BinaryInitialParameters
from tests.tests_analytics.fitting_helpers import lc_parameters


STANDARD_PARAMETERS = ("primary@mass", "secondary@mass")
COMMUNITY_PARAMETERS = ("system@mass_ratio", "system@semi_major_axis")


@pytest.fixture(params=["detached", "over-contact"])
def morphology(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(
    params=[
        pytest.param({"value": 1.0, "fixed": True}, id="fixed"),
        pytest.param({"value": 1.0, "fixed": False, "min": 0.1, "max": 2.0}, id="free"),
        pytest.param({"constraint": "1.0"}, id="constrained"),
    ],
)
def parameter_definition(request: pytest.FixtureRequest) -> dict[str, Any]:
    return deepcopy(request.param)


@pytest.fixture
def lc_config() -> dict[str, Any]:
    """Keep required orbital/stellar inputs and an independent free inclination."""
    config, _ = lc_parameters()
    del config["system"]["mass_ratio"]
    del config["system"]["semi_major_axis"]
    return config


def add_parameters(
    config: dict[str, Any], names: tuple[str, ...], definition: dict[str, Any],
) -> BinaryInitialParameters:
    """Give each physical input independent fixed, free or constrained metadata."""
    for name in names:
        component, property_name = name.split("@")
        config[component][property_name] = deepcopy(definition)
    return BinaryInitialParameters(**config)


@pytest.mark.parametrize(
    "parameter_names",
    [
        pytest.param((), id="no-mass-parameterization"),
        pytest.param(("primary@mass",), id="primary-mass-only"),
        pytest.param(("secondary@mass",), id="secondary-mass-only"),
        pytest.param(("system@mass_ratio",), id="community-without-axis"),
        pytest.param(("system@semi_major_axis",), id="community-without-ratio"),
    ],
)
def test_incomplete_lc_parameterization_is_rejected(
    lc_config: dict[str, Any],
    parameter_definition: dict[str, Any],
    parameter_names: tuple[str, ...],
    morphology: str,
) -> None:
    initial = add_parameters(lc_config, parameter_names, parameter_definition)

    with pytest.raises(ValidationError, match="Incomplete LC parameterization"):
        initial.validate_lc_parameters(morphology)


@pytest.mark.parametrize(
    "parameter_names",
    [
        pytest.param(COMMUNITY_PARAMETERS + ("primary@mass",), id="community-with-primary-mass"),
        pytest.param(COMMUNITY_PARAMETERS + ("secondary@mass",), id="community-with-secondary-mass"),
        pytest.param(STANDARD_PARAMETERS + ("system@mass_ratio",), id="standard-with-ratio"),
        pytest.param(STANDARD_PARAMETERS + ("system@semi_major_axis",), id="standard-with-axis"),
        pytest.param(STANDARD_PARAMETERS + COMMUNITY_PARAMETERS, id="both-complete-parameterizations"),
        pytest.param(("primary@mass", "system@mass_ratio"), id="partial-masses-with-ratio"),
        pytest.param(("secondary@mass", "system@semi_major_axis"), id="partial-masses-with-axis"),
    ],
)
def test_mixed_lc_parameterizations_are_rejected(
    lc_config: dict[str, Any],
    parameter_definition: dict[str, Any],
    parameter_names: tuple[str, ...],
    morphology: str,
) -> None:
    initial = add_parameters(lc_config, parameter_names, parameter_definition)

    with pytest.raises(ValidationError, match="LC parameters must use either"):
        initial.validate_lc_parameters(morphology)


@pytest.mark.parametrize(
    "parameter_names",
    [
        pytest.param(STANDARD_PARAMETERS, id="standard"),
        pytest.param(COMMUNITY_PARAMETERS, id="community"),
    ],
)
def test_complete_lc_parameterization_is_accepted(
    lc_config: dict[str, Any],
    parameter_definition: dict[str, Any],
    parameter_names: tuple[str, ...],
    morphology: str,
) -> None:
    initial = add_parameters(lc_config, parameter_names, parameter_definition)

    initial.validate_lc_parameters(morphology)

    assert set(parameter_names) <= initial.data.keys()


@pytest.fixture
def stellar_phenomena() -> dict[str, Any]:
    return {
        "spots": [{
            "label": "cool",
            "longitude": {"value": 90.0, "fixed": True},
            "latitude": {"value": 60.0, "fixed": True},
            "angular_radius": {"value": 15.0, "fixed": True},
            "temperature_factor": {"value": 0.9, "fixed": True},
        }],
        "pulsations": [{
            "label": "mode",
            "l": {"value": 1, "fixed": True},
            "m": {"value": 0, "fixed": True},
            "amplitude": {"value": 50.0, "fixed": True},
            "frequency": {"value": 16.0, "fixed": True},
            "start_phase": {"value": 0.0, "fixed": True},
            "mode_axis_phi": {"value": 0.0, "fixed": True},
            "mode_axis_theta": {"value": 30.0, "fixed": True},
        }],
    }


@pytest.mark.parametrize("standard", [True, False], ids=["standard", "community"])
def test_valid_lc_inputs_preserve_optional_parameters_and_stellar_phenomena(
    standard: bool, stellar_phenomena: dict[str, Any], morphology: str,
) -> None:
    config, _ = lc_parameters(standard=standard)
    config["system"]["primary_minimum_time"] = {"value": 2450000.0, "fixed": True}
    config["system"]["phase_shift"] = {"value": 0.1, "fixed": True}
    config["system"]["additional_light"] = {"value": 0.05, "fixed": True}
    config["primary"].update(stellar_phenomena)
    initial = BinaryInitialParameters(**config)
    original_names = set(initial.data)

    initial.validate_lc_parameters(morphology)

    assert set(initial.data) == original_names
    assert initial.data["nuisance@ln_f"].value == -2.0
    assert initial.data["primary@spot@cool@angular_radius"].value == 15.0
    assert initial.data["primary@pulsation@mode@frequency"].value == 16.0
    assert initial.data["secondary@t_eff"].constraint == "primary@t_eff * 0.7692307692307693"


@pytest.mark.parametrize("fit_class", [LCFitLeastSquares, LCFitMCMC], ids=["lsq", "mcmc"])
def test_public_lc_fit_rejects_a_constrained_mass_before_forward_model_preparation(
    fit_class: type[LCFit],
) -> None:
    config, _ = lc_parameters()
    config["primary"]["mass"] = {"constraint": "-1"}
    initial = BinaryInitialParameters(**config)
    fit = fit_class("detached", None, None)

    with patch.object(fit.fit_method_instance, "fit") as run_fit:
        with pytest.raises(ValidationError, match="LC parameters must use either"):
            fit.fit(x0=initial, data={})

    run_fit.assert_not_called()
    assert initial.data["primary@mass"].constraint == "-1"
