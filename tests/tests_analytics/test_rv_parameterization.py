"""RV parameter validation accepts complete alternatives and rejects mixed inputs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from jsonschema import ValidationError

from elisa.analytics.params.parameters import BinaryInitialParameters


STANDARD_PARAMETERS = ("primary@mass", "secondary@mass", "system@inclination")
COMMUNITY_PARAMETERS = ("system@mass_ratio", "system@asini")


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
def orbital_config() -> dict[str, Any]:
    # Gamma is independently free so every case is a valid fitting metadata object.
    return {
        "system": {
            "period": {"value": 3.0, "fixed": True},
            "eccentricity": {"value": 0.15, "fixed": True},
            "argument_of_periastron": {"value": 25.0, "fixed": True},
            "gamma": {"value": 12000.0, "fixed": False, "min": 10000.0, "max": 14000.0},
        },
    }


def add_parameters(
    config: dict[str, Any], names: tuple[str, ...], definition: dict[str, Any],
) -> BinaryInitialParameters:
    """Build initial metadata without sharing mutable definitions across parameters."""
    for name in names:
        component, property_name = name.split("@")
        component_config = config.setdefault(component, {})
        component_config[property_name] = deepcopy(definition)
    return BinaryInitialParameters(**config)


@pytest.mark.parametrize(
    "parameter_names",
    [
        pytest.param((), id="no-mass-parameterization"),
        pytest.param(("primary@mass",), id="primary-mass-only"),
        pytest.param(("secondary@mass",), id="secondary-mass-only"),
        pytest.param(("primary@mass", "secondary@mass"), id="standard-without-inclination"),
        pytest.param(("system@mass_ratio",), id="community-without-asini"),
        pytest.param(("system@asini",), id="community-without-ratio"),
    ],
)
def test_incomplete_parameterization_is_rejected(
    orbital_config: dict[str, Any], parameter_definition: dict[str, Any], parameter_names: tuple[str, ...],
) -> None:
    initial = add_parameters(orbital_config, parameter_names, parameter_definition)

    with pytest.raises(ValidationError, match="Incomplete RV parameterization"):
        initial.validate_rv_parameters()


@pytest.mark.parametrize(
    "parameter_names",
    [
        pytest.param(COMMUNITY_PARAMETERS + ("primary@mass",), id="community-with-primary-mass"),
        pytest.param(COMMUNITY_PARAMETERS + ("secondary@mass",), id="community-with-secondary-mass"),
        pytest.param(STANDARD_PARAMETERS + ("system@mass_ratio",), id="standard-with-ratio"),
        pytest.param(STANDARD_PARAMETERS + ("system@asini",), id="standard-with-asini"),
        pytest.param(STANDARD_PARAMETERS + COMMUNITY_PARAMETERS, id="both-complete-parameterizations"),
    ],
)
def test_conflicting_parameterizations_are_rejected(
    orbital_config: dict[str, Any], parameter_definition: dict[str, Any], parameter_names: tuple[str, ...],
) -> None:
    initial = add_parameters(orbital_config, parameter_names, parameter_definition)

    with pytest.raises(ValidationError, match="either component masses or mass_ratio and asini"):
        initial.validate_rv_parameters()


@pytest.mark.parametrize(
    "parameter_names",
    [
        pytest.param(STANDARD_PARAMETERS, id="standard"),
        pytest.param(COMMUNITY_PARAMETERS, id="community"),
        pytest.param(COMMUNITY_PARAMETERS + ("system@inclination",), id="community-with-inclination"),
    ],
)
def test_complete_parameterization_preserves_optional_orbital_and_nuisance_inputs(
    orbital_config: dict[str, Any], parameter_definition: dict[str, Any], parameter_names: tuple[str, ...],
) -> None:
    orbital_config["system"]["primary_minimum_time"] = {"value": 2450000.0, "fixed": True}
    orbital_config["nuisance"] = {"ln_f": {"value": -3.0, "fixed": True}}
    initial = add_parameters(orbital_config, parameter_names, parameter_definition)

    initial.validate_rv_parameters()

    assert set(parameter_names) <= initial.data.keys()
    assert {"system@primary_minimum_time", "nuisance@ln_f"} <= initial.data.keys()


def test_community_asini_can_depend_on_an_optional_free_inclination(orbital_config: dict[str, Any]) -> None:
    orbital_config["system"]["mass_ratio"] = {"value": 0.7, "fixed": True}
    orbital_config["system"]["inclination"] = {"value": 85.0, "fixed": False, "min": 80.0, "max": 90.0}
    orbital_config["system"]["asini"] = {"constraint": "10 * sin(radians(system@inclination))"}
    initial = BinaryInitialParameters(**orbital_config)

    initial.validate_rv_parameters()

    assert initial.data["system@asini"].constraint == "10 * sin(radians(system@inclination))"
