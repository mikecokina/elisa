"""Model JSON rejection uses the model exception and retains schema diagnostics."""

from copy import deepcopy

import pytest
from jsonschema import SchemaError, ValidationError

from elisa import settings
from elisa.base.error import ElisaParametersError
from elisa.binary_system import utils as binary_utils
from elisa.binary_system.system import BinarySystem
from elisa.single_system import utils as single_utils
from elisa.single_system.system import SingleSystem


@pytest.mark.parametrize("validator", [
    binary_utils.validate_binary_json, single_utils.validate_single_json,
    BinarySystem.from_json, SingleSystem.from_json,
])
@pytest.mark.parametrize("payload", [
    {}, None, [], [["system", {}]], "invalid", 1,
    {"system": None}, {"system": {}, "primary": None, "secondary": []},
])
def test_invalid_json_retains_schema_failure(validator, payload):
    with pytest.raises(ElisaParametersError) as caught:
        validator(payload)
    assert isinstance(caught.value.__cause__, ValidationError)
    assert caught.value.__cause__.schema is not None


def binary_payload(community=False):
    star = dict(surface_potential=10, synchronicity=1, t_eff=5772)
    system = dict(inclination=90, period=10, eccentricity=0,
                  argument_of_periastron=90, gamma=0)
    if community:
        system.update(mass_ratio=0.5, semi_major_axis=10)
    else:
        star["mass"] = 1
    return dict(system=system, primary=deepcopy(star), secondary=deepcopy(star))


@pytest.mark.parametrize("community", [False, True])
def test_binary_alternative_schemas_still_accept_valid_input(community):
    assert binary_utils.validate_binary_json(binary_payload(community)) is True


@pytest.mark.parametrize("community", [False, True])
def test_binary_mixed_parameterizations_use_model_error(community):
    payload = binary_payload(community)
    if community:
        payload["primary"]["mass"] = 1
    else:
        payload["system"]["mass_ratio"] = 0.5
    with pytest.raises(ElisaParametersError, match=r"use either \(M1, M2\) or \(q, a\)"):
        binary_utils.validate_binary_json(payload)


def test_schema_failure_keeps_nested_field_path():
    payload = binary_payload(community=True)
    payload["system"]["period"] = []
    with pytest.raises(ElisaParametersError) as caught:
        binary_utils.validate_binary_json(payload)
    assert list(caught.value.__cause__.absolute_path) == ["system", "period"]


@pytest.mark.parametrize("validator", [
    binary_utils.validate_binary_json, single_utils.validate_single_json,
])
def test_ambiguous_schemas_use_model_error(monkeypatch, validator):
    monkeypatch.setattr(settings.SCHEMA_REGISTRY, "get_schema", lambda name: {})
    with pytest.raises(ElisaParametersError, match="combination") as caught:
        validator({})
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("module", [binary_utils, single_utils])
@pytest.mark.parametrize("error_type", [RuntimeError, SchemaError])
def test_internal_failures_are_not_reclassified(monkeypatch, module, error_type):
    failure = error_type("internal failure")

    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(module, "validate", fail)
    validator = (module.validate_binary_json if module is binary_utils
                 else module.validate_single_json)
    with pytest.raises(error_type) as caught:
        validator({})
    assert caught.value is failure
