"""Result and chain JSON regressions without optimization or sampling."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from astropy.units import UnitBase

from elisa.analytics.binary_fit.lc_fit import LCFitLeastSquares
from elisa.analytics.binary_fit.mixins import MCMCMixin
from elisa.analytics.binary_fit.rv_fit import RVFitLeastSquares
from elisa.analytics.binary_fit.shared import AbstractFit
from elisa.analytics.params import parameters
from elisa.analytics.params.conf import DEFAULT_FLOAT_UNITS, JOINT_POSTERIOR_PERCENTILES_ESTIMATE
from elisa.analytics.params.parameters import BinaryInitialParameters, InitialParameter
from elisa.analytics.params.transform import BinaryInitialProperties
from elisa.json_encoder import ElisaJSONEncoder


@pytest.fixture(params=["lc", "rv"])
def lsq_result(request):
    if request.param == "lc":
        fit = LCFitLeastSquares("detached", None, None)
        free, constrained = "primary@t_eff", "secondary@t_eff"
        free_meta = {"value": 6500.0, "fixed": False, "min": 5000.0, "max": 8000.0, "unit": "K"}
        expression = "primary@t_eff * 0.8"
    else:
        fit = RVFitLeastSquares()
        free, constrained = "system@asini", "system@mass_ratio"
        free_meta = {"value": 10.0, "fixed": False, "min": 1.0, "max": 20.0, "unit": "solRad"}
        expression = "system@asini * 0.07"
    initial = BinaryInitialParameters(**parameters.serialize_result({
        free: free_meta,
        constrained: {"constraint": expression},
        "system@period": {"value": 2.0, "fixed": True, "unit": "d"},
    }))
    # Use the same result builder as LSQ completion, without running an optimizer.
    flat = AbstractFit.eval_constrained_results(
        {free: free_meta, "system@period": {"value": 2.0, "fixed": True, "unit": "d"}},
        initial.get_constrained(),
    )
    fit.result = parameters.serialize_result(flat)
    return fit, constrained


def test_ordinary_result_serialization_is_unchanged(tmp_path: Path, lsq_result) -> None:
    fit, constrained = lsq_result
    flat = parameters.deserialize_result(fit.result)
    flat[constrained].pop("constraint")
    flat[constrained]["fixed"] = True
    fit.set_result(parameters.serialize_result(flat))
    before = deepcopy(fit.result)
    path = tmp_path / "ordinary.json"
    fit.save_result(path)
    assert path.read_text() == json.dumps(before, separators=(",", ": "), indent=4)
    assert json.loads(path.read_text()) == before
    assert fit.result == before


def test_constrained_lsq_save_load_save(tmp_path: Path, lsq_result) -> None:
    fit, constrained = lsq_result
    original = deepcopy(fit.result)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    fit.save_result(first)
    assert json.loads(first.read_text()) == original
    fit.load_result(first)
    unit = fit.flat_result[constrained]["unit"]
    assert isinstance(unit, UnitBase)  # Constraint reevaluation introduces this object.
    fit.save_result(second)
    assert json.loads(second.read_text()) == json.loads(first.read_text())
    assert json.loads(second.read_text()) == original
    assert fit.flat_result[constrained]["unit"] is unit


def test_constrained_lsq_repeated_serialization(tmp_path: Path, lsq_result) -> None:
    fit, constrained = lsq_result
    fit.set_result(fit.result)
    unit = fit.flat_result[constrained]["unit"]
    assert isinstance(unit, UnitBase)
    expected = json.loads(json.dumps(fit.result, cls=ElisaJSONEncoder))
    path = tmp_path / "repeated.json"
    for _ in range(3):
        fit.save_result(path)
        assert json.loads(path.read_text()) == expected
        assert fit.flat_result[constrained]["unit"] is unit
        fit.load_result(path)
        unit = fit.flat_result[constrained]["unit"]
        assert isinstance(unit, UnitBase)


@pytest.mark.parametrize("name", DEFAULT_FLOAT_UNITS)
def test_unit_objects_match_existing_parameter_metadata(name: str) -> None:
    parameter = InitialParameter(BinaryInitialProperties, param=name, constraint="1")
    encoded = json.dumps({"unit": DEFAULT_FLOAT_UNITS[name]}, cls=ElisaJSONEncoder)
    assert json.loads(encoded) == {"unit": parameter.to_dict()["unit"]}


def test_initial_parameter_json_representation_supports_numpy_scalars() -> None:
    parameter = InitialParameter(BinaryInitialProperties, param="period", value=2.0, fixed=True, min=1.0, max=3.0)
    parameter.value = np.float32(2.0)
    parameter.fixed = np.bool_(True)
    expected = {"param": "period", "value": 2.0, "fixed": True, "constraint": None,
                "min": None, "max": None, "sigma": None, "unit": "d"}
    assert json.loads(repr(parameter)) == expected
    assert json.loads(str(parameter)) == expected


@pytest.mark.parametrize("structured", [False, True])
def test_chain_serialization_preserves_format_with_numpy_metadata(tmp_path: Path, structured: bool) -> None:
    parameter = InitialParameter(
        BinaryInitialProperties, param="period", value=2.0, fixed=False, min=1.0, max=3.0,
    )
    parameter.fixed = np.bool_(False)
    chain = np.array([[0.25], [0.75]])
    norm = {"system@period": (np.int64(1), np.float32(3.0))}
    kwargs = {"chain_shape": (1, 2, 1)} if structured else {}
    path = MCMCMixin.save_flat_chain(
        chain, {"system@period": parameter}, norm, str(tmp_path / "chain.json"), **kwargs,
    )
    expected = {
        "flat_chain": [[0.25], [0.75]], "fitable_parameters": ["system@period"],
        "normalization": {"system@period": [1, 3.0]},
        "fitable": {"system@period": {**parameter.to_dict(), "fixed": False}},
    }
    if structured:
        expected.update(chain_shape=[1, 2, 1], chain_order="step,walker,parameter")
    assert json.loads(Path(path).read_text()) == expected


def test_mcmc_constrained_summary_survives_json_roundtrips(tmp_path: Path, mcmc_result) -> None:
    initial = BinaryInitialParameters(system={
        "gamma": {"value": 0.0, "fixed": False, "min": -4.0, "max": 4.0},
        "mass_ratio": {"constraint": "system@gamma**2"},
    })
    samples = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.5])
    chain = ((samples + 4.0) / 8.0)[:, None]
    normalization = {"system@gamma": (-4.0, 4.0)}
    flat = MCMCMixin.resolve_mcmc_result(chain, initial.get_fitable(), normalization)
    flat.update(MCMCMixin.resolve_constrained_result(
        chain, initial.get_fitable(), normalization, initial.get_constrained(), initial.get_fixed(),
    ))
    expected = deepcopy(flat["system@mass_ratio"])
    assert expected["value"] == np.median(samples**2)
    assert expected["value"] != np.median(samples)**2
    assert expected["estimate"] == JOINT_POSTERIOR_PERCENTILES_ESTIMATE
    mcmc_result.set_result(parameters.serialize_result(flat))
    path = tmp_path / "posterior.json"
    for _ in range(3):
        mcmc_result.save_result(path)
        assert json.loads(path.read_text())["system"]["mass_ratio"] == expected
        mcmc_result.load_result(path)
        assert mcmc_result.flat_result["system@mass_ratio"] == expected
