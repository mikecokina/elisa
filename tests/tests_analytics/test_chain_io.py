"""Chain serialization, flattening, discard and public fit lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from elisa.analytics.binary_fit import chain_tools as chains
from elisa.analytics.binary_fit import io_tools, plot
from elisa.analytics.binary_fit.lc_fit import LCFitMCMC
from elisa.analytics.binary_fit.mixins import MCMCMixin
from elisa.analytics.binary_fit.rv_fit import RVFitMCMC
from elisa.analytics.params.parameters import BinaryInitialParameters, InitialParameter
from elisa.analytics.params.transform import StarInitialProperties
from elisa.analytics.tasks import RVBinaryAnalyticsTask
from elisa.base.error import ElisaMonteCarloError
from elisa.conf.enums import DiscardUnit
from tests.tests_analytics.fitting_helpers import lc_parameters, rv_problem

LABELS = ["primary@t_eff", "secondary@t_eff"]


def chain_payload(chain: NDArray[np.float64] | None = None) -> dict[str, Any]:
    chain = np.arange(40.0, dtype=float).reshape(5, 4, 2) if chain is None else chain
    return {
        "flat_chain": chain.reshape(-1, chain.shape[-1]).tolist(),
        "chain_shape": list(chain.shape),
        "chain_order": "step,walker,parameter",
        "fitable_parameters": LABELS[: chain.shape[-1]],
        "normalization": {key: (0.0, 1.0) for key in LABELS[: chain.shape[-1]]},
        "fitable": {},
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (DiscardUnit.STEPS, DiscardUnit.STEPS),
        (DiscardUnit.ROWS, DiscardUnit.ROWS),
        ("steps", DiscardUnit.STEPS),
        ("rows", DiscardUnit.ROWS),
    ],
)
def test_discard_unit_normalization(value, expected) -> None:
    assert chains.validate_discard(0, value) is expected


def test_structured_step_discard_before_flattening() -> None:
    original = np.arange(40.0).reshape(5, 4, 2)
    for unit in (None, DiscardUnit.STEPS, "steps"):
        flat, chain = chains.prepare_chain_samples(chain_payload(original), 2, unit)
        assert_array_equal(chain, original[2:])
        assert_array_equal(flat, original[2:].reshape(-1, 2))


def test_legacy_row_discard_without_guessing_walkers() -> None:
    data = chain_payload()
    data.pop("chain_shape")
    data.pop("chain_order")
    for unit in (None, DiscardUnit.ROWS, "rows"):
        flat, chain = chains.prepare_chain_samples(data, 3, unit)
        assert chain is None
        assert_array_equal(flat, np.asarray(data["flat_chain"])[3:])
    with pytest.raises(ElisaMonteCarloError, match="requires temporal chain metadata"):
        chains.prepare_chain_samples(data, 0, DiscardUnit.STEPS)


@pytest.mark.parametrize(("discard", "retains"), [(0, True), (1, False), (4, True), (7, False), (16, True)])
def test_explicit_row_discard_preserves_only_complete_ensembles(discard: int, retains: bool) -> None:
    original = np.arange(40.0).reshape(5, 4, 2)
    flat, chain = chains.prepare_chain_samples(chain_payload(original), discard, DiscardUnit.ROWS)
    assert_array_equal(flat, original.reshape(-1, 2)[discard:])
    assert (chain is not None) == retains
    if retains:
        assert_array_equal(chain, original[discard // 4 :])


@pytest.mark.parametrize(
    ("discard", "unit", "exception"),
    [
        (-1, None, ElisaMonteCarloError),
        (True, None, ElisaMonteCarloError),
        (1.5, None, ElisaMonteCarloError),
        (np.bool_(True), None, ElisaMonteCarloError),
        (0, "walkers", ElisaMonteCarloError),
        (5, None, ElisaMonteCarloError),
        (6, "steps", ElisaMonteCarloError),
        (20, "rows", ElisaMonteCarloError),
        (100, "rows", ElisaMonteCarloError),
    ],
)
def test_invalid_or_excessive_discard(discard: int, unit, exception) -> None:
    with pytest.raises(exception):
        chains.prepare_chain_samples(chain_payload(), discard, unit)


@pytest.mark.parametrize(
    "metadata",
    [
        {"chain_shape": None},
        {"chain_shape": [5, 4]},
        {"chain_shape": [5, 4, True]},
        {"chain_shape": [5, 4.0, 2]},
        {"chain_shape": [5, 0, 2]},
        {"chain_shape": [-5, 4, 2]},
        {"chain_shape": [4, 4, 2]},
        {"chain_order": "walker,step,parameter"},
        {"chain_order": None},
    ],
)
def test_invalid_temporal_metadata(metadata) -> None:
    data = chain_payload()
    data.update(metadata)
    with pytest.raises(ElisaMonteCarloError, match="temporal chain metadata"):
        chains.prepare_chain_samples(data)


@pytest.mark.parametrize("missing", ["chain_shape", "chain_order"])
def test_partial_metadata_is_rejected(missing) -> None:
    data = chain_payload()
    data.pop(missing)
    with pytest.raises(ElisaMonteCarloError, match="temporal chain metadata"):
        chains.prepare_chain_samples(data)


@pytest.mark.parametrize("samples", [[], [1, 2], [[np.nan, 0]], [[np.inf, 0]], [[[1, 2]]]])
def test_invalid_sample_matrix(samples) -> None:
    data = chain_payload()
    data["flat_chain"] = samples
    with pytest.raises(ElisaMonteCarloError, match="finite nonempty matrix"):
        chains.prepare_chain_samples(data)


@pytest.mark.parametrize("labels", [[], ["a"], ["a", "a"], ["a", 2]])
def test_invalid_labels(labels) -> None:
    data = chain_payload()
    data["fitable_parameters"] = labels
    with pytest.raises(ElisaMonteCarloError, match="labels"):
        chains.prepare_chain_samples(data)


def save_chain(tmp_path: Path, structured: bool = True) -> tuple[NDArray[np.float64], str]:
    original = np.arange(40.0).reshape(5, 4, 2) / 40
    # noinspection PyTypeChecker
    fitable = {
        label: InitialParameter(
            StarInitialProperties,
            param="t_eff",
            value=6000.0,
            fixed=False,
            min=1000.0,
            max=10000.0
        )
        for label in LABELS
    }
    kwargs = {"chain_shape": original.shape} if structured else {}
    path = MCMCMixin.save_flat_chain(
        original.reshape(-1, 2), fitable, dict.fromkeys(LABELS, (0.0, 1.0)), str(tmp_path / "chain.json"), **kwargs
    )
    return original, path


@pytest.mark.parametrize("structured", [True, False])
def test_json_roundtrip_keeps_flat_public_format_and_order(tmp_path: Path, structured: bool) -> None:
    original, path = save_chain(tmp_path, structured)
    data = MCMCMixin.load_flat_chain(path)
    assert_array_equal(data["flat_chain"], original.reshape(-1, 2))
    assert set(data) == {"flat_chain", "fitable", "normalization", "fitable_parameters"} | (
        {"chain_shape", "chain_order"} if structured else set()
    )
    flat, chain = chains.prepare_chain_samples(data)
    assert_array_equal(flat, original.reshape(-1, 2))
    if structured:
        assert_array_equal(chain, original)
    else:
        assert chain is None


def test_invalid_export_does_not_overwrite_saved_file(tmp_path: Path) -> None:
    _, path = save_chain(tmp_path)
    before = Path(path).read_bytes()
    with pytest.raises(ElisaMonteCarloError, match="metadata"):
        MCMCMixin.save_flat_chain(
            np.zeros((20, 2)),
            dict.fromkeys(
                LABELS,
                InitialParameter(
                    StarInitialProperties, param="t_eff", value=6000.0, fixed=False, min=1000.0, max=10000.0
                ),
            ),
            {},
            path,
            chain_shape=(4, 4, 2),
        )
    assert Path(path).read_bytes() == before


def test_existing_saved_json_fixture_loads_unchanged() -> None:
    fixture = Path(__file__).resolve().parents[2] / "jupyter_tutorials/demo_data/assets/lc_mcmc_chain.json"
    original = json.loads(fixture.read_text())
    loaded = MCMCMixin.load_flat_chain(str(fixture))
    assert loaded == original
    flat, chain = chains.prepare_chain_samples(loaded, 7)
    assert chain is None
    assert_array_equal(flat, np.asarray(original["flat_chain"])[7:])


def test_csv_samples_remain_flat_without_external_temporal_metadata(tmp_path: Path) -> None:
    # Core chain I/O is JSON; CSV arrays are imported by their caller, with labels.
    original = np.arange(40.0).reshape(20, 2)
    path = tmp_path / "legacy.csv"
    np.savetxt(path, original, delimiter=",")
    data = {"flat_chain": np.loadtxt(path, delimiter=","), "fitable_parameters": LABELS}
    flat, chain = chains.prepare_chain_samples(data, 3)
    assert_array_equal(flat, original[3:])
    assert chain is None


def test_public_load_initialization_and_temporal_state(tmp_path: Path, mcmc_result: LCFitMCMC | RVFitMCMC) -> None:
    fit = mcmc_result
    assert fit.chain is None
    original, path = save_chain(tmp_path)
    with patch.object(io_tools, "update_solution"):
        result = fit.load_chain(path, discard=1)
        assert_array_equal(fit.chain, original[1:])
        assert_array_equal(result[0], original[1:].reshape(-1, 2))
        fit.load_chain(path, discard=1, discard_unit=DiscardUnit.ROWS)
        assert fit.chain is None
        # Reload legacy after structured must clear previous temporal data.
        fit.load_chain(path)
        _, legacy = save_chain(tmp_path, structured=False)
        fit.load_chain(legacy, discard=1)
        assert fit.chain is None


def test_filtering_invalidates_temporal_diagnostics() -> None:
    original = np.arange(40.0).reshape(5, 4, 2) / 40
    fit = RVFitMCMC()
    fit.chain = original
    fit.flat_chain = original.reshape(-1, 2)
    fit.variable_labels = LABELS
    fit.normalization = dict.fromkeys(LABELS, (0.0, 1.0))
    fit.flat_result = dict.fromkeys(LABELS, {})
    with patch.object(io_tools, "update_solution"):
        flat = io_tools.filter_chain(fit, **{LABELS[0]: (0.2, 0.8)})
    expected = original.reshape(-1, 2)
    assert_array_equal(flat, expected[(expected[:, 0] > 0.2) & (expected[:, 0] < 0.8)])
    assert fit.chain is None
    with pytest.raises(ElisaMonteCarloError, match="Temporal autocorrelation"):
        plot.autocorrelation(fit)


def test_invalid_load_does_not_mutate_fit_state() -> None:
    fit = RVFitMCMC()
    original = np.ones((2, 3, 2))
    fit.chain = original
    fit.flat_chain = original.reshape(-1, 2)
    flat_before = fit.flat_chain
    with (
        patch.object(MCMCMixin, "load_flat_chain", return_value=chain_payload()),
        pytest.raises(ElisaMonteCarloError, match="no samples"),
    ):
        io_tools.load_chain(fit, "unused", discard=5)
    assert fit.chain is original
    assert fit.flat_chain is flat_before


def test_public_fit_extracts_ensemble_without_synthesis(mcmc_result: LCFitMCMC | RVFitMCMC) -> None:
    original = np.arange(40.0).reshape(5, 4, 2)
    fit = mcmc_result
    method = fit.fit_method_instance
    sampler = Mock()
    sampler.get_chain.return_value = original
    method.last_sampler = sampler
    method.flat_result = {}
    method.flat_chain_path = ""
    method.normalization = {}
    config, _ = lc_parameters() if isinstance(fit, LCFitMCMC) else rv_problem(constrained=True)
    initial = BinaryInitialParameters(**config)
    method.fitable = initial.get_fitable()
    with patch.object(method, "fit", return_value={}), patch.object(fit, "fit_summary"):
        fit.fit(initial, {})
    sampler.get_chain.assert_called_once_with()
    assert_array_equal(fit.chain, original)
    assert_array_equal(fit.flat_chain, original.reshape(-1, 2))


@pytest.mark.parametrize("unit", [DiscardUnit.STEPS, DiscardUnit.ROWS, "steps", "rows"])
def test_task_forwards_explicit_discard_unit(unit) -> None:
    _, data = rv_problem()
    task = RVBinaryAnalyticsTask(method="mcmc", data=data)
    with patch.object(task.fit_cls, "load_chain") as load:
        assert task.load_chain("chain.json", discard=3, discard_unit=unit) is task
    load.assert_called_once_with("chain.json", 3, None, discard_unit=unit)


def test_periodic_export_preserves_sampler_axes_without_diagnostics() -> None:
    original = np.arange(40.0).reshape(5, 4, 2)
    sampler = Mock()
    sampler.get_chain.side_effect = lambda **kwargs: original.reshape(-1, 2) if kwargs.get("flat") else original
    sampler.sample.return_value = iter([None])
    with (
        patch("elisa.analytics.binary_fit.mixins.time", side_effect=[0.0, 1e12, 1e12]),
        patch.object(MCMCMixin, "save_flat_chain") as save_chain,
        patch.object(chains, "ensemble_autocorrelation", side_effect=AssertionError("diagnostics during sampling")),
    ):
        MCMCMixin.worker(sampler, original[0], 1, 0, save=True, fitable={}, normalization={})
    kwargs = save_chain.call_args.kwargs
    assert kwargs["chain_shape"] == original.shape
    assert_array_equal(save_chain.call_args.args[0], original.reshape(-1, 2))
    sampler.reset.assert_called_once()


def test_worker_excludes_unstored_burn_in_from_production() -> None:
    # The actual sampler boundary resets after un-stored burn-in; no burn-in rows
    # are available to serialize or discard a second time automatically.
    sampler = Mock()
    initial = np.zeros((4, 2))
    burned = np.ones((4, 2))
    sampler.run_mcmc.side_effect = [(burned, None, None), (burned, None, None)]
    MCMCMixin.worker(sampler, initial, 5, 2)
    burn_call, production_call = sampler.run_mcmc.call_args_list
    assert_array_equal(burn_call.args[0], initial)
    assert burn_call.args[1] == 2
    assert burn_call.kwargs == {"progress": False, "store": False}
    assert_array_equal(production_call.args[0], burned)
    assert production_call.args[1] == 5
    assert production_call.kwargs == {"progress": False}
    sampler.reset.assert_called_once()
