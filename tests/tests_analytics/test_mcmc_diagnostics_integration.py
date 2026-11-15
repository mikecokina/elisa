"""Exercise real emcee and persistence without physical synthesis."""
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import emcee
import numpy as np
import pytest

from elisa import settings
from elisa.analytics.binary_fit import mcmc, mcmc_diagnostics, mixins
from elisa.analytics.params.parameters import InitialParameter
from elisa.analytics.params.transform import BinaryInitialProperties
from elisa.base.error import MorphologyError


class AnalyticFit(mcmc.MCMCFit):
    """Pickleable analytic likelihood using the real posterior failure boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.initial_vector = [0.5]
        self.normalization = {"system@gamma": (0.0, 1.0)}
        self.fitable = {
            "system@gamma": InitialParameter(
                BinaryInitialProperties, param="gamma", value=0.5, min=0.0, max=1.0,
            ),
        }

    def likelihood(self, xn: np.ndarray) -> float:
        if xn[0] > 0.65:
            raise MorphologyError("analytic candidate boundary")
        return -10 * (xn[0] - 0.5) ** 2

    def fit(self) -> None:
        raise NotImplementedError


def _run(fit_id: str | None) -> AnalyticFit:
    fit = AnalyticFit()
    # emcee initializes its legacy RandomState from NumPy's global state.
    with patch("numpy.random.get_state", return_value=np.random.RandomState(2026).get_state()):
        fit._fit(8, 1, 6, 2, np.linspace(0.2, 0.6, 8)[:, None], save=True, fit_id=fit_id)
    return fit


@pytest.mark.parametrize("absolute_home", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("fit_id", [None, "fit", "nested/run", "existing/run.json", "absolute"])
def test_saved_fit_paths(
    absolute_home: bool, workers: int, fit_id: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing").mkdir()
    if fit_id == "absolute":
        fit_id = str(tmp_path / "external" / "run.json")
    clock = Mock(wraps=mixins.datetime)
    clock.now.return_value = datetime(2026, 11, 8, 12, tzinfo=UTC)
    monkeypatch.setattr(mixins, "datetime", clock)
    with settings.override(
        HOME=str(tmp_path / "home") if absolute_home else "home",
        NUMBER_OF_PROCESSES=1, NUMBER_OF_MCMC_PROCESSES=workers,
        MCMC_DIAGNOSTICS_INTERVAL_STEPS=4, MCMC_SAVE_INTERVAL=0,
    ):
        expected = AnalyticFit.chain_path(fit_id).absolute()
        fit = _run(fit_id)
        assert Path(fit.flat_chain_path) == expected
        data = AnalyticFit.load_flat_chain(str(expected))
        np.testing.assert_array_equal(data["flat_chain"], fit.last_sampler.get_chain(flat=True))
        assert data["chain_shape"] == [6, 8, 1]
        import json

        diagnostics = json.loads(expected.with_suffix(".diagnostics.json").read_text())
        assert diagnostics["burn_in"]["completed_steps"] == 2
        assert [item["step"] for item in diagnostics["burn_in"]["history"]] == [2]
        assert [item["step"] for item in diagnostics["production"]["history"]] == [4, 6]
        assert diagnostics["production"]["total_failures"] > 0
        assert diagnostics["production"]["total_failures"] == sum(
            item["rejected_in_interval"] for item in diagnostics["production"]["history"]
        )
        assert len(list(tmp_path.rglob("*.json"))) == 2

        reference = emcee.EnsembleSampler(8, 1, fit.ln_probability)
        reference.random_state = np.random.RandomState(2026).get_state()
        coordinates, _, _ = reference.run_mcmc(np.linspace(0.2, 0.6, 8)[:, None], 2, store=False)
        reference.reset()
        reference.run_mcmc(coordinates, 6)
        np.testing.assert_array_equal(fit.last_sampler.get_chain(), reference.get_chain())
        np.testing.assert_array_equal(fit.last_sampler.get_log_prob(), reference.get_log_prob())


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("broken", ["context", "allocation", "json", "cleanup"])
def test_saved_fit_survives_diagnostics_failures(
    workers: int, broken: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DiagnosticError(Exception):
        pass

    def fail(*_args: object, **_kwargs: object) -> None:
        raise DiagnosticError("injected diagnostics error")

    warnings = []
    monkeypatch.setattr(mcmc_diagnostics.logger, "warning", lambda *args: warnings.append(args))
    with settings.override(
        NUMBER_OF_PROCESSES=1, NUMBER_OF_MCMC_PROCESSES=workers, MCMC_DIAGNOSTICS_INTERVAL_STEPS=2,
    ):
        reference = _run(str(tmp_path / "reference.json"))
        if broken == "context":
            monkeypatch.setattr(mcmc_diagnostics, "get_mp_context", fail)
        elif broken == "allocation":
            monkeypatch.setattr(mcmc_diagnostics.get_mp_context(), "Array", fail)
        elif broken == "json":
            monkeypatch.setattr(mcmc_diagnostics.json, "dump", fail)
        else:
            monkeypatch.setattr(mcmc_diagnostics.RunDiagnostics, "write", fail)
        actual = _run(str(tmp_path / "actual.json"))
        np.testing.assert_array_equal(actual.last_sampler.get_chain(), reference.last_sampler.get_chain())
        np.testing.assert_array_equal(actual.last_sampler.get_log_prob(), reference.last_sampler.get_log_prob())
        assert Path(actual.flat_chain_path).is_file()
        assert len(warnings) == 1


def test_actual_pool_failure_still_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        raise RuntimeError("sampling pool initialization failed")

    monkeypatch.setattr(mcmc, "get_mp_context", fail)
    with (
        settings.override(NUMBER_OF_PROCESSES=1, NUMBER_OF_MCMC_PROCESSES=2),
        pytest.raises(RuntimeError, match="sampling pool initialization failed"),
    ):
        _run(str(tmp_path / "fit.json"))


def test_diagnostics_context_lookup_failure_is_nonfatal(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    def fail() -> None:
        raise RuntimeError("context lookup failed")

    monkeypatch.setattr(mcmc_diagnostics, "_active", SimpleNamespace(get=fail))
    assert mcmc_diagnostics.count_failure(MorphologyError("invalid candidate"))
    mcmc_diagnostics.diagnostics_failed(RuntimeError("diagnostic failure"))
    fit = AnalyticFit()
    fit.prepare_priors()
    assert fit.ln_probability(np.array([0.7])) == -np.inf
