"""Run-scoped counter and parent snapshot regressions."""
import json
import multiprocessing
from pathlib import Path

import pytest

from elisa import settings
from elisa.analytics.binary_fit.mcmc_diagnostics import RunDiagnostics, count_failure
from elisa.base.error import ElisaSpotError, MorphologyError


def _fail_many(n: int) -> None:
    for _ in range(n):
        count_failure(MorphologyError("private"))
        count_failure(ElisaSpotError("private"))


@pytest.mark.parametrize("workers", [0, 4])
def test_counts_and_isolation(workers: int, tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    path = tmp_path / "chain.json"
    with settings.override(MCMC_DIAGNOSTICS_INTERVAL_STEPS=2):
        with RunDiagnostics(context, {"accepted": 4}, path) as run:
            if workers:
                with context.Pool(workers) as pool:
                    pool.map(_fail_many, [1000] * workers)
            else:
                _fail_many(1000)
            run.report("burn_in", 1, 1)
            run.begin_production()
            _fail_many(3)
            run.report("production", 1, 3)
            assert run.data["production"]["history"] == []
            run.report("production", 2, 3)
            _fail_many(2)
            run.report("production", 3, 3)
        data = json.loads(path.with_suffix(".diagnostics.json").read_text())
        assert data["burn_in"]["total_failures"] == 2000 * (workers or 1)
        assert data["production"]["total_failures"] == 10
        assert [x["rejected_in_interval"] for x in data["production"]["history"]] == [6, 4]
        assert data["production"]["failures"] == {"MorphologyError": 5, "ElisaSpotError": 5}
        assert list(tmp_path.iterdir()) == [path.with_suffix(".diagnostics.json")]
        with RunDiagnostics(context, {}) as run:
            assert sum(run.counters) == 0
    assert not count_failure(MorphologyError("outside"))


def test_penalty_boundary_is_quiet_and_unexpected_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from elisa.analytics.binary_fit import failure_policy
    from tests.tests_analytics.test_candidate_failure_policy import callbacks

    warnings = []
    monkeypatch.setattr(failure_policy.logger, "warning", lambda *args: warnings.append(args))
    with RunDiagnostics(multiprocessing.get_context("fork"), {}) as run:
        fit, calls = callbacks(monkeypatch, MorphologyError("private"))
        assert calls[2](fit, np.array([])) == -np.inf
        assert sum(run.counters) == 1
        assert warnings == []
        assert fit._candidate_failure_diagnostics.last == {}
        fit, calls = callbacks(monkeypatch, RuntimeError("unexpected"))
        with settings.override(PENALIZE_ALL_EXCEPTIONS=False), pytest.raises(RuntimeError, match="unexpected"):
            calls[2](fit, np.array([]))
        assert sum(run.counters) == 1


def test_sampling_matches_uninstrumented_chain() -> None:
    import emcee
    import numpy as np

    from elisa.analytics.binary_fit.mixins import MCMCMixin

    def posterior(x: np.ndarray) -> float:
        return -0.5 * np.dot(x, x)

    p0 = np.random.default_rng(42).normal(size=(8, 2))
    reference = emcee.EnsembleSampler(8, 2, posterior)
    actual = emcee.EnsembleSampler(8, 2, posterior)
    actual.random_state = reference.random_state
    MCMCMixin.worker(reference, p0, 5, 3)
    with (
        settings.override(MCMC_DIAGNOSTICS_INTERVAL_STEPS=2),
        RunDiagnostics(multiprocessing.get_context("fork"), {}) as run,
    ):
        MCMCMixin.worker(actual, p0, 5, 3, diagnostics=run)
    np.testing.assert_array_equal(actual.get_chain(), reference.get_chain())
    np.testing.assert_array_equal(actual.get_log_prob(), reference.get_log_prob())
    assert [x["step"] for x in run.data["burn_in"]["history"]] == [3]
    assert [x["step"] for x in run.data["production"]["history"]] == [2, 4, 5]


def test_atomic_snapshot_survives_interruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "chain.diagnostics.json"
    with pytest.raises(KeyboardInterrupt), RunDiagnostics(  # noqa: PT012
        multiprocessing.get_context("fork"), {}, tmp_path / "chain.json",
    ) as run:
        _fail_many(1)
        run.report("production", 1, 1)
        raise KeyboardInterrupt
    original = destination.read_text()

    def fail_replace(*_args: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    run.write()
    assert run.disabled
    assert destination.read_text() == original
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_positive_integer_setting(interval: object) -> None:
    from elisa.base.error import ElisaSettingsError

    with pytest.raises(ElisaSettingsError), settings.override(MCMC_DIAGNOSTICS_INTERVAL_STEPS=interval):
        pass


@pytest.mark.parametrize("broken", ["initialization", "counter", "snapshot", "json", "info", "warning"])
def test_diagnostic_failures_do_not_affect_sampling(  # noqa: C901
    broken: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import emcee
    import numpy as np

    from elisa.analytics.binary_fit import mcmc_diagnostics as diagnostics
    from elisa.analytics.binary_fit.mixins import MCMCMixin
    from tests.tests_analytics.test_candidate_failure_policy import callbacks

    warnings = []

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("diagnostic injection")

    def warn(*args: object) -> None:
        warnings.append(args)
        if broken == "warning":
            fail()

    monkeypatch.setattr(diagnostics.logger, "warning", warn)
    context = multiprocessing.get_context("fork")
    if broken == "initialization":
        monkeypatch.setattr(context, "Array", fail)

    def posterior(x: np.ndarray) -> float:
        return -0.5 * np.dot(x, x)

    p0 = np.random.default_rng(10).normal(size=(8, 2))
    reference = emcee.EnsembleSampler(8, 2, posterior)
    actual = emcee.EnsembleSampler(8, 2, posterior)
    actual.random_state = reference.random_state
    MCMCMixin.worker(reference, p0, 4, 2)
    with settings.override(MCMC_DIAGNOSTICS_INTERVAL_STEPS=1), RunDiagnostics(
        context, {}, tmp_path / "chain.json",
    ) as run:
        if broken in {"counter", "warning"}:
            monkeypatch.setattr(run.counters, "get_lock", fail)
        elif broken == "snapshot":
            monkeypatch.setattr(diagnostics, "_failures", fail)
        elif broken == "json":
            monkeypatch.setattr(diagnostics.json, "dump", fail)
        elif broken == "info":
            monkeypatch.setattr(diagnostics.logger, "info", fail)
        fit, calls = callbacks(monkeypatch, MorphologyError("private"))
        for _ in range(3):
            assert calls[2](fit, np.array([])) == -np.inf
        assert fit._candidate_failure_diagnostics.total == 3
        assert sum(fit._candidate_failure_diagnostics.counts.values()) == 3
        MCMCMixin.worker(actual, p0, 4, 2, diagnostics=run)
    np.testing.assert_array_equal(actual.get_chain(), reference.get_chain())
    np.testing.assert_array_equal(actual.get_log_prob(), reference.get_log_prob())
    assert run.disabled
    assert len(warnings) == 1


def test_original_burn_in_coordinates() -> None:
    from types import SimpleNamespace

    import numpy as np

    from elisa.analytics.binary_fit.mixins import MCMCMixin

    initial = np.zeros((4, 1))
    coordinates = np.ones((4, 1))
    calls = []

    def burn(p0: np.ndarray, steps: int, **kwargs: object) -> tuple:
        assert p0 is initial
        assert steps == 3
        assert kwargs == {"progress": False, "store": False}
        calls.append("burn")
        _fail_many(2)
        return coordinates, None, None

    def sample(p0: np.ndarray, **kwargs: object) -> list:
        assert p0 is coordinates
        assert calls == ["burn", "reset"]
        return [None] * kwargs["iterations"]

    sampler = SimpleNamespace(run_mcmc=burn, sample=sample, reset=lambda: calls.append("reset"))
    with RunDiagnostics(multiprocessing.get_context("fork"), {}) as run:
        MCMCMixin.worker(sampler, initial, 2, 3, diagnostics=run)
    assert run.data["burn_in"]["total_failures"] == 4
    assert [entry["step"] for entry in run.data["burn_in"]["history"]] == [3]
    assert run.data["production"]["total_failures"] == 0


def test_cleanup_preserves_original_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from elisa.analytics.binary_fit import mcmc_diagnostics as diagnostics

    warnings = []
    monkeypatch.setattr(diagnostics.logger, "warning", lambda *args: warnings.append(args))

    def fail() -> None:
        raise OSError("cleanup injection")

    with pytest.raises(RuntimeError, match="sampler error"), RunDiagnostics(  # noqa: PT012
        multiprocessing.get_context("fork"), {},
    ) as run:
        monkeypatch.setattr(run, "write", fail)
        raise RuntimeError("sampler error")
    assert len(warnings) == 1
    assert not count_failure(MorphologyError("outside"))


def test_activation_and_reset_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextvars import ContextVar
    from types import SimpleNamespace

    from elisa.analytics.binary_fit import mcmc_diagnostics as diagnostics

    real_active = ContextVar("test_mcmc_diagnostics", default=None)
    warnings = []
    monkeypatch.setattr(diagnostics.logger, "warning", lambda *args: warnings.append(args))

    def fail(*_args: object) -> None:
        raise OSError("context injection")

    monkeypatch.setattr(diagnostics, "_active", SimpleNamespace(set=fail, get=real_active.get))
    with RunDiagnostics(multiprocessing.get_context("fork"), {}) as run:
        run.report("production", 1, 1)
    assert run.disabled
    assert len(warnings) == 1
    warnings.clear()
    monkeypatch.setattr(
        diagnostics, "_active", SimpleNamespace(set=real_active.set, get=real_active.get, reset=fail),
    )
    with pytest.raises(RuntimeError, match="original"), RunDiagnostics(
        multiprocessing.get_context("fork"), {},
    ):
        raise RuntimeError("original")
    assert real_active.get() is None
    assert len(warnings) == 1


def test_worker_failure_warns_once_in_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    from elisa.analytics.binary_fit import mcmc_diagnostics as diagnostics

    warnings = []
    monkeypatch.setattr(diagnostics.logger, "warning", lambda *args: warnings.append(args))

    def fail() -> None:
        raise OSError("worker counter injection")

    with RunDiagnostics(multiprocessing.get_context("fork"), {}) as run:
        monkeypatch.setattr(run.counters, "get_lock", fail)
        with multiprocessing.get_context("fork").Pool(2) as pool:
            pool.map(_fail_many, [10, 10])
        assert run.failed.value
        run.report("production", 1, 1)
        run.report("production", 1, 1)
    assert len(warnings) == 1
