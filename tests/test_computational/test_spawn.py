"""Exercise production pools with fresh interpreters, including on Windows."""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from elisa import BinarySystem, Observer, get_default_binary_definition, settings
from elisa.analytics.binary_fit import mcmc_diagnostics
from elisa.analytics.binary_fit.summary import _manage_chain_evaluation
from elisa.observer import mp_manager
from tests.tests_analytics.test_mcmc_diagnostics_integration import AnalyticFit


@pytest.fixture(autouse=True)
def spawn_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the Windows fallback even on a host that also supports fork."""
    monkeypatch.setattr(mp_manager.multiprocessing, "get_all_start_methods", lambda: ["spawn"])


def _worker_state() -> tuple[str, int, str, int]:
    return (
        multiprocessing.get_start_method(), settings.REFLECTION_EFFECT_ITERATIONS,
        settings.HOME, os.getpid(),
    )


def _summarize_state(chain: np.ndarray) -> np.ndarray:
    method, value, _home, pid = _worker_state()
    assert method == "spawn"
    return np.column_stack((chain[:, 0], np.full(len(chain), value), np.full(len(chain), pid)))


def _fail(*_args: object) -> None:
    raise ValueError("spawned worker failed")


def test_worker_settings_snapshot_and_paths(tmp_path: Path) -> None:
    with settings.override(HOME=str(tmp_path / "data with spaces"), REFLECTION_EFFECT_ITERATIONS=23):
        with mp_manager.get_mp_context().Pool(
            1, initializer=mp_manager.initialize_worker, initargs=(settings.settings_serializer(),),
        ) as pool:
            first = pool.apply_async(_worker_state).get(timeout=60)
            settings.configure(REFLECTION_EFFECT_ITERATIONS=24)
            retained = pool.apply_async(_worker_state).get(timeout=60)
        with mp_manager.get_mp_context().Pool(
            1, initializer=mp_manager.initialize_worker, initargs=(settings.settings_serializer(),),
        ) as pool:
            latest = pool.apply_async(_worker_state).get(timeout=60)
        assert first[:3] == ("spawn", 23, settings.HOME)
        assert retained == first
        assert latest[:3] == ("spawn", 24, settings.HOME)
        assert first[3] != os.getpid()
        assert latest[3] != os.getpid()


def test_chain_summary_uses_effective_settings() -> None:
    with settings.override(NUMBER_OF_MCMC_PROCESSES=2, REFLECTION_EFFECT_ITERATIONS=23):
        result = _manage_chain_evaluation(np.arange(4).reshape(-1, 1), _summarize_state)
    np.testing.assert_array_equal(result[:, 0], np.arange(4))
    np.testing.assert_array_equal(result[:, 1], [23] * 4)
    assert np.all(result[:, 2] != os.getpid())


def test_physical_light_and_radial_velocity_curves() -> None:
    """Use blackbody atmospheres and explicit LD so CI needs no external tables."""
    with settings.override(
        NUMBER_OF_PROCESSES=1, NUMBER_OF_MCMC_PROCESSES=1, LIMB_DARKENING_LAW="square_root",
        REFLECTION_EFFECT=False, MESH_GENERATOR="trapezoidal",
    ):
        definition = get_default_binary_definition()
        for component in ("primary", "secondary"):
            definition[component].update(
                atmosphere="bb", discretization_factor=10,
                limb_darkening_coefficients={"bolometric": [0.5, 0.2], "rv_band": [0.5, 0.2]},
            )
        observer = Observer(system=BinarySystem.from_json(definition), passband=["bolometric"])
        phases = np.array([0.12, 0.22, 0.32, 0.42])
        serial_lc = observer.lc(phases=phases)
        serial_rv = observer.rv(phases=phases, method="radiometric")
        settings.configure(NUMBER_OF_PROCESSES=2)
        parallel_lc = observer.lc(phases=phases)
        parallel_rv = observer.rv(phases=phases, method="radiometric")
        for serial, parallel in ((serial_lc, parallel_lc), (serial_rv, parallel_rv)):
            np.testing.assert_array_equal(serial[0], parallel[0])
            for band, values in serial[1].items():
                assert np.isfinite(values).all()
                np.testing.assert_allclose(parallel[1][band], values, rtol=1e-10, atol=1e-10)


def test_physical_curves_with_parallel_mcmc_user_configuration() -> None:
    """The physical smoke test must not depend on a user's MCMC worker count."""
    with settings.override(NUMBER_OF_PROCESSES=1, NUMBER_OF_MCMC_PROCESSES=2):
        test_physical_light_and_radial_velocity_curves()
        assert settings.NUMBER_OF_MCMC_PROCESSES == 2


class ConfiguredFit(AnalyticFit):
    """Require a nondefault setting in every real emcee likelihood evaluation."""

    def likelihood(self, xn: np.ndarray) -> float:
        assert settings.REFLECTION_EFFECT_ITERATIONS == 23
        return super().likelihood(xn)


def _run_fit(path: Path, workers: int) -> ConfiguredFit:
    fit = ConfiguredFit()
    with (
        settings.override(
            NUMBER_OF_PROCESSES=1, NUMBER_OF_MCMC_PROCESSES=workers,
            REFLECTION_EFFECT_ITERATIONS=23, MCMC_DIAGNOSTICS_INTERVAL_STEPS=2,
            MCMC_SAVE_INTERVAL=0,
        ),
        patch("numpy.random.get_state", return_value=np.random.RandomState(2026).get_state()),
    ):
        fit._fit(8, 1, 6, 2, np.linspace(0.2, 0.6, 8)[:, None], save=True, fit_id=str(path))
    return fit


def test_mcmc_chain_and_diagnostics_match_serial(tmp_path: Path) -> None:
    serial = _run_fit(tmp_path / "serial.json", 1)
    spawned = _run_fit(tmp_path / "spawn.json", 2)
    np.testing.assert_array_equal(serial.last_sampler.get_chain(), spawned.last_sampler.get_chain())
    np.testing.assert_array_equal(serial.last_sampler.get_log_prob(), spawned.last_sampler.get_log_prob())
    serial_counts = json.loads((tmp_path / "serial.diagnostics.json").read_text())
    spawn_counts = json.loads((tmp_path / "spawn.diagnostics.json").read_text())
    for phase in ("burn_in", "production"):
        assert spawn_counts[phase] == serial_counts[phase]
    assert spawn_counts["production"]["total_failures"] > 0
    assert not mcmc_diagnostics.count_failure(ValueError("outside a run"))


def test_spawn_survives_disabled_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = _run_fit(tmp_path / "reference.json", 2)

    def fail() -> None:
        raise RuntimeError("diagnostic context unavailable")

    monkeypatch.setattr(mcmc_diagnostics, "get_mp_context", fail)
    actual = _run_fit(tmp_path / "disabled.json", 2)
    np.testing.assert_array_equal(reference.last_sampler.get_chain(), actual.last_sampler.get_chain())
    assert Path(actual.flat_chain_path).is_file()


@pytest.mark.parametrize("operation", ["observation", "summary"])
def test_spawn_worker_errors_propagate_and_children_exit(operation: str) -> None:
    previous = {child.pid for child in multiprocessing.active_children()}
    with (
        settings.override(
            NUMBER_OF_PROCESSES=2 if operation == "observation" else 1,
            NUMBER_OF_MCMC_PROCESSES=2 if operation == "summary" else 1,
        ),
        pytest.raises(ValueError, match="spawned worker failed"),
    ):
        if operation == "observation":
            mp_manager.manage_observations(_fail, (None, None), np.arange(4))
        else:
            _manage_chain_evaluation(np.arange(4).reshape(-1, 1), _fail)
    assert {child.pid for child in multiprocessing.active_children()} == previous
