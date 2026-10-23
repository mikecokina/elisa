"""Fork context, configuration inheritance and production pool integration."""
from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from elisa import settings
from elisa.analytics.binary_fit.mcmc import MCMCFit
from elisa.analytics.binary_fit.mixins import MCMCMixin
from elisa.analytics.binary_fit.summary import _manage_chain_evaluation
from elisa.observer import mp_manager


FORK_AVAILABLE = "fork" in multiprocessing.get_all_start_methods()
requires_fork = pytest.mark.skipif(not FORK_AVAILABLE, reason="fork is unavailable")


def _worker_state():
    return multiprocessing.get_start_method(), settings.REFLECTION_EFFECT_ITERATIONS, os.getpid()


def _observe_state(_system, _curve, positions, _kwargs):
    method, value, pid = _worker_state()
    return {"method": np.full(len(positions), method),
            "value": np.full(len(positions), value), "pid": np.full(len(positions), pid)}


def _summarize_state(chain):
    method, value, pid = _worker_state()
    assert method == "fork"
    return np.column_stack((chain[:, 0], np.full(len(chain), value), np.full(len(chain), pid)))


def _probability(vector):
    if multiprocessing.current_process().name != "MainProcess":
        assert multiprocessing.get_start_method() == "fork"
    return -float(np.sum((vector - settings.REFLECTION_EFFECT_ITERATIONS / 100.0) ** 2))


@requires_fork
def test_fork_pool_inherits_configuration_at_creation():
    original = settings.REFLECTION_EFFECT_ITERATIONS
    try:
        settings.configure(REFLECTION_EFFECT_ITERATIONS=11)
        with mp_manager.get_mp_context().Pool(1) as pool:
            first = pool.apply_async(_worker_state).get(timeout=10)
            settings.configure(REFLECTION_EFFECT_ITERATIONS=12)
            retained = pool.apply_async(_worker_state).get(timeout=10)
        with mp_manager.get_mp_context().Pool(1) as pool:
            latest = pool.apply_async(_worker_state).get(timeout=10)
    finally:
        settings.configure(REFLECTION_EFFECT_ITERATIONS=original)
    assert first[:2] == ("fork", 11)
    assert retained == first
    assert latest[:2] == ("fork", 12)
    assert latest[2] != os.getpid()


@requires_fork
def test_observations_and_summaries_use_fork():
    with patch.object(settings, "NUMBER_OF_PROCESSES", 2), \
            patch.object(settings, "NUMBER_OF_MCMC_PROCESSES", 2), \
            patch.object(settings, "REFLECTION_EFFECT_ITERATIONS", 23):
        observed = mp_manager.manage_observations(_observe_state, (None, None), np.arange(4))
        summary = _manage_chain_evaluation(np.arange(4).reshape(-1, 1), _summarize_state)
    np.testing.assert_array_equal(observed["method"], ["fork"] * 4)
    np.testing.assert_array_equal(observed["value"], [23] * 4)
    assert np.all(observed["pid"] != os.getpid())
    np.testing.assert_array_equal(summary[:, 0], np.arange(4))
    np.testing.assert_array_equal(summary[:, 1], [23] * 4)
    assert np.all(summary[:, 2] != os.getpid())


@pytest.mark.parametrize("workers", [1, 2])
@requires_fork
def test_mcmc_sampling_with_real_emcee(workers):
    # Exercise production _fit/worker with a small analytic likelihood, not mocked sampling.
    fit = SimpleNamespace(
        initial_vector=[0.37], fitable={"x": None}, normalization={"x": (0.0, 1.0)},
        normalized_sigma=lambda vector: np.full(vector.shape, np.nan),
        generate_initial_states=MCMCFit.generate_initial_states,
        ln_probability=_probability, worker=MCMCMixin.worker,
    )
    with patch.object(settings, "NUMBER_OF_MCMC_PROCESSES", workers), \
            patch.object(settings, "REFLECTION_EFFECT_ITERATIONS", 37), \
            patch("elisa.analytics.binary_fit.mcmc.get_mp_context", wraps=mp_manager.get_mp_context) as context:
        sampler = MCMCFit._fit(fit, nwalkers=4, ndim=1, nsteps=3, nsteps_burn_in=1,
                              p0=np.array([[0.2], [0.4], [0.6], [0.8]]))
    if workers > 1:
        context.assert_called_once_with()
    else:
        context.assert_not_called()
    assert sampler.get_chain().shape == (3, 4, 1)
    np.testing.assert_allclose(sampler.get_log_prob(), -np.sum((sampler.get_chain() - 0.37) ** 2, axis=2))
    if workers > 1:
        assert sampler.pool._ctx.get_start_method() == "fork"
    else:
        assert sampler.pool is None


def test_unavailable_fork_is_explicit_and_serial_still_works():
    with patch.object(mp_manager.multiprocessing, "get_context", side_effect=ValueError("no fork")):
        with pytest.raises(RuntimeError, match="requires the 'fork'.*Use serial execution"):
            mp_manager.get_mp_context()
        with patch.object(settings, "NUMBER_OF_PROCESSES", 1), \
                patch.object(settings, "NUMBER_OF_MCMC_PROCESSES", 1):
            with patch.object(mp_manager, "get_mp_context", side_effect=AssertionError("serial requested a pool")):
                observed = mp_manager.manage_observations(_observe_state, (None, None), np.arange(3))
            chain = np.arange(3).reshape(-1, 1)
            np.testing.assert_array_equal(_manage_chain_evaluation(chain, np.negative), -chain)
    np.testing.assert_array_equal(observed["pid"], [os.getpid()] * 3)


@pytest.mark.parametrize("preset", [None, "forkserver"])
def test_imports_and_context_leave_application_start_method_unchanged(tmp_path, preset):
    if preset is not None and preset not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{preset} is unavailable")
    config = tmp_path / "elisa.ini"
    config.write_text(f"[general]\nhome={tmp_path}\nsuppress_warnings=true\n")
    code = """
import multiprocessing as mp
import sys
import os
if sys.argv[1] != 'None':
    mp.set_start_method(sys.argv[1])  # application choice, before importing ELISA
before = mp.get_start_method(allow_none=True)
import elisa
from elisa.observer.mp_manager import get_mp_context
from elisa.analytics.binary_fit import mcmc, summary
assert mp.get_start_method(allow_none=True) == before
if 'fork' in mp.get_all_start_methods():
    assert get_mp_context().get_start_method() == 'fork'
    with get_mp_context().Pool(1) as pool:
        assert pool.apply_async(os.getpid).get(timeout=10) != os.getpid()
assert mp.get_start_method(allow_none=True) == before
"""
    env = {**os.environ, "ELISA_CONFIG": str(config), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-c", code, str(preset)], env=env, check=True, timeout=30,
                   capture_output=True, text=True)


def _fail(*args):
    raise ValueError("worker failed")


@requires_fork
@pytest.mark.parametrize("operation", ["observation", "summary"])
def test_worker_exceptions_still_reach_caller(operation):
    previous_children = {child.pid for child in multiprocessing.active_children()}
    with patch.object(settings, "NUMBER_OF_PROCESSES", 2), \
            patch.object(settings, "NUMBER_OF_MCMC_PROCESSES", 2):
        with pytest.raises(ValueError, match="worker failed"):
            if operation == "observation":
                mp_manager.manage_observations(_fail, (None, None), np.arange(4))
            else:
                _manage_chain_evaluation(np.arange(4).reshape(-1, 1), _fail)
    assert {child.pid for child in multiprocessing.active_children()} == previous_children
