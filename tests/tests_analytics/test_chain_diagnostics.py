"""Independent chain integrity regressions; no fitting or synthesis is performed."""

from __future__ import annotations

from unittest.mock import patch

import emcee
import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from elisa.base.error import ElisaGraphicsError, ElisaMonteCarloError
from elisa.analytics.binary_fit import chain_tools as chains
from elisa.analytics.binary_fit import plot
from elisa.analytics.binary_fit.rv_fit import RVFitMCMC

LABELS = ["primary@t_eff", "secondary@t_eff"]


@pytest.fixture(scope="module")
def correlated() -> NDArray[np.float64]:
    # Independent AR(1) walkers, with equilibrium transient removed.
    rng = np.random.default_rng(401)
    noise = rng.normal(size=(17000, 8, 2))
    result = np.zeros_like(noise)
    for step in range(1, len(result)):
        result[step] = 0.95 * result[step - 1] + noise[step]
    return result[1000:]


def test_ensemble_time_axis_and_independent_acf(correlated: NDArray[np.float64]) -> None:
    acf, tau = chains.ensemble_autocorrelation(correlated)
    assert acf.shape == (16000, 2)
    # Direct time-domain dot products, independent of emcee's FFT implementation.
    for param in range(2):
        centered = correlated[:, :, param] - correlated[:, :, param].mean(axis=0)
        denom = np.sum(centered**2, axis=0)
        for lag in (0, 1, 5, 20):
            end = len(centered) - lag
            expected = np.mean(np.sum(centered[:end] * centered[lag:], axis=0) / denom)
            assert_allclose(acf[lag, param], expected, atol=1e-14)
    # AR(1) theoretical integrated time: (1 + rho)/(1 - rho) = 39.
    assert_allclose(tau, 39.0, rtol=0.20)
    assert_allclose(tau, emcee.autocorr.integrated_time(correlated))


def test_flattening_correlated_walkers_gives_wrong_answer(correlated: NDArray[np.float64]) -> None:
    _, temporal = chains.ensemble_autocorrelation(correlated)
    incorrect = emcee.autocorr.integrated_time(correlated.reshape(-1, 2), has_walkers=False)
    assert np.all(temporal > 10 * incorrect)
    with pytest.raises(ElisaMonteCarloError, match="steps, walkers, parameters"):
        chains.ensemble_autocorrelation(correlated.reshape(-1, 2))


@pytest.mark.parametrize("length", [1, 2, 8, 20])
def test_insufficient_chain_is_unavailable(length: int, monkeypatch) -> None:
    from elisa import settings
    monkeypatch.setattr(settings, "MCMC_AUTOCORR_TOL", 50)
    # Smooth short trajectory has far fewer than 50 autocorrelation times.
    chain = np.broadcast_to(np.arange(length)[:, None, None], (length, 4, 2))
    _, tau = chains.ensemble_autocorrelation(chain)
    assert np.isnan(tau).all()


def test_one_stuck_walker_invalidates_only_its_parameter(correlated: NDArray[np.float64]) -> None:
    chain = correlated.copy()
    chain[:, 2, 0] = 1.0
    acf, tau = chains.ensemble_autocorrelation(chain)
    assert np.isnan(acf[:, 0]).all()
    assert np.isnan(tau[0])
    assert np.isfinite(tau[1])


def test_fully_constant_chain_has_no_runtime_warnings() -> None:
    with np.errstate(all="raise"):
        acf, tau = chains.ensemble_autocorrelation(np.ones((100, 4, 2)))
    assert np.isnan(acf).all()
    assert np.isnan(tau).all()


@pytest.mark.parametrize("value", [None, [], np.ones((8, 2)), np.full((8, 2, 1), np.nan)])
def test_unavailable_temporal_structure(value) -> None:
    with pytest.raises(ElisaMonteCarloError, match="Temporal autocorrelation"):
        chains.ensemble_autocorrelation(value)


def test_plot_selection_uses_parameter_axis_without_flattening(correlated: NDArray[np.float64]) -> None:
    fit = RVFitMCMC()
    fit.chain = correlated
    fit.variable_labels = LABELS
    with patch.object(plot.MCMCPlot, "autocorr") as draw:
        plot.autocorrelation(fit, correlations_to_plot=[LABELS[1]])
    args = draw.call_args.kwargs
    assert args["correlations_to_plot"] == [LABELS[1]]
    assert args["autocorr_fns"].shape == (16000, 2)
    assert_allclose(args["autocorr_time"], emcee.autocorr.integrated_time(correlated))
    # Selecting/reordering columns retains their time and walker axes.
    with patch.object(plot.MCMCPlot, "autocorr") as draw:
        plot.autocorrelation(fit, flat_chain=correlated[:, :, 1:], variable_labels=LABELS[1:])
    assert_allclose(draw.call_args.kwargs["autocorr_time"], args["autocorr_time"][1:])
    with pytest.raises(ElisaGraphicsError, match="parameter dimension"):
        plot.autocorrelation(fit, variable_labels=LABELS[:1])
