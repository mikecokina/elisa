"""Gaussian likelihood normalization and numerical limits, with independent references."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

from elisa.analytics.models import cost_fns


@pytest.mark.parametrize("ln_f", [-1000.0, -3.0, 0.0, 3.0, 1000.0])
def test_gaussian_likelihood_matches_log_variance_reference(ln_f: float) -> None:
    rng = np.random.default_rng(2410)
    observed = {"b": rng.normal(size=13), "a": rng.normal(size=7)}
    errors = {key: rng.uniform(0.1, 2.0, size=len(value)) for key, value in observed.items()}
    synthetic = {key: value + 0.1 for key, value in observed.items()}
    # Independent original Gaussian formula in log-variance coordinates.
    expected = 0.0
    for key, data in observed.items():
        log_variance = 2 * np.log(errors[key]) + np.logaddexp(0.0, 2 * ln_f)
        residual = (data - synthetic[key]) * np.exp(-0.5 * log_variance)
        expected -= 0.5 * np.sum(residual**2 + np.log(2 * np.pi) + log_variance)
    assert_allclose(cost_fns.likelihood_fn(observed, errors, synthetic, ln_f), expected, rtol=2e-13, atol=2e-12)
    if abs(ln_f) < 4:
        reference = sum(
            norm.logpdf(observed[key], loc=synthetic[key], scale=errors[key] * np.sqrt(1 + np.exp(2 * ln_f))).sum()
            for key in observed
        )
        assert_allclose(cost_fns.likelihood_fn(observed, errors, synthetic, ln_f), reference, rtol=2e-13)
    expected_residuals = np.concatenate([(observed[key] - synthetic[key]) / errors[key] for key in observed])
    assert_array_equal(cost_fns.standardized_residuals(observed, errors, synthetic), expected_residuals)


@pytest.mark.parametrize("error", [1e-320, 1e-200, 0.1, 1e200])
@pytest.mark.parametrize("ln_f", [-1000.0, 0.0, 1000.0])
def test_extreme_positive_errors_and_noise_are_stable(error: float, ln_f: float) -> None:
    observed = {"V": np.zeros(3)}
    errors = {"V": np.full(3, error)}
    synthetic = {"V": np.zeros(3)}
    actual = cost_fns.likelihood_fn(observed, errors, synthetic, ln_f)
    expected = -1.5 * (np.log(2 * np.pi) + 2 * np.log(error) + np.logaddexp(0.0, 2 * ln_f))
    assert np.isfinite(actual)
    assert_allclose(actual, expected, rtol=2e-13, atol=2e-12)


def test_subnormal_nonzero_residual_matches_log_coordinate_reference() -> None:
    error = np.float64(1e-320)
    observed = {"V": np.zeros(1)}
    synthetic = {"V": np.array([error / 2])}
    errors = {"V": np.array([error])}
    for ln_f in [-1000.0, 0.0, 1000.0]:
        log_scale = 0.5 * np.logaddexp(0.0, 2 * ln_f)
        z = (synthetic["V"][0] / error) * np.exp(-log_scale)
        expected = -0.5 * (z * z + np.log(2 * np.pi) + 2 * np.log(error) + 2 * log_scale)
        assert_allclose(cost_fns.likelihood_fn(observed, errors, synthetic, ln_f), expected, rtol=2e-13, atol=2e-12)


def test_unrepresentable_log_density_raises_numerical_failure() -> None:
    with pytest.raises(FloatingPointError):
        cost_fns.likelihood_fn({"a": np.ones(2)}, {"a": np.ones(2)}, {"a": np.ones(2)}, 1e308)


@pytest.mark.parametrize("ln_f", [-1000.0, -3.0, 0.0, 3.0, 1000.0])
def test_prepared_likelihood_matches_standalone_for_extreme_errors(ln_f: float) -> None:
    errors = {"V": np.array([1e-320, 1e-200, 0.1, 1e200])}
    observed = {"V": np.zeros(4)}
    synthetic = {"V": errors["V"] * 0.5}
    prepared = cost_fns.prepare_observations(observed, errors)
    assert_allclose(
        cost_fns.prepared_likelihood(prepared, synthetic, ln_f),
        cost_fns.likelihood_fn(observed, errors, synthetic, ln_f),
        rtol=2e-13,
    )


@pytest.mark.parametrize(
    ("observed", "error", "model", "ln_f", "expected"),
    [
        (0.0, 1.0, 1.5e154, -1000.0, -1.125e308),
        (1e308, 1e308, -1e308, -1000.0, -2.0 - np.log(1e308) - 0.5 * np.log(2 * np.pi)),
        (0.0, 1.0, 0.0, 1e308, -1e308),
    ],
)
def test_representable_densities_survive_intermediate_overflow(observed, error, model, ln_f, expected) -> None:
    actual = cost_fns.likelihood_fn(
        {"V": np.array([observed])}, {"V": np.array([error])}, {"V": np.array([model])}, ln_f,
    )
    assert np.isfinite(actual)
    assert_allclose(actual, expected, rtol=2e-13)


@pytest.mark.parametrize("error", [0.0, -1.0, np.nan, np.inf])
def test_standalone_rejects_invalid_uncertainties(error: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        cost_fns.likelihood_fn({"V": np.zeros(1)}, {"V": np.array([error])}, {"V": np.zeros(1)}, 0.0)


@pytest.mark.parametrize("ln_f", [np.nan, np.inf, -np.inf, np.array([0.0])])
def test_standalone_rejects_invalid_noise_parameters(ln_f) -> None:
    with pytest.raises(ValueError, match="finite and scalar"):
        cost_fns.likelihood_fn({"V": np.zeros(1)}, {"V": np.ones(1)}, {"V": np.zeros(1)}, ln_f)


def test_prepared_mcmc_reuses_statistics_and_refreshes_on_lc_normalization() -> None:
    from unittest.mock import patch

    from elisa.analytics.binary_fit.mcmc import CentralRadialVelocity, DetachedLightCurveFit
    from elisa.analytics.params.parameters import BinaryInitialParameters
    from elisa.binary_system.curves.community import RadialVelocitySystem
    from tests.tests_analytics.fitting_helpers import V, lc_setup, make_lc, rv_problem

    lc = DetachedLightCurveFit()
    lc_setup(lc, {V: make_lc(errors=[0.1] * 3)})
    rv = CentralRadialVelocity()
    cfg, data = rv_problem()
    rv.set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    for fit in (lc, rv):
        predictions = {key: values + 0.01 for key, values in fit.y_data.items()}
        expected = cost_fns.likelihood_fn(fit.y_data, fit.y_err, predictions, -3.0)
        prepared = fit._prepared_observations
        with patch.object(cost_fns, "prepare_observations", side_effect=AssertionError("Repeated setup")):
            for _ in range(3):
                assert_allclose(fit.likelihood_fn(predictions, -3.0), expected, rtol=2e-13)
                assert fit._prepared_observations is prepared
    old = lc._prepared_observations
    lc.normalize_data(kind="maximum")
    assert lc._prepared_observations is not old
    assert_allclose(
        lc.likelihood_fn(lc.y_data, 0.0),
        cost_fns.likelihood_fn(lc.y_data, lc.y_err, lc.y_data, 0.0),
        rtol=2e-13,
    )
