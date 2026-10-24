"""Prepared fitting state, static-validation counts and numerical equivalence."""

from __future__ import annotations

import pickle
import sys
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

from elisa import units as u
from elisa.analytics import LCData
from elisa.analytics.binary_fit import lc_prediction, least_squares, mcmc, shared
from elisa.analytics.dataset.base import DataSet
from elisa.analytics.models import cost_fns, rv
from elisa.analytics.params import parameters
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from tests.tests_analytics.test_lc_prediction_consistency import LCScientificConsistencyTestCase, V, X
from tests.tests_analytics.test_lc_preparation_and_lsq import rv_problem

if TYPE_CHECKING:
    from elisa.types import Float


@pytest.fixture
def science():
    fixture = LCScientificConsistencyTestCase()
    try:
        fixture.setUp()
        yield fixture
    finally:
        fixture.doCleanups()


def normalized(fit) -> np.ndarray:
    return np.asarray(parameters.vector_normalizer(fit.initial_vector, fit.fitable, fit.normalization))


@pytest.mark.parametrize("kind", ["lc", "rv"])
@pytest.mark.parametrize("method", ["ls", "mcmc"])
def test_static_preparation_once_and_absent_from_complete_callbacks(science, kind, method):
    tracked = [
        cost_fns.prepare_observations,
        parameters.prepare_fit_parameters,
        parameters.prepare_constraint_expressions,
        lc_prediction.validate_observations,
        lc_prediction.prepare_lc_support,
        DataSet.check_data_validity,
        parameters.check_for_invalid_constraint,
        cost_fns.likelihood_fn,
        cost_fns.standardized_residuals,
    ]
    calls = dict.fromkeys(tracked, 0)
    by_code = {fn.__code__: fn for fn in tracked}

    def profile(frame, event, result):
        if event == "call" and frame.f_code in by_code:
            calls[by_code[frame.f_code]] += 1

    module = least_squares if method == "ls" else mcmc
    if kind == "lc":
        cfg, physical = science.physical()
        values = science.synth(X, [V], physical)[V]
        data = {
            V: LCData(
                x_data=X,
                y_data=values,
                y_err=np.full(len(X), 0.01),
                x_unit=u.dimensionless_unscaled,
                y_unit=u.dimensionless_unscaled,
            )
        }
        cls = module.DetachedLightCurveFit
        options = dict(
            passband=[V], discretization=10.0, interp_treshold=100, observer_system_cls=BinarySystem, samples="uniform"
        )
    else:
        cfg, data = rv_problem(constrained=True)
        cls, options = module.CentralRadialVelocity, dict(observer_system_cls=RadialVelocitySystem)
    fit = cls()
    initial = parameters.BinaryInitialParameters(**cfg)
    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        fit.set_up(initial, data, **options)
        for fn in (
            cost_fns.prepare_observations,
            parameters.prepare_fit_parameters,
            parameters.prepare_constraint_expressions,
            lc_prediction.validate_observations,
        ):
            assert calls[fn] == 1
        assert calls[lc_prediction.prepare_lc_support] == (1 if kind == "lc" else 0)
        assert calls[DataSet.check_data_validity] == len(data)
        prepared_counts = calls.copy()
        xn = normalized(fit)
        callback = (
            fit.ln_probability
            if method == "mcmc"
            else (fit.model_to_fit if kind == "lc" else fit.central_rv_model_to_fit)
        )
        for _ in range(3):
            assert np.all(np.isfinite(callback(xn)))
        assert calls == prepared_counts
        # A fresh setup creates new records; it is the supported mutation boundary.
        old_stats, old_params = fit._prepared_observations, fit._prepared_parameters
        fit.set_up(parameters.BinaryInitialParameters(**deepcopy(cfg)), data, **options)
        assert fit._prepared_observations is not old_stats
        assert fit._prepared_parameters is not old_params
        assert calls[cost_fns.prepare_observations] == 2
        assert calls[parameters.prepare_constraint_expressions] == 2
    finally:
        sys.setprofile(previous)
    # Records must survive multiprocessing pickling, including constrained expressions.
    restored = pickle.loads(pickle.dumps(fit))
    assert_allclose(restored.prepare_model_parameters(normalized(restored))["system@period"], 3.0)


@pytest.mark.parametrize("ln_f", [-1000.0, -3.0, 0.0, 3.0, 1000.0])
@pytest.mark.parametrize("fixed", [False, True])
def test_prepared_gaussian_matches_original_formula_and_scipy(ln_f: Float, fixed: bool):
    rng = np.random.default_rng(2410)
    observed = {"b": rng.normal(size=13), "a": rng.normal(size=7)}
    errors = {key: rng.uniform(0.1, 2.0, size=len(value)) for key, value in observed.items()}
    synthetic = {key: value + 0.1 for key, value in observed.items()}
    prepared = cost_fns.prepare_observations(observed, errors, fixed_ln_f=ln_f if fixed else None)
    # Independent original Gaussian formula in log-variance coordinates.
    expected = 0.0
    for key, data in observed.items():
        log_variance = 2 * np.log(errors[key]) + np.logaddexp(0.0, 2 * ln_f)
        residual = (data - synthetic[key]) * np.exp(-0.5 * log_variance)
        expected -= 0.5 * np.sum(residual**2 + np.log(2 * np.pi) + log_variance)
    assert_allclose(cost_fns.prepared_likelihood(prepared, synthetic, ln_f), expected, rtol=2e-13, atol=2e-12)
    if abs(ln_f) < 4:
        reference = sum(
            norm.logpdf(observed[key], loc=synthetic[key], scale=errors[key] * np.sqrt(1 + np.exp(2 * ln_f))).sum()
            for key in observed
        )
        assert_allclose(cost_fns.prepared_likelihood(prepared, synthetic, ln_f), reference, rtol=2e-13)
    expected_residuals = np.concatenate([(observed[key] - synthetic[key]) / errors[key] for key in observed])
    assert_array_equal(cost_fns.prepared_residuals(prepared, synthetic), expected_residuals)


@pytest.mark.parametrize("error", [1e-320, 1e-200, 0.1, 1e200])
@pytest.mark.parametrize("ln_f", [-1000.0, 0.0, 1000.0])
def test_extreme_positive_errors_and_noise_are_stable(error: Float, ln_f: Float):
    observed = {"V": np.zeros(3)}
    errors = {"V": np.full(3, error)}
    synthetic = {"V": np.zeros(3)}
    actual = cost_fns.likelihood_fn(observed, errors, synthetic, ln_f)
    expected = -1.5 * (np.log(2 * np.pi) + 2 * np.log(error) + np.logaddexp(0.0, 2 * ln_f))
    assert np.isfinite(actual)
    assert_allclose(actual, expected, rtol=2e-13, atol=2e-12)


@pytest.mark.parametrize("kind", ["lc", "rv"])
@pytest.mark.parametrize("defect", ["missing", "column", "short", "nan", "inf"])
def test_dynamic_bad_output_is_not_hidden_by_preparation(science, kind, defect):
    if kind == "lc":
        fit = science.prepare({V: X})[0]
        model = fit.prepare_model_parameters(normalized(fit))

        def evaluate(raw):
            return fit.predict_light_curves(model, synthesis=lambda *args, **kwargs: raw)

        raw = {V: np.ones(len(fit.x_data_reduced))}
        key = V
    else:
        cfg, data = rv_problem()
        fit = least_squares.CentralRadialVelocity()
        fit.set_up(parameters.BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
        raw = {key: np.ones(len(fit.x_data_reduced)) for key in data}
        evaluate, key = fit.map_rv_predictions, "primary"
    if defect == "missing":
        raw.pop(key)
    elif defect == "column":
        raw[key] = raw[key][:, None]
    elif defect == "short":
        raw[key] = raw[key][:-1]
    else:
        raw[key][-1] = np.nan if defect == "nan" else np.inf
    with pytest.raises(lc_prediction.PredictionAlignmentError):
        evaluate(raw)


def test_dynamic_rv_ephemeris_matches_safe_standalone_forward_model():
    cfg, data = rv_problem()
    cfg["system"]["period"] = dict(value=3.0, fixed=False, min=2.5, max=3.5)
    cfg["system"]["primary_minimum_time"] = dict(value=2450000.0, fixed=False, min=2449999.0, max=2450001.0)
    for dataset in data.values():
        dataset.x_data = 2450000.0 + 3 * dataset.x_data
        dataset.x_unit = u.d
    fit = least_squares.CentralRadialVelocity()
    fit.set_up(parameters.BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    assert fit._rv_phases is None
    xn = normalized(fit)
    for period, epoch in [(3.0, 2450000.0), (3.1, 2450000.02)]:
        xn[list(fit.fitable).index("system@period")] = parameters.normalize_value(
            period, *fit.normalization["system@period"]
        )
        xn[list(fit.fitable).index("system@primary_minimum_time")] = parameters.normalize_value(
            epoch, *fit.normalization["system@primary_minimum_time"]
        )
        physical = fit.prepare_model_parameters(xn)
        expected = rv.central_rv_synthetic(fit.x_data_reduced, fit.observer, **physical)
        actual = fit.prepare_synthetic(xn)
        for key in expected:
            assert_allclose(actual[key], expected[key], rtol=2e-14, atol=1e-8)


def test_prepared_constraints_preserve_scalar_precedence_and_public_arithmetic():
    from elisa.analytics.params.transform import StarInitialProperties

    fitable = {
        "primary@t_eff": parameters.InitialParameter(
            StarInitialProperties, param="t_eff", value=1.0, fixed=False, min=-2.0, max=2.0
        )
    }
    fixed = {"system@period": 3.0}
    constraints = {"secondary@t_eff": "primary@t_eff ** 2 + sin(system@period)"}
    prepared = parameters.prepare_fit_parameters(fitable, {"primary@t_eff": (-2.0, 2.0)}, fixed, constraints)
    for value in [-2.0, -1.0, 0.0, 0.5, 2.0]:
        actual = parameters.evaluate_fit_parameters(np.array([(value + 2) / 4]), prepared)
        expected = parameters.prepare_properties_set([value], fitable, constraints, fixed)
        assert_allclose(actual["secondary@t_eff"], expected["secondary@t_eff"], rtol=2e-14)


@pytest.mark.parametrize("base", [shared.AbstractLCFit, shared.AbstractRVFit])
def test_prepared_attributes_work_without_instance_dictionary(base):
    class SlottedFit(base):
        __slots__ = ()

        def fit(self):
            raise NotImplementedError

    fit = SlottedFit()
    assert not hasattr(fit, "__dict__")
    assert fit._prepared_parameters is None and fit._prepared_observations is None
    slots = [name for cls in type(fit).__mro__ for name in cls.__dict__.get("__slots__", ())]
    assert len(slots) == len(set(slots))
    fit._prepared_parameters = None
    fit._prepared_observations = None


def test_subnormal_nonzero_residual_matches_log_coordinate_reference():
    error = np.float64(1e-320)
    observed = {'V': np.zeros(1)}
    synthetic = {'V': np.array([error / 2])}
    errors = {'V': np.array([error])}
    for ln_f in [-1000., 0., 1000.]:
        log_scale = .5*np.logaddexp(0., 2*ln_f)
        z = (synthetic['V'][0] / error) * np.exp(-log_scale)
        expected = -.5 * (z*z + np.log(2*np.pi) + 2*np.log(error) + 2*log_scale)
        assert_allclose(cost_fns.likelihood_fn(observed, errors, synthetic, ln_f), expected,
                        rtol=2e-13, atol=2e-12)


def test_unrepresentable_log_density_raises_numerical_failure():
    with pytest.raises(FloatingPointError):
        cost_fns.likelihood_fn({'a': np.ones(2)}, {'a': np.ones(2)}, {'a': np.ones(2)}, 1e308)
