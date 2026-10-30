"""Prepared LC mappings and residual-vector LS tests using real LC/RV synthesis."""

from __future__ import annotations

import inspect
import json
import pickle
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares as solve

from elisa import const
from elisa import units as u
from elisa.analytics import LCData
from elisa.analytics.binary_fit import lc_prediction, least_squares, mcmc, shared
from elisa.analytics.binary_fit.shared import lc_r_squared
from elisa.analytics.models import cost_fns, lc
from elisa.analytics.params import parameters as p
from elisa.base.error import ElisaAnalyticsDataError, ElisaAnalyticsTaskError
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from elisa.conf.enums import SamplingMethod
from tests.tests_analytics.fitting_helpers import (
    LC_PHASES,
    B,
    V,
    lc_parameters,
    normalized,
    physical,
    prepare_lc_fits,
    rv_problem,
    synthesize_lc,
)


@contextmanager
def validation_calls() -> Iterator[list]:
    """Count validation while executing the real checks."""
    with (
        patch.object(lc_prediction, "validate_observations", wraps=lc_prediction.validate_observations) as validate,
        patch.object(shared, "validate_observations", new=validate),
    ):
        yield validate.call_args_list


def lc_problem() -> tuple[dict[str, Any], dict[str, LCData]]:
    """Noiseless two-band LC with two free values and a temperature constraint."""
    cfg, flat = lc_parameters()
    cfg["system"]["inclination"]["value"] = 83.0
    cfg["primary"]["t_eff"]["value"] = 6300.0
    data = {}
    for band, xs in {
        V: np.linspace(0.0, 0.98, 31),
        B: np.array([0.5, 0.02, 0.12, 0.25, 0.47, 0.0, 0.6, 0.8, 0.02]),
    }.items():
        flux = synthesize_lc(xs, [band], flat)[band]
        data[band] = LCData(
            x_data=xs,
            y_data=flux,
            y_err=np.full(len(xs), 0.002),
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled,
        )
    return cfg, data


def capture_fit(callback: Callable, *args, **kwargs) -> tuple[dict[str, Any], OptimizeResult]:
    """Run the real SciPy solver and retain its diagnostics alongside ELISA's report."""
    results = []

    def solve_and_record(*solver_args, **solver_options):
        result = solve(*solver_args, **solver_options)
        results.append(result)
        return result

    with patch.object(least_squares, "least_squares", side_effect=solve_and_record):
        report = callback(*args, **kwargs)
    assert len(results) == 1
    return report, results[0]


@pytest.mark.requires_ld(passbands=(V, B, "bolometric"), laws=("linear",))
def test_static_validation_once_per_setup_and_reused_by_evaluations(science_settings: None) -> None:
    with validation_calls() as calls:
        fits = prepare_lc_fits({V: LC_PHASES, B: [0.25]})
        assert len(calls) == 2
        for fit in fits:
            mappings = fit._validated_reducers
            assert mappings is not None
            arrays = tuple(mappings.values())
            obs = fit.observer
            for _ in range(3):
                if isinstance(fit, mcmc.LightCurveFit):
                    fit.likelihood(normalized(fit))
                else:
                    fit.model_to_fit(normalized(fit))
                fit.prediction_r_squared(physical(fit))
            assert fit.observer is obs
            assert fit._validated_reducers is mappings
            assert all(a is b for a, b in zip(arrays, mappings.values(), strict=True))
        assert len(calls) == 2
        # Standalone public API still validates on every call.
        fit = fits[0]
        score = lc_r_squared(
            lc.synthetic_binary,
            fit.x_data_reduced,
            fit.y_data,
            fit.observer.passband,
            fit.discretization,
            fit.x_data_reducer,
            0.01,
            fit.interp_treshold,
            BinarySystem,
            **physical(fit),
        )
        assert_allclose(score, fit.prediction_r_squared(physical(fit)), atol=2e-12)
        assert len(calls) == 3


@pytest.mark.requires_ld(passbands=(V, B, "bolometric"), laws=("linear",))
def test_second_setup_replaces_and_validates_data(science_settings: None) -> None:
    fits = prepare_lc_fits({V: LC_PHASES})
    cfg, data = lc_problem()
    with validation_calls() as calls:
        for fit in fits:
            old = fit._validated_reducers
            fit.set_up(
                p.BinaryInitialParameters(**deepcopy(cfg)),
                data,
                passband=data.keys(),
                discretization=10.0,
                interp_treshold=100,
                observer_system_cls=BinarySystem,
                samples="uniform",
            )
            assert fit._validated_reducers is not old
            assert fit._validated_reducers is not None
            assert set(fit._validated_reducers) == set(data)
            predictions = fit.predict_light_curves(physical(fit))
            assert {b: len(y) for b, y in predictions.items()} == {b: len(d.y_data) for b, d in data.items()}
        assert len(calls) == 2
        data[V].y_err = np.ones(2)
        for fit in fits:
            with pytest.raises(ElisaAnalyticsDataError, match="same shape"):
                fit.set_up(
                    p.BinaryInitialParameters(**deepcopy(cfg)),
                    data,
                    passband=data.keys(),
                    discretization=10.0,
                    interp_treshold=100,
                    observer_system_cls=BinarySystem,
                    samples="uniform",
                )
        assert len(calls) == 2  # Malformed errors fail before reducer validation.


@pytest.mark.requires_ld(passbands=(V, "bolometric"), laws=("linear",))
def test_normalization_and_explicit_manual_revalidation(science_settings: None) -> None:
    fit = prepare_lc_fits({V: LC_PHASES})[0]
    with validation_calls() as calls:
        fit.normalize_data(kind="average")
        assert len(calls) == 1
        before = fit.predict_light_curves(physical(fit))
        # Corresponding edits remain usable after the explicit revalidation step.
        fit.x_data[V] = fit.x_data[V][::-1]
        fit.y_data[V] = fit.y_data[V][::-1]
        fit.y_err[V] = fit.y_err[V][::-1]
        fit.x_data_reducer[V] = fit.x_data_reducer[V][::-1]
        fit.validate_observations()
        assert len(calls) == 2
        after = fit.predict_light_curves(physical(fit))
        assert_allclose(after[V], before[V][::-1], atol=2e-12)
        assert len(calls) == 2


@pytest.mark.parametrize("defect", ["short", "column", "passband"])
@pytest.mark.requires_ld(passbands=(V, "bolometric"), laws=("linear",))
def test_dynamic_malformed_real_synthesis_output_is_rejected(science_settings: None, defect) -> None:
    fit = prepare_lc_fits({V: LC_PHASES})[0]

    def malformed(*args, **kwargs):
        raw = lc.synthetic_binary(*args, **kwargs)
        if defect == "passband":
            return {"other": raw[V]}
        raw[V] = raw[V][:-1] if defect == "short" else raw[V][:, None]
        return raw

    with validation_calls() as calls, pytest.raises(lc_prediction.PredictionAlignmentError):
        fit.predict_light_curves(physical(fit), synthesis=malformed)
    assert not calls


@pytest.mark.requires_ld(passbands=(V, B, "bolometric"), laws=("linear",))
def test_lc_parameter_recovery_result_and_residual_jacobian(science_settings: None) -> None:
    cfg, data = lc_problem()
    fit = least_squares.DetachedLightCurveFit()
    fit.atmosphere_model = fit.limb_darkening_coefficients = None
    report, result = capture_fit(
        fit.fit,
        data,
        p.BinaryInitialParameters(**cfg),
        discretization=10.0,
        interp_treshold=100,
        ftol=1e-10,
        gtol=1e-9,
        max_nfev=80,
    )
    assert result.success
    assert result.jac.shape == (40, 2)
    assert np.linalg.matrix_rank(result.jac) == 2
    assert_allclose(result.cost, 0.5 * np.dot(result.fun, result.fun), atol=1e-12)
    assert_allclose(report["system"]["inclination"]["value"], 85.0, atol=0.03)
    assert_allclose(report["primary"]["t_eff"]["value"], 6500.0, atol=5.0)
    assert_allclose(
        report["secondary"]["t_eff"]["value"], report["primary"]["t_eff"]["value"] * 0.7692307692307693, atol=1e-10
    )
    assert report["system"]["period"]["value"] == 3.0
    assert report["system"]["period"]["fixed"]
    assert report["secondary"]["t_eff"]["constraint"] == "primary@t_eff * 0.7692307692307693"
    assert_allclose(report["r_squared"]["value"], 1.0, atol=1e-8)
    assert_allclose(fit.model_to_fit(result.x), result.fun, atol=2e-12)


@pytest.mark.parametrize("constrained", [False, True])
def test_rv_residuals_parameter_recovery_and_results(constrained) -> None:
    cfg, data = rv_problem(constrained)
    fit = least_squares.CentralRadialVelocity()
    fit.set_up(p.BinaryInitialParameters(**deepcopy(cfg)), data, observer_system_cls=RadialVelocitySystem)
    predictions = fit.prepare_synthetic(normalized(fit))
    expected = np.concatenate([(d.y_data - predictions[b][fit.x_data_reducer[b]]) / d.y_err for b, d in data.items()])
    assert_allclose(fit.central_rv_model_to_fit(normalized(fit)), expected, atol=1e-12)
    mapped = {b: predictions[b][fit.x_data_reducer[b]] for b in data}
    assert_allclose(np.dot(expected, expected), cost_fns.wssr(fit.y_data, fit.y_err, mapped), rtol=2e-15)
    report, result = capture_fit(fit.fit, data, p.BinaryInitialParameters(**cfg), ftol=1e-12, gtol=1e-10)
    assert result.success
    assert result.jac.shape == (27, 2 if constrained else 3)
    gamma_column = list(fit.fitable).index("system@gamma")
    assert_allclose(result.jac[:, gamma_column], -250.0, atol=2e-5)
    assert_allclose(result.cost, 0.5 * np.dot(result.fun, result.fun), atol=1e-12)
    for key, value in {"asini": 10.0, "mass_ratio": 0.7, "gamma": 12000.0}.items():
        assert_allclose(report["system"][key]["value"], value, atol=1e-7)
    assert report["system"]["period"]["fixed"]
    if constrained:
        assert report["system"]["mass_ratio"]["constraint"] == "system@asini * 0.07"
    assert_allclose(report["r_squared"]["value"], 1.0, atol=1e-12)
    assert_allclose(fit.central_rv_model_to_fit(result.x), result.fun, atol=1e-10)


def test_robust_rv_loss_acts_on_each_standardized_observation() -> None:
    cfg, data = rv_problem(outlier=True)
    records = {}
    for loss in ("linear", "soft_l1"):
        fit = least_squares.CentralRadialVelocity()
        report, result = capture_fit(fit.fit, data, p.BinaryInitialParameters(**deepcopy(cfg)), loss=loss)
        expected = 0.5 * np.dot(result.fun, result.fun) if loss == "linear" else np.sum(np.sqrt(1 + result.fun**2) - 1)
        assert_allclose(result.cost, expected, rtol=2e-12)
        records[loss] = report["system"]["gamma"]["value"]
    assert_allclose(records["linear"], 12000.0 + 20000.0 / 27, atol=0.01)
    assert abs(records["soft_l1"] - 12000.0) < 10.0


@pytest.mark.requires_ld(passbands=(V, B, "bolometric"), laws=("linear",))
def test_lc_invalid_penalty_has_fixed_length_and_finite_linear_cost(science_settings: None) -> None:
    fit = prepare_lc_fits({V: LC_PHASES, B: [0.25]})[0]
    xn = normalized(fit)
    fit.fixed["primary@surface_potential"].value = 0.1
    residuals = fit.model_to_fit(xn)
    assert residuals.shape == (9,)
    assert np.all(np.isfinite(residuals))
    assert_allclose(np.dot(residuals, residuals), const.MAX_USABLE_FLOAT, rtol=2e-15)
    result = solve(fit.model_to_fit, xn, bounds=(0, 1), max_nfev=2)
    assert np.isfinite(result.cost)
    assert_allclose(result.cost, 0.5 * const.MAX_USABLE_FLOAT, rtol=2e-15)
    # A flat invalid region can still satisfy gtol, as the old penalty intended.
    assert result.jac.shape == (9, len(xn))


def test_residual_order_scalar_uncertainty_and_broadcasting_guard() -> None:
    observed = {"b": np.array([3.0, 5.0]), "a": np.array([7.0])}
    model = {"a": np.array([5.0]), "b": np.array([1.0, 1.0])}
    errors = {"a": np.array([0.5]), "b": np.array(2.0)}
    assert_array_equal(cost_fns.standardized_residuals(observed, errors, model), [1.0, 2.0, 4.0])
    with pytest.raises(lc_prediction.PredictionAlignmentError, match="prediction shape"):
        cost_fns.standardized_residuals(observed, errors, {**model, "a": np.ones(3)})


def test_rv_invalid_physical_synthesis_returns_existing_penalty() -> None:
    cfg, data = rv_problem()
    fit = least_squares.CentralRadialVelocity()
    fit.set_up(p.BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    fit.fixed["system@eccentricity"].value = 1.1
    count = sum(fit.num_of_points.values())
    assert_array_equal(fit.central_rv_model_to_fit(normalized(fit)),
                       np.full(count, np.sqrt(const.MAX_USABLE_FLOAT / count)))


@pytest.mark.parametrize("strategy", list(SamplingMethod))
@pytest.mark.requires_ld(passbands=(V, "bolometric"), laws=("linear",))
def test_sampling_enum_matches_legacy_strings_for_both_fitters(science_settings: None, strategy) -> None:
    enum_fits = prepare_lc_fits({V: LC_PHASES}, 4, samples=strategy)
    legacy_fits = prepare_lc_fits({V: LC_PHASES}, 4, samples=strategy.value)
    for enum_fit, legacy_fit in zip(enum_fits, legacy_fits, strict=True):
        assert_array_equal(enum_fit.fit_xs, legacy_fit.fit_xs)
        enum_prediction = enum_fit.predict_light_curves(physical(enum_fit))
        legacy_prediction = legacy_fit.predict_light_curves(physical(legacy_fit))
        assert_array_equal(enum_prediction[V], legacy_prediction[V])


@pytest.mark.requires_ld(passbands=(V, "bolometric"), laws=("linear",))
def test_sampling_enum_defaults_serialization_and_invalid_strategy(science_settings: None) -> None:
    assert inspect.signature(least_squares.LightCurveFit.fit).parameters["samples"].default is SamplingMethod.UNIFORM
    assert inspect.signature(mcmc.LightCurveFit.fit).parameters["samples"].default is SamplingMethod.UNIFORM
    assert json.dumps(SamplingMethod.ADAPTIVE) == '"adaptive"'
    assert str(SamplingMethod.UNIFORM) == "uniform"
    assert pickle.loads(pickle.dumps(SamplingMethod.ADAPTIVE)) is SamplingMethod.ADAPTIVE
    with pytest.raises(ElisaAnalyticsTaskError, match="SamplingMethod.UNIFORM"):
        prepare_lc_fits({V: LC_PHASES}, 4, samples="unknown")
