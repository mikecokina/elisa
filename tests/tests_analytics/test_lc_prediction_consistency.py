"""Scientific regression tests: actual binary synthesis, deterministic data.

Requires the real ELISA linear LD and CK04 support tables. By default these
are read from ~/.elisa; ELISA_SCIENCE_SUPPORT can select another installation.
Synthesis, interpolation, parameters, reducers and costs use real calculations.
Forwarding wrappers record the inputs and outputs used by each fitter.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from elisa import const
from elisa.analytics.binary_fit import least_squares, mcmc, shared
from elisa.analytics.binary_fit.lc_prediction import PredictionAlignmentError
from elisa.analytics.binary_fit.shared import AbstractLCFit, lc_r_squared
from elisa.analytics.models import lc
from elisa.analytics.params import parameters as p
from elisa.analytics.tools.utils import time_layer_resolver
from elisa.base.error import PhysicalStateError
from elisa.binary_system.system import BinarySystem
from elisa.conf.enums import SamplingMethod
from tests.tests_analytics.fitting_helpers import (
    EQUAL_LENGTH_PHASES,
    LC_PHASES,
    MANUAL_PHASES,
    SECONDARY_PHASES,
    B,
    V,
    prepare_lc_fits,
    synthesize_lc,
)

pytestmark = pytest.mark.usefixtures("science_settings")


def capture_prediction(callback: Callable, *args, **kwargs) -> tuple[Any, dict[str, Any]]:
    """Record real synthesis inputs and predictions through a forwarding wrapper."""
    captured = {}
    predict = shared.predict_light_curves

    def record_prediction(*prediction_args, **options):
        synthesis = options.pop("synthesis", lc.synthetic_binary)

        def record_synthesis(phases, discretization, observer, **physical):
            captured["support"] = phases.copy()
            captured["physical"] = deepcopy(physical)
            raw = synthesis(phases, discretization, observer, **physical)
            captured["raw"] = deepcopy(raw)
            return raw

        result = predict(*prediction_args, synthesis=record_synthesis, **options)
        captured["prediction"] = deepcopy(result)
        return result

    with patch.object(shared, "predict_light_curves", side_effect=record_prediction):
        value = callback(*args, **kwargs)
    return value, captured


def assert_observation_mapping(fit: AbstractLCFit, physical: dict[str, Any], captured: dict[str, Any]) -> None:
    """Compare the mapped prediction to direct synthesis or cubic interpolation."""
    phases, _ = time_layer_resolver(fit.x_data_reduced, **physical)
    if fit.fit_xs is not None:
        assert_array_equal(captured["support"], fit.fit_xs)
    # Independent mapping/interpolation on the actual raw synthesized support.
    for band, curve in captured["raw"].items():
        target = phases[fit.x_data_reducer[band]]
        if np.array_equal(captured["support"], phases):
            mapped = curve[fit.x_data_reducer[band]]
            own = synthesize_lc(target, [band], captured["physical"])[band]
            assert_allclose(mapped / mapped.mean(), own / own.mean(), atol=2e-12, rtol=2e-12)
        else:
            mapped = interp1d(captured["support"], curve, kind="cubic")(target)
        predicted = captured["prediction"][band]
        assert predicted.shape == fit.y_data[band].shape
        assert_allclose(predicted, mapped / mapped.mean(), atol=2e-12, rtol=2e-12)
        assert_allclose(predicted.mean(), 1.0, atol=3e-15)


def assert_statistical_cost(
    fit: AbstractLCFit,
    physical: dict[str, Any],
    normalized_values: NDArray,
    value: Any,
    predictions: dict[str, NDArray],
) -> None:
    chi2 = sum(
        np.dot((fit.y_data[b] - m) / fit.y_err[b], (fit.y_data[b] - m) / fit.y_err[b]) for b, m in predictions.items()
    )
    if isinstance(fit, least_squares.LightCurveFit):
        independent = np.concatenate([(fit.y_data[b] - predictions[b]) / fit.y_err[b] for b in fit.y_data])
        assert value.ndim == 1
        assert_allclose(value, independent, atol=2e-12)
        assert_allclose(np.dot(value, value), chi2, rtol=2e-13, atol=2e-12)
    else:
        assert isinstance(fit, mcmc.LightCurveFit)
        scale = 1 + np.exp(2 * physical["nuisance@ln_f"])
        expected = -0.5 * (
            chi2 / scale
            + sum(
                np.log(2 * np.pi * scale * np.broadcast_to(fit.y_err[b], y.shape) ** 2).sum()
                for b, y in fit.y_data.items()
            )
        )
        assert_allclose(value, expected, rtol=2e-13, atol=2e-12)
        fit.norm_init_vector, fit.sigmas = normalized_values, np.full(len(normalized_values), np.nan)
        assert_allclose(fit.ln_probability(normalized_values), value, atol=2e-12)


def assert_prediction_r_squared(fit: AbstractLCFit, physical: dict[str, Any], predictions: dict[str, NDArray]) -> None:
    atm_physical = p.extend_json_with_atm_params(
        physical, atmosphere_model=fit.atmosphere_model, limb_darkening_coefficients=fit.limb_darkening_coefficients
    )
    r2, score_capture = capture_prediction(
        lc_r_squared,
        lc.synthetic_binary,
        fit.x_data_reduced,
        fit.y_data,
        fit.observer.passband,
        fit.discretization,
        fit.x_data_reducer,
        1 / fit.interp_treshold,
        fit.interp_treshold,
        BinarySystem,
        fit_xs=fit.fit_xs,
        x_data=fit.x_data,
        **atm_physical,
    )
    residual = sum(np.sum((fit.y_data[b] - m) ** 2) for b, m in predictions.items())
    variance = sum(np.sum((y - y.mean()) ** 2) for y in fit.y_data.values())
    assert_allclose(r2, 1 - residual / variance, atol=2e-12)
    for band in fit.y_data:
        assert_allclose(score_capture["prediction"][band], predictions[band], atol=2e-12, rtol=2e-12)


def assert_consistent_predictions(
    fits: tuple[least_squares.DetachedLightCurveFit, mcmc.DetachedLightCurveFit],
    *,
    change_vector: bool = False,
) -> list[dict[str, Any]]:
    """Check the same scientific expectations for LSQ and MCMC predictions."""
    captures = []
    for fit in fits:
        normalized_values = np.asarray(p.vector_normalizer(fit.initial_vector, fit.fitable, fit.normalization))
        if change_vector:
            normalized_values[list(fit.fitable).index("system@inclination")] = 0.4
            normalized_values[list(fit.fitable).index("primary@t_eff")] = 0.6
        values = p.vector_renormalizer(normalized_values, fit.fitable, fit.normalization)
        physical = p.prepare_properties_set(np.asarray(values), fit.fitable, fit.constrained, fit.fixed)
        assert_allclose(physical["secondary@t_eff"], physical["primary@t_eff"] * 0.7692307692307693)

        callback = fit.model_to_fit if isinstance(fit, least_squares.LightCurveFit) else fit.likelihood
        value, captured = capture_prediction(callback, normalized_values)
        assert_observation_mapping(fit, physical, captured)
        assert_statistical_cost(fit, physical, normalized_values, value, captured["prediction"])
        assert_prediction_r_squared(fit, physical, captured["prediction"])
        if fit.atmosphere_model:
            assert captured["physical"]["primary@atmosphere"] == "ck04"
        if fit.limb_darkening_coefficients:
            assert captured["physical"]["primary@limb_darkening_coefficients"][V] == [0.1]
        captures.append(captured)

    for band in fits[0].y_data:
        assert_allclose(captures[0]["prediction"][band], captures[1]["prediction"][band], atol=2e-12, rtol=2e-12)
    return captures


@pytest.mark.parametrize(
    "grids",
    [
        pytest.param({V: LC_PHASES}, id="sorted"),
        pytest.param({V: [0.25, 0.0, 0.5, 0.12]}, id="unsorted"),
        pytest.param({V: [0.0, 0.1, 0.1, 0.5]}, id="repeated"),
        pytest.param({V: LC_PHASES, B: SECONDARY_PHASES}, id="different"),
        pytest.param({V: [0.0, 0.1, 0.25, 0.5], B: [0.0, 0.12, 0.3, 0.5]}, id="different_equal_lengths"),
        pytest.param({V: LC_PHASES, B: [0.25]}, id="singleton"),
        pytest.param({V: [0.5, 0.0, 0.1, 0.1], B: [0.3, 0.1, 0.0]}, id="unsorted_repeated_multiband"),
    ],
)
def test_direct_observation_mapping(grids) -> None:
    assert_consistent_predictions(prepare_lc_fits(grids))


@pytest.mark.parametrize(
    ("threshold", "samples"),
    [
        pytest.param(4, SamplingMethod.UNIFORM, id="uniform"),
        pytest.param(4, SamplingMethod.ADAPTIVE, id="adaptive"),
        pytest.param(4, MANUAL_PHASES, id="manual"),
        pytest.param(4, np.asarray(MANUAL_PHASES), id="manual_array"),
        pytest.param(4, LC_PHASES, id="manual_exact_coordinates"),
        pytest.param(4, EQUAL_LENGTH_PHASES, id="manual_equal_length"),
        pytest.param(6, SamplingMethod.UNIFORM, id="auto_equal_length"),
        pytest.param(8, SamplingMethod.UNIFORM, id="threshold_boundary"),
    ],
)
def test_prepared_support(threshold: int, samples) -> None:
    assert_consistent_predictions(prepare_lc_fits({V: LC_PHASES}, threshold, samples))


@pytest.mark.parametrize(("jd", "threshold"), [(False, 100), (False, 4), (True, 100), (True, 4)])
def test_multiband_phase_and_jd(jd: bool, threshold: int) -> None:
    assert_consistent_predictions(prepare_lc_fits({V: [0.6, 0.0, 0.1, 0.1, 0.5], B: [0.25]}, threshold, jd=jd))


@pytest.mark.parametrize(
    ("threshold", "atmosphere", "custom_ld"), [(100, False, True), (100, True, False), (4, True, True)]
)
def test_explicit_atmosphere_and_ld(threshold: int, atmosphere, custom_ld: bool) -> None:
    grids = {V: LC_PHASES, B: SECONDARY_PHASES}
    fits = prepare_lc_fits(
        grids,
        threshold,
        atmosphere={"primary": "ck04", "secondary": "ck04"} if atmosphere else None,
        custom_ld=custom_ld,
    )
    caps = assert_consistent_predictions(fits)
    baseline = prepare_lc_fits(grids, threshold)
    physical = p.prepare_properties_set(
        np.asarray(baseline[0].initial_vector), baseline[0].fitable.keys(), baseline[0].constrained, baseline[0].fixed
    )
    control = baseline[0].predict_light_curves(physical)
    assert max(np.max(np.abs(caps[0]["prediction"][b] - control[b])) for b in grids) > 1e-5


@pytest.mark.parametrize(("standard", "jd"), [(False, False), (True, False), (False, True), (True, True)])
def test_standard_community_constraints_and_free_noise(standard: bool, jd: bool) -> None:
    assert_consistent_predictions(
        prepare_lc_fits({V: LC_PHASES, B: SECONDARY_PHASES}, 4, jd=jd, standard=standard, free_noise=True),
        change_vector=True,
    )


@pytest.mark.parametrize("threshold", [100, 4])
def test_missing_error_scalar_noise_model(threshold: int) -> None:
    fits = prepare_lc_fits({V: LC_PHASES, B: [0.25]}, threshold, missing_errors=True)
    for fit in fits:
        for band, error in fit.y_err.items():
            assert error.shape == fit.y_data[band].shape
            assert_allclose(error, 0.05)
    assert_consistent_predictions(fits)


def test_invalid_physical_synthesis_retains_penalties() -> None:
    fits = prepare_lc_fits({V: LC_PHASES})
    for fit in fits:
        fit.fixed["primary@surface_potential"].value = 0.1
        xn = np.asarray(p.vector_normalizer(fit.initial_vector, fit.fitable.keys(), fit.normalization))
        if isinstance(fit, least_squares.LightCurveFit):
            penalty = fit.model_to_fit(xn)
            assert penalty.shape == (len(LC_PHASES),)
            assert np.all(np.isfinite(penalty))
            assert_allclose(np.dot(penalty, penalty), const.MAX_USABLE_FLOAT, rtol=2e-15)
        else:
            fit.norm_init_vector, fit.sigmas = xn, np.full(len(xn), np.nan)
            with pytest.raises(PhysicalStateError):
                fit.likelihood(xn)
            assert fit.ln_probability(xn) == -np.inf


def test_jd_period_epoch_changes() -> None:
    fits = prepare_lc_fits({V: [1.6, 1.0, 1.1, 1.1, 1.5], B: [1.25]}, 4, jd=True)
    for fit in fits:
        fit.fixed["system@period"].value = 3.1
        fit.fixed["system@primary_minimum_time"].value = 2450000.02
    assert_consistent_predictions(fits)


def test_default_threshold_equal_length() -> None:
    assert_consistent_predictions(prepare_lc_fits({V: np.linspace(0.0, 0.99, 302)}, 300))


def test_dense_support_approximation() -> None:
    fits = prepare_lc_fits(
        {V: np.linspace(0.0, 0.99, 41), B: SECONDARY_PHASES}, 4, np.linspace(-0.01, 1.01, 302).tolist()
    )
    caps = assert_consistent_predictions(fits)
    for band, raw_x in fits[0].x_data.items():
        direct = synthesize_lc(raw_x, [band], caps[0]["physical"])[band]
        assert_allclose(caps[0]["prediction"][band], direct / direct.mean(), atol=0.005, rtol=0)


@pytest.mark.parametrize(
    "defect", ["coordinates", "reducer_length", "reducer_bounds", "reducer_type", "uncertainty_length"]
)
def test_invalid_observation_mapping_requires_revalidation(defect) -> None:
    for fit in prepare_lc_fits({V: LC_PHASES, B: [0.25]}):
        if defect == "coordinates":
            fit.x_data_reducer[V] = fit.x_data_reducer[V][::-1]
        elif defect == "reducer_length":
            fit.x_data_reducer[V] = fit.x_data_reducer[V][:-1]
        elif defect == "reducer_bounds":
            fit.x_data_reducer[V][0] = len(fit.x_data_reduced)
        elif defect == "reducer_type":
            fit.x_data_reducer[V] = fit.x_data_reducer[V].astype(float)
        else:
            fit.y_err[B] = np.ones(8)
        with pytest.raises(PredictionAlignmentError):
            fit.validate_observations()


@pytest.mark.parametrize(
    "support",
    [
        np.array([-0.1, 0.0, 0.5, 0.5, 1.1]),
        np.array([0.0, 0.5, 1.0]),
        np.linspace(0.1, 0.7, 8),
    ],
    ids=["duplicate", "too_short", "uncovered"],
)
def test_invalid_support_escapes_physical_penalty_handlers(support) -> None:
    for fit in prepare_lc_fits({V: LC_PHASES, B: [0.25]}):
        fit.fit_xs = support
        normalized_values = np.asarray(p.vector_normalizer(fit.initial_vector, fit.fitable, fit.normalization))
        if isinstance(fit, mcmc.LightCurveFit):
            fit.prepare_priors()
            callback = fit.ln_probability
        else:
            callback = fit.model_to_fit
        with pytest.raises(PredictionAlignmentError):
            callback(normalized_values)


def test_missing_passband_escapes_physical_penalty_handlers() -> None:
    for fit in prepare_lc_fits({V: LC_PHASES, B: [0.25]}):
        fit.observer.passband.pop(B)
        normalized_values = np.asarray(p.vector_normalizer(fit.initial_vector, fit.fitable, fit.normalization))
        if isinstance(fit, mcmc.LightCurveFit):
            fit.prepare_priors()
            callback = fit.ln_probability
        else:
            callback = fit.model_to_fit
        with pytest.raises(PredictionAlignmentError):
            callback(normalized_values)
