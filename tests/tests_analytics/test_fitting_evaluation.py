"""Configured LC/RV evaluations, serialization and changing ephemerides."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from elisa import const
from elisa import units as u
from elisa.analytics.binary_fit import lc_prediction, least_squares, mcmc, shared
from elisa.analytics.binary_fit.shared import AbstractFit
from elisa.analytics.models import cost_fns, rv
from elisa.analytics.params import parameters
from elisa.base.error import PhysicalStateError, PredictionAlignmentError
from elisa.binary_system.curves.community import RadialVelocitySystem
from tests.tests_analytics.fitting_helpers import (
    LC_PHASES,
    V,
    lc_setup,
    make_lc,
    normalized,
    physical,
    prepare_lc_fits,
    rv_problem,
)


def evaluate_fit(fit: AbstractFit) -> float | NDArray[np.float64]:
    if isinstance(fit, mcmc.MCMCFit):
        fit.prepare_priors()
        return float(fit.ln_probability(normalized(fit)))
    if isinstance(fit, least_squares.LightCurveFit):
        return fit.model_to_fit(normalized(fit))
    assert isinstance(fit, least_squares.CentralRadialVelocity)
    return fit.central_rv_model_to_fit(normalized(fit))


@pytest.mark.parametrize("kind", ["lc", "rv"])
@pytest.mark.parametrize("method", ["lsq", "mcmc"])
def test_configured_fit_survives_multiprocessing_pickle(science_settings: None, kind: str, method: str) -> None:
    if kind == "lc":
        fit = prepare_lc_fits({V: LC_PHASES})[0 if method == "lsq" else 1]
    else:
        config, data = rv_problem(constrained=True)
        fit = least_squares.CentralRadialVelocity() if method == "lsq" else mcmc.CentralRadialVelocity()
        fit.set_up(parameters.BinaryInitialParameters(**config), data, observer_system_cls=RadialVelocitySystem)
    expected = evaluate_fit(fit)

    restored = pickle.loads(pickle.dumps(fit))
    assert_allclose(evaluate_fit(restored), expected, rtol=2e-14, atol=1e-8)
    assert physical(restored)["system@period"] == 3.0


def test_dynamic_rv_ephemeris_matches_safe_standalone_forward_model() -> None:
    cfg, data = rv_problem()
    cfg["system"]["period"] = {"value": 3.0, "fixed": False, "min": 2.5, "max": 3.5}
    cfg["system"]["primary_minimum_time"] = {"value": 2450000.0, "fixed": False, "min": 2449999.0, "max": 2450001.0}
    for dataset in data.values():
        dataset.x_data = 2450000.0 + 3 * dataset.x_data
        dataset.x_unit = u.d
    fit = least_squares.CentralRadialVelocity()
    fit.set_up(parameters.BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    xn = normalized(fit)
    for period, epoch in [(3.0, 2450000.0), (3.1, 2450000.02)]:
        xn[list(fit.fitable).index("system@period")] = parameters.normalize_value(
            period, *fit.normalization["system@period"]
        )
        xn[list(fit.fitable).index("system@primary_minimum_time")] = parameters.normalize_value(
            epoch, *fit.normalization["system@primary_minimum_time"]
        )
        physical = parameters.prepare_properties_set(
            np.asarray(parameters.vector_renormalizer(xn, fit.fitable, fit.normalization)),
            fit.fitable,
            fit.constrained,
            fit.fixed,
        )
        expected = rv.central_rv_synthetic(fit.x_data_reduced, fit.observer, **physical)
        actual = fit.prepare_synthetic(xn)
        for key in expected:
            assert_allclose(actual[key], expected[key], rtol=2e-14, atol=1e-8)


@pytest.fixture(params=[least_squares.DetachedLightCurveFit, mcmc.DetachedLightCurveFit])
def lc_fit(request: pytest.FixtureRequest) -> least_squares.DetachedLightCurveFit | mcmc.DetachedLightCurveFit:
    fit: least_squares.DetachedLightCurveFit | mcmc.DetachedLightCurveFit = request.param()
    lc_setup(fit, {V: make_lc(errors=[0.1, 0.1, 0.1])})
    if isinstance(fit, mcmc.MCMCFit):
        fit.prepare_priors()
    return fit


@pytest.fixture(params=[least_squares.CentralRadialVelocity, mcmc.CentralRadialVelocity])
def rv_fit(request: pytest.FixtureRequest) -> least_squares.CentralRadialVelocity | mcmc.CentralRadialVelocity:
    config, data = rv_problem()
    fit: least_squares.CentralRadialVelocity | mcmc.CentralRadialVelocity = request.param()
    fit.set_up(parameters.BinaryInitialParameters(**config), data, observer_system_cls=RadialVelocitySystem)
    if isinstance(fit, mcmc.MCMCFit):
        fit.prepare_priors()
    return fit


@contextmanager
def use_lc_synthesis(
    fit: least_squares.DetachedLightCurveFit | mcmc.DetachedLightCurveFit,
    synthesis: Callable[..., dict[str, NDArray]],
) -> Iterator[None]:
    """Supply synthesis to both callbacks, including MCMC's bound default."""
    predict = fit.predict_light_curves

    def predict_from_source(model_parameters: dict[str, Any], **options: Any) -> dict:
        options["synthesis"] = synthesis
        return predict(model_parameters, **options)

    with patch.object(fit, "predict_light_curves", side_effect=predict_from_source):
        yield


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param({}, id="missing_passband"),
        pytest.param({"unexpected": np.ones(3)}, id="wrong_passband"),
        pytest.param({V: np.ones(3), "unexpected": np.ones(3)}, id="extra_passband"),
        pytest.param({V: np.ones((3, 1))}, id="column"),
        pytest.param({V: np.ones((1, 3))}, id="row"),
        pytest.param({V: np.array(1.0)}, id="scalar"),
        pytest.param({V: np.ones(2)}, id="short"),
        pytest.param({V: np.ones(4)}, id="oversized"),
        pytest.param({V: np.array([1.0, 1.0, np.nan])}, id="nan"),
        pytest.param({V: np.array([1.0, 1.0, np.inf])}, id="inf"),
        pytest.param({V: np.array([1.0, 1.0, -np.inf])}, id="negative-inf"),
    ],
)
def test_lc_malformed_output_propagates_through_callbacks(
    lc_fit: least_squares.DetachedLightCurveFit | mcmc.DetachedLightCurveFit, raw: dict[str, NDArray]
) -> None:
    normalized_values = normalized(lc_fit)
    with use_lc_synthesis(lc_fit, lambda *args, **kwargs: raw):
        if isinstance(lc_fit, mcmc.MCMCFit):
            with pytest.raises(PredictionAlignmentError):
                lc_fit.likelihood(normalized_values)
            with pytest.raises(PredictionAlignmentError):
                lc_fit.ln_probability(normalized_values)
        else:
            with pytest.raises(PredictionAlignmentError):
                lc_fit.model_to_fit(normalized_values)


def malformed_rv_curves(defect: str, size: int) -> dict[str, NDArray]:
    raw = {"primary": np.ones(size), "secondary": np.ones(size)}
    if defect == "missing":
        raw.pop("primary")
    elif defect == "replacement":
        raw["tertiary"] = raw.pop("primary")
    else:
        defective_curves = {
            "column": np.ones((size, 1)),
            "row": np.ones((1, size)),
            "scalar": np.array(1.0),
            "short": np.ones(size - 1),
            "oversized": np.ones(size + 1),
            "nan": np.append(np.ones(size - 1), np.nan),
            "inf": np.append(np.ones(size - 1), np.inf),
            "negative-inf": np.append(np.ones(size - 1), -np.inf),
        }
        raw["primary"] = defective_curves[defect]
    return raw


@pytest.mark.parametrize(
    "defect", ["missing", "replacement", "column", "row", "scalar", "short", "oversized", "nan", "inf", "negative-inf"],
)
def test_rv_malformed_output_propagates_through_callbacks(
    rv_fit: least_squares.CentralRadialVelocity | mcmc.CentralRadialVelocity, defect: str
) -> None:
    raw = malformed_rv_curves(defect, len(rv_fit.x_data_reduced))
    normalized_values = normalized(rv_fit)
    with patch.object(rv, "central_rv_synthetic", return_value=raw):
        if isinstance(rv_fit, mcmc.MCMCFit):
            with pytest.raises(PredictionAlignmentError):
                rv_fit.likelihood(normalized_values)
            with pytest.raises(PredictionAlignmentError):
                rv_fit.ln_probability(normalized_values)
        else:
            with pytest.raises(PredictionAlignmentError):
                rv_fit.central_rv_model_to_fit(normalized_values)


def test_rv_truncated_output_is_checked_before_reducer_indexing() -> None:
    coordinates = np.array([0.0, 0.1, 0.2, 0.3])
    reducers = {"primary": np.array([3, 0, 3]), "secondary": np.array([1])}
    raw = {"primary": np.ones(3), "secondary": np.ones(4)}
    with pytest.raises(PredictionAlignmentError, match="primary.*coordinate grid"):
        shared.map_rv_predictions(raw, coordinates, reducers)


@pytest.mark.parametrize("defect", ["missing", "replacement", "column", "short", "oversized"])
def test_rv_r_squared_rejects_malformed_output(
    rv_fit: least_squares.CentralRadialVelocity | mcmc.CentralRadialVelocity, defect: str
) -> None:
    raw = malformed_rv_curves(defect, len(rv_fit.x_data_reduced))
    with pytest.raises(PredictionAlignmentError):
        shared.rv_r_squared(
            lambda *args, **kwargs: raw,
            rv_fit.x_data_reduced,
            rv_fit.y_data,
            rv_fit.x_data_reducer,
            RadialVelocitySystem,
            **physical(rv_fit),
        )


def test_lc_synthesis_alignment_error_is_not_a_penalty_or_rejection(
    lc_fit: least_squares.DetachedLightCurveFit | mcmc.DetachedLightCurveFit,
) -> None:
    error = PredictionAlignmentError("broken synthesis integration")

    def broken_synthesis(*args, **kwargs):
        raise error

    with use_lc_synthesis(lc_fit, broken_synthesis), pytest.raises(PredictionAlignmentError) as raised:
        evaluate_fit(lc_fit)
    assert raised.value is error


def test_adaptive_sampling_preserves_synthesis_alignment_error() -> None:
    fit = least_squares.DetachedLightCurveFit()
    error = PredictionAlignmentError("broken adaptive synthesis")
    lc_setup(fit, {V: make_lc(errors=[0.1, 0.1, 0.1])})
    with (
        patch("elisa.analytics.models.lc.synthetic_binary", side_effect=error),
        pytest.raises(PredictionAlignmentError) as raised,
    ):
        fit.adaptive_sampling()
    assert raised.value is error


def test_lc_valid_predictions_keep_residuals_and_likelihood(
    lc_fit: least_squares.DetachedLightCurveFit | mcmc.DetachedLightCurveFit,
) -> None:
    raw = {V: np.array([2.0, 4.0, 6.0])}
    with use_lc_synthesis(lc_fit, lambda *args, **kwargs: raw):
        actual = evaluate_fit(lc_fit)
    residuals = np.array([5.0, 10.0, -15.0])
    if isinstance(lc_fit, mcmc.MCMCFit):
        expected = -0.5 * (np.dot(residuals, residuals) + 3 * np.log(2 * np.pi / 900))
        assert_allclose(actual, expected, rtol=2e-13, atol=2e-12)
    else:
        assert_allclose(actual, residuals, rtol=2e-13, atol=2e-12)


def test_rv_valid_predictions_preserve_coordinate_order_and_duplicates(
    rv_fit: least_squares.CentralRadialVelocity | mcmc.CentralRadialVelocity,
) -> None:
    coordinates = rv_fit.x_data_reduced
    raw = {"primary": 10000 + 1000 * coordinates, "secondary": -5000 + 2000 * coordinates}
    coefficients = {"primary": (10000, 1000), "secondary": (-5000, 2000)}
    residuals = []
    for component, observed in rv_fit.y_data.items():
        offset, slope = coefficients[component]
        prediction = offset + slope * rv_fit.x_data[component]
        residuals.append((observed - prediction) / rv_fit.y_err[component])
    expected_residuals = np.concatenate(residuals)

    with patch.object(rv, "central_rv_synthetic", return_value=raw):
        actual = evaluate_fit(rv_fit)
    if isinstance(rv_fit, mcmc.MCMCFit):
        log_variances = sum(np.log(2 * np.pi * errors**2).sum() for errors in rv_fit.y_err.values())
        expected = -0.5 * (np.dot(expected_residuals, expected_residuals) + log_variances)
        assert_allclose(actual, expected, rtol=2e-13, atol=2e-12)
    else:
        assert_allclose(actual, expected_residuals, rtol=2e-13, atol=2e-12)


@pytest.mark.parametrize("predicted", [{}, {"v": np.ones((2, 1))}, {"v": np.ones(3)}])
def test_standalone_residuals_report_prediction_alignment(predicted: dict[str, NDArray]) -> None:
    with pytest.raises(PredictionAlignmentError):
        cost_fns.standardized_residuals({"v": np.ones(2)}, {"v": np.ones(2)}, predicted)


@pytest.mark.parametrize(
    ("observed", "errors"),
    [
        ({"v": np.ones(2)}, {}),
        ({"v": np.ones(2)}, {"v": np.ones(3)}),
        ({"v": np.ones((2, 1))}, {"v": np.ones(2)}),
    ],
)
def test_standalone_residuals_keep_observation_and_uncertainty_value_errors(observed, errors) -> None:
    with pytest.raises(ValueError) as raised:
        cost_fns.standardized_residuals(observed, errors, {"v": np.ones(2)})
    assert type(raised.value) is ValueError


def test_alignment_error_keeps_legacy_import_and_value_error_compatibility() -> None:
    assert lc_prediction.PredictionAlignmentError is PredictionAlignmentError
    assert issubclass(PredictionAlignmentError, ValueError)
    assert not issubclass(PredictionAlignmentError, PhysicalStateError)


def test_lc_other_synthesis_value_errors_keep_existing_handling(
    lc_fit: least_squares.DetachedLightCurveFit | mcmc.DetachedLightCurveFit,
) -> None:
    def invalid_synthesis(*args, **kwargs):
        raise ValueError("invalid physical synthesis")

    with use_lc_synthesis(lc_fit, invalid_synthesis):
        if isinstance(lc_fit, mcmc.MCMCFit):
            with pytest.raises(ValueError, match="invalid physical synthesis"):
                lc_fit.ln_probability(normalized(lc_fit))
        else:
            residuals = lc_fit.model_to_fit(normalized(lc_fit))
            assert residuals.shape == (3,)
            assert np.all(np.isfinite(residuals))
            assert_allclose(np.dot(residuals, residuals), const.MAX_USABLE_FLOAT, rtol=2e-15)


def test_lc_mcmc_still_rejects_explicit_physical_exclusions() -> None:
    lc_fit = mcmc.DetachedLightCurveFit()
    lc_setup(lc_fit, {V: make_lc(errors=[0.1, 0.1, 0.1])})
    lc_fit.prepare_priors()

    def excluded_synthesis(*args, **kwargs):
        raise PhysicalStateError("excluded geometry")

    with use_lc_synthesis(lc_fit, excluded_synthesis):
        with pytest.raises(PhysicalStateError, match="excluded geometry"):
            lc_fit.likelihood(normalized(lc_fit))
        assert lc_fit.ln_probability(normalized(lc_fit)) == -np.inf


def test_rv_mcmc_still_rejects_explicit_physical_exclusions() -> None:
    config, data = rv_problem()
    rv_fit = mcmc.CentralRadialVelocity()
    rv_fit.set_up(parameters.BinaryInitialParameters(**config), data, observer_system_cls=RadialVelocitySystem)
    rv_fit.prepare_priors()
    with patch.object(rv, "central_rv_synthetic", side_effect=PhysicalStateError("excluded geometry")):
        with pytest.raises(PhysicalStateError, match="excluded geometry"):
            rv_fit.likelihood(normalized(rv_fit))
        assert rv_fit.ln_probability(normalized(rv_fit)) == -np.inf


@pytest.mark.parametrize("kind", ["lc", "rv"])
def test_mcmc_unrepresentable_likelihood_propagates_with_finite_predictions(kind: str) -> None:
    fit: mcmc.DetachedLightCurveFit | mcmc.CentralRadialVelocity
    if kind == "lc":
        fit = mcmc.DetachedLightCurveFit()
        lc_setup(fit, {V: make_lc(errors=[0.1, 0.1, 0.1])})
        raw = {V: np.ones(len(fit.x_data_reduced))}
        synthesis = use_lc_synthesis(fit, lambda *args, **kwargs: raw)
    else:
        config, data = rv_problem()
        fit = mcmc.CentralRadialVelocity()
        fit.set_up(parameters.BinaryInitialParameters(**config), data, observer_system_cls=RadialVelocitySystem)
        raw = {component: np.ones(len(fit.x_data_reduced)) for component in data}
        synthesis = patch.object(rv, "central_rv_synthetic", return_value=raw)
    # Finite inputs reach the real Gaussian calculation, but its total log
    # normalization exceeds float64 range for multiple observations.
    fit.fixed["nuisance@ln_f"].value = 1e308
    fit.prepare_priors()
    xn = normalized(fit)
    with synthesis:
        with pytest.raises(FloatingPointError, match="Gaussian log density is not representable"):
            fit.likelihood(xn)
        with pytest.raises(FloatingPointError, match="Gaussian log density is not representable"):
            fit.ln_probability(xn)
