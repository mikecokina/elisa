"""Adaptive sampling fails setup on bad initial models without hiding other failures."""
from unittest.mock import patch

import numpy as np
import pytest

from elisa import BinarySystem
from elisa.analytics.binary_fit import least_squares
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.base.error import (
    ElisaInitialParametersError,
    ElisaNumericError,
    ElisaParametersError,
    ElisaSynthesisError,
    MaxIterationError,
    ElisaPassbandError,
    ElisaPredictionAlignmentError,
    ElisaSettingsError,
    ElisaUnexpectedError,
)
from elisa.conf.enums import SamplingMethod
from tests.tests_analytics.fitting_helpers import V, lc_parameters, lc_setup, make_lc


@pytest.fixture
def prepared_fit():
    fit = least_squares.DetachedLightCurveFit()
    lc_setup(fit, {V: make_lc(errors=[0.1, 0.1, 0.1])})
    return fit


@pytest.mark.parametrize("category", [ElisaParametersError, ElisaSynthesisError, MaxIterationError, ElisaNumericError])
def test_initial_model_failure_becomes_initial_parameters_error(prepared_fit, category):
    failure = category("initial model failed")
    with patch("elisa.analytics.models.lc.synthetic_binary", side_effect=failure):
        with pytest.raises(ElisaInitialParametersError, match="initial model") as caught:
            prepared_fit.adaptive_sampling()
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("category", [
    ElisaPassbandError, ElisaPredictionAlignmentError, ElisaUnexpectedError, ElisaSettingsError,
])
def test_other_errors_propagate_in_their_original_type(prepared_fit, category):
    failure = category("not an initial-model rejection")
    with patch("elisa.analytics.models.lc.synthetic_binary", side_effect=failure):
        with pytest.raises(category) as caught:
            prepared_fit.adaptive_sampling()
    assert caught.value is failure


def test_failed_adaptive_setup_can_be_retried_with_valid_synthesis():
    fit = least_squares.DetachedLightCurveFit()
    config, _ = lc_parameters()
    # Exceed the interpolation threshold so setup actually requests adaptive samples.
    data = {V: make_lc(values=np.ones(120), errors=np.full(120, 0.1))}

    def set_up():
        fit.set_up(
            BinaryInitialParameters(**config), data, passband=[V],
            discretization=10.0, interp_treshold=100, samples=SamplingMethod.ADAPTIVE,
            observer_system_cls=BinarySystem,
        )

    failure = ElisaNumericError("initial numerical failure")
    with patch("elisa.analytics.models.lc.synthetic_binary", side_effect=failure):
        with pytest.raises(ElisaInitialParametersError) as caught:
            set_up()
    assert caught.value.__cause__ is failure

    def valid_synthesis(phases, *args, **kwargs):
        return {"bolometric": 1.0 + 0.1 * np.sin(2.0 * np.pi * phases)}

    with patch("elisa.analytics.models.lc.synthetic_binary", side_effect=valid_synthesis):
        set_up()
    assert fit.fit_xs.shape == (100,)
    assert np.all(np.isfinite(fit.fit_xs))
    assert np.all(np.diff(fit.fit_xs) > 0)


@pytest.mark.parametrize("category", [TypeError, ValueError, RuntimeError, KeyError])
def test_unclassified_sampling_failures_keep_context_and_original_cause(prepared_fit, category):
    failure = category("original failure")
    with patch("elisa.analytics.models.lc.synthetic_binary", side_effect=failure):
        with pytest.raises(ElisaUnexpectedError, match="Adaptive phase sampling failed") as caught:
            prepared_fit.adaptive_sampling()
    assert caught.value.__cause__ is failure


def test_unclassified_normalization_failure_keeps_context_and_original_cause(prepared_fit):
    failure = RuntimeError("normalization failure")
    with (
        patch("elisa.analytics.models.lc.synthetic_binary", return_value={"bolometric": np.ones(10)}),
        patch("elisa.analytics.binary_fit.shared.normalize_light_curve", side_effect=failure),
        pytest.raises(ElisaUnexpectedError, match="normalizing the initial model") as caught,
    ):
        prepared_fit.adaptive_sampling()
    assert caught.value.__cause__ is failure
