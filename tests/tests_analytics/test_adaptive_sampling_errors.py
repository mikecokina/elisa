"""Adaptive sampling fails setup on bad initial models without hiding other failures."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from elisa import BinarySystem, settings
from elisa.analytics.binary_fit import least_squares
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.base.error import (
    ElisaAtmosphereError,
    ElisaInitialParametersError,
    ElisaLimbDarkeningError,
    ElisaMaxIterationError,
    ElisaNumericError,
    ElisaParametersError,
    ElisaPassbandError,
    ElisaPredictionAlignmentError,
    ElisaSettingsError,
    ElisaSpotError,
    ElisaTemperatureError,
    ElisaUnexpectedError,
    MorphologyError,
)
from elisa.conf.enums import SamplingMethod
from tests.tests_analytics.fitting_helpers import V, lc_parameters, lc_setup, make_lc

if TYPE_CHECKING:
    from numpy.typing import NDArray

    import elisa.analytics.binary_fit.least_squares
    import elisa.base.error


@pytest.fixture
def prepared_fit() -> elisa.analytics.binary_fit.least_squares.DetachedLightCurveFit:
    fit = least_squares.DetachedLightCurveFit()
    lc_setup(fit, {V: make_lc(errors=[0.1, 0.1, 0.1])})
    return fit


@pytest.mark.parametrize(
    "category",
    [
        ElisaAtmosphereError,
        ElisaLimbDarkeningError,
        ElisaParametersError,
        MorphologyError,
        ElisaTemperatureError,
        ElisaSpotError,
        ElisaMaxIterationError,
        ElisaNumericError,
    ],
)
def test_initial_model_failure_becomes_initial_parameters_error(
    prepared_fit: elisa.analytics.binary_fit.least_squares.DetachedLightCurveFit,
    category: type[
        elisa.base.error.ElisaAtmosphereError
        | elisa.base.error.ElisaLimbDarkeningError
        | elisa.base.error.ElisaMaxIterationError
        | elisa.base.error.ElisaNumericError
        | elisa.base.error.ElisaParametersError
        | elisa.base.error.ElisaSpotError
        | elisa.base.error.ElisaTemperatureError
        | elisa.base.error.MorphologyError
    ],
) -> None:
    failure = category("initial model failed")
    with (
        patch("elisa.analytics.models.lc.synthetic_binary", side_effect=failure),
        pytest.raises(ElisaInitialParametersError, match="initial model") as caught,
    ):
        prepared_fit.adaptive_sampling()
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    "category",
    [
        ElisaPassbandError,
        ElisaPredictionAlignmentError,
        ElisaUnexpectedError,
        ElisaSettingsError,
    ],
)
def test_other_errors_propagate_in_their_original_type(
    prepared_fit: elisa.analytics.binary_fit.least_squares.DetachedLightCurveFit,
    category: type[
        elisa.base.error.ElisaPassbandError
        | elisa.base.error.ElisaPredictionAlignmentError
        | elisa.base.error.ElisaSettingsError
        | elisa.base.error.ElisaUnexpectedError
    ],
) -> None:
    failure = category("not an initial-model rejection")
    with (
        patch("elisa.analytics.models.lc.synthetic_binary", side_effect=failure),
        pytest.raises(category) as caught,
    ):
        prepared_fit.adaptive_sampling()
    assert caught.value is failure


def test_failed_adaptive_setup_can_be_retried_with_valid_synthesis() -> None:
    fit = least_squares.DetachedLightCurveFit()
    config, _ = lc_parameters()
    # Exceed the interpolation threshold so setup actually requests adaptive samples.
    data = {V: make_lc(values=np.ones(120), errors=np.full(120, 0.1))}

    def set_up() -> None:
        fit.set_up(
            BinaryInitialParameters(**config),
            data,
            passband=[V],
            discretization=10.0,
            interp_treshold=100,
            samples=SamplingMethod.ADAPTIVE,
            observer_system_cls=BinarySystem,
        )

    failure = ElisaNumericError("initial numerical failure")
    with (
        patch("elisa.analytics.models.lc.synthetic_binary", side_effect=failure),
        pytest.raises(ElisaInitialParametersError) as caught,
    ):
        set_up()
    assert caught.value.__cause__ is failure

    def valid_synthesis(
        phases: NDArray[np.float64],
        *_args,
        **_kwargs,
    ) -> dict[str, NDArray[np.float64]]:
        return {"bolometric": 1.0 + 0.1 * np.sin(2.0 * np.pi * phases)}

    with patch("elisa.analytics.models.lc.synthetic_binary", side_effect=valid_synthesis):
        set_up()
    assert fit.fit_xs.shape == (100,)
    assert np.all(np.isfinite(fit.fit_xs))
    assert np.all(np.diff(fit.fit_xs) > 0)


@pytest.mark.parametrize("category", [TypeError, ValueError, RuntimeError, KeyError])
def test_unclassified_sampling_failures_keep_context_and_original_cause(
    prepared_fit: elisa.analytics.binary_fit.least_squares.DetachedLightCurveFit,
    category: type[KeyError | RuntimeError | TypeError | ValueError],
) -> None:
    failure = category("original failure")
    with (
        patch("elisa.analytics.models.lc.synthetic_binary", side_effect=failure),
        pytest.raises(ElisaUnexpectedError, match="Adaptive phase sampling failed") as caught,
    ):
        prepared_fit.adaptive_sampling()
    assert caught.value.__cause__ is failure


def test_unclassified_normalization_failure_keeps_context_and_original_cause(
    prepared_fit: elisa.analytics.binary_fit.least_squares.DetachedLightCurveFit,
) -> None:
    failure = RuntimeError("normalization failure")
    with (
        patch("elisa.analytics.models.lc.synthetic_binary", return_value={"bolometric": np.ones(10)}),
        patch("elisa.analytics.binary_fit.shared.normalize_light_curve", side_effect=failure),
        pytest.raises(ElisaUnexpectedError, match="normalizing the initial model") as caught,
    ):
        prepared_fit.adaptive_sampling()
    assert caught.value.__cause__ is failure


def test_curve_length_sampling_interpolates_phases_without_synthesis() -> None:
    from elisa.analytics.binary_fit.shared import AbstractLCFit

    # Segment lengths are 1 and 2, so four samples fall at arc lengths 0, 1, 2, 3.
    phases = np.array([0.0, 1.0, 2.0])
    flux = np.array([0.0, 0.0, np.sqrt(3.0)])
    result = AbstractLCFit._sample_phases_by_curve_length(phases, flux, 4)
    np.testing.assert_allclose(result, [0.0, 1.0, 1.5, 2.0])


@pytest.mark.parametrize("category", [ElisaNumericError, ElisaPassbandError, RuntimeError])
def test_emergency_penalization_does_not_change_adaptive_setup(
    prepared_fit: elisa.analytics.binary_fit.least_squares.DetachedLightCurveFit,
    category: type[RuntimeError | elisa.base.error.ElisaNumericError | elisa.base.error.ElisaPassbandError],
) -> None:
    """Candidate catch-all must not convert setup failures into penalties."""
    failure = category("setup synthesis failed")
    expected = {
        ElisaNumericError: ElisaInitialParametersError,
        ElisaPassbandError: ElisaPassbandError,
        RuntimeError: ElisaUnexpectedError,
    }[category]
    with (
        settings.override(PENALIZE_ALL_EXCEPTIONS=True),
        patch("elisa.analytics.models.lc.synthetic_binary", side_effect=failure),
        pytest.raises(expected) as caught,
    ):
        prepared_fit.adaptive_sampling()
    if category is ElisaPassbandError:
        assert caught.value is failure
    else:
        assert caught.value.__cause__ is failure
