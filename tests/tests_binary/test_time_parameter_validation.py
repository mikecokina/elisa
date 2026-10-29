"""Time conversion preserves units and rejects invalid periods before arithmetic."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from jsonschema import ValidationError
from numpy.testing import assert_allclose

from elisa.analytics.tools.utils import time_layer_resolver
from elisa.base.error import PhysicalStateError
from elisa.binary_system import t_layer


INVALID_PERIODS = [
    pytest.param(0.0, PhysicalStateError, id="zero-period"),
    pytest.param(-1.0, PhysicalStateError, id="negative-period"),
    pytest.param(np.nan, FloatingPointError, id="nan-period"),
    pytest.param(np.inf, FloatingPointError, id="positive-infinite-period"),
    pytest.param(-np.inf, FloatingPointError, id="negative-infinite-period"),
]


@pytest.mark.parametrize(
    "convert",
    [
        pytest.param(t_layer.jd_to_phase, id="jd-to-phase"),
        pytest.param(t_layer.phase_to_jd, id="phase-to-jd"),
    ],
)
@pytest.mark.parametrize(("period", "expected_error"), INVALID_PERIODS)
def test_period_failure_precedes_invalid_epoch_and_coordinates(
    convert: Callable[..., Any], period: float, expected_error: type[Exception],
) -> None:
    invalid_epoch = -1.0
    invalid_coordinates = object()

    with pytest.raises(expected_error, match="period") as caught:
        convert(invalid_epoch, period, invalid_coordinates)

    assert type(caught.value) is expected_error


@pytest.mark.parametrize("pop", [False, True], ids=["retain-epoch", "remove-epoch"])
@pytest.mark.parametrize(("period", "expected_error"), INVALID_PERIODS)
def test_fitting_time_resolver_preserves_period_exception_type(
    period: float, expected_error: type[Exception], pop: bool,
) -> None:
    observation_times = np.array([2450000.0, 2450001.0])
    model_parameters = {
        "system@primary_minimum_time": 2450000.0,
        "system@period": period,
    }

    with pytest.raises(expected_error, match="period") as caught:
        time_layer_resolver(observation_times, pop=pop, **model_parameters)

    assert type(caught.value) is expected_error


@pytest.mark.parametrize(
    "time_scale",
    [pytest.param(1.0, id="days"), pytest.param(86400.0, id="seconds")],
)
def test_time_conversion_preserves_the_callers_numeric_time_scale(time_scale: float) -> None:
    epoch = 3.0 * time_scale
    period = 2.0 * time_scale
    observation_times = np.array([3.0, 3.5, 4.0]) * time_scale

    phases = t_layer.jd_to_phase(epoch, period, observation_times)
    restored_times = t_layer.phase_to_jd(epoch, period, phases)

    assert_allclose(phases, [0.0, 0.25, 0.5])
    assert_allclose(restored_times, observation_times)
    assert t_layer.jd_to_phase(epoch, period, 3.5 * time_scale) == 0.25
    assert t_layer.phase_to_jd(epoch, period, 0.25) == 3.5 * time_scale


def test_valid_period_preserves_the_existing_epoch_validation() -> None:
    with pytest.raises(ValidationError, match="Primary minimum time"):
        t_layer.jd_to_phase(0.0, 2.0, np.array([3.0, 3.5, 4.0]))
