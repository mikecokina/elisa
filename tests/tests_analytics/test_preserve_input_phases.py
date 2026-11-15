"""Coordinate policy and multi-cycle support regression checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa import settings
from elisa.analytics.binary_fit.lc_prediction import _synthesis_support, extend_phase_support, predict_light_curves
from elisa.analytics.tools.utils import time_layer_resolver
from elisa.observer.observer import Observer

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_resolver_preserves_only_phase_domain() -> None:
    phases = np.array([-1.5, 2.5, 0.25])
    parameters = {"system@period": 2.0}
    folded, _ = time_layer_resolver(phases, **parameters)
    preserved, _ = time_layer_resolver(phases, preserve_phases=True, **parameters)
    assert_array_equal(folded, phases % 1)
    assert_array_equal(preserved, phases)
    for period in (2, 3):
        for epoch in (1, 1.2):
            kwargs = {"system@period": period, "system@primary_minimum_time": epoch}
            converted, _ = time_layer_resolver(phases, **kwargs)
            converted_with_preservation, _ = time_layer_resolver(phases, preserve_phases=True, **kwargs)
            assert_array_equal(converted, converted_with_preservation)


def test_uniform_support_density() -> None:
    for cycles in (1, 4, 10):
        phases = np.linspace(-1.5, -1.5 + cycles, 100)
        support = _synthesis_support(phases, 20, None, preserve_phases=True)
        assert support[0] < phases[0]
        assert support[-1] > phases[-1]
        assert np.max(np.diff(support)) <= 1.1 / 21 + 1e-14


def test_adaptive_support_repeats_features() -> None:
    reference = np.array([-0.01, 0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.01])
    support = extend_phase_support(reference, np.array([-1.5, 2.5]))
    assert np.all(np.diff(support) > 0)
    assert support[0] < -1.5
    assert support[-1] > 2.5
    for cycle in (-1, 0, 1, 2):
        assert np.any(np.isclose(support, cycle + 0.01))


def test_prediction_preserves_order_cycles_and_duplicates() -> None:
    xs = np.array([-1.5, 0.25, 1.25, 10.25])
    reducers = {"a": np.array([3, 0, 1, 1]), "b": np.array([2, 1])}
    observed = {band: np.ones(len(indices)) for band, indices in reducers.items()}

    def synthesis(
        support: NDArray[np.float64],
        _discretization: float,
        _observer: Observer,
        **_parameters: Any,
    ) -> dict[str, NDArray[np.float64]]:
        assert_array_equal(support, xs)
        return dict.fromkeys(reducers, 2 + support / 20)

    with settings.override(PRESERVE_INPUT_PHASES=True):
        result = predict_light_curves(
            x_data_reduced=xs,
            y_data=observed,
            x_data_reducer=reducers,
            discretization=10,
            observer=Observer(passband="bolometric"),
            interp_treshold=100,
            model_parameters={},
            synthesis=synthesis,
        )
    for band, indices in reducers.items():
        expected = 2 + xs[indices] / 20
        assert_allclose(result[band], expected / expected.mean())


@pytest.mark.parametrize(
    ("lower", "upper", "retained"),
    [
        (0.41, 0.42, [0.25, 0.5]),
        (0.49, 0.51, [0.25, 0.5, 0.75]),
        (0.999, 1.001, [0.75, 1.0, 1.25]),
    ],
)
def test_narrow_adaptive_support_adds_local_samples(lower: float, upper: float, retained: list[float]) -> None:
    phases = np.linspace(lower, upper, 100)
    support = extend_phase_support(np.linspace(0, 1, 5), phases)
    expected = np.unique(np.concatenate((retained, np.linspace(lower, upper, 4))))
    assert_array_equal(support, expected)
    assert np.all(np.diff(support) > 0)
    assert support[0] <= lower
    assert support[-1] >= upper

    def synthesis(
        xs: NDArray[np.float64],
        _discretization: float,
        _observer: Observer,
        **_parameters: Any,
    ) -> dict[str, NDArray[np.float64]]:
        assert_array_equal(xs, support)
        return {"a": 2 + xs + xs**3}

    with settings.override(PRESERVE_INPUT_PHASES=True):
        result = predict_light_curves(
            x_data_reduced=phases,
            y_data={"a": np.ones(phases.size)},
            x_data_reducer={"a": np.arange(phases.size)},
            discretization=10,
            observer=Observer(passband="bolometric"),
            interp_treshold=20,
            model_parameters={},
            fit_xs=support,
            synthesis=synthesis,
        )["a"]
    direct = 2 + phases + phases**3
    assert np.all(np.isfinite(result))
    assert_allclose(result, direct / direct.mean(), rtol=1e-12)


def test_sufficient_adaptive_support_is_unchanged() -> None:
    reference = np.linspace(0, 1, 9)
    assert_array_equal(extend_phase_support(reference, np.array([0.2, 0.6])), reference[1:6])


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (0.4, 0.4),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.4, np.nextafter(0.4, 1.0)),
    ],
)
def test_degenerate_adaptive_support_reuses_neighbours(lower: float, upper: float) -> None:
    reference = np.linspace(0, 1, 5)
    phases = np.unique([lower, upper])
    support = extend_phase_support(reference, phases)
    assert support.size >= 4
    assert np.all(np.diff(support) > 0)
    assert support[0] <= lower
    assert support[-1] >= upper
    expanded = np.concatenate([reference + k for k in range(int(np.floor(lower)), int(np.ceil(upper)) + 1)])
    assert np.all(np.isin(support, np.concatenate((expanded, phases))))

    def synthesis(
        xs: NDArray[np.float64],
        _discretization: float,
        _observer: Observer,
        **_parameters: Any,
    ) -> dict[str, NDArray[np.float64]]:
        return {"a": 2 + xs}

    with settings.override(PRESERVE_INPUT_PHASES=True):
        result = predict_light_curves(
            x_data_reduced=phases,
            y_data={"a": np.ones(phases.size)},
            x_data_reducer={"a": np.arange(phases.size)},
            discretization=10,
            observer=Observer(passband="bolometric"),
            interp_treshold=20,
            model_parameters={},
            fit_xs=support,
            synthesis=synthesis,
        )["a"]
    assert np.all(np.isfinite(result))
    direct = 2 + phases
    assert_allclose(result, direct / direct.mean(), rtol=1e-12)
