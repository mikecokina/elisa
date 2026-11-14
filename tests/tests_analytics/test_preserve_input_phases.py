"""Coordinate policy and multi-cycle support regression checks."""
from typing import Any

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

from elisa import settings
from elisa.analytics.binary_fit.lc_prediction import _synthesis_support, extend_phase_support, predict_light_curves
from elisa.analytics.tools.utils import time_layer_resolver


def test_resolver_preserves_only_phase_domain() -> None:
    phases = np.array([-1.5, 2.5, 0.25])
    assert_array_equal(time_layer_resolver(phases, **{"system@period": 2})[0], phases % 1)
    assert_array_equal(time_layer_resolver(phases, preserve_phases=True, **{"system@period": 2})[0], phases)
    for period in (2, 3):
        for epoch in (1, 1.2):
            kwargs = {"system@period": period, "system@primary_minimum_time": epoch}
            assert_array_equal(time_layer_resolver(phases, **kwargs)[0],
                               time_layer_resolver(phases, preserve_phases=True, **kwargs)[0])


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

    def synthesis(support: NDArray, *_args: Any, **_kwargs: Any) -> dict:
        assert_array_equal(support, xs)
        return dict.fromkeys(reducers, 2 + support / 20)

    with settings.override(PRESERVE_INPUT_PHASES=True):
        result = predict_light_curves(xs, observed, reducers, 10, None, 100, {}, synthesis=synthesis)
    for band, indices in reducers.items():
        expected = 2 + xs[indices] / 20
        assert_allclose(result[band], expected / expected.mean())
