"""Observation times retain the cycle clock used by surface variability."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests.tests_observer.test_bolometric_preparation import _make_observer


@pytest.mark.parametrize("case", ["pulsating", "async_spot", "circular", "eccentric"])
@pytest.mark.parametrize("observable", ["lc", "rv"])
def test_time_phase_observables(case: str, observable: str) -> None:
    observer = _make_observer(case, ["bolometric"], "linear", custom_coefficients=True)
    if case == "pulsating":
        observer._system.primary.pulsations[0].angular_frequency *= 1.013
    for star in (observer._system.primary, observer._system.secondary):
        star.limb_darkening_coefficients["rv_band"] = [0.5]
    phases = np.array([2.2, -0.8, 0.2, 1.2])
    calculate = getattr(observer, observable)
    kwargs = {} if observable == "lc" else {"method": "radiometric"}
    _, expected = calculate(phases=phases, **kwargs)
    _, actual = calculate(times=phases * observer._system.period, **kwargs)
    for component in expected:
        assert_allclose(actual[component], expected[component], rtol=1e-8, atol=0)
    if case in ("pulsating", "async_spot"):
        assert any(np.ptp(values) > 1e-6 * np.max(np.abs(values)) for values in expected.values())
    else:
        reduced, inverse = observer.phase_interval_reduce(phases)
        assert len(reduced) == 1
        assert_allclose(reduced[inverse], 0.2)


@pytest.mark.parametrize("shift", [-0.3, 0.0, 0.3])
def test_clock_boundaries_and_ranges(shift: float) -> None:
    observer = _make_observer("pulsating", ["bolometric"], "linear", custom_coefficients=True)
    observer._system.t0 = 2450000.0
    observer._system.phase_shift = shift
    phases = np.array([10000.0, -2.0, 0.0, 1.2, -0.8])
    times = observer._system.t0 + observer._system.period * phases
    assert_allclose(observer.manage_time_series(times=times), phases - shift, atol=1e-9)
    assert_allclose(observer.manage_time_series(phases=phases), phases - shift)
    assert_allclose(
        observer.manage_time_series(from_time=2450000.0, to_time=2450007.5, time_step=2.5),
        observer.manage_time_series(from_phase=0.0, to_phase=3.0, phase_step=1.0),
    )


def test_single_cycle_clock() -> None:
    from elisa import Observer, SingleSystem
    from tests.tests_observer.test_observer import SingleSystemMock

    system = SingleSystemMock(p=True)
    system.period = 2.0
    observer = Observer(system=system)
    observer.system_cls = SingleSystem
    phases = observer.manage_time_series(times=[0.4, 2.4, 4.4])
    reduced, inverse = observer.phase_interval_reduce(phases)
    assert_allclose(reduced[inverse], [0.2, 1.2, 2.2])


@pytest.mark.parametrize("case", ["circular", "eccentric"])
def test_unwrapped_orbital_geometry(case: str) -> None:
    observer = _make_observer(case, ["bolometric"], "linear", custom_coefficients=True)
    phases = np.array([-100.8, -2.8, 0.2, 1.2, 10000.2])
    positions = observer._system.orbit.orbital_motion(phases)
    assert_allclose(positions[:, :3], np.tile(positions[2, :3], (len(phases), 1)), atol=1e-9)
    assert_allclose(positions[:, 3], phases)
