"""Regression coverage for single-system orbit period normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from elisa import SingleSystem, Star
from elisa.base.error import PhysicalStateError
from elisa.single_system.orbit.orbit import Orbit, angular_velocity

if TYPE_CHECKING:
    from elisa.types import AstropyQuantity as Quantity
    from elisa.types import Float, Int


@pytest.mark.parametrize("period", [10, 10.0, np.float64(10.0), 240 * u.hour])
def test_rotation_period_is_normalized_to_days(period: Quantity | Float | Int) -> None:
    orbit = Orbit(rotation_period=period, inclination=1.0)

    assert isinstance(orbit.rotation_period, float)
    assert orbit.rotation_period == 10.0
    assert orbit.period == orbit.rotation_period
    assert orbit.kwargs["rotation_period"] == 10.0
    assert orbit.inclination == 1.0
    assert_allclose(angular_velocity(orbit.rotation_period), 2 * np.pi / (10 * 86400))


@pytest.mark.parametrize("period", [0.0, -1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("quantity", [False, True])
def test_invalid_rotation_period_is_rejected(period: float, quantity: bool) -> None:
    supplied = period * u.hour if quantity else period
    error_type = PhysicalStateError if np.isfinite(period) else FloatingPointError

    with pytest.raises(error_type, match="period"):
        Orbit(rotation_period=supplied, inclination=1.0)


@pytest.mark.parametrize("period", [10.0, 240 * u.hour])
def test_single_system_period_and_orbital_calculations(period: Quantity | Float) -> None:
    star = Star(mass=1.0, t_eff=5772.0, polar_log_g=4.43775, gravity_darkening=0.32)
    system = SingleSystem(star=star, rotation_period=period, inclination=90.0)

    assert system.rotation_period == system.period == 10.0
    assert system.orbit.rotation_period == system.orbit.period == 10.0
    assert_allclose(system.orbit.inclination, np.pi / 2)
    assert_allclose(system.angular_velocity, 2 * np.pi / (10 * 86400))

    phases = np.array([-0.25, 0.0, 0.25, 1.0])
    expected = np.column_stack((2 * np.pi * phases, np.full(phases.shape, np.nan), phases))
    assert_allclose(system.orbit.orbital_motion(phases), expected)
    assert_allclose(system.orbit.orbital_motion(0.25), [[np.pi / 2, np.nan, 0.25]])
    positions = system.calculate_lines_of_sight(phases, return_nparray=True)
    assert_allclose(positions[:, 2:], expected)
    assert_allclose(
        system.calculate_lines_of_sight(2 * np.pi * phases, return_nparray=True, calculate_from="azimuth"),
        positions,
    )
