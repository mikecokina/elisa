"""First-branch angular regressions against independent Decimal geometry."""

from math import pi

import numpy as np
import pytest

from elisa.base.error import ElisaNumericError
from elisa.binary_system.radius import calculate_radius
from scripts.diagnostics.roche_reference import angular_reference

REGULAR_DELTA = 0.1
ABSENT_DELTA = -1e-5
REFERENCE_PRECISION = 1e-60

GEOMETRIES = [(0.001, 1.0, 0.4), (1000.0, 0.3, 1.6), (0.7, 3.0, 0.7)]
ANGLES = [
    (0.01, pi / 2),
    (1e-9, pi / 2),
    (2 * pi, pi / 2),
    (np.nextafter(2 * pi, 0.0), pi / 2),
    (-0.01, pi / 2),
    (0.2, 1.3),
    (2.0, 0.4),
]


@pytest.mark.parametrize("delta", [1.0, 0.1, 0.001])
def test_review_reproduction(delta: float) -> None:
    critical = angular_reference(0.001, 1.0, 0.4, "primary", 0.01, pi / 2)
    omega = float(critical["critical"]) + delta
    expected = angular_reference(0.001, 1.0, 0.4, "primary", 0.01, pi / 2, omega)
    obtained = calculate_radius(1.0, 0.001, omega, "primary", 0.4, 0.01, pi / 2)
    assert obtained == pytest.approx(float(expected["root"]), rel=1e-12)
    assert obtained < float(critical["stationary"])


@pytest.mark.parametrize("turns", [-2, -1, 1, 2])
def test_full_turn_retains_forward_intersection(turns: int) -> None:
    expected = angular_reference(0.001, 1.0, 0.4, "primary", turns * 2 * pi, pi / 2, 4.0)
    axial = calculate_radius(1.0, 0.001, 4.0, "primary", 0.4, 0.0, pi / 2)
    obtained = calculate_radius(1.0, 0.001, 4.0, "primary", 0.4, turns * 2 * pi, pi / 2)
    assert obtained == pytest.approx(float(expected["root"]), rel=1e-12)
    assert obtained == pytest.approx(axial, rel=1e-14)


@pytest.fixture(params=[(g, a, c) for g in GEOMETRIES for a in ANGLES for c in ("primary", "secondary")])
def angular_case(request: pytest.FixtureRequest) -> tuple:
    geometry, angles, component = request.param
    critical = angular_reference(*geometry, component, *angles)
    return geometry, angles, component, critical


@pytest.mark.parametrize("delta", [0.1, 1e-10, "ulp16", -1e-5])
def test_angular_inner_branch(angular_case: tuple, delta: float | str) -> None:
    (q, spin, distance), (phi, theta), component, critical = angular_case
    omega = float(critical["critical"])
    if delta == "ulp16":
        for _ in range(16):
            omega = float(np.nextafter(omega, np.inf))
    else:
        omega += delta
    expected = angular_reference(q, spin, distance, component, phi, theta, omega)
    if delta == ABSENT_DELTA:
        assert expected["root"] is None
        with pytest.raises(ElisaNumericError, match=r"inner radial branch|critical limit"):
            calculate_radius(spin, q, omega, component, distance, phi, theta)
        return
    assert expected["root"] is not None
    obtained = calculate_radius(spin, q, omega, component, distance, phi, theta)
    tolerance = 1e-11 if delta == REGULAR_DELTA else 1e-7
    assert obtained == pytest.approx(float(expected["root"]), rel=tolerance, abs=5e-13 * distance)
    assert obtained <= float(critical["stationary"]) + 2e-14 * distance


def test_angular_reference_precision() -> None:
    low = angular_reference(0.001, 1.0, 0.4, "primary", 0.01, pi / 2, 3.0)
    high = angular_reference(0.001, 1.0, 0.4, "primary", 0.01, pi / 2, 3.0, precision=95)
    for key in ("stationary", "critical", "root"):
        assert abs(low[key] - high[key]) < REFERENCE_PRECISION
