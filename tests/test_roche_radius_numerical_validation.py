"""Scientific radius contracts against an independent high-precision oracle.

Critical float targets may straddle the exact minimum by a few ULP. Such cases
explicitly test the limiting radius, not existence of an exact float-target root.
The tolerances bound local radius/area error; they do not certify full LC accuracy.
Expected failures expose current policy/branch problems and are not xfailed.
"""

# ruff: noqa: S101 - assertions express scientific regression contracts
from contextlib import suppress
from decimal import Decimal, localcontext
from unittest.mock import patch

import numpy as np
import pytest

from elisa.base.error import ElisaError
from elisa.binary_system import radius
from elisa.binary_system.system import BinarySystem
from scripts.diagnostics.roche_reference import Ray, reference

ORDINARY_RTOL = 1e-10
CRITICAL_RTOL = 1e-7  # At most ~2e-7 relative error for a local circular area.
ABSOLUTE_PER_SEPARATION = 1e-10
GEOMETRIES = [(0.7, 1.0, 1.0), (0.1, 0.6, 0.7), (3.0, 1.8, 1.4)]
AXES = [("primary", "forward"), ("primary", "backward"), ("secondary", "backward")]


def solve(q: float, spin: float, distance: float, component: str, direction: str, omega: float) -> float:
    """Exercise the public directional radius API, including its acceptance policy."""
    return getattr(radius, f"calculate_{direction}_radius")(
        synchronicity=spin,
        mass_ratio=q,
        components_distance=distance,
        surface_potential=omega,
        component=component,
    )


@pytest.mark.parametrize(("q", "spin", "distance"), GEOMETRIES)
@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize("direction", ["polar", "side", "backward", "forward"])
def test_regular_radius_matches_first_intersection(
    q: float,
    spin: float,
    distance: float,
    component: str,
    direction: str,
) -> None:
    critical = max(reference(q, spin, distance, component, ray)["critical"] for ray in ("forward", "backward", "side"))
    omega = float(critical) + 0.1
    expected = reference(q, spin, distance, component, direction, omega)["root"]
    assert expected is not None
    obtained = solve(q, spin, distance, component, direction, omega)
    assert obtained == pytest.approx(float(expected), rel=ORDINARY_RTOL, abs=5e-13 * distance)


@pytest.mark.parametrize(("component", "direction"), AXES)
@pytest.mark.parametrize("delta", [1e-5, 1e-10, "ulp16"])
def test_near_critical_radius_keeps_inner_branch(component: str, direction: str, delta: float | str) -> None:
    critical = reference(0.7, 1.0, 1.0, component, direction)
    omega = float(critical["critical"])
    if delta == "ulp16":
        for _ in range(16):
            omega = float(np.nextafter(omega, np.inf))
    else:
        omega += delta
    expected = reference(0.7, 1.0, 1.0, component, direction, omega)["root"]
    assert expected is not None
    obtained = solve(0.7, 1.0, 1.0, component, direction, omega)
    assert obtained == pytest.approx(float(expected), rel=CRITICAL_RTOL, abs=ABSOLUTE_PER_SEPARATION)
    # The uncertainty band allows indistinguishable near-double roots only.
    assert obtained <= float(critical["stationary"]) + CRITICAL_RTOL * float(expected)


@pytest.mark.parametrize(
    ("q", "component", "direction"),
    [
        (0.6999999999999998, "primary", "forward"),
        (1.0, "primary", "forward"),
        (0.7, "primary", "backward"),
        (0.7, "secondary", "backward"),
        (1000.0, "primary", "backward"),
    ],
)
def test_rounded_critical_potential_recovers_limiting_radius(q: float, component: str, direction: str) -> None:
    critical = reference(q, 1.0, 1.0, component, direction)
    obtained = solve(q, 1.0, 1.0, component, direction, float(critical["critical"]))
    assert obtained == pytest.approx(float(critical["stationary"]), rel=CRITICAL_RTOL, abs=ABSOLUTE_PER_SEPARATION)


@pytest.mark.parametrize(("component", "direction"), AXES)
def test_below_directional_minimum_has_no_relevant_root(component: str, direction: str) -> None:
    critical = reference(0.7, 1.0, 1.0, component, direction)
    omega = float(critical["critical"]) - 1e-5
    assert reference(0.7, 1.0, 1.0, component, direction, omega)["root"] is None
    with pytest.raises(ElisaError):
        solve(0.7, 1.0, 1.0, component, direction, omega)


def test_forward_solver_must_not_accept_root_beyond_companion() -> None:
    # Both environments report success for a remote root beyond d=0.4. There
    # is no component-local forward surface intersection in the required (0,d).
    omega = 3.2034863800799114
    assert reference(0.01, 3.0, 0.4, "primary", "forward", omega)["root"] is None
    with pytest.raises(ElisaError):
        solve(0.01, 3.0, 0.4, "primary", "forward", omega)


def test_raw_critical_candidate_quality_is_independent_of_solver_status() -> None:
    q, omega = 0.6999999999999998, 3.243288166372893
    captured = []
    original = radius.fsolve

    def record(*args, **kwargs) -> tuple:
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    # Inspect the candidate even when today's status policy rejects it.
    with patch.object(radius, "fsolve", side_effect=record), suppress(ElisaError):
        solve(q, 1.0, 1.0, "primary", "forward", omega)
    candidate = float(captured[0][0][0])
    expected = reference(q, 1.0, 1.0, "primary", "forward")["stationary"]
    assert candidate == pytest.approx(float(expected), rel=CRITICAL_RTOL, abs=ABSOLUTE_PER_SEPARATION)
    with localcontext() as ctx:
        ctx.prec = 65
        residual = Ray(q, 1.0, 1.0, "primary", "forward").potential(Decimal(candidate)) - Decimal(omega)
    assert abs(residual) < Decimal("1e-13")


@pytest.mark.parametrize(("q", "spin", "distance"), GEOMETRIES)
@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_generalized_critical_potential_matches_stationary_reference(
    q: float,
    spin: float,
    distance: float,
    component: str,
) -> None:
    expected = reference(q, spin, distance, component, "forward")["critical"]
    obtained = BinarySystem.critical_potential_static(component, distance, q, spin)
    assert obtained == pytest.approx(float(expected), rel=2e-13, abs=2e-13)


@pytest.mark.parametrize("q", [0.001, 0.7, 1.0, 3.0, 1000.0])
def test_lagrange_points_use_distinct_geometric_intervals(q: float) -> None:
    l3 = -float(reference(q, 1.0, 1.0, "primary", "backward")["stationary"])
    l1 = float(reference(q, 1.0, 1.0, "primary", "forward")["stationary"])
    l2 = 1 + float(reference(q, 1.0, 1.0, "secondary", "backward")["stationary"])
    expected = [l3, l1, l2] if q < 1 else [l2, l1, l3]
    assert BinarySystem.lagrangian_points_static(1.0, q) == pytest.approx(expected, abs=2e-12, rel=0)


def test_decimal_oracle_certifies_symmetric_l1_and_precision() -> None:
    symmetric = reference(1.0, 1.0, 1.0, "primary", "forward", Decimal("3.75"))
    assert abs(symmetric["stationary"] - Decimal("0.5")) < Decimal("1e-60")
    assert abs(symmetric["root"] - Decimal("0.5")) < Decimal("1e-30")
    for component, direction in AXES:
        low = reference(0.7, 1.0, 1.0, component, direction, precision=65)
        high = reference(0.7, 1.0, 1.0, component, direction, precision=95)
        assert abs(low["stationary"] - high["stationary"]) < Decimal("1e-60")
        assert abs(low["critical"] - high["critical"]) < Decimal("1e-60")


def test_l1_at_library_rounded_critical_potential_recovers_limiting_radius() -> None:
    q = 0.6999999999999998
    omega = BinarySystem.libration_potentials_static(1.0, q)[1]
    expected = reference(q, 1.0, 1.0, "primary", "forward")["stationary"]
    obtained = solve(q, 1.0, 1.0, "primary", "forward", omega)
    assert obtained == pytest.approx(float(expected), rel=CRITICAL_RTOL, abs=ABSOLUTE_PER_SEPARATION)


def test_ulp_above_l3_respects_radius_accuracy_budget() -> None:
    critical = reference(1000.0, 1.0, 1.0, "primary", "backward")
    omega = float(np.nextafter(float(critical["critical"]), np.inf))
    expected = reference(1000.0, 1.0, 1.0, "primary", "backward", omega)["root"]
    assert expected is not None
    obtained = solve(1000.0, 1.0, 1.0, "primary", "backward", omega)
    assert obtained == pytest.approx(float(expected), rel=CRITICAL_RTOL, abs=ABSOLUTE_PER_SEPARATION)
