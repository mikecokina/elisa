"""Branch, input and failure contracts for the bracketed Roche solver."""

from decimal import Decimal, localcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from elisa.base.error import ElisaMaxIterationError, ElisaNumericError
from elisa.binary_system import _radius_solver as solver
from elisa.binary_system import radius
from scripts.diagnostics.roche_reference import Ray, bisect, reference


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize("q", [0.001, 0.7, 1000.0])
@pytest.mark.parametrize("direction", ["forward", "backward", "side"])
@pytest.mark.parametrize("ulps", [-2, -1, 0, 1, 2, 16])
def test_critical_ulp_neighborhood(component: str, q: float, direction: str, ulps: int) -> None:
    critical = reference(q, 1.0, 1.0, component, direction)
    omega = float(critical["critical"])
    for _ in range(abs(ulps)):
        omega = float(np.nextafter(omega, np.inf if ulps > 0 else -np.inf))
    expected = reference(q, 1.0, 1.0, component, direction, omega)["root"]
    if expected is None:
        with localcontext() as ctx:
            ctx.prec = 65
            ray = Ray(q, 1.0, 1.0, component, direction)
            uncertainty = (2 * (critical["critical"] - Decimal(omega)) / ray.curvature(critical["stationary"])).sqrt()
        if uncertainty > Decimal("1e-7") * critical["stationary"]:
            with pytest.raises(ElisaNumericError):
                getattr(radius, f"calculate_{direction}_radius")(1.0, q, 1.0, omega, component)
            return
    expected = critical["stationary"] if expected is None else expected
    obtained = getattr(radius, f"calculate_{direction}_radius")(1.0, q, 1.0, omega, component)
    assert obtained == pytest.approx(float(expected), rel=1e-7, abs=1e-10)
    assert obtained <= float(critical["stationary"]) + 2e-15


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize(("phi", "theta"), [(0.2, 1.3), (2.0, 0.4), (1.2, 2.1), (0.001, np.pi / 2)])
def test_arbitrary_angle_matches_independent_decimal_intersection(component: str, phi: float, theta: float) -> None:
    q, spin, distance, omega = 0.7, 1.3, 0.8, 8.0
    with localcontext() as ctx:
        ctx.prec = 65
        a, b = (Decimal(1), Decimal(q)) if component == "primary" else (Decimal(q), Decimal(1))
        c = Decimal(float(np.cos(phi) * np.sin(theta)))
        d = Decimal(distance)
        rotation = Decimal(spin) ** 2 * (1 + Decimal(q)) * Decimal(float(np.sin(theta))) ** 2 / 2
        offset = Decimal(0) if component == "primary" else (1 - Decimal(q)) / 2

        def residual(r: Decimal) -> Decimal:
            return (
                a / r
                + b / (d * d + r * r - 2 * d * c * r).sqrt()
                - b * c * r / d**2
                + rotation * r * r
                + offset
                - Decimal(omega)
            )

        expected = bisect(residual, Decimal("1e-20"), Decimal(".3"))
    obtained = radius.calculate_radius(spin, q, omega, component, distance, phi, theta)
    assert obtained == pytest.approx(float(expected), rel=1e-12)


def test_forward_batch_is_order_independent() -> None:
    distances = np.array([0.4, 1.6, 0.7, 1.0])
    potentials = np.array([8.0, 5.0, 6.0, 5.0])
    expected = [
        float(reference(0.1, 0.6, float(d), "secondary", "forward", float(o))["root"])
        for d, o in zip(distances, potentials, strict=True)
    ]
    obtained = radius.calculate_forward_radii(distances, potentials, 0.1, 0.6, "secondary")
    reverse = radius.calculate_forward_radii(distances[::-1], potentials[::-1], 0.1, 0.6, "secondary")
    assert obtained == pytest.approx(expected, rel=1e-12)
    assert reverse[::-1] == obtained


def test_bracket_nonconvergence_is_typed() -> None:
    with (
        patch.object(solver, "brentq", return_value=(0.5, SimpleNamespace(converged=False))),
        pytest.raises(ElisaMaxIterationError),
    ):
        radius.calculate_forward_radius(1.0, 0.7, 1.0, 5.0, "primary")


def test_unexpected_solver_failure_is_preserved() -> None:
    cause = TypeError("broken callback")
    with patch.object(solver, "brentq", side_effect=cause), pytest.raises(TypeError) as caught:
        radius.calculate_forward_radius(1.0, 0.7, 1.0, 5.0, "primary")
    assert caught.value is cause


@pytest.mark.parametrize("omega", [np.nan, np.inf, -np.inf, -1.0])
def test_invalid_or_nonexistent_polar_surface(omega: float) -> None:
    with pytest.raises(ElisaNumericError):
        radius.calculate_polar_radius(1.0, 0.7, 1.0, omega, "primary")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.5, 31.0])
def test_bracket_rejects_invalid_returned_radius(value: float) -> None:
    with (
        patch.object(solver, "brentq", return_value=(value, SimpleNamespace(converged=True))),
        pytest.raises(ElisaNumericError),
    ):
        radius.calculate_forward_radius(1.0, 0.7, 1.0, 5.0, "primary")


@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_slow_rotation_backward_radius_can_exceed_separation(component: str) -> None:
    q, spin, distance = (0.7 if component == "primary" else 1 / 0.7), 0.1, 1.0
    critical = reference(q, spin, distance, component, "backward")
    omega = float(critical["critical"]) + 1e-3
    expected = reference(q, spin, distance, component, "backward", omega)["root"]
    obtained = radius.calculate_backward_radius(spin, q, distance, omega, component)
    assert obtained > distance
    assert obtained == pytest.approx(float(expected), rel=1e-11)


def test_nearly_axial_ray_does_not_become_a_companion_singularity() -> None:
    phi, theta = 1e-9, np.pi / 2
    # cos(phi) rounds to one; the nonzero transverse separation must survive.
    ray = solver.RocheRay(1.0, 0.7, 1.0, 1.0, 0.85, 0.0, np.sin(phi) ** 2)
    assert ray.separation(1.0) == pytest.approx(1e-9, rel=1e-15)
    expected = reference(0.7, 1.0, 1.0, "primary", "forward", 4.0)["root"]
    obtained = radius.calculate_radius(1.0, 0.7, 4.0, "primary", 1.0, phi, theta)
    assert obtained == pytest.approx(float(expected), rel=1e-12)
