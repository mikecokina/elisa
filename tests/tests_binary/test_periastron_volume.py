"""Periastron reference contracts with independent solid-angle volume integration."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from elisa import BinarySystem, Star
from elisa.binary_system import radius

if TYPE_CHECKING:
    import elisa.binary_system.system


def binary(eccentricity: float = 0.3, spin: float = 1.0) -> elisa.binary_system.system.BinarySystem:
    return BinarySystem(
        primary=Star(mass=2.0, t_eff=8000, surface_potential=6.0, synchronicity=spin),
        secondary=Star(mass=1.4, t_eff=7000, surface_potential=6.0, synchronicity=spin),
        eccentricity=eccentricity,
        period=3.0,
        inclination=85,
        argument_of_periastron=90,
    )


def volume(
    system: elisa.binary_system.system.BinarySystem,
    component: str,
    potential: float | np.float64,
    distance: float | np.float64,
) -> np.float64:
    # Integrate r^3/3 over solid angle, independently of the elliptical slices
    # and equivalent-radius rescaling used by correct_potentials.
    mu, weights = np.polynomial.legendre.leggauss(24)
    phis = np.arange(48) * (2 * np.pi / 48)
    star = getattr(system, component)
    return (
        sum(
            weight
            * sum(
                radius.calculate_radius(
                    star.synchronicity,
                    system.mass_ratio,
                    potential,
                    component,
                    distance,
                    phi,
                    np.arccos(z),
                )
                ** 3
                for phi in phis
            )
            for z, weight in zip(mu, weights, strict=True)
        )
        * (2 * np.pi / 48)
        / 3
    )


@pytest.mark.parametrize("spin", [1.0, 1.3])
@pytest.mark.parametrize("eccentricity", [0.0, 0.3])
def test_identity_and_equivalent_radius(eccentricity: float, spin: float) -> None:
    system = binary(eccentricity, spin)
    peri = system.orbit.periastron_distance
    with patch.object(system, "generate_equator_and_meridian_points", side_effect=AssertionError):
        corrected = system.correct_potentials(distances=np.array([peri, peri]))
    for component in ("primary", "secondary"):
        star = getattr(system, component)
        np.testing.assert_array_equal(corrected[component], [star.surface_potential] * 2)
        expected_volume = volume(system, component, star.surface_potential, peri)
        assert system.calculate_equivalent_radius(component)[component] == pytest.approx(
            (3 * expected_volume / (4 * np.pi)) ** (1 / 3),
            rel=2e-3,
        )


def test_mixed_distances_conserve_volume_and_order() -> None:
    system = binary()
    distances = np.array([1.3, 0.7, 1.0, 0.7, 1.3])
    corrected = system.correct_potentials(distances=distances)
    for component in ("primary", "secondary"):
        star = getattr(system, component)
        values = corrected[component]
        assert values.shape == distances.shape
        assert values.dtype == np.float64
        assert values[1] == values[3] == star.surface_potential
        assert values[0] == values[4]
        reference = volume(system, component, star.surface_potential, 0.7)
        for distance, potential in zip(distances[:3], values[:3], strict=True):
            assert volume(system, component, potential, distance) == pytest.approx(reference, rel=3e-3)


def test_near_critical_periastron_radii_use_input_potential() -> None:
    system = binary()
    for star in (system.primary, system.secondary):
        star.surface_potential = star.critical_surface_potential + 1e-4
    assert system.compute_morphology() == "detached"
    with patch.object(radius, "solve_radius", wraps=radius.solve_radius) as solver:
        radii = system.calculate_components_radii(system.orbit.periastron_distance)
    for call in solver.call_args_list:
        component = "secondary" if call.kwargs["secondary"] else "primary"
        assert call.args[2] == system.orbit.periastron_distance
        assert call.args[3] == getattr(system, component).surface_potential
    for component in ("primary", "secondary"):
        assert radii[component]["forward_radius"] > radii[component]["polar_radius"] > 0


@pytest.mark.parametrize("distance", [0.7, 1.3])
def test_horizon_mesh_uses_instantaneous_surface(distance: float) -> None:
    from elisa import const
    from elisa.analytics.tools.horizon import _horizon_base_component
    from elisa.binary_system import model

    system = binary()
    system.inclination = const.HALF_PI
    position = const.Position(0, distance, const.HALF_PI, 0, 0)
    container = _horizon_base_component(system, position, analytic=False)
    expected = system.correct_potentials(distances=np.array([distance]))["primary"][0]
    assert container.primary.surface_potential == expected
    # Mesh points must lie on the instantaneous equipotential.
    from elisa import utils

    spherical = utils.cartesian_to_spherical(container.primary.points)
    r, phi, theta = spherical.T
    coefficients = model.pre_calculate_for_potential_value_primary(
        system.primary.synchronicity,
        system.mass_ratio,
        distance,
        phi,
        theta,
        return_as_tuple=True,
    )
    residual = model.potential_primary_fn(r, (system.mass_ratio, *coefficients), expected)
    np.testing.assert_allclose(residual, 0, atol=1e-7)
