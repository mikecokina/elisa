"""Critical Roche limits and summary selection against an independent oracle."""

from __future__ import annotations

import warnings
from copy import deepcopy
from unittest.mock import patch

import pytest

from elisa import BinarySystem, Star
from elisa.analytics.binary_fit.shared import check_for_boundary_surface_potentials
from scripts.diagnostics.roche_reference import reference


@pytest.mark.parametrize("q", [0.1, 0.7, 1.0, 3.0])
def test_classical_l1_matches_component_critical_normalization(q: float) -> None:
    classical = BinarySystem.libration_potentials_static(1.0, q)[1]
    for component in ("primary", "secondary"):
        expected = float(reference(q, 1.0, 1.0, component, "forward")["critical"])
        actual = BinarySystem.critical_potential_static(component, 1.0, q, 1.0)
        assert actual == pytest.approx(expected, rel=2e-13)
        assert actual == pytest.approx(classical, rel=2e-12)


@pytest.mark.parametrize(("eccentricity", "spins"), [(0.3, (1.0, 1.0)), (0.0, (0.8, 1.4)), (0.3, (0.8, 1.4))])
def test_summary_generalized_limits_use_each_spin_without_classical_solver(
    eccentricity: float,
    spins: tuple[float, ...],
) -> None:
    q = 0.7
    distance = 1 - eccentricity
    expected = [
        float(reference(q, spin, distance, component, "forward")["critical"])
        for component, spin in zip(("primary", "secondary"), spins, strict=True)
    ]
    results = {"system@eccentricity": {"value": eccentricity}, "system@mass_ratio": {"value": q}}
    for component, spin, critical in zip(("primary", "secondary"), spins, expected, strict=True):
        results[f"{component}@synchronicity"] = {"value": spin}
        results[f"{component}@surface_potential"] = {"value": critical - 0.0001, "fixed": False}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with patch.object(BinarySystem, "libration_potentials_static", side_effect=AssertionError("classical called")):
            corrected = check_for_boundary_surface_potentials(deepcopy(results))
    for component, critical in zip(("primary", "secondary"), expected, strict=True):
        assert corrected[f"{component}@surface_potential"]["value"] == pytest.approx(critical + 1e-8, abs=2e-13)


def test_equal_generalized_potentials_do_not_skip_component_checks() -> None:
    results = {"system@eccentricity": {"value": 0.3}, "system@mass_ratio": {"value": 0.7}}
    for component, spin in zip(("primary", "secondary"), (0.8, 1.4), strict=True):
        results[f"{component}@surface_potential"] = {"value": 5.0, "fixed": False}
        results[f"{component}@synchronicity"] = {"value": spin}
    with (
        patch.object(BinarySystem, "critical_potential_static", return_value=5.0001) as critical,
        patch.object(BinarySystem, "libration_potentials_static", side_effect=AssertionError),
    ):
        check_for_boundary_surface_potentials(results)
    assert critical.call_count == 2


def test_circular_summary_retains_classical_outer_boundary() -> None:
    results = {"system@eccentricity": {"value": 0.0}, "system@mass_ratio": {"value": 1.0}}
    for component, potential in [("primary", 3.0), ("secondary", 3.0001)]:
        results[f"{component}@surface_potential"] = {"value": potential, "fixed": False}
    with patch.object(
        BinarySystem,
        "libration_potentials_static",
        wraps=BinarySystem.libration_potentials_static,
    ) as classical:
        check_for_boundary_surface_potentials(results, "over-contact")
    assert classical.call_count == 2
    assert results["primary@surface_potential"]["value"] == results["secondary@surface_potential"]["value"]


@pytest.mark.parametrize("eccentricity", [0.0, 0.3])
def test_initialization_sets_periastron_limits_with_individual_spins(eccentricity: float) -> None:
    system = BinarySystem(
        primary=Star(mass=2.0, t_eff=8000.0, surface_potential=6.0, synchronicity=0.8),
        secondary=Star(mass=1.4, t_eff=6500.0, surface_potential=6.5, synchronicity=1.4),
        eccentricity=eccentricity,
        period=3.0,
        inclination=87.0,
        argument_of_periastron=90.0,
    )
    assert system.morphology == "detached"
    for component in ("primary", "secondary"):
        star = getattr(system, component)
        expected = float(reference(0.7, star.synchronicity, 1 - eccentricity, component, "forward")["critical"])
        assert star.critical_surface_potential == pytest.approx(expected, rel=2e-13)
