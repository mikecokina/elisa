"""Morphology rejection identifies the failed condition and its numerical inputs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from elisa import BinarySystem, Star
from elisa.base.error import MorphologyError

if TYPE_CHECKING:
    import elisa.binary_system.system


def test_eccentric_model_reports_surface_and_periastron_critical_potentials() -> None:
    primary = Star(mass=2.15, t_eff=10000.0, surface_potential=3.6, synchronicity=1.0)
    secondary = Star(mass=1.2, t_eff=7000.0, surface_potential=4.0, synchronicity=1.0)
    with pytest.raises(MorphologyError) as caught:
        BinarySystem(
            primary=primary,
            secondary=secondary,
            eccentricity=0.3,
            period=2.5,
            inclination=85.0,
            argument_of_periastron=58.0,
        )

    message = str(caught.value)
    assert "eccentric/asynchronous morphology conditions failed" in message
    assert "Critical potentials are evaluated at periastron" in message
    assert "abs(surface_potential - critical_surface_potential) < 1e-08" in message
    for name, star in [("primary", primary), ("secondary", secondary)]:
        assert f"  {name}:\n    surface_potential: {star.surface_potential:.12g}" in message
        assert f"    critical_surface_potential: {star.critical_surface_potential:.12g}" in message
        assert f"    difference: {star.surface_potential - star.critical_surface_potential:.12g}" in message
    assert "    above_critical: False\n    at_critical: False" in message
    assert "  eccentricity: 0.3" in message


def contact_model(primary_potential: float, secondary_potential: float) -> elisa.binary_system.system.BinarySystem:
    model = BinarySystem.__new__(BinarySystem)
    model.eccentricity = 0.0
    model.primary = SimpleNamespace(synchronicity=1.0, surface_potential=primary_potential)
    model.secondary = SimpleNamespace(synchronicity=1.0, surface_potential=secondary_potential)
    # Fixed potentials isolate classification and diagnostic arithmetic.
    model.libration_potentials = lambda: [2.0, 4.0, 3.0]
    return model


def test_overcontact_mismatch_reports_actual_factors_and_tolerance() -> None:
    model = contact_model(3.5, 3.75)
    with pytest.raises(MorphologyError) as caught:
        model.compute_morphology()
    message = str(caught.value)
    assert "potentials of components are not the same" in message
    assert "primary: surface_potential=3.5, filling_factor=0.5" in message
    assert "secondary: surface_potential=3.75, filling_factor=0.25" in message
    assert "Filling-factor difference=0.25 exceeds tolerance=1e-08" in message
    assert "inner_potential=4, outer_potential=3" in message


def test_excessive_filling_reports_values_exceeding_one() -> None:
    model = contact_model(2.5, 2.5)
    with pytest.raises(MorphologyError) as caught:
        model.compute_morphology()
    message = str(caught.value)
    assert "filling_factor is greater than 1" in message
    assert "primary: surface_potential=2.5, filling_factor=1.5" in message
    assert "secondary: surface_potential=2.5, filling_factor=1.5" in message
    assert "inner_potential=4, outer_potential=3" in message
