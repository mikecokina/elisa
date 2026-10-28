"""The historical 90-degree synthesis default and explicit angle conventions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from elisa import const, settings
from elisa.analytics.models import lc, rv
from elisa.analytics.models.serializers import serialize_system_kwargs
from elisa.binary_system.orbit.orbit import Orbit
from elisa.observer.observer import Observer


ANGLES = [
    pytest.param(None, np.pi / 2, id="omitted"),
    pytest.param(0.0, 0.0, id="numeric-zero"),
    pytest.param(25.0, np.radians(25.0), id="numeric-degrees"),
    pytest.param(90.0, np.pi / 2, id="numeric-right-angle"),
    pytest.param(25.0 * u.deg, np.radians(25.0), id="quantity-degrees"),
    pytest.param(0.7 * u.rad, 0.7, id="quantity-radians"),
]
PHASES = np.array([0.0, 0.125, 0.25, 0.5, 0.75])


def test_serializer_default_has_explicit_angular_units() -> None:
    angle = serialize_system_kwargs(**{"system@period": 3.0})["argument_of_periastron"]
    assert isinstance(angle, u.Quantity)
    assert_allclose(angle.to_value(u.deg), 90.0)


@pytest.mark.parametrize("eccentricity", [0.0, 0.3], ids=["circular", "eccentric"])
@pytest.mark.parametrize(("supplied", "expected"), ANGLES)
def test_lc_preparation_periastron(eccentricity: float, supplied: Any, expected: float) -> None:
    parameters = {
        "system@period": 3.0,
        "system@eccentricity": eccentricity,
        "system@inclination": 85.0,
        "primary@mass": 1.0,
        "secondary@mass": 0.7,
    }
    for component in ("primary", "secondary"):
        parameters.update({
            f"{component}@surface_potential": 100.0,
            f"{component}@t_eff": 6000.0,
            f"{component}@gravity_darkening": 0.32,
            f"{component}@albedo": 0.5,
        })
    if supplied is not None:
        parameters["system@argument_of_periastron"] = supplied
        serialized = serialize_system_kwargs(**parameters)
        assert serialized["argument_of_periastron"] is supplied

    binary = lc.prepare_binary(**parameters)
    assert_allclose(binary.argument_of_periastron, expected)
    assert_allclose(binary.orbit.argument_of_periastron, expected)
    reference = Orbit(
        period=3.0, inclination=np.radians(85.0), eccentricity=eccentricity,
        argument_of_periastron=expected,
    )
    assert_allclose(binary.orbit.orbital_motion(PHASES), reference.orbital_motion(PHASES), atol=1e-12)


@pytest.mark.parametrize("kind", ["standard", "community"])
@pytest.mark.parametrize("eccentricity", [0.0, 0.3], ids=["circular", "eccentric"])
@pytest.mark.parametrize(("supplied", "expected"), ANGLES)
def test_rv_synthesis_periastron(kind: str, eccentricity: float, supplied: Any, expected: float) -> None:
    parameters = {"system@period": 3.0, "system@eccentricity": eccentricity}
    if supplied is not None:
        parameters["system@argument_of_periastron"] = supplied
    if kind == "standard":
        parameters.update({"primary@mass": 1.0, "secondary@mass": 0.7, "system@inclination": 85.0})
        inclination = np.radians(85.0)
        total_mass = (1.7 * u.solMass).to_value(u.kg)
        semi_major_axis = (const.G * total_mass * (3 * 86400)**2 / (4 * np.pi**2))**(1 / 3)
        projected_axis = semi_major_axis * np.sin(inclination)
    else:
        parameters.update({"system@mass_ratio": 0.7, "system@asini": 10.0})
        inclination = np.pi / 2
        projected_axis = (10.0 * u.solRad).to_value(u.m)

    observer = Observer(passband="bolometric", system=None)
    with settings.override(RV_METHOD="kinematic"):
        curves = rv.central_rv_synthetic(PHASES, observer, **parameters)
    assert_allclose(observer._system.argument_of_periastron, expected)
    assert_allclose(observer._system.orbit.argument_of_periastron, expected)

    reference = Orbit(
        period=3.0, inclination=inclination, eccentricity=eccentricity,
        argument_of_periastron=expected,
    )
    true_anomalies = reference.orbital_motion(PHASES)[:, 2]
    scale = 2 * np.pi * projected_axis / (3 * 86400 * np.sqrt(1 - eccentricity**2))
    velocity = scale * (np.cos(true_anomalies + expected) + eccentricity * np.cos(expected))
    assert_allclose(curves["primary"], 0.7 / 1.7 * velocity, rtol=2e-14, atol=1e-8)
    assert_allclose(curves["secondary"], -1 / 1.7 * velocity, rtol=2e-14, atol=1e-8)
