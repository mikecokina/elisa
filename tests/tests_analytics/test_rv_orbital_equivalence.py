"""Compare the orbit-only fitting model against physical BinarySystem kinematics."""

from __future__ import annotations

import numpy as np
import pytest

from elisa import settings
from elisa.analytics.models import lc, rv
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.observer.observer import Observer


@pytest.mark.parametrize("eccentricity", [0.0, 0.4])
@pytest.mark.parametrize("omega", [0.0, 37.0, 240.0])
@pytest.mark.parametrize("inclination", [35.0, 85.0])
@pytest.mark.parametrize("times", [False, True])
def test_mass_model_matches_binary(eccentricity: float, omega: float, inclination: float, times: bool) -> None:
    flat = {
        "system@eccentricity": eccentricity,
        "system@argument_of_periastron": omega,
        "system@inclination": inclination,
        "system@period": 3.7,
        "system@gamma": 12345.0,
        "primary@mass": 1.3,
        "secondary@mass": 0.6,
    }
    physical = dict(flat)
    for component in ("primary", "secondary"):
        physical.update(
            {
                f"{component}@t_eff": 6000.0,
                f"{component}@surface_potential": 100.0,
                f"{component}@albedo": 0.5,
                f"{component}@gravity_darkening": 0.32,
            },
        )
    binary = lc.prepare_binary(**physical)
    phases = np.array([-0.3, 0.0, 0.1, 0.25, 0.6, 1.3])
    with settings.override(RV_METHOD="kinematic"):
        _, reference = Observer(system=binary).observe.rv(phases=phases)
        observer = Observer(system=None)
        if times:
            flat["system@primary_minimum_time"] = 2450000.0
            inputs = phases * 3.7 + 2450000.0
        else:
            inputs = phases
        actual = rv.central_rv_synthetic(inputs, observer, **flat)
    assert isinstance(observer._system, RadialVelocitySystem)
    for component in reference:
        np.testing.assert_allclose(actual[component], reference[component], rtol=2e-14, atol=1e-8)
