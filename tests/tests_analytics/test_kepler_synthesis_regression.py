"""Real RV and LC predictions agree between configured Kepler solvers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa import const, settings
from elisa import units as u
from elisa.analytics.models import lc, rv
from elisa.binary_system.orbit.orbit import Orbit
from elisa.binary_system.system import BinarySystem
from elisa.conf.enums import MeanAnomalySolver
from elisa.observer.observer import Observer
from tests.kepler_helpers import scalar_orbital_motion
from tests.tests_analytics.fitting_helpers import V, lc_parameters


@pytest.fixture(autouse=True)
def synthesis_settings(science_settings: None) -> Iterator[None]:
    with settings.override(REFLECTION_EFFECT=False, RV_METHOD="kinematic"):
        yield


@pytest.mark.parametrize("synthesis_kind", ["community", "standard"])
@pytest.mark.parametrize(
    ("eccentricity", "periastron_degrees"),
    [
        pytest.param(0.0, 270.0, id="circular"),
        pytest.param(0.3, 25.0, id="moderately-eccentric"),
        pytest.param(0.95, 75.0, id="highly-eccentric"),
    ],
)
def test_real_rv_curves_match_scalar_orbit_and_velocity_formula(
    synthesis_kind: str,
    eccentricity: float,
    periastron_degrees: float,
) -> None:
    phases = np.array([0.5, -0.1, 0.0, 0.125, 0.5, 1.0, 1.1])
    original_phases = phases.copy()
    period_days = 3.0
    systemic_velocity = 12000.0
    mass_ratio = 0.7
    model_parameters = {
        "system@eccentricity": eccentricity,
        "system@argument_of_periastron": periastron_degrees,
        "system@period": period_days,
        "system@gamma": systemic_velocity,
    }

    if synthesis_kind == "community":
        inclination = np.pi / 2
        model_parameters.update({"system@asini": 10.0, "system@mass_ratio": mass_ratio})
        projected_axis = (10.0 * u.solRad).to_value(u.m)
    else:
        inclination = np.radians(85.0)
        model_parameters.update(
            {
                "primary@mass": 1.5,
                "secondary@mass": 1.05,
                "system@inclination": 85.0,
            }
        )
        total_mass = (2.55 * u.solMass).to_value(u.kg)
        semi_major_axis = (const.G * total_mass * (period_days * 86400.0) ** 2 / (4 * np.pi**2)) ** (1.0 / 3.0)
        projected_axis = semi_major_axis * np.sin(inclination)

    reference_orbit = Orbit(
        period=period_days,
        inclination=inclination,
        eccentricity=eccentricity,
        argument_of_periastron=np.radians(periastron_degrees),
    )
    true_anomalies = scalar_orbital_motion(reference_orbit, phases % 1.0)[:, 2]
    velocity_scale = 2 * np.pi * projected_axis / (period_days * 86400.0 * np.sqrt(1 - eccentricity**2))
    orbital_velocity = velocity_scale * (
        np.cos(true_anomalies + np.radians(periastron_degrees)) + eccentricity * np.cos(np.radians(periastron_degrees))
    )
    expected_velocities = {
        "primary": systemic_velocity + mass_ratio / (1 + mass_ratio) * orbital_velocity,
        "secondary": systemic_velocity - 1 / (1 + mass_ratio) * orbital_velocity,
    }
    observer = Observer(passband="bolometric", system=None)

    with settings.override(MEAN_ANOMALY_SOLVER=MeanAnomalySolver.SCALAR):
        scalar_velocities = rv.central_rv_synthetic(phases, observer, **model_parameters)
    with settings.override(MEAN_ANOMALY_SOLVER=MeanAnomalySolver.VECTORIZED):
        vectorized_velocities = rv.central_rv_synthetic(phases, observer, **model_parameters)
    single_phase_velocities = rv.central_rv_synthetic(phases[:1], observer, **model_parameters)

    assert set(vectorized_velocities) == set(expected_velocities)
    for component, expected_curve in expected_velocities.items():
        scalar_curve = scalar_velocities[component]
        vectorized_curve = vectorized_velocities[component]

        assert vectorized_curve.shape == phases.shape
        assert_allclose(vectorized_curve, scalar_curve, rtol=2e-14, atol=1e-8)
        assert_allclose(vectorized_curve, expected_curve, rtol=2e-14, atol=1e-8)
        assert vectorized_curve[0] == vectorized_curve[4]
        assert_allclose(single_phase_velocities[component], expected_curve[:1], rtol=2e-14, atol=1e-8)
    assert_array_equal(phases, original_phases)


@pytest.mark.parametrize("eccentricity", [0.0, 0.3], ids=["circular", "eccentric"])
def test_real_lc_and_binary_positions_match_scalar_orbit(eccentricity: float) -> None:
    _, model_parameters = lc_parameters()
    model_parameters.update(
        {
            "system@eccentricity": eccentricity,
            "system@argument_of_periastron": 25.0,
        }
    )
    phases = np.array([0.5, 0.0, 0.12, 0.04, 0.5, 0.8, 0.25])
    observer = Observer(passband=[V], system=None)
    observer.system_cls = BinarySystem
    binary = lc.prepare_binary(discretization=10.0, **model_parameters)

    with settings.override(MEAN_ANOMALY_SOLVER=MeanAnomalySolver.SCALAR):
        scalar_positions = binary.calculate_orbital_motion(phases, return_nparray=True)
        scalar_flux = lc.synthetic_binary(phases, 10.0, observer, **model_parameters)[V]
    with settings.override(MEAN_ANOMALY_SOLVER=MeanAnomalySolver.VECTORIZED):
        vectorized_positions = binary.calculate_orbital_motion(phases, return_nparray=True)
        vectorized_flux = lc.synthetic_binary(phases, 10.0, observer, **model_parameters)[V]

    reference_positions = scalar_orbital_motion(binary.orbit, phases)
    assert_allclose(vectorized_positions, scalar_positions, rtol=0.0, atol=1e-10)
    assert_array_equal(vectorized_positions[:, 0], np.arange(len(phases)))
    assert_allclose(vectorized_positions[:, 1:], reference_positions, rtol=0.0, atol=1e-10)
    assert vectorized_flux.shape == phases.shape
    assert_allclose(vectorized_flux, scalar_flux, rtol=2e-14, atol=1e-12)
    assert vectorized_flux[0] == vectorized_flux[4]
