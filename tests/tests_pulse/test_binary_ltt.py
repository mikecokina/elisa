"""Orbital propagation delays relative to a binary's barycentre."""

from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import const
from elisa import units as u
from elisa.base.container import StarContainer
from elisa.base.star import Star
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.system import BinarySystem
from tests.tests_pulse.helpers import BINARY_STAR, BINARY_SYSTEM
from tests.utils import ElisaTestCase


@pytest.mark.parametrize("distance", [1.0, 0.7, 1.4])
@pytest.mark.parametrize("light_crossing_seconds", [1.0, 7.0])
@pytest.mark.parametrize(("inclination", "projection"), [(0.0, 0.0), (np.pi / 6, 0.5), (np.pi / 2, 1.0)])
@pytest.mark.parametrize(("azimuth", "direction"), [(0.0, 0), (np.pi / 2, 1), (np.pi, 0), (3 * np.pi / 2, -1)])
def test_geometric_emission_times(
    distance: float,
    light_crossing_seconds: float,
    inclination: float,
    projection: float,
    azimuth: float,
    direction: int,
) -> None:
    # For M1=2 M2, the barycentre divides the separation into d/3 and 2d/3.
    # At A=pi/2 the primary is farther away, the secondary nearer; at 3pi/2
    # these reverse. At the nodes (A=0, pi) both have the barycentre's path.
    stars = [StarContainer(), StarContainer()]
    for star in stars:
        star.synchronicity = 1.0
        star.pulsations = {0: object()}
    system = OrbitalPositionContainer(
        *stars,
        position=const.Position(0, 1.0, 0.0, 0.0, 0.0),
        period=1.0,
        mass_ratio=0.5,
        inclination=inclination,
        semi_major_axis=const.C * light_crossing_seconds,
    )
    # Use the same position update as curve synthesis, retaining actual rotations.
    system.set_on_position_params(const.Position(0, distance, azimuth, 0.0, 100 / 86400))
    system.set_time()
    path_seconds = direction * distance * projection * light_crossing_seconds
    expected = [100 - path_seconds / 3, 100 + 2 * path_seconds / 3]
    with patch("elisa.binary_system.surface.pulsations.generate_harmonics") as generate:
        system.build_harmonics(components_distance=distance)
    assert generate.call_count == 2
    for call, star, time, com_x in zip(generate.call_args_list, stars, expected, [0.0, distance], strict=True):
        assert call.args == (star,)
        assert call.kwargs["time"] == pytest.approx(time, abs=1e-12)
        assert call.kwargs["com_x"] == com_x
        assert call.kwargs["phase"] == 0.0
    assert system.time == pytest.approx(100.0)


class TestBinaryLttSynthesis(ElisaTestCase):
    def test_real_binary_harmonics(self) -> None:
        """Check real harmonic generation and perturbations against RMS-normalized radial Y_0^0."""
        pulse = [{
            "l": 0, "m": 0, "amplitude": 1 * u.m / u.s,
            "frequency": 0.013 / u.s, "start_phase": 0.23,
            "horizontal_to_radial_amplitude_ratio": 0.0, "temperature_amplitude_factor": 0.01,
        }]
        primary = Star(pulsations=pulse, **{**BINARY_STAR, "discretization_factor": 10})
        secondary = Star(pulsations=pulse, **{
            **BINARY_STAR, "mass": 1 * u.solMass, "discretization_factor": 10,
        })
        # With omega=0 and phase=0, circular conjunction is A=pi/2.
        binary = BinarySystem(primary=primary, secondary=secondary, **{
            **BINARY_SYSTEM, "eccentricity": 0.0, "inclination": 30 * u.deg,
        })
        container = binary.build_container(phase=0.0)
        assert container.position.azimuth == pytest.approx(np.pi / 2)
        assert binary.mass_ratio == pytest.approx(0.5)
        # Kepler's law in SI supplies an independent physical length scale.
        period_seconds = 2 * 86400
        separation_metres = (const.G * (primary.mass + secondary.mass)
                             * (period_seconds / (2 * np.pi)) ** 2) ** (1 / 3)
        assert binary.semi_major_axis == pytest.approx(separation_metres)
        crossing_seconds = separation_metres / 299792458.0
        # Half the orbital radius is along the line of sight at i=30 degrees.
        emission_times = [-crossing_seconds / 6, crossing_seconds / 3]
        for star, time in zip([container.primary, container.secondary], emission_times, strict=True):
            mode = star.pulsations[0]
            expected = np.exp(-1j * (2 * np.pi * 0.013 * time + 0.23))
            assert_allclose(mode.point_harmonics, expected, rtol=1e-12, atol=1e-12)
            assert np.isfinite(star.points).all()
            assert np.isfinite(star.velocities).all()
            assert np.isfinite(star.temperatures).all()
