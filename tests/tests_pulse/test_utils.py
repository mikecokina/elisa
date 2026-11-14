from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from elisa import const
from elisa import units as u
from elisa.base.star import Star
from elisa.binary_system.system import BinarySystem
from elisa.pulse import utils as putils
from elisa.single_system.system import SingleSystem
from tests.tests_pulse.helpers import BINARY_STAR, BINARY_SYSTEM, STAR_PARAMS, SYSTEM_PARMAS
from tests.utils import ElisaTestCase


class TestPulseUtils(ElisaTestCase):
    """Unit tests for elisa.pulse.utils module functions."""

    def setUp(self) -> None:
        super().setUp()
        self.base_path = str(Path(__file__).resolve().parent.parent)
        self.lc_base_path = str(Path(__file__).resolve().parent.joinpath("data", "light_curves"))

    def test_phase_correction_with_synchronicity(self) -> None:
        """Test phase_correction with valid synchronicity values."""
        phase = 0.5
        synchronicity = 1.5

        result = putils.phase_correction(phase, synchronicity)
        expected = (synchronicity - 1) * phase * const.FULL_ARC

        assert_almost_equal(result, expected)

    def test_phase_correction_with_synchronicity_one(self) -> None:
        """Test phase_correction with synchronicity = 1 (no correction)."""
        phase = 0.5
        synchronicity = 1.0

        result = putils.phase_correction(phase, synchronicity)
        expected = 0.0

        assert_almost_equal(result, expected)

    def test_phase_correction_with_nan_synchronicity(self) -> None:
        """Test phase_correction when synchronicity is NaN."""
        phase = 0.5

        result = putils.phase_correction(phase, np.nan)
        expected = phase * const.FULL_ARC

        assert_almost_equal(result, expected)

    def test_generate_time_exponential(self) -> None:
        """Test generate_time_exponential returns correct complex exponential."""
        pulse_meta = [
            {
                "l": 1,
                "m": 1,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
            },
        ]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        mode = single.star.pulsations[0]

        time = 1.0
        result = putils.generate_time_exponential(mode, time)

        exponent = mode.angular_frequency * time + mode.start_phase
        expected = np.exp(complex(0, -exponent))

        assert_almost_equal(result, expected)

    def test_generate_phase_shift(self) -> None:
        """Test generate_phase_shift returns correct phase shift factor."""
        shift = np.pi / 4

        result = putils.generate_phase_shift(shift)
        expected = np.exp(complex(0, -shift))

        assert_almost_equal(result, expected)

    def test_generate_phase_shift_zero(self) -> None:
        """Test generate_phase_shift with zero shift."""
        shift = 0.0

        result = putils.generate_phase_shift(shift)
        expected = 1.0 + 0j

        assert_almost_equal(result, expected)

    def test_tilt_mode_coordinates_no_tilt(self) -> None:
        """Test tilt_mode_coordinates when phi and theta are zero (no tilt)."""
        points = np.array([[1.0, 0.0, np.pi / 2], [1.0, np.pi / 4, np.pi / 3], [1.0, np.pi / 2, np.pi / 6]])
        phi = 0.0
        theta = 0.0

        result = putils.tilt_mode_coordinates(points, phi, theta)

        assert_array_equal(result, points)

    def test_tilt_mode_coordinates_with_tilt(self) -> None:
        """Test tilt_mode_coordinates applies rotation correctly."""
        points = np.array([[1.0, 0.0, np.pi / 2], [1.0, np.pi / 4, np.pi / 3]])
        phi = np.pi / 6
        theta = np.pi / 4

        result = putils.tilt_mode_coordinates(points, phi, theta)

        # Result should have same shape and first column (radius) unchanged
        assert result.shape == points.shape
        assert_array_equal(result[:, 0], points[:, 0])

        # Phi and theta should be modified
        assert not np.array_equal(result[:, 1:], points[:, 1:])

    def test_derotate_surface_points_no_rotation(self) -> None:
        """Test derotate_surface_points when phi and theta are zero (no rotation)."""
        points = np.array([[1.0, 0.0, np.pi / 2], [1.0, np.pi / 4, np.pi / 3]])
        phi = 0.0
        theta = 0.0

        result = putils.derotate_surface_points(points, phi, theta)

        assert_array_equal(result, points)

    def test_derotate_surface_points_with_rotation(self) -> None:
        """Test derotate_surface_points applies derotation correctly."""
        points = np.array([[1.0, 0.0, np.pi / 2], [1.0, np.pi / 4, np.pi / 3]])
        phi = np.pi / 6
        theta = np.pi / 4

        result = putils.derotate_surface_points(points, phi, theta)

        # Result should have same shape and first column (radius) unchanged
        assert result.shape == points.shape
        assert_array_equal(result[:, 0], points[:, 0])

        # Phi and theta should be modified
        assert not np.array_equal(result[:, 1:], points[:, 1:])

    def test_generate_tilt_coordinates_without_tidally_locked(self) -> None:
        """Test generate_tilt_coordinates with non-tidally-locked modes."""
        pulse_meta = [
            {
                "l": 1,
                "m": 1,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
                "mode_axis_theta": np.pi / 6,
                "mode_axis_phi": np.pi / 4,
                "tidally_locked": False,
            },
        ]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = SingleSystem.build_container(single, phase=0.0)

        star = system_container.star
        phase = 0.5

        phi, theta = putils.generate_tilt_coordinates(star, phase)

        # Phi should include the correction
        assert phi is not None
        assert theta is not None
        assert np.isfinite(phi)
        assert np.isfinite(theta)

    @pytest.mark.requires_ld
    def test_generate_tilt_coordinates_tidally_locked(self) -> None:
        """Test generate_tilt_coordinates with tidally-locked modes."""
        pulse_meta = [
            {
                "l": 1,
                "m": 1,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
                "mode_axis_theta": np.pi / 6,
                "mode_axis_phi": np.pi / 4,
                "tidally_locked": True,
            },
        ]

        # For tidally locked, we need a binary system
        primary = Star(pulsations=pulse_meta, **BINARY_STAR)
        secondary = Star(pulsations=pulse_meta, **BINARY_STAR)
        binary = BinarySystem(primary=primary, secondary=secondary, **BINARY_SYSTEM)
        system_container = BinarySystem.build_container(binary, phase=0.0)

        star = system_container.primary
        phase = 0.5

        phi, _theta = putils.generate_tilt_coordinates(star, phase)

        # For tidally locked, phi_corr should be 0
        expected_phi = star.pulsations[0].mode_axis_phi
        assert_almost_equal(phi, expected_phi)
