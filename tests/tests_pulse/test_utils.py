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

    def test_derotate_surface_displacements_no_rotation(self) -> None:
        """Test derotate_surface_displacements when axis_phi and axis_theta are zero."""
        velocity = np.array([[0.1, 0.01, 0.02], [0.15, 0.015, 0.025]])
        tilted_points = np.array([[1.0, 0.0, np.pi / 2], [1.0, np.pi / 4, np.pi / 3]])
        points = np.array([[1.0, 0.0, np.pi / 2], [1.0, np.pi / 4, np.pi / 3]])
        axis_phi = 0.0
        axis_theta = 0.0

        result = putils.derotate_surface_displacements(velocity, tilted_points, points, axis_phi, axis_theta)

        assert_array_equal(result, velocity)

    def test_derotate_surface_displacements_with_rotation(self) -> None:
        """Test derotate_surface_displacements applies derotation correctly."""
        velocity = np.array([[0.1, 0.01, 0.02], [0.15, 0.015, 0.025]])
        tilted_points = np.array([[1.0, 0.0, np.pi / 2], [1.0, np.pi / 4, np.pi / 3]])
        points = np.array([[1.0, 0.0, np.pi / 2], [1.0, np.pi / 4, np.pi / 3]])
        axis_phi = np.pi / 6
        axis_theta = np.pi / 4

        result = putils.derotate_surface_displacements(velocity, tilted_points, points, axis_phi, axis_theta)

        # Result should have same shape and first column (radial) unchanged
        assert result.shape == velocity.shape
        assert_array_equal(result[:, 0], velocity[:, 0])

    def test_transform_spherical_displacement_to_cartesian_simple(self) -> None:
        """Test spherical to cartesian displacement transformation."""
        sph_displacement = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        surf_points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        com_x = 0.0

        result = putils.transform_spherical_displacement_to_cartesian(sph_displacement, surf_points, com_x)

        # Result should have same shape
        assert result.shape == (3, 3)

        # Result should be finite (no NaNs or infs)
        assert np.all(np.isfinite(result))

    def test_transform_spherical_displacement_to_cartesian_nonzero_com(self) -> None:
        """Test spherical to cartesian transformation with non-zero center of mass."""
        sph_displacement = np.array([[0.1, 0.01, 0.02], [0.15, 0.015, 0.025]])
        surf_points = np.array([[1.5, 0.5, 0.2], [1.2, 0.8, 0.3]])
        com_x = 0.5

        result = putils.transform_spherical_displacement_to_cartesian(sph_displacement, surf_points, com_x)

        # Result should have same shape
        assert result.shape == (2, 3)

        # Result should be finite
        assert np.all(np.isfinite(result))

    def test_horizontal_component_zero_displacement(self) -> None:
        """Test horizontal_component with zero displacement."""
        displacement = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, np.pi / 2], [1.0, np.pi / 4, np.pi / 3]])

        result = putils.horizontal_component(displacement, points)

        expected = np.array([0.0, 0.0])
        assert_almost_equal(result, expected)

    def test_horizontal_component_radial_displacement_only(self) -> None:
        """Test horizontal_component with only radial displacement (should be zero)."""
        displacement = np.array([[0.1, 0.0, 0.0], [0.15, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, np.pi / 2], [1.0, np.pi / 4, np.pi / 3]])

        result = putils.horizontal_component(displacement, points)

        expected = np.array([0.0, 0.0])
        assert_almost_equal(result, expected)

    def test_horizontal_component_with_displacement(self) -> None:
        """Test horizontal_component calculates distance correctly."""
        displacement = np.array([[0.0, 0.1, 0.1], [0.0, 0.05, 0.05]])
        points = np.array([[1.0, 0.0, np.pi / 2], [1.0, np.pi / 4, np.pi / 3]])

        result = putils.horizontal_component(displacement, points)

        # Result should be positive for non-zero displacement
        assert np.all(result >= 0.0)

        # Result should be finite
        assert np.all(np.isfinite(result))

    def test_horizontal_component_treat_poles(self) -> None:
        """Test horizontal_component with treat_poles=True."""
        displacement = np.array([[0.0, 0.1, 0.1], [0.0, 0.5, 0.5]])
        points = np.array([[1.0, 0.0, np.pi / 2], [1.0, np.pi / 4, np.pi / 3]])

        result = putils.horizontal_component(displacement, points, treat_poles=True)

        # Result should be finite and reasonable
        assert np.all(np.isfinite(result))
        # All values should be positive and valid
        assert np.all(result >= 0.0)

    def test_pole_neighbours_single_system(self) -> None:
        """Test pole_neighbours finds correct pole indices."""
        pulse_meta = [
            {
                "l": 2,
                "m": 1,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
            },
        ]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = SingleSystem.build_container(single, phase=0.0)

        star = system_container.star
        putils.pole_neighbours(star)

        # Check that pole indices were assigned
        assert star.pole_idx is not None
        assert star.pole_idx_neighbour is not None

        # Check that pole indices are valid
        assert len(star.pole_idx) == 2
        assert len(star.pole_idx_neighbour) == 2

        # Check that indices are within valid range
        n_points = len(star.points_spherical)
        assert np.all(star.pole_idx >= 0)
        assert np.all(star.pole_idx < n_points)
        assert np.all(star.pole_idx_neighbour >= 0)
        assert np.all(star.pole_idx_neighbour < n_points)

    def test_pole_neighbours_different_indices(self) -> None:
        """Test that pole_neighbours identifies different poles."""
        pulse_meta = [
            {
                "l": 2,
                "m": 1,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
            },
        ]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = SingleSystem.build_container(single, phase=0.0)

        star = system_container.star
        putils.pole_neighbours(star)

        # The two poles should be different
        assert star.pole_idx[0] != star.pole_idx[1]

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
