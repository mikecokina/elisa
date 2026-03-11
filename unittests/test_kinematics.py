"""Unit tests for elisa.pulse.surface.kinematics module."""

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from elisa import settings
from elisa import units as u
from elisa.base.star import Star
from elisa.pulse.mode import PulsationMode
from elisa.pulse.surface import kinematics
from elisa.single_system.system import SingleSystem
from unittests.utils import ElisaTestCase

STAR_PARAMS = {
    "mass": 2.0 * u.solMass,
    "t_eff": 10000 * u.K,
    "gravity_darkening": 1.0,
    "discretization_factor": 5,
    "albedo": 1.0,
    "metallicity": 0.0,
    "polar_log_g": 4.0 * u.dex(u.cm / u.s ** 2),
}

SYSTEM_PARAMS = {
    "gamma": 0 * u.km / u.s,
    "inclination": 80 * u.deg,
    "rotation_period": 30 * u.d,
}


class TestCalculateHorizontalDisplacements(ElisaTestCase):
    """Unit tests for calculate_horizontal_displacements function."""

    def setUp(self):
        """Set up test fixtures."""
        self.thetas = np.linspace(0.1, np.pi - 0.1, 10)
        self.harmonics_derivatives = np.random.random((2, 10)) + 1j * np.random.random((2, 10))
        self.radius = 1.0
        self.scale = 1.0

    def test_radial_mode_returns_zeros(self):
        """Test that radial mode (l=0) returns zero displacements."""
        mode = PulsationMode(
            l=0,
            m=0,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            horizontal_to_radial_amplitude_ratio=0.0,
            temperature_amplitude_factor=0.01,
        )

        phi_disp, theta_disp = kinematics.calculate_horizontal_displacements(
            mode, self.thetas, self.harmonics_derivatives, self.radius, self.scale,
        )

        assert_array_equal(phi_disp, np.zeros(10))
        assert_array_equal(theta_disp, np.zeros(10))

    def test_non_radial_mode_returns_complex(self):
        """Test that non-radial mode returns complex displacements."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            horizontal_to_radial_amplitude_ratio=0.5,
        )
        mode.horizontal_amplitude = 0.1

        phi_disp, theta_disp = kinematics.calculate_horizontal_displacements(
            mode, self.thetas, self.harmonics_derivatives, self.radius, self.scale,
        )

        # Should return complex arrays
        self.assertTrue(np.iscomplexobj(phi_disp))
        self.assertTrue(np.iscomplexobj(theta_disp))

        # Should have correct shape
        self.assertEqual(phi_disp.shape, self.thetas.shape)
        self.assertEqual(theta_disp.shape, self.thetas.shape)

        # Should be finite
        self.assertTrue(np.all(np.isfinite(phi_disp)))
        self.assertTrue(np.all(np.isfinite(theta_disp)))

    def test_horizontal_displacements_with_zero_amplitude(self):
        """Test horizontal displacements with zero horizontal amplitude."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            horizontal_to_radial_amplitude_ratio=0.0,
            temperature_amplitude_factor=0.01,
        )
        mode.horizontal_amplitude = 0.0

        phi_disp, theta_disp = kinematics.calculate_horizontal_displacements(
            mode, self.thetas, self.harmonics_derivatives, self.radius, self.scale,
        )

        # Should return zeros when horizontal amplitude is zero
        assert_array_almost_equal(phi_disp, np.zeros(10))
        assert_array_almost_equal(theta_disp, np.zeros(10))

    def test_horizontal_displacements_scale_dependence(self):
        """Test that scale parameter affects the displacements correctly."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            horizontal_to_radial_amplitude_ratio=0.5,
        )
        mode.horizontal_amplitude = 0.1

        # Calculate with scale = 1.0
        phi_disp_1, theta_disp_1 = kinematics.calculate_horizontal_displacements(
            mode, self.thetas, self.harmonics_derivatives, self.radius, 1.0,
        )

        # Calculate with scale = 2.0
        phi_disp_2, theta_disp_2 = kinematics.calculate_horizontal_displacements(
            mode, self.thetas, self.harmonics_derivatives, self.radius, 2.0,
        )

        # Different scales should give different results
        self.assertFalse(np.allclose(phi_disp_1, phi_disp_2))
        self.assertFalse(np.allclose(theta_disp_1, theta_disp_2))

    def test_horizontal_displacements_pole_treatment(self):
        """Test that pole region handling works correctly."""
        # Use angles away from exact poles
        thetas_no_poles = np.array([0.1, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi - 0.1])
        harmonics_derivatives = np.ones((2, 5)) + 1j * np.ones((2, 5))

        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            horizontal_to_radial_amplitude_ratio=0.5,
        )
        mode.horizontal_amplitude = 0.1

        phi_disp, theta_disp = kinematics.calculate_horizontal_displacements(
            mode, thetas_no_poles, harmonics_derivatives, self.radius, self.scale,
        )

        # All results should be finite
        self.assertTrue(np.all(np.isfinite(phi_disp)))
        self.assertTrue(np.all(np.isfinite(theta_disp)))

        # Shape should match input
        self.assertEqual(phi_disp.shape, thetas_no_poles.shape)
        self.assertEqual(theta_disp.shape, thetas_no_poles.shape)


class TestCalculateDisplacementCoordinates(ElisaTestCase):
    """Unit tests for calculate_displacement_coordinates function."""

    def setUp(self):
        """Set up test fixtures."""
        self.points = np.array([[1.0, 0.0, 0.1], [1.0, 0.1, 0.5], [1.0, 0.2, 1.0]])
        self.harmonics = np.array([1.0 + 0.1j, 0.9 + 0.2j, 0.8 + 0.3j])
        self.harmonics_derivatives = np.array([
            [0.1 + 0.05j, 0.15 + 0.1j, 0.2 + 0.15j],
            [0.2 + 0.1j, 0.25 + 0.15j, 0.3 + 0.2j],
        ])
        self.radius = 1.0

    def test_displacement_coordinates_basic(self):
        """Test basic displacement coordinates calculation."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            horizontal_to_radial_amplitude_ratio=0.5,
        )
        mode.radial_amplitude = 0.1
        mode.horizontal_amplitude = 0.05

        result = kinematics.calculate_displacement_coordinates(
            mode, self.points, self.harmonics, self.harmonics_derivatives, self.radius,
        )

        # Should return array with shape (n_points, 3)
        self.assertEqual(result.shape, (3, 3))

        # Should be finite
        self.assertTrue(np.all(np.isfinite(result)))

    def test_displacement_coordinates_radial_component(self):
        """Test radial component of displacement."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            horizontal_to_radial_amplitude_ratio=0.5,
        )
        mode.radial_amplitude = 0.1
        mode.horizontal_amplitude = 0.05

        result = kinematics.calculate_displacement_coordinates(
            mode, self.points, self.harmonics, self.harmonics_derivatives, self.radius, scale=1.0,
        )

        # Radial component should match radial_amplitude * harmonics
        expected_radial = 0.1 * self.harmonics / 1.0
        assert_array_almost_equal(result[:, 0], expected_radial)

    def test_displacement_coordinates_scale_factor(self):
        """Test that scale factor affects displacement correctly."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            horizontal_to_radial_amplitude_ratio=0.5,
        )
        mode.radial_amplitude = 0.1
        mode.horizontal_amplitude = 0.05

        result_scale_1 = kinematics.calculate_displacement_coordinates(
            mode, self.points, self.harmonics, self.harmonics_derivatives, self.radius, scale=1.0,
        )

        result_scale_2 = kinematics.calculate_displacement_coordinates(
            mode, self.points, self.harmonics, self.harmonics_derivatives, self.radius, scale=2.0,
        )

        # Radial component should be scaled by 1/scale
        assert_array_almost_equal(
            result_scale_1[:, 0] / result_scale_2[:, 0],
            np.full(3, 2.0),
        )

    def test_displacement_coordinates_invalid_model(self):
        """Test that invalid pulsation model raises error."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            horizontal_to_radial_amplitude_ratio=0.5,
        )
        mode.radial_amplitude = 0.1
        mode.horizontal_amplitude = 0.05

        # Save original setting
        original_model = settings.PULSATION_MODEL

        try:
            # Set to invalid model
            settings.PULSATION_MODEL = "invalid_model"

            with self.assertRaises(NotImplementedError):
                kinematics.calculate_displacement_coordinates(
                    mode, self.points, self.harmonics, self.harmonics_derivatives, self.radius,
                )
        finally:
            # Restore original setting
            settings.PULSATION_MODEL = original_model


class TestCalculateModeAngularDisplacement(ElisaTestCase):
    """Unit tests for calculate_mode_angular_displacement function."""

    def test_real_part_extraction(self):
        """Test that real part is extracted from complex displacement."""
        displacement = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])

        result = kinematics.calculate_mode_angular_displacement(displacement)

        expected = np.array([1.0, 3.0, 5.0])
        assert_array_equal(result, expected)

    def test_real_displacement_unchanged(self):
        """Test that real displacement is unchanged."""
        displacement = np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])

        result = kinematics.calculate_mode_angular_displacement(displacement)

        assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_pure_imaginary_gives_zero(self):
        """Test that pure imaginary displacement gives zero."""
        displacement = np.array([0.0 + 1.0j, 0.0 + 2.0j, 0.0 + 3.0j])

        result = kinematics.calculate_mode_angular_displacement(displacement)

        assert_array_almost_equal(result, np.array([0.0, 0.0, 0.0]))


class TestCalculateRadialDisplacement(ElisaTestCase):
    """Unit tests for calculate_radial_displacement function."""

    def test_radial_displacement_basic(self):
        """Test basic radial displacement calculation."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            horizontal_to_radial_amplitude_ratio=0.5,
        )
        mode.radial_amplitude = 0.1

        harmonics = np.array([1.0 + 0.1j, 0.9 + 0.2j, 0.8 + 0.3j])

        result = kinematics.calculate_radial_displacement(mode, harmonics)

        expected = 0.1 * harmonics
        assert_array_almost_equal(result, expected)

    def test_radial_displacement_preserves_complex(self):
        """Test that radial displacement preserves complex nature."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            horizontal_to_radial_amplitude_ratio=0.5,
        )
        mode.radial_amplitude = 0.1

        harmonics = np.array([1.0 + 0.1j, 0.9 + 0.2j])

        result = kinematics.calculate_radial_displacement(mode, harmonics)

        self.assertTrue(np.iscomplexobj(result))

    def test_radial_displacement_zero_amplitude(self):
        """Test radial displacement with zero amplitude."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            horizontal_to_radial_amplitude_ratio=0.5,
        )
        mode.radial_amplitude = 0.0

        harmonics = np.array([1.0 + 0.1j, 0.9 + 0.2j])

        result = kinematics.calculate_radial_displacement(mode, harmonics)

        assert_array_almost_equal(result, np.zeros_like(harmonics))


class TestCalculateModeDerivatives(ElisaTestCase):
    """Unit tests for calculate_mode_derivatives function."""

    def test_derivatives_basic(self):
        """Test basic derivative calculation."""
        displacement = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])
        angular_frequency = 2.0

        result = kinematics.calculate_mode_derivatives(displacement, angular_frequency)

        expected = angular_frequency * np.imag(displacement)
        expected = np.array([4.0, 8.0, 12.0])
        assert_array_almost_equal(result, expected)

    def test_derivatives_real_displacement(self):
        """Test that real displacement gives zero derivatives."""
        displacement = np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])
        angular_frequency = 2.0

        result = kinematics.calculate_mode_derivatives(displacement, angular_frequency)

        assert_array_almost_equal(result, np.array([0.0, 0.0, 0.0]))

    def test_derivatives_frequency_scaling(self):
        """Test that angular frequency scales the derivatives."""
        displacement = np.array([1.0 + 1.0j, 2.0 + 2.0j])

        result_1 = kinematics.calculate_mode_derivatives(displacement, 1.0)
        result_2 = kinematics.calculate_mode_derivatives(displacement, 2.0)

        assert_array_almost_equal(result_2, 2.0 * result_1)


class TestCalculateModeSecondDerivatives(ElisaTestCase):
    """Unit tests for calculate_mode_second_derivatives function."""

    def test_second_derivatives_basic(self):
        """Test basic second derivative calculation."""
        displacement = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])
        angular_frequency = 2.0

        result = kinematics.calculate_mode_second_derivatives(displacement, angular_frequency)

        expected = -angular_frequency**2 * np.real(displacement)
        expected = np.array([-4.0, -12.0, -20.0])
        assert_array_almost_equal(result, expected)

    def test_second_derivatives_imaginary_displacement(self):
        """Test that imaginary displacement gives zero second derivatives."""
        displacement = np.array([0.0 + 1.0j, 0.0 + 2.0j, 0.0 + 3.0j])
        angular_frequency = 2.0

        result = kinematics.calculate_mode_second_derivatives(displacement, angular_frequency)

        assert_array_almost_equal(result, np.array([0.0, 0.0, 0.0]))

    def test_second_derivatives_negative_sign(self):
        """Test that second derivatives have negative sign."""
        displacement = np.array([1.0 + 0.0j, 2.0 + 0.0j])
        angular_frequency = 2.0

        result = kinematics.calculate_mode_second_derivatives(displacement, angular_frequency)

        self.assertTrue(np.all(result <= 0.0))

    def test_second_derivatives_frequency_squared_scaling(self):
        """Test that angular frequency squared scales the second derivatives."""
        displacement = np.array([1.0 + 1.0j, 2.0 + 2.0j])

        result_1 = kinematics.calculate_mode_second_derivatives(displacement, 1.0)
        result_2 = kinematics.calculate_mode_second_derivatives(displacement, 2.0)

        assert_array_almost_equal(result_2, 4.0 * result_1)


class TestCalculateTemperaturePertFactor(ElisaTestCase):
    """Unit tests for calculate_temperature_pert_factor function."""

    def test_temperature_perturbation_basic(self):
        """Test basic temperature perturbation factor calculation."""
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
            "horizontal_to_radial_amplitude_ratio": 0.5,
            "temperature_amplitude_factor": 0.01,
        }]

        star = Star(pulsations=pulse_meta, **STAR_PARAMS)
        system = SingleSystem(star=star, **SYSTEM_PARAMS)

        mode = system.star.pulsations[0]
        mode.complex_displacement = np.array([
            [1.0 + 1.0j, 0.1 + 0.05j, 0.2 + 0.1j],
            [0.9 + 1.1j, 0.15 + 0.1j, 0.25 + 0.15j],
        ])
        mode.radial_amplitude = 0.1

        result = kinematics.calculate_temperature_pert_factor(mode, scale=1.0)

        # Should return array with same shape as first column of complex_displacement
        self.assertEqual(result.shape, (2,))

        # Should be real-valued (imaginary part eliminated by np.real)
        self.assertTrue(np.isrealobj(result))

        # Should be finite
        self.assertTrue(np.all(np.isfinite(result)))

    def test_temperature_perturbation_scale_dependence(self):
        """Test that scale parameter affects temperature perturbation."""
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
            "horizontal_to_radial_amplitude_ratio": 0.5,
            "temperature_amplitude_factor": 0.01,
        }]

        star = Star(pulsations=pulse_meta, **STAR_PARAMS)
        system = SingleSystem(star=star, **SYSTEM_PARAMS)

        mode = system.star.pulsations[0]
        mode.complex_displacement = np.array([
            [1.0 + 1.0j, 0.1 + 0.05j],
            [0.9 + 1.1j, 0.15 + 0.1j],
        ])
        mode.radial_amplitude = 0.1

        result_1 = kinematics.calculate_temperature_pert_factor(mode, scale=1.0)
        result_2 = kinematics.calculate_temperature_pert_factor(mode, scale=2.0)

        # Different scales should give different results
        self.assertFalse(np.allclose(result_1, result_2))

        # Ratio should match scale ratio
        assert_array_almost_equal(result_2, 2.0 * result_1)

    def test_temperature_perturbation_phase_shift(self):
        """Test that phase shift affects temperature perturbation."""
        pulse_meta_1 = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
            "temperature_perturbation_phase_shift": 0.0,
            "horizontal_to_radial_amplitude_ratio": 0.5,
            "temperature_amplitude_factor": 0.01,
        }]

        pulse_meta_2 = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
            "temperature_perturbation_phase_shift": np.pi / 4,
            "horizontal_to_radial_amplitude_ratio": 0.5,
            "temperature_amplitude_factor": 0.01,
        }]

        star_1 = Star(pulsations=pulse_meta_1, **STAR_PARAMS)
        system_1 = SingleSystem(star=star_1, **SYSTEM_PARAMS)

        star_2 = Star(pulsations=pulse_meta_2, **STAR_PARAMS)
        system_2 = SingleSystem(star=star_2, **SYSTEM_PARAMS)

        mode_1 = system_1.star.pulsations[0]
        mode_2 = system_2.star.pulsations[0]

        displacement = np.array([
            [1.0 + 1.0j, 0.1 + 0.05j],
            [0.9 + 1.1j, 0.15 + 0.1j],
        ])

        mode_1.complex_displacement = displacement
        mode_2.complex_displacement = displacement
        mode_1.radial_amplitude = 0.1
        mode_2.radial_amplitude = 0.1

        result_1 = kinematics.calculate_temperature_pert_factor(mode_1, scale=1.0)
        result_2 = kinematics.calculate_temperature_pert_factor(mode_2, scale=1.0)

        # Different phase shifts should give different results
        self.assertFalse(np.allclose(result_1, result_2))

