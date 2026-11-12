"""Unit tests for elisa.pulse.surface.kinematics module."""

import numpy as np
from numpy.testing import assert_array_almost_equal

from elisa import units as u
from elisa.base.star import Star
from elisa.pulse.mode import PulsationMode
from elisa.pulse.surface import kinematics
from elisa.single_system.system import SingleSystem
from tests.utils import ElisaTestCase

STAR_PARAMS = {
    "mass": 2.0 * u.solMass,
    "t_eff": 10000 * u.K,
    "gravity_darkening": 1.0,
    "discretization_factor": 5,
    "albedo": 1.0,
    "metallicity": 0.0,
    "polar_log_g": 4.0 * u.dex(u.cm / u.s**2),
}

SYSTEM_PARAMS = {
    "gamma": 0 * u.km / u.s,
    "inclination": 80 * u.deg,
    "rotation_period": 30 * u.d,
}


class TestCalculateRadialDisplacement(ElisaTestCase):
    """Unit tests for calculate_radial_displacement function."""

    def test_radial_displacement_basic(self) -> None:
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

    def test_radial_displacement_preserves_complex(self) -> None:
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

        assert np.iscomplexobj(result)

    def test_radial_displacement_zero_amplitude(self) -> None:
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

    def test_derivatives_basic(self) -> None:
        """Test basic derivative calculation."""
        displacement = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])
        angular_frequency = 2.0

        result = kinematics.calculate_mode_derivatives(displacement, angular_frequency)

        expected = angular_frequency * np.imag(displacement)
        expected = np.array([4.0, 8.0, 12.0])
        assert_array_almost_equal(result, expected)

    def test_derivatives_real_displacement(self) -> None:
        """Test that real displacement gives zero derivatives."""
        displacement = np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])
        angular_frequency = 2.0

        result = kinematics.calculate_mode_derivatives(displacement, angular_frequency)

        assert_array_almost_equal(result, np.array([0.0, 0.0, 0.0]))

    def test_derivatives_frequency_scaling(self) -> None:
        """Test that angular frequency scales the derivatives."""
        displacement = np.array([1.0 + 1.0j, 2.0 + 2.0j])

        result_1 = kinematics.calculate_mode_derivatives(displacement, 1.0)
        result_2 = kinematics.calculate_mode_derivatives(displacement, 2.0)

        assert_array_almost_equal(result_2, 2.0 * result_1)


class TestCalculateModeSecondDerivatives(ElisaTestCase):
    """Unit tests for calculate_mode_second_derivatives function."""

    def test_second_derivatives_basic(self) -> None:
        """Test basic second derivative calculation."""
        displacement = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])
        angular_frequency = 2.0

        result = kinematics.calculate_mode_second_derivatives(displacement, angular_frequency)

        expected = -(angular_frequency**2) * np.real(displacement)
        expected = np.array([-4.0, -12.0, -20.0])
        assert_array_almost_equal(result, expected)

    def test_second_derivatives_imaginary_displacement(self) -> None:
        """Test that imaginary displacement gives zero second derivatives."""
        displacement = np.array([0.0 + 1.0j, 0.0 + 2.0j, 0.0 + 3.0j])
        angular_frequency = 2.0

        result = kinematics.calculate_mode_second_derivatives(displacement, angular_frequency)

        assert_array_almost_equal(result, np.array([0.0, 0.0, 0.0]))

    def test_second_derivatives_negative_sign(self) -> None:
        """Test that second derivatives have negative sign."""
        displacement = np.array([1.0 + 0.0j, 2.0 + 0.0j])
        angular_frequency = 2.0

        result = kinematics.calculate_mode_second_derivatives(displacement, angular_frequency)

        assert np.all(result <= 0.0)

    def test_second_derivatives_frequency_squared_scaling(self) -> None:
        """Test that angular frequency squared scales the second derivatives."""
        displacement = np.array([1.0 + 1.0j, 2.0 + 2.0j])

        result_1 = kinematics.calculate_mode_second_derivatives(displacement, 1.0)
        result_2 = kinematics.calculate_mode_second_derivatives(displacement, 2.0)

        assert_array_almost_equal(result_2, 4.0 * result_1)


class TestCalculateTemperaturePertFactor(ElisaTestCase):
    """Unit tests for calculate_temperature_pert_factor function."""

    def test_temperature_perturbation_basic(self) -> None:
        """Test basic temperature perturbation factor calculation."""
        pulse_meta = [
            {
                "l": 2,
                "m": 1,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": 0.5,
                "temperature_amplitude_factor": 0.01,
            },
        ]

        star = Star(pulsations=pulse_meta, **STAR_PARAMS)
        system = SingleSystem(star=star, **SYSTEM_PARAMS)

        mode = system.star.pulsations[0]
        mode.complex_displacement = np.array(
            [
                [1.0 + 1.0j, 0.1 + 0.05j, 0.2 + 0.1j],
                [0.9 + 1.1j, 0.15 + 0.1j, 0.25 + 0.15j],
            ],
        )
        mode.complex_radial_displacement = mode.complex_displacement[:, 0]
        mode.radial_amplitude = 0.1

        result = kinematics.calculate_temperature_pert_factor(mode, scale=1.0)

        # Should return array with same shape as first column of complex_displacement
        assert result.shape == (2,)

        # Should be real-valued (imaginary part eliminated by np.real)
        assert np.isrealobj(result)

        # Should be finite
        assert np.all(np.isfinite(result))

    def test_temperature_perturbation_scale_dependence(self) -> None:
        """Test that scale parameter affects temperature perturbation."""
        pulse_meta = [
            {
                "l": 2,
                "m": 1,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": 0.5,
                "temperature_amplitude_factor": 0.01,
            },
        ]

        star = Star(pulsations=pulse_meta, **STAR_PARAMS)
        system = SingleSystem(star=star, **SYSTEM_PARAMS)

        mode = system.star.pulsations[0]
        mode.complex_displacement = np.array(
            [
                [1.0 + 1.0j, 0.1 + 0.05j],
                [0.9 + 1.1j, 0.15 + 0.1j],
            ],
        )
        mode.radial_amplitude = 0.1

        mode.complex_radial_displacement = mode.complex_displacement[:, 0]
        result_1 = kinematics.calculate_temperature_pert_factor(mode, scale=1.0)
        result_2 = kinematics.calculate_temperature_pert_factor(mode, scale=2.0)

        # Different scales should give different results
        assert not np.allclose(result_1, result_2)

        # Ratio should match scale ratio
        assert_array_almost_equal(result_2, 2.0 * result_1)

    def test_temperature_perturbation_phase_shift(self) -> None:
        """Test that phase shift affects temperature perturbation."""
        pulse_meta_1 = [
            {
                "l": 2,
                "m": 1,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
                "temperature_perturbation_phase_shift": 0.0,
                "horizontal_to_radial_amplitude_ratio": 0.5,
                "temperature_amplitude_factor": 0.01,
            },
        ]

        pulse_meta_2 = [
            {
                "l": 2,
                "m": 1,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
                "temperature_perturbation_phase_shift": np.pi / 4,
                "horizontal_to_radial_amplitude_ratio": 0.5,
                "temperature_amplitude_factor": 0.01,
            },
        ]

        star_1 = Star(pulsations=pulse_meta_1, **STAR_PARAMS)
        system_1 = SingleSystem(star=star_1, **SYSTEM_PARAMS)

        star_2 = Star(pulsations=pulse_meta_2, **STAR_PARAMS)
        system_2 = SingleSystem(star=star_2, **SYSTEM_PARAMS)

        mode_1 = system_1.star.pulsations[0]
        mode_2 = system_2.star.pulsations[0]

        displacement = np.array(
            [
                [1.0 + 1.0j, 0.1 + 0.05j],
                [0.9 + 1.1j, 0.15 + 0.1j],
            ],
        )

        mode_1.complex_displacement = displacement
        mode_2.complex_displacement = displacement
        mode_1.complex_radial_displacement = displacement[:, 0]
        mode_2.complex_radial_displacement = displacement[:, 0]
        mode_1.radial_amplitude = 0.1
        mode_2.radial_amplitude = 0.1

        result_1 = kinematics.calculate_temperature_pert_factor(mode_1, scale=1.0)
        result_2 = kinematics.calculate_temperature_pert_factor(mode_2, scale=1.0)

        # Different phase shifts should give different results
        assert not np.allclose(result_1, result_2)
