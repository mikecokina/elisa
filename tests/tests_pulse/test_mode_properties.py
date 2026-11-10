import numpy as np
import pytest

from elisa import const
from elisa import units as u
from elisa.base.error import ElisaParametersError
from tests.utils import ElisaTestCase


class TestPulsationModeProperties(ElisaTestCase):
    """Unit tests for elisa.pulse.transform.PulsationModeProperties validation methods."""

    def setUp(self) -> None:
        super().setUp()
        from elisa.pulse.transform import PulsationModeProperties

        self.props = PulsationModeProperties

    def test_l_valid_int(self) -> None:
        """Test l property with valid integer input."""
        result = self.props.l(2)
        assert result == 2
        assert isinstance(result, (int, np.integer))

    def test_l_valid_float_convertible_to_int(self) -> None:
        """Test l property with float that can be converted to int."""
        result = self.props.l(2.0)
        assert result == 2
        assert isinstance(result, (int, np.integer))

    def test_l_valid_numpy_int(self) -> None:
        """Test l property with numpy integer."""
        result = self.props.l(np.int32(3))
        assert result == 3

    def test_l_invalid_float(self) -> None:
        """Test l property raises ElisaParametersError for non-integer float."""
        with pytest.raises(ElisaParametersError):
            self.props.l(2.5)

    def test_l_invalid_string(self) -> None:
        """Test l property raises ElisaParametersError for string input."""
        with pytest.raises(ElisaParametersError):
            self.props.l("2")

    def test_m_valid_int(self) -> None:
        """Test m property with valid integer input."""
        result = self.props.m(1)
        assert result == 1
        assert isinstance(result, (int, np.integer))

    def test_m_valid_float_convertible_to_int(self) -> None:
        """Test m property with float that can be converted to int."""
        result = self.props.m(1.0)
        assert result == 1
        assert isinstance(result, (int, np.integer))

    def test_m_valid_numpy_int(self) -> None:
        """Test m property with numpy integer."""
        result = self.props.m(np.int64(-1))
        assert result == -1

    def test_m_invalid_float(self) -> None:
        """Test m property raises ElisaParametersError for non-integer float."""
        with pytest.raises(ElisaParametersError):
            self.props.m(1.5)

    def test_m_invalid_string(self) -> None:
        """Test m property raises ElisaParametersError for string input."""
        with pytest.raises(ElisaParametersError):
            self.props.m("1")

    def test_amplitude_valid_float(self) -> None:
        """Test amplitude property with valid float input."""
        result = self.props.amplitude(50.0)
        self.assertAlmostEqual(result, 50.0)

    def test_amplitude_valid_int(self) -> None:
        """Test amplitude property with valid integer input."""
        result = self.props.amplitude(50)
        self.assertAlmostEqual(result, 50.0)

    def test_amplitude_valid_quantity(self) -> None:
        """Test amplitude property with astropy Quantity."""
        value = 1.0 * u.km / u.s
        result = self.props.amplitude(value)
        assert np.isfinite(result)
        self.assertAlmostEqual(result, np.float64(value.to(u.VELOCITY_UNIT).value))

    def test_amplitude_valid_quantity_string(self) -> None:
        """Test amplitude property with quantity string."""
        result = self.props.amplitude("1 km / s")
        assert np.isfinite(result)

    def test_amplitude_negative_raises_error(self) -> None:
        """Test amplitude property raises ElisaParametersError for negative values."""
        with pytest.raises(ElisaParametersError):
            self.props.amplitude(-10.0)

    def test_amplitude_invalid_type(self) -> None:
        """Test amplitude property raises ElisaParametersError for invalid type."""
        with pytest.raises(ElisaParametersError):
            self.props.amplitude("invalid")

    def test_frequency_valid_float(self) -> None:
        """Test frequency property with valid float input (assumes default unit)."""
        result = self.props.frequency(1.0)
        assert np.isfinite(result)

    def test_frequency_valid_int(self) -> None:
        """Test frequency property with valid integer input."""
        result = self.props.frequency(2)
        assert np.isfinite(result)

    def test_frequency_valid_quantity(self) -> None:
        """Test frequency property with astropy Quantity."""
        value = 16.0 / u.d
        result = self.props.frequency(value)
        assert np.isfinite(result)
        self.assertAlmostEqual(result, np.float64(value.to(u.FREQUENCY_UNIT).value))

    def test_frequency_valid_quantity_string(self) -> None:
        """Test frequency property with frequency quantity string."""
        result = self.props.frequency("1 Hz")
        assert np.isfinite(result)

    def test_frequency_negative_raises_error(self) -> None:
        """Test frequency property raises ElisaParametersError for negative values."""
        with pytest.raises(ElisaParametersError):
            self.props.frequency(-1.0)

    def test_frequency_invalid_type(self) -> None:
        """Test frequency property raises ElisaParametersError for invalid type."""
        with pytest.raises(ElisaParametersError):
            self.props.frequency("invalid_freq")

    def test_start_phase_zero(self) -> None:
        """Test start_phase property with zero value."""
        result = self.props.start_phase(0.0)
        self.assertAlmostEqual(result, 0.0)

    def test_start_phase_valid_float(self) -> None:
        """Test start_phase property with valid float input."""
        result = self.props.start_phase(0.75)
        assert np.isfinite(result)

    def test_start_phase_valid_quantity(self) -> None:
        """Test start_phase property with astropy Quantity."""
        value = 45 * u.deg
        result = self.props.start_phase(value)
        assert np.isfinite(result)

    def test_mode_axis_theta_valid(self) -> None:
        """Test mode_axis_theta property with valid value in range (0, pi)."""
        result = self.props.mode_axis_theta(45.0)  # 45 degrees
        assert np.isfinite(result)
        assert 0 <= result < const.PI

    def test_mode_axis_theta_valid_quantity(self) -> None:
        """Test mode_axis_theta property with quantity."""
        value = 45 * u.deg
        result = self.props.mode_axis_theta(value)
        assert np.isfinite(result)
        assert 0 <= result < const.PI

    def test_mode_axis_theta_zero(self) -> None:
        """Test mode_axis_theta property with zero (boundary value)."""
        result = self.props.mode_axis_theta(0.0)
        self.assertAlmostEqual(result, 0.0)

    def test_mode_axis_theta_below_pi(self) -> None:
        """Test mode_axis_theta property with value just below pi."""
        result = self.props.mode_axis_theta(179.9)
        assert 0 <= result < const.PI

    def test_mode_axis_theta_above_pi_raises_error(self) -> None:
        """Test mode_axis_theta property raises ElisaParametersError for value >= pi."""
        with pytest.raises(ElisaParametersError):
            self.props.mode_axis_theta(180.0)  # 180 degrees = pi radians

    def test_mode_axis_theta_negative_raises_error(self) -> None:
        """Test mode_axis_theta property raises ElisaParametersError for negative values."""
        with pytest.raises(ElisaParametersError):
            self.props.mode_axis_theta(-45.0)

    def test_mode_axis_phi_valid(self) -> None:
        """Test mode_axis_phi property with valid value."""
        result = self.props.mode_axis_phi(45.0)
        assert np.isfinite(result)

    def test_mode_axis_phi_zero(self) -> None:
        """Test mode_axis_phi property with zero value."""
        result = self.props.mode_axis_phi(0.0)
        self.assertAlmostEqual(result, 0.0)

    def test_mode_axis_phi_valid_quantity(self) -> None:
        """Test mode_axis_phi property with quantity."""
        value = 90 * u.deg
        result = self.props.mode_axis_phi(value)
        assert np.isfinite(result)

    def test_temperature_perturbation_phase_shift_valid(self) -> None:
        """Test temperature_perturbation_phase_shift with valid value."""
        result = self.props.temperature_perturbation_phase_shift(0.5)
        assert np.isfinite(result)

    def test_temperature_perturbation_phase_shift_zero(self) -> None:
        """Test temperature_perturbation_phase_shift with zero."""
        result = self.props.temperature_perturbation_phase_shift(0.0)
        assert np.isfinite(result)

    def test_temperature_perturbation_phase_shift_with_quantity(self) -> None:
        """Test temperature_perturbation_phase_shift with quantity."""
        value = 30 * u.deg
        result = self.props.temperature_perturbation_phase_shift(value)
        assert np.isfinite(result)

    def test_horizontal_to_radial_amplitude_ratio_valid_float(self) -> None:
        """Test horizontal_to_radial_amplitude_ratio with valid float."""
        result = self.props.horizontal_to_radial_amplitude_ratio(0.5)
        self.assertAlmostEqual(result, 0.5)

    def test_horizontal_to_radial_amplitude_ratio_zero(self) -> None:
        """Test horizontal_to_radial_amplitude_ratio with zero."""
        result = self.props.horizontal_to_radial_amplitude_ratio(0.0)
        self.assertAlmostEqual(result, 0.0)

    def test_horizontal_to_radial_amplitude_ratio_large_value(self) -> None:
        """Test horizontal_to_radial_amplitude_ratio with large value."""
        result = self.props.horizontal_to_radial_amplitude_ratio(2.5)
        self.assertAlmostEqual(result, 2.5)

    def test_horizontal_to_radial_amplitude_ratio_int(self) -> None:
        """Test horizontal_to_radial_amplitude_ratio with integer."""
        result = self.props.horizontal_to_radial_amplitude_ratio(1)
        self.assertAlmostEqual(result, 1.0)

    def test_horizontal_to_radial_amplitude_ratio_numpy_float(self) -> None:
        """Test horizontal_to_radial_amplitude_ratio with numpy float."""
        result = self.props.horizontal_to_radial_amplitude_ratio(np.float64(0.75))
        self.assertAlmostEqual(result, 0.75)

    def test_horizontal_to_radial_amplitude_ratio_invalid_type(self) -> None:
        """Test horizontal_to_radial_amplitude_ratio raises ElisaParametersError for invalid type."""
        with pytest.raises(ElisaParametersError):
            self.props.horizontal_to_radial_amplitude_ratio("0.5")

    def test_tidally_locked_true(self) -> None:
        """Test tidally_locked property with True."""
        result = self.props.tidally_locked(True)
        assert result
        assert isinstance(result, bool)

    def test_tidally_locked_false(self) -> None:
        """Test tidally_locked property with False."""
        result = self.props.tidally_locked(False)
        assert not result
        assert isinstance(result, bool)

    def test_tidally_locked_invalid_int(self) -> None:
        """Test tidally_locked property raises ElisaParametersError for integer."""
        with pytest.raises(ElisaParametersError):
            self.props.tidally_locked(1)

    def test_tidally_locked_invalid_string(self) -> None:
        """Test tidally_locked property raises ElisaParametersError for string."""
        with pytest.raises(ElisaParametersError):
            self.props.tidally_locked("True")

    def test_tidally_locked_invalid_none(self) -> None:
        """Test tidally_locked property raises ElisaParametersError for None."""
        with pytest.raises(ElisaParametersError):
            self.props.tidally_locked(None)
