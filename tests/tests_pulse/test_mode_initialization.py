from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
import pytest

from elisa import const, settings
from elisa import units as u
from elisa.base.error import ElisaInvalidKwargsError, ElisaMissingKwargsError, ElisaParametersError
from elisa.pulse.mode import PulsationMode
from elisa.pulse.transform import PulsationModeProperties
from tests.utils import ElisaTestCase

if TYPE_CHECKING:
    import decimal
    import fractions


class TestPulsationModeInitialization(ElisaTestCase):
    """Unit tests for elisa.pulse.mode.PulsationMode class initialization and properties."""

    def test_init_with_mandatory_params_only(self) -> None:
        """Test PulsationMode initialization with mandatory parameters only."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        assert mode.l == 2
        assert mode.m == 1
        self.assertAlmostEqual(mode.amplitude, 1.0)
        assert np.isfinite(mode.frequency)

    def test_init_with_all_parameters(self) -> None:
        """Test PulsationMode initialization with all parameters."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            start_phase=0.5,
            mode_axis_theta=np.pi / 4,
            mode_axis_phi=np.pi / 3,
            temperature_perturbation_phase_shift=0.1,
            horizontal_to_radial_amplitude_ratio=0.5,
            temperature_amplitude_factor=0.01,
            tidally_locked=True,
        )

        assert mode.l == 2
        assert mode.m == 1
        self.assertAlmostEqual(mode.start_phase, 0.5)
        assert 0 <= mode.mode_axis_theta < const.PI
        assert np.isfinite(mode.mode_axis_phi)
        self.assertAlmostEqual(mode.horizontal_to_radial_amplitude_ratio, 0.5)
        self.assertAlmostEqual(mode.temperature_amplitude_factor, 0.01)
        assert mode.tidally_locked

    def test_init_missing_mandatory_param_l(self) -> None:
        """Test PulsationMode raises error when l is missing."""
        with pytest.raises(ElisaMissingKwargsError):
            PulsationMode(
                m=1,
                amplitude=1.0 * u.m / u.s,
                frequency=1.0 / u.d,
            )

    def test_init_missing_mandatory_param_m(self) -> None:
        """Test PulsationMode raises error when m is missing."""
        with pytest.raises(ElisaMissingKwargsError):
            PulsationMode(
                l=2,
                amplitude=1.0 * u.m / u.s,
                frequency=1.0 / u.d,
            )

    def test_init_missing_mandatory_param_amplitude(self) -> None:
        """Test PulsationMode raises error when amplitude is missing."""
        with pytest.raises(ElisaMissingKwargsError):
            PulsationMode(
                l=2,
                m=1,
                frequency=1.0 / u.d,
            )

    def test_init_missing_mandatory_param_frequency(self) -> None:
        """Test PulsationMode raises error when frequency is missing."""
        with pytest.raises(ElisaMissingKwargsError):
            PulsationMode(
                l=2,
                m=1,
                amplitude=1.0 * u.m / u.s,
            )

    def test_init_invalid_kwarg(self) -> None:
        """Test PulsationMode raises error for invalid keyword argument."""
        with pytest.raises(ElisaInvalidKwargsError):
            PulsationMode(
                l=2,
                m=1,
                amplitude=1.0 * u.m / u.s,
                frequency=1.0 / u.d,
                invalid_param=42,
            )

    def test_angular_frequency_calculation(self) -> None:
        """Test that angular_frequency is correctly calculated from frequency."""
        freq = 1.0 / u.d
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=freq,
        )

        expected_angular_freq = const.FULL_ARC * float(freq.to(u.FREQUENCY_UNIT).value)
        self.assertAlmostEqual(mode.angular_frequency, expected_angular_freq)

    def test_renorm_constant_value(self) -> None:
        """Test that renorm_const is correctly set."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        expected_renorm = 2 * const.PI**0.5
        self.assertAlmostEqual(mode.renorm_const, expected_renorm)

    def test_validate_mode_valid(self) -> None:
        """Test mode validation with valid parameters (|m| <= l)."""
        # Should not raise any error
        mode = PulsationMode(
            l=2,
            m=2,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )
        assert mode.l == 2
        assert mode.m == 2

    def test_validate_mode_m_equals_l(self) -> None:
        """Test mode validation when |m| = l."""
        mode = PulsationMode(
            l=3,
            m=3,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )
        assert mode.l == 3
        assert mode.m == 3

    def test_validate_mode_negative_m(self) -> None:
        """Test mode validation with negative m within valid range."""
        mode = PulsationMode(
            l=2,
            m=-1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )
        assert mode.l == 2
        assert mode.m == -1

    def test_validate_mode_invalid_m_exceeds_l(self) -> None:
        """Test mode validation raises error when |m| > l."""
        with pytest.raises(ElisaParametersError):
            PulsationMode(
                l=2,
                m=3,
                amplitude=1.0 * u.m / u.s,
                frequency=1.0 / u.d,
            )

    def test_validate_mode_invalid_negative_m_exceeds_l(self) -> None:
        """Test mode validation raises error when |m| > l with negative m."""
        with pytest.raises(ElisaParametersError):
            PulsationMode(
                l=2,
                m=-4,
                amplitude=1.0 * u.m / u.s,
                frequency=1.0 / u.d,
            )

    def test_default_input_units_property(self) -> None:
        """Test that default_input_units property returns correct units."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        input_units = mode.default_input_units
        assert input_units is not None

    def test_default_internal_units_property(self) -> None:
        """Test that default_internal_units property returns correct units."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        internal_units = mode.default_internal_units
        assert internal_units is not None

    def test_default_values_assigned(self) -> None:
        """Test that default values are correctly assigned to optional parameters."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        # Check default values
        self.assertAlmostEqual(mode.start_phase, 0.0)
        self.assertAlmostEqual(mode.mode_axis_theta, 0.0)
        self.assertAlmostEqual(mode.mode_axis_phi, 0.0)
        self.assertAlmostEqual(
            mode.temperature_perturbation_phase_shift,
            settings.DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT,
        )
        assert mode.horizontal_to_radial_amplitude_ratio is None
        assert not mode.tidally_locked
        assert mode.temperature_amplitude_factor is None

    def test_amplitude_initialization_with_quantity(self) -> None:
        """Test amplitude initialization with astropy Quantity."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.km / u.s,
            frequency=1.0 / u.d,
        )

        assert np.isfinite(mode.amplitude)
        # 1 km/s = 1000 m/s internally
        self.assertAlmostEqual(mode.amplitude, 1000.0)

    def test_amplitude_initialization_with_string(self) -> None:
        """Test amplitude initialization with string quantity."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude="1 km / s",
            frequency=1.0 / u.d,
        )

        assert np.isfinite(mode.amplitude)
        # 1 km/s = 1000 m/s internally
        self.assertAlmostEqual(mode.amplitude, 1000.0)

    def test_frequency_initialization_with_string(self) -> None:
        """Test frequency initialization with string quantity."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency="1 Hz",
        )

        assert np.isfinite(mode.frequency)

    def test_l_and_m_as_floats(self) -> None:
        """Test that l and m can be initialized as floats and converted to int."""
        mode = PulsationMode(
            l=2.0,
            m=1.0,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        assert mode.l == 2
        assert mode.m == 1
        assert isinstance(mode.l, (int, np.integer))
        assert isinstance(mode.m, (int, np.integer))

    def test_mode_axis_theta_range_validation(self) -> None:
        """Constructor angles become internal radians in the valid range [0, pi)."""
        expected_theta = 7 * np.pi / 20  # 63 degrees, not a symmetry-axis angle.
        for theta in (63.0, 63.0 * u.deg, expected_theta * u.rad):
            with self.subTest(theta=theta):
                mode = PulsationMode(
                    l=2,
                    m=1,
                    amplitude=1.0 * u.m / u.s,
                    frequency=1.0 / u.d,
                    mode_axis_theta=theta,
                )
                self.assertAlmostEqual(mode.mode_axis_theta, expected_theta)
                assert 0 <= mode.mode_axis_theta < const.PI

    def test_amplitude_negative_raises_error(self) -> None:
        """Test that negative amplitude raises error."""
        with pytest.raises(ElisaParametersError):
            PulsationMode(
                l=2,
                m=1,
                amplitude=-1.0 * u.m / u.s,
                frequency=1.0 / u.d,
            )

    def test_frequency_negative_raises_error(self) -> None:
        """Test that negative frequency raises error."""
        with pytest.raises(ElisaParametersError):
            PulsationMode(
                l=2,
                m=1,
                amplitude=1.0 * u.m / u.s,
                frequency=-1.0 / u.d,
            )

    def test_radial_and_horizontal_amplitude_initialization(self) -> None:
        """Test that radial and horizontal amplitude are None after init."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        assert mode.radial_amplitude is None
        assert mode.horizontal_amplitude is None

    def test_points_initialization(self) -> None:
        """Test that points are None after initialization."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        assert mode.points is None
        assert mode.point_harmonics is None
        assert mode.point_harmonics_derivatives is None
        assert mode.complex_displacement is None

    def test_tilt_angles_initialization(self) -> None:
        """Test that tilt angles are None after initialization."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        assert mode.tilt_phi is None
        assert mode.tilt_theta is None

    def test_radial_mode_l_equals_zero(self) -> None:
        """Test radial mode with l=0."""
        mode = PulsationMode(
            l=0,
            m=0,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        assert mode.l == 0
        assert mode.m == 0

    def test_high_degree_mode(self) -> None:
        """Test mode with high degree value."""
        mode = PulsationMode(
            l=10,
            m=5,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        assert mode.l == 10
        assert mode.m == 5

    def test_mode_with_custom_start_phase(self) -> None:
        """Test mode initialization with custom start_phase."""
        start_phase = 1.5
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            start_phase=start_phase,
        )

        self.assertAlmostEqual(mode.start_phase, start_phase)

    def test_mode_with_custom_horizontal_ratio(self) -> None:
        """Test mode initialization with custom horizontal_to_radial_amplitude_ratio."""
        ratio = 0.75
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            horizontal_to_radial_amplitude_ratio=ratio,
        )

        self.assertAlmostEqual(mode.horizontal_to_radial_amplitude_ratio, ratio)

    def test_mode_with_temperature_amplitude_factor(self) -> None:
        """Test mode initialization with temperature_amplitude_factor."""
        temp_factor = 0.01
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            temperature_amplitude_factor=temp_factor,
        )

        self.assertAlmostEqual(mode.temperature_amplitude_factor, temp_factor)


@pytest.mark.parametrize("value", [-1, -1.0, np.int32(-2), Decimal(-1), Fraction(-1, 1)])
def test_degree_transform_rejects_negative_integers(
    value: decimal.Decimal | float | fractions.Fraction | np.int32,
) -> None:
    with pytest.raises(ElisaParametersError, match="must be non-negative"):
        PulsationModeProperties.l(value)


def test_degree_zero_and_negative_azimuthal_order_remain_valid() -> None:
    mode = {"l": 1, "m": 1, "frequency": 16.0, "amplitude": 50.0}
    assert PulsationModeProperties.l(0) == 0
    assert PulsationModeProperties.m(-1) == -1
    assert PulsationMode(**mode | {"l": 0, "m": 0}).l == 0
    assert PulsationMode(**mode | {"m": -1}).m == -1


@pytest.mark.parametrize("operation", ["harmonics", "displacement"])
def test_pulsations_require_caller_to_flatten_container(operation: str) -> None:
    from elisa.base.container import StarContainer
    from elisa.base.error import ElisaUnexpectedError
    from elisa.pulse import container_ops

    star = StarContainer()
    star._flatten = False
    with pytest.raises(ElisaUnexpectedError, match="has not been flattened"):  # noqa: PT012 - scenario includes context cleanup or selection of the operation under test
        if operation == "harmonics":
            container_ops.generate_harmonics(star, 0.0, 0.0, 0.0)
        else:
            container_ops.complex_displacement(star, 1.0)
