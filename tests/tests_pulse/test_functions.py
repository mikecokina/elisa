from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from elisa import units as u
from elisa import utils
from elisa.base.error import PulsationModeError
from elisa.base.star import Star
from elisa.pulse import pulsations
from elisa.pulse import utils as putils
from elisa.single_system.system import SingleSystem
from tests import utils as testutils
from tests.tests_pulse.helpers import STAR_PARAMS, SYSTEM_PARMAS
from tests.utils import ElisaTestCase


class TestPulsationsFunctions(ElisaTestCase):
    """Unit tests for elisa.pulse.pulsations module functions."""

    def setUp(self) -> None:
        super().setUp()
        self.base_path = str(Path(__file__).resolve().parent.parent)
        self.lc_base_path = str(Path(__file__).resolve().parent.joinpath("data", "light_curves"))

    def test_spherical_harmonics_basic(self) -> None:
        """Test spherical_harmonics calculates harmonics correctly."""
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
        system_container = testutils.prepare_single_system_container(single)
        system_container.build_mesh()
        points = utils.cartesian_to_spherical(system_container.star.points)

        mode = single.star.pulsations[0]
        time = 0.0
        exponential = putils.generate_time_exponential(mode, time)

        result = pulsations.spherical_harmonics(mode, points, exponential)

        # Result should have same length as points
        assert result.shape[0] == points.shape[0]

        # Result should be finite (no NaNs or infs)
        assert np.all(np.isfinite(result))

        # Result should be complex
        assert np.iscomplexobj(result)

    def test_spherical_harmonics_with_custom_degree(self) -> None:
        """Test spherical_harmonics with custom degree parameter."""
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
        system_container = testutils.prepare_single_system_container(single)
        system_container.build_mesh()
        points = utils.cartesian_to_spherical(system_container.star.points)

        mode = single.star.pulsations[0]
        time = 0.0
        exponential = putils.generate_time_exponential(mode, time)

        # Call with custom degree
        result = pulsations.spherical_harmonics(mode, points, exponential, degree=2)

        # Result should be finite
        assert np.all(np.isfinite(result))

    def test_spherical_harmonics_with_custom_order(self) -> None:
        """Test spherical_harmonics with custom order parameter."""
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
        system_container = testutils.prepare_single_system_container(single)
        system_container.build_mesh()
        points = utils.cartesian_to_spherical(system_container.star.points)

        mode = single.star.pulsations[0]
        time = 0.0
        exponential = putils.generate_time_exponential(mode, time)

        # Call with custom order
        result = pulsations.spherical_harmonics(mode, points, exponential, order=2)

        # Result should be finite
        assert np.all(np.isfinite(result))

    def test_diff_spherical_harmonics_by_phi_basic(self) -> None:
        """Test diff_spherical_harmonics_by_phi calculates derivative correctly."""
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
        system_container = testutils.prepare_single_system_container(single)
        system_container.build_mesh()
        points = utils.cartesian_to_spherical(system_container.star.points)

        mode = single.star.pulsations[0]
        time = 0.0
        exponential = putils.generate_time_exponential(mode, time)

        harmonics_0 = pulsations.spherical_harmonics(mode, points, exponential)
        harmonics_1 = pulsations.spherical_harmonics(mode, points, exponential, order=mode.m + 1)

        result = pulsations.diff_spherical_harmonics_by_phi(mode, [harmonics_0, harmonics_1])

        # Result should have same length as points
        assert result.shape[0] == points.shape[0]

        # Result should be finite
        assert np.all(np.isfinite(result))

        # Result should be complex
        assert np.iscomplexobj(result)

    def test_diff_spherical_harmonics_by_phi_zero_m(self) -> None:
        """Test diff_spherical_harmonics_by_phi with m=0."""
        pulse_meta = [
            {
                "l": 2,
                "m": 0,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
            },
        ]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = testutils.prepare_single_system_container(single)
        system_container.build_mesh()
        points = utils.cartesian_to_spherical(system_container.star.points)

        mode = single.star.pulsations[0]
        time = 0.0
        exponential = putils.generate_time_exponential(mode, time)

        harmonics_0 = pulsations.spherical_harmonics(mode, points, exponential)
        harmonics_1 = pulsations.spherical_harmonics(mode, points, exponential, order=mode.m + 1)

        result = pulsations.diff_spherical_harmonics_by_phi(mode, [harmonics_0, harmonics_1])

        # For m=0, derivative should be zero
        assert_almost_equal(result, np.zeros_like(result))

    def test_assign_amplitudes_basic(self) -> None:
        """Test assign_amplitudes assigns correct amplitude values."""
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

        pulsations.assign_amplitudes(system_container.star)

        mode = system_container.star.pulsations[0]

        # Radial amplitude should be assigned and positive
        assert mode.radial_amplitude is not None
        assert mode.radial_amplitude > 0.0

        # Horizontal amplitude should be assigned and positive
        assert mode.horizontal_amplitude is not None
        assert mode.horizontal_amplitude > 0.0

        # Temperature amplitude factor should be assigned
        assert mode.temperature_amplitude_factor is not None

    def test_assign_amplitudes_with_custom_ratio(self) -> None:
        """Test assign_amplitudes respects custom horizontal_to_radial_amplitude_ratio."""
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

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = SingleSystem.build_container(single, phase=0.0)

        pulsations.assign_amplitudes(system_container.star)

        mode = system_container.star.pulsations[0]

        # Check that ratio is preserved
        if mode.radial_amplitude > 0:
            ratio = mode.horizontal_amplitude / mode.radial_amplitude
            assert ratio == pytest.approx(0.5, rel=0.0, abs=5e-05)

    def test_assign_amplitudes_normalization_constant(self) -> None:
        """Test assign_amplitudes with custom normalization constant."""
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

        # Call with custom normalization constant
        pulsations.assign_amplitudes(system_container.star, normalization_constant=2.0)

        mode = system_container.star.pulsations[0]

        # Amplitudes should be assigned
        assert mode.radial_amplitude is not None
        assert mode.horizontal_amplitude is not None

    def test_temp_amplitude_basic(self) -> None:
        """Test temp_amplitude calculates temperature amplitude."""
        pulse_meta = [
            {
                "l": 2,
                "m": 1,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": 0.5,
            },
        ]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        mode = single.star.pulsations[0]

        result = pulsations.temp_amplitude(mode)

        # Result should be finite
        assert np.isfinite(result)

    def test_temp_amplitude_radial_mode(self) -> None:
        """Test temp_amplitude with radial mode (l=0, m=0)."""
        pulse_meta = [
            {
                "l": 0,
                "m": 0,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": 0.0,
                "temperature_amplitude_factor": 0.01,  # Must provide this for radial modes
            },
        ]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        mode = single.star.pulsations[0]

        with pytest.raises(PulsationModeError):
            pulsations.temp_amplitude(mode)

    def test_temp_amplitude_different_ratios(self) -> None:
        """Test temp_amplitude with different amplitude ratios."""
        for ratio in [0.1, 0.5, 1.0, 2.0]:
            pulse_meta = [
                {
                    "l": 2,
                    "m": 1,
                    "amplitude": 1.0 * u.m / u.s,
                    "frequency": 1.0 / u.d,
                    "start_phase": 0.0,
                    "horizontal_to_radial_amplitude_ratio": ratio,
                },
            ]

            single = SingleSystem(
                star=Star(pulsations=pulse_meta, **STAR_PARAMS),
                **SYSTEM_PARMAS,
            )
            mode = single.star.pulsations[0]

            result = pulsations.temp_amplitude(mode)

            # Result should be finite
            assert np.isfinite(result)

    def test_assign_amplitudes_radial_mode_error(self) -> None:
        """Test assign_amplitudes raises error for radial modes without temperature_amplitude_factor."""
        pulse_meta = [
            {
                "l": 0,
                "m": 0,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
            },
        ]

        # Should raise PulsationModeError during SingleSystem initialization for radial modes
        # without temperature_amplitude_factor
        with pytest.raises(PulsationModeError):
            SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
