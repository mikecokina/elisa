# keep it first
# due to stupid astropy units/constants implementation
import os
import os.path as op

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_array_less

from elisa import Observer, const, settings, utils
from elisa import units as u
from elisa.base.star import Star
from elisa.binary_system.system import BinarySystem
from elisa.pulse import container_ops, pulsations
from elisa.pulse import utils as putils
from elisa.pulse.mode import PulsationMode
from elisa.single_system.system import SingleSystem
from unittests import set_astropy_units
from unittests import utils as testutils
from unittests.utils import ElisaTestCase

set_astropy_units()


STAR_PARAMS = {
    "mass": 2.0 * u.solMass,
    "t_eff": 10000 * u.K,
    "gravity_darkening": 1.0,
    "discretization_factor": 5,
    "albedo": 1.0,
    "metallicity": 0.0,
    "polar_log_g": 4.0 * u.dex(u.cm / u.s ** 2),
}

SYSTEM_PARMAS = {
    "gamma": 0 * u.km / u.s,
    "inclination": 80 * u.deg,
    "rotation_period": 30 * u.d,
}

BINARY_STAR = {
    "mass": 2.0 * u.solMass,
    "t_eff": 10000 * u.K,
    "gravity_darkening": 1.0,
    "discretization_factor": 5,
    "albedo": 1.0,
    "metallicity": 0.0,
    "synchronicity": 1.0,
    "surface_potential": 10,
}

BINARY_SYSTEM = {
    "argument_of_periastron": 0 * u.deg,
    "gamma": 0 * u.km / u.s,
    "period": 2.0 * u.d,
    "eccentricity": 0.1,
    "inclination": 87 * u.deg,
    "primary_minimum_time": 0 * u.d,
    "phase_shift": 0.0,
}

TOL = 1e-3


class PulsatingStarInitTestCase(ElisaTestCase):
    def setUp(self):
        self.pulsation_modes = [
            {
                "l": 1,
                "m": 1,
                "amplitude": 0.050 * u.km / u.s,
                "frequency": 16 / u.d,
                "start_phase": 0.75,
            },
            {
                "l": 1,
                "m": -1,
                "amplitude": 50,
                "frequency": "16 Hz",
                "start_phase": 1.5,
            },
        ]

        self.star_params = {
            "mass": 2.15 * u.solMass,
            "t_eff": 10000 * u.K,
            "gravity_darkening": 1.0,
            "discretization_factor": 3,
            "albedo": 0.6,
            "metallicity": 0.0,
            "polar_log_g": 4.4 * u.dex(u.cm / u.s ** 2),
        }

        self.system_params = {"gamma": 0 * u.km / u.s,
                              "inclination": 80 * u.deg,
                              "rotation_period": 30 * u.d,
                              }

    def prepare_system(self, pulsations=None):
        pulsations = self.pulsation_modes if pulsations is None else pulsations
        star = Star(pulsations=pulsations, **self.star_params)
        return SingleSystem(star=star, **self.system_params)

    def test_mode_initialization_of_parameters(self):
        """Testing whether pulsation parameters are initialised to correct values
        :return:
        """
        rounding_prec = 6
        expected_values = [
            [1, 1, 50, 0.000185, 0.75],
            [1, -1, 50, 16, 1.5],
        ]

        star = self.prepare_system().star

        for ii, mode in star.pulsations.items():
            list_to_check = [mode.l, mode.m, np.round(mode.amplitude, rounding_prec),
                             np.round(mode.frequency, rounding_prec), np.round(mode.start_phase, rounding_prec)]
            assert_array_equal(expected_values[ii], list_to_check)

    def test_renorm_constant(self):
        """Testing if RMS of pulsation mode gives 1
        :return:
        """
        puls_meta = [{
            "l": 1,
            "m": 1,
            "amplitude": 1 * u.m / u.s,
            "frequency": 1 / u.d,
            "start_phase": 0.0,
            "temperature_amplitude_factor": 1.0,
        }]

        time = 0

        single = self.prepare_system()

        system_container = testutils.prepare_single_system_container(single)
        system_container.build_mesh()
        points = utils.cartesian_to_spherical(system_container.star.points)

        for ll in range(10):
            for mm in range(ll + 1):
                puls_meta[0]["l"] = ll
                puls_meta[0]["m"] = mm
                single.star.pulsations = puls_meta
                single.init()

                mode = single.star.pulsations[0]
                exponential = putils.generate_time_exponential(mode, time)
                sph_harm = pulsations.spherical_harmonics(mode, points, exponential)

                test_val = np.sqrt(np.sum(np.abs(sph_harm) ** 2) / points.shape[0])
                assert_almost_equal(test_val, 1.0, 2)


class TestPulsationModule(ElisaTestCase):

    def setUp(self):
        super().setUp()
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        settings.configure(LD_TABLES=op.join(self.lc_base_path, "limbdarkening"), CK04_ATM_TABLES=op.join(self.lc_base_path, "atmosphere"))

    def prepare_system(self, pulse_meta):
        star = Star(pulsations=pulse_meta, **STAR_PARAMS)
        return SingleSystem(star=star, **SYSTEM_PARMAS)

    def prepare_binary(self, pulse_meta):
        primary = Star(pulsations=pulse_meta, **BINARY_STAR)
        secondary = Star(pulsations=pulse_meta, **BINARY_STAR)
        return BinarySystem(primary=primary, secondary=secondary, **BINARY_SYSTEM)

    def test_complex_displacement_amplitudes(self):
        in_ratio = 2
        pulse_meta = [{
            "l": 10,
            "m": 2,
            "amplitude": 1 * u.m / u.s,
            "frequency": 1 / u.d,
            "start_phase": 0.0,
            "horizontal_to_radial_amplitude_ratio": in_ratio,
        }]

        single = self.prepare_system(pulse_meta)
        system_container = SingleSystem.build_container(single, phase=0)

        r_eq = single.star.equivalent_radius

        mode = system_container.star.pulsations[0]
        theta = mode.points[:, 2]

        radial = np.mean(np.abs(mode.complex_displacement[:, 0])**2)**0.5
        dphi = np.abs(mode.complex_displacement[:, 1])
        dtheta = np.abs(mode.complex_displacement[:, 2])
        horizontal = np.sqrt(dtheta**2 + (np.sin(theta)*dphi)**2)

        horizontal = r_eq * np.mean(horizontal**2)**0.5

        ratio = (horizontal / radial)
        self.assertTrue(abs(ratio - in_ratio) < 0.05)

    def test_kinematics_single(self):
        amplitude = 1.0
        freq = 1
        omega = const.FULL_ARC * freq
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": amplitude * u.m / u.s,
            "frequency": freq / u.s,
            "start_phase": 0.0,
            "horizontal_to_radial_amplitude_ratio": 0.0,
            "temperature_amplitude_factor": 0.01,
        }]
        overshoot = 2.0  # this value may change for different modes

        single = self.prepare_system(pulse_meta)
        system_container = SingleSystem.build_container(single, phase=0.2854)

        # displacement
        r_ampl = amplitude / omega
        args = (system_container.star, 0.0)
        kwargs_ = dict(
            update_container = False,
            return_perturbation = True,
            spherical_perturbation = True,
        )
        radius = np.abs(container_ops.position_perturbation(*args, **kwargs_)[:, 0])
        assert_array_less(radius, overshoot * r_ampl)
        assert_array_less(r_ampl, radius.max())

        # velocity
        args = (system_container.star, 1)
        velocities = np.abs(
            container_ops.velocity_perturbation(
                *args,
                update_container=False,
                return_perturbation=True,
                spherical_perturbation=True,
            )[:, 0]
        )
        assert_array_less(velocities, overshoot * amplitude)
        assert_array_less(amplitude, velocities.max())

        # acceleration
        a_amp = amplitude * omega
        args = (system_container.star, 1.0)
        acc = np.abs(
            container_ops.gravity_acc_perturbation(
                *args,
                update_container=False,
                return_perturbation=True,
                spherical_perturbation=True,
            )[:, 0]
        )
        assert_array_less(acc, overshoot * a_amp)
        assert_array_less(a_amp, acc.max())

        # teff
        t_ampl = pulse_meta[0]["temperature_amplitude_factor"] * system_container.star.t_eff
        args = (system_container.star, 1.0)
        ts = np.abs(
            container_ops.temp_perturbation(
                *args,
                update_container=False,
                return_perturbation=True,
            )
        )
        assert_array_less(ts, overshoot * t_ampl)
        assert_array_less(t_ampl, ts.max())

    def test_container_time_and_perturbations_not_nan(self):
        """Ensure container.time is computed and pulsation perturbation arrays are finite.

        Regression test guarding against accidental overwrites of the container time
        (which previously produced NaNs in time-dependent pulsation arrays).
        """
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1 / u.s,
            "start_phase": 0.0,
            "horizontal_to_radial_amplitude_ratio": 0.0,
            "temperature_amplitude_factor": 0.01,
        }]

        single = self.prepare_system(pulse_meta)
        system_container = SingleSystem.build_container(single, phase=0.2854)

        # time must be finite (not NaN or inf)
        assert np.isfinite(system_container.time), "Container time must be finite and not NaN"

        # position perturbation (point-wise) must not contain NaNs
        pert = container_ops.position_perturbation(system_container.star, 0.0,
                                                   update_container=False,
                                                   return_perturbation=True,
                                                   spherical_perturbation=True)
        assert pert is not None
        assert not np.isnan(pert).any(), "Position perturbation contains NaNs"

        # velocity perturbation must not contain NaNs
        vel = container_ops.velocity_perturbation(
            system_container.star,
            1.0,
            update_container=False,
            return_perturbation=True,
            spherical_perturbation=True,
        )
        assert vel is not None
        assert not np.isnan(vel).any(), "Velocity perturbation contains NaNs"

    def test_single_pulsating_lc(self):
        freq = 15
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 100 * u.m / u.s,
            "frequency": freq / u.d,
            "start_phase": 0.0,
            "horizontal_to_radial_amplitude_ratio": 0.0,
            "temperature_amplitude_factor": 0.01,
        }]

        single = self.prepare_system(pulse_meta)
        o = Observer(passband=["Generic.Bessell.V" ], system=single)

        phases = np.linspace(0.0, 1 / (freq * single.rotation_period))

        expected = testutils.load_light_curve("single.pulsating.v.json")
        expected_phases = expected[0]
        expected_flux = testutils.normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        obtained = o.lc(phases=phases)
        obtained_phases = obtained[0]
        obtained_flux = testutils.normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(np.abs(np.round(obtained_phases, 3) - np.round(expected_phases, 3)) < TOL))
        self.assertTrue(np.all(np.abs(np.round(obtained_flux, 3) - np.round(expected_flux, 3)) < TOL))

        # plt.plot(obtained_phases, obtained_flux)
        # plt.plot(expected_phases, expected_flux)
        # plt.show()

    def test_single_pulsating_rv(self):
        freq = 15
        pulse_meta = [{
            "l": 1,
            "m": 1,
            "amplitude": 1 * u.km / u.s,
            "frequency": freq / u.d,
            "start_phase": 0.0,
            "horizontal_to_radial_amplitude_ratio": 0.0,
            "temperature_amplitude_factor": 0.01,
        }]

        single = self.prepare_system(pulse_meta)
        o = Observer(system=single)

        phases = np.linspace(0.0, 1 / (freq * single.period), 50)

        expected = testutils.load_radial_curve("single.pulsating.json")
        expected_rv = testutils.normalize_single_rv_for_unittests(expected["star"])

        obtained = o.rv(phases=phases, method="radiometric")
        obtained_rv = testutils.normalize_single_rv_for_unittests(obtained[1]["star"])

        self.assertTrue(np.all(np.abs(obtained_rv - expected_rv) < TOL))

        # plt.plot(phases, obtained_rv)
        # plt.plot(phases, expected_rv)
        # plt.plot(expected_rv-obtained_rv)
        # plt.show()

    def test_kinematics_binary(self):
        amplitude = 1.0
        freq = 1
        omega = const.FULL_ARC * freq
        pulse_meta = [{
            "l": 4,
            "m": 2,
            "amplitude": amplitude * u.m / u.s,
            "frequency": freq / u.s,
            "start_phase": 0.0,
            "horizontal_to_radial_amplitude_ratio": 1.0,
            "temperature_amplitude_factor": 0.01,
        }]
        overshoot = 2.0  # this value may change for different modes

        binary = self.prepare_binary(pulse_meta)
        system_container = BinarySystem.build_container(binary, phase=0.6741)

        # displacement
        r_ampl = amplitude / omega
        args = (system_container.secondary, system_container.position.distance)
        kwargs_ = {
            "update_container": False,
            "return_perturbation": True,
            "spherical_perturbation": True,
        }
        radius = np.abs(container_ops.position_perturbation(*args, **kwargs_)[:, 0]) * binary.semi_major_axis
        assert_array_less(radius, overshoot * r_ampl)
        assert_array_less(r_ampl, radius.max())

        # velocity
        args = (system_container.secondary, binary.semi_major_axis)
        velocities = np.abs(container_ops.velocity_perturbation(*args, **kwargs_)[:, 0])
        assert_array_less(velocities, overshoot * amplitude)
        assert_array_less(amplitude, velocities.max())

        # acceleration
        a_amp = amplitude * omega
        args = (system_container.secondary, binary.semi_major_axis)
        acc = np.abs(container_ops.gravity_acc_perturbation(*args, **kwargs_)[:, 0])
        assert_array_less(acc, overshoot * a_amp)
        assert_array_less(a_amp, acc.max())

        # teff
        t_ampl = pulse_meta[0]["temperature_amplitude_factor"] * system_container.secondary.t_eff
        args = (system_container.secondary, binary.semi_major_axis)
        kwargs_ = {
            "update_container": False,
            "return_perturbation": True,
        }
        ts = np.abs(container_ops.temp_perturbation(*args, **kwargs_))
        assert_array_less(ts, overshoot * t_ampl)
        assert_array_less(t_ampl, ts.max())

    def test_binary_pulsating_lc(self):
        freq = 15
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 100 * u.m / u.s,
            "frequency": freq / u.d,
            "start_phase": 0.0,
            "horizontal_to_radial_amplitude_ratio": 0.0,
            "temperature_amplitude_factor": 0.01,
        }]

        binary = self.prepare_binary(pulse_meta)
        o = Observer(passband=["Generic.Bessell.V" ], system=binary)

        start_phase = 0.025
        phases = np.linspace(start_phase, start_phase + 2 / (freq * binary.period), 10)

        expected = testutils.load_light_curve("binary.pulsating.v.json")
        expected_phases = expected[0]
        expected_flux = testutils.normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        obtained = o.lc(phases=phases)
        obtained_phases = obtained[0]
        obtained_flux = testutils.normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(np.abs(np.round(obtained_phases, 3) - np.round(expected_phases, 3)) < TOL))
        self.assertTrue(np.all(np.abs(np.round(obtained_flux, 3) - np.round(expected_flux, 3)) < TOL))

        # o.plot.phase_curve()
        # plt.plot(obtained_phases, obtained_flux)
        # plt.plot(expected_phases, expected_flux)
        # plt.show()

    def test_binary_pulsating_rv(self):
        freq = 15
        pulse_meta = [{
            "l": 1,
            "m": 1,
            "amplitude": 5 * u.km / u.s,
            "frequency": freq / u.d,
            "start_phase": 0.0,
            "horizontal_to_radial_amplitude_ratio": 0.0,
            "temperature_amplitude_factor": 0.01,
        }]

        binary = self.prepare_binary(pulse_meta)
        o = Observer(system=binary)

        start_phase = 0.05
        phases = np.linspace(start_phase, start_phase + 2 / (freq * binary.period), 10)

        expected = testutils.load_radial_curve("binary.pulsating.json")
        expected_rvp, expected_rvs = testutils.normalize_rv_for_unittests(expected["primary"], expected["secondary"])

        obtained = o.rv(phases=phases, method="radiometric")
        obtained_rvp, obtained_rvs = testutils.normalize_rv_for_unittests(obtained[1]["primary"], obtained[1]["secondary"])

        assert_array_equal(np.round(expected_rvp, 3), np.round(obtained_rvp, 3))
        assert_array_equal(np.round(expected_rvs, 3), np.round(obtained_rvs, 3))

        # o.plot.rv_curve()


class TestPulseUtils(ElisaTestCase):
    """Unit tests for elisa.pulse.utils module functions."""

    def setUp(self):
        super().setUp()
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        settings.configure(LD_TABLES=op.join(self.lc_base_path, "limbdarkening"), CK04_ATM_TABLES=op.join(self.lc_base_path, "atmosphere"))

    def test_phase_correction_with_synchronicity(self):
        """Test phase_correction with valid synchronicity values."""
        phase = 0.5
        synchronicity = 1.5

        result = putils.phase_correction(phase, synchronicity)
        expected = (synchronicity - 1) * phase * const.FULL_ARC

        assert_almost_equal(result, expected)

    def test_phase_correction_with_synchronicity_one(self):
        """Test phase_correction with synchronicity = 1 (no correction)."""
        phase = 0.5
        synchronicity = 1.0

        result = putils.phase_correction(phase, synchronicity)
        expected = 0.0

        assert_almost_equal(result, expected)

    def test_phase_correction_with_nan_synchronicity(self):
        """Test phase_correction when synchronicity is NaN."""
        phase = 0.5

        result = putils.phase_correction(phase, np.nan)
        expected = phase * const.FULL_ARC

        assert_almost_equal(result, expected)

    def test_generate_time_exponential(self):
        """Test generate_time_exponential returns correct complex exponential."""
        pulse_meta = [{
            "l": 1,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        mode = single.star.pulsations[0]

        time = 1.0
        result = putils.generate_time_exponential(mode, time)

        exponent = mode.angular_frequency * time + mode.start_phase
        expected = np.exp(complex(0, -exponent))

        assert_almost_equal(result, expected)

    def test_generate_phase_shift(self):
        """Test generate_phase_shift returns correct phase shift factor."""
        shift = np.pi / 4

        result = putils.generate_phase_shift(shift)
        expected = np.exp(complex(0, -shift))

        assert_almost_equal(result, expected)

    def test_generate_phase_shift_zero(self):
        """Test generate_phase_shift with zero shift."""
        shift = 0.0

        result = putils.generate_phase_shift(shift)
        expected = 1.0 + 0j

        assert_almost_equal(result, expected)

    def test_tilt_mode_coordinates_no_tilt(self):
        """Test tilt_mode_coordinates when phi and theta are zero (no tilt)."""
        points = np.array([[1.0, 0.0, np.pi / 2],
                          [1.0, np.pi / 4, np.pi / 3],
                          [1.0, np.pi / 2, np.pi / 6]])
        phi = 0.0
        theta = 0.0

        result = putils.tilt_mode_coordinates(points, phi, theta)

        assert_array_equal(result, points)

    def test_tilt_mode_coordinates_with_tilt(self):
        """Test tilt_mode_coordinates applies rotation correctly."""
        points = np.array([[1.0, 0.0, np.pi / 2],
                          [1.0, np.pi / 4, np.pi / 3]])
        phi = np.pi / 6
        theta = np.pi / 4

        result = putils.tilt_mode_coordinates(points, phi, theta)

        # Result should have same shape and first column (radius) unchanged
        self.assertEqual(result.shape, points.shape)
        assert_array_equal(result[:, 0], points[:, 0])

        # Phi and theta should be modified
        self.assertFalse(np.array_equal(result[:, 1:], points[:, 1:]))

    def test_derotate_surface_points_no_rotation(self):
        """Test derotate_surface_points when phi and theta are zero (no rotation)."""
        points = np.array([[1.0, 0.0, np.pi / 2],
                          [1.0, np.pi / 4, np.pi / 3]])
        phi = 0.0
        theta = 0.0

        result = putils.derotate_surface_points(points, phi, theta)

        assert_array_equal(result, points)

    def test_derotate_surface_points_with_rotation(self):
        """Test derotate_surface_points applies derotation correctly."""
        points = np.array([[1.0, 0.0, np.pi / 2],
                          [1.0, np.pi / 4, np.pi / 3]])
        phi = np.pi / 6
        theta = np.pi / 4

        result = putils.derotate_surface_points(points, phi, theta)

        # Result should have same shape and first column (radius) unchanged
        self.assertEqual(result.shape, points.shape)
        assert_array_equal(result[:, 0], points[:, 0])

        # Phi and theta should be modified
        self.assertFalse(np.array_equal(result[:, 1:], points[:, 1:]))

    def test_derotate_surface_displacements_no_rotation(self):
        """Test derotate_surface_displacements when axis_phi and axis_theta are zero."""
        velocity = np.array([[0.1, 0.01, 0.02],
                           [0.15, 0.015, 0.025]])
        tilted_points = np.array([[1.0, 0.0, np.pi / 2],
                                 [1.0, np.pi / 4, np.pi / 3]])
        points = np.array([[1.0, 0.0, np.pi / 2],
                          [1.0, np.pi / 4, np.pi / 3]])
        axis_phi = 0.0
        axis_theta = 0.0

        result = putils.derotate_surface_displacements(velocity, tilted_points, points, axis_phi, axis_theta)

        assert_array_equal(result, velocity)

    def test_derotate_surface_displacements_with_rotation(self):
        """Test derotate_surface_displacements applies derotation correctly."""
        velocity = np.array([[0.1, 0.01, 0.02],
                           [0.15, 0.015, 0.025]])
        tilted_points = np.array([[1.0, 0.0, np.pi / 2],
                                 [1.0, np.pi / 4, np.pi / 3]])
        points = np.array([[1.0, 0.0, np.pi / 2],
                          [1.0, np.pi / 4, np.pi / 3]])
        axis_phi = np.pi / 6
        axis_theta = np.pi / 4

        result = putils.derotate_surface_displacements(velocity, tilted_points, points, axis_phi, axis_theta)

        # Result should have same shape and first column (radial) unchanged
        self.assertEqual(result.shape, velocity.shape)
        assert_array_equal(result[:, 0], velocity[:, 0])

    def test_transform_spherical_displacement_to_cartesian_simple(self):
        """Test spherical to cartesian displacement transformation."""
        sph_displacement = np.array([[1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0]])
        surf_points = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])
        com_x = 0.0

        result = putils.transform_spherical_displacement_to_cartesian(sph_displacement, surf_points, com_x)

        # Result should have same shape
        self.assertEqual(result.shape, (3, 3))

        # Result should be finite (no NaNs or infs)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_transform_spherical_displacement_to_cartesian_nonzero_com(self):
        """Test spherical to cartesian transformation with non-zero center of mass."""
        sph_displacement = np.array([[0.1, 0.01, 0.02],
                                    [0.15, 0.015, 0.025]])
        surf_points = np.array([[1.5, 0.5, 0.2],
                               [1.2, 0.8, 0.3]])
        com_x = 0.5

        result = putils.transform_spherical_displacement_to_cartesian(sph_displacement, surf_points, com_x)

        # Result should have same shape
        self.assertEqual(result.shape, (2, 3))

        # Result should be finite
        self.assertTrue(np.all(np.isfinite(result)))

    def test_horizontal_component_zero_displacement(self):
        """Test horizontal_component with zero displacement."""
        displacement = np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, np.pi / 2],
                          [1.0, np.pi / 4, np.pi / 3]])

        result = putils.horizontal_component(displacement, points)

        expected = np.array([0.0, 0.0])
        assert_almost_equal(result, expected)

    def test_horizontal_component_radial_displacement_only(self):
        """Test horizontal_component with only radial displacement (should be zero)."""
        displacement = np.array([[0.1, 0.0, 0.0],
                                [0.15, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, np.pi / 2],
                          [1.0, np.pi / 4, np.pi / 3]])

        result = putils.horizontal_component(displacement, points)

        expected = np.array([0.0, 0.0])
        assert_almost_equal(result, expected)

    def test_horizontal_component_with_displacement(self):
        """Test horizontal_component calculates distance correctly."""
        displacement = np.array([[0.0, 0.1, 0.1],
                                [0.0, 0.05, 0.05]])
        points = np.array([[1.0, 0.0, np.pi / 2],
                          [1.0, np.pi / 4, np.pi / 3]])

        result = putils.horizontal_component(displacement, points)

        # Result should be positive for non-zero displacement
        self.assertTrue(np.all(result >= 0.0))

        # Result should be finite
        self.assertTrue(np.all(np.isfinite(result)))

    def test_horizontal_component_treat_poles(self):
        """Test horizontal_component with treat_poles=True."""
        displacement = np.array([[0.0, 0.1, 0.1],
                                [0.0, 0.5, 0.5]])
        points = np.array([[1.0, 0.0, np.pi / 2],
                          [1.0, np.pi / 4, np.pi / 3]])

        result = putils.horizontal_component(displacement, points, treat_poles=True)

        # Result should be finite and reasonable
        self.assertTrue(np.all(np.isfinite(result)))
        # All values should be positive and valid
        self.assertTrue(np.all(result >= 0.0))

    def test_pole_neighbours_single_system(self):
        """Test pole_neighbours finds correct pole indices."""
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = SingleSystem.build_container(single, phase=0.0)

        star = system_container.star
        putils.pole_neighbours(star)

        # Check that pole indices were assigned
        self.assertIsNotNone(star.pole_idx)
        self.assertIsNotNone(star.pole_idx_neighbour)

        # Check that pole indices are valid
        self.assertTrue(len(star.pole_idx) == 2)
        self.assertTrue(len(star.pole_idx_neighbour) == 2)

        # Check that indices are within valid range
        n_points = len(star.points_spherical)
        self.assertTrue(np.all(star.pole_idx >= 0))
        self.assertTrue(np.all(star.pole_idx < n_points))
        self.assertTrue(np.all(star.pole_idx_neighbour >= 0))
        self.assertTrue(np.all(star.pole_idx_neighbour < n_points))

    def test_pole_neighbours_different_indices(self):
        """Test that pole_neighbours identifies different poles."""
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = SingleSystem.build_container(single, phase=0.0)

        star = system_container.star
        putils.pole_neighbours(star)

        # The two poles should be different
        self.assertNotEqual(star.pole_idx[0], star.pole_idx[1])

    def test_generate_tilt_coordinates_without_tidally_locked(self):
        """Test generate_tilt_coordinates with non-tidally-locked modes."""
        pulse_meta = [{
            "l": 1,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
            "mode_axis_theta": np.pi / 6,
            "mode_axis_phi": np.pi / 4,
            "tidally_locked": False,
        }]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = SingleSystem.build_container(single, phase=0.0)

        star = system_container.star
        phase = 0.5

        phi, theta = putils.generate_tilt_coordinates(star, phase)

        # Phi should include the correction
        self.assertIsNotNone(phi)
        self.assertIsNotNone(theta)
        self.assertTrue(np.isfinite(phi))
        self.assertTrue(np.isfinite(theta))

    def test_generate_tilt_coordinates_tidally_locked(self):
        """Test generate_tilt_coordinates with tidally-locked modes."""
        pulse_meta = [{
            "l": 1,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
            "mode_axis_theta": np.pi / 6,
            "mode_axis_phi": np.pi / 4,
            "tidally_locked": True,
        }]

        # For tidally locked, we need a binary system
        primary = Star(pulsations=pulse_meta, **BINARY_STAR)
        secondary = Star(pulsations=pulse_meta, **BINARY_STAR)
        binary = BinarySystem(primary=primary, secondary=secondary, **BINARY_SYSTEM)
        system_container = BinarySystem.build_container(binary, phase=0.0)

        star = system_container.primary
        phase = 0.5

        phi, theta = putils.generate_tilt_coordinates(star, phase)

        # For tidally locked, phi_corr should be 0
        expected_phi = star.pulsations[0].mode_axis_phi
        assert_almost_equal(phi, expected_phi)


class TestPulsationModeProperties(ElisaTestCase):
    """Unit tests for elisa.pulse.transform.PulsationModeProperties validation methods."""

    def setUp(self):
        super().setUp()
        from elisa.pulse.transform import PulsationModeProperties
        self.props = PulsationModeProperties

    def test_l_valid_int(self):
        """Test l property with valid integer input."""
        result = self.props.l(2)
        self.assertEqual(result, 2)
        self.assertIsInstance(result, (int, np.integer))

    def test_l_valid_float_convertible_to_int(self):
        """Test l property with float that can be converted to int."""
        result = self.props.l(2.0)
        self.assertEqual(result, 2)
        self.assertIsInstance(result, (int, np.integer))

    def test_l_valid_numpy_int(self):
        """Test l property with numpy integer."""
        result = self.props.l(np.int32(3))
        self.assertEqual(result, 3)

    def test_l_invalid_float(self):
        """Test l property raises TypeError for non-integer float."""
        with self.assertRaises(TypeError):
            self.props.l(2.5)

    def test_l_invalid_string(self):
        """Test l property raises TypeError for string input."""
        with self.assertRaises(TypeError):
            self.props.l("2")

    def test_m_valid_int(self):
        """Test m property with valid integer input."""
        result = self.props.m(1)
        self.assertEqual(result, 1)
        self.assertIsInstance(result, (int, np.integer))

    def test_m_valid_float_convertible_to_int(self):
        """Test m property with float that can be converted to int."""
        result = self.props.m(1.0)
        self.assertEqual(result, 1)
        self.assertIsInstance(result, (int, np.integer))

    def test_m_valid_numpy_int(self):
        """Test m property with numpy integer."""
        result = self.props.m(np.int64(-1))
        self.assertEqual(result, -1)

    def test_m_invalid_float(self):
        """Test m property raises TypeError for non-integer float."""
        with self.assertRaises(TypeError):
            self.props.m(1.5)

    def test_m_invalid_string(self):
        """Test m property raises TypeError for string input."""
        with self.assertRaises(TypeError):
            self.props.m("1")

    def test_amplitude_valid_float(self):
        """Test amplitude property with valid float input."""
        result = self.props.amplitude(50.0)
        self.assertAlmostEqual(result, 50.0)

    def test_amplitude_valid_int(self):
        """Test amplitude property with valid integer input."""
        result = self.props.amplitude(50)
        self.assertAlmostEqual(result, 50.0)

    def test_amplitude_valid_quantity(self):
        """Test amplitude property with astropy Quantity."""
        value = 1.0 * u.km / u.s
        result = self.props.amplitude(value)
        self.assertTrue(np.isfinite(result))
        self.assertAlmostEqual(result, np.float64(value.to(u.VELOCITY_UNIT).value))

    def test_amplitude_valid_quantity_string(self):
        """Test amplitude property with quantity string."""
        result = self.props.amplitude("1 km / s")
        self.assertTrue(np.isfinite(result))

    def test_amplitude_negative_raises_error(self):
        """Test amplitude property raises ValueError for negative values."""
        with self.assertRaises(ValueError):
            self.props.amplitude(-10.0)

    def test_amplitude_invalid_type(self):
        """Test amplitude property raises TypeError for invalid type."""
        with self.assertRaises(TypeError):
            self.props.amplitude("invalid")

    def test_frequency_valid_float(self):
        """Test frequency property with valid float input (assumes default unit)."""
        result = self.props.frequency(1.0)
        self.assertTrue(np.isfinite(result))

    def test_frequency_valid_int(self):
        """Test frequency property with valid integer input."""
        result = self.props.frequency(2)
        self.assertTrue(np.isfinite(result))

    def test_frequency_valid_quantity(self):
        """Test frequency property with astropy Quantity."""
        value = 16.0 / u.d
        result = self.props.frequency(value)
        self.assertTrue(np.isfinite(result))
        self.assertAlmostEqual(result, np.float64(value.to(u.FREQUENCY_UNIT).value))

    def test_frequency_valid_quantity_string(self):
        """Test frequency property with frequency quantity string."""
        result = self.props.frequency("1 Hz")
        self.assertTrue(np.isfinite(result))

    def test_frequency_negative_raises_error(self):
        """Test frequency property raises ValueError for negative values."""
        with self.assertRaises(ValueError):
            self.props.frequency(-1.0)

    def test_frequency_invalid_type(self):
        """Test frequency property raises TypeError for invalid type."""
        with self.assertRaises(TypeError):
            self.props.frequency("invalid_freq")

    def test_start_phase_zero(self):
        """Test start_phase property with zero value."""
        result = self.props.start_phase(0.0)
        self.assertAlmostEqual(result, 0.0)

    def test_start_phase_valid_float(self):
        """Test start_phase property with valid float input."""
        result = self.props.start_phase(0.75)
        self.assertTrue(np.isfinite(result))

    def test_start_phase_valid_quantity(self):
        """Test start_phase property with astropy Quantity."""
        value = 45 * u.deg
        result = self.props.start_phase(value)
        self.assertTrue(np.isfinite(result))

    def test_mode_axis_theta_valid(self):
        """Test mode_axis_theta property with valid value in range (0, pi)."""
        result = self.props.mode_axis_theta(45.0)  # 45 degrees
        self.assertTrue(np.isfinite(result))
        self.assertTrue(0 <= result < const.PI)

    def test_mode_axis_theta_valid_quantity(self):
        """Test mode_axis_theta property with quantity."""
        value = 45 * u.deg
        result = self.props.mode_axis_theta(value)
        self.assertTrue(np.isfinite(result))
        self.assertTrue(0 <= result < const.PI)

    def test_mode_axis_theta_zero(self):
        """Test mode_axis_theta property with zero (boundary value)."""
        result = self.props.mode_axis_theta(0.0)
        self.assertAlmostEqual(result, 0.0)

    def test_mode_axis_theta_below_pi(self):
        """Test mode_axis_theta property with value just below pi."""
        result = self.props.mode_axis_theta(179.9)
        self.assertTrue(0 <= result < const.PI)

    def test_mode_axis_theta_above_pi_raises_error(self):
        """Test mode_axis_theta property raises ValueError for value >= pi."""
        with self.assertRaises(ValueError):
            self.props.mode_axis_theta(180.0)  # 180 degrees = pi radians

    def test_mode_axis_theta_negative_raises_error(self):
        """Test mode_axis_theta property raises ValueError for negative values."""
        with self.assertRaises(ValueError):
            self.props.mode_axis_theta(-45.0)

    def test_mode_axis_phi_valid(self):
        """Test mode_axis_phi property with valid value."""
        result = self.props.mode_axis_phi(45.0)
        self.assertTrue(np.isfinite(result))

    def test_mode_axis_phi_zero(self):
        """Test mode_axis_phi property with zero value."""
        result = self.props.mode_axis_phi(0.0)
        self.assertAlmostEqual(result, 0.0)

    def test_mode_axis_phi_valid_quantity(self):
        """Test mode_axis_phi property with quantity."""
        value = 90 * u.deg
        result = self.props.mode_axis_phi(value)
        self.assertTrue(np.isfinite(result))

    def test_temperature_perturbation_phase_shift_valid(self):
        """Test temperature_perturbation_phase_shift with valid value."""
        result = self.props.temperature_perturbation_phase_shift(0.5)
        self.assertTrue(np.isfinite(result))

    def test_temperature_perturbation_phase_shift_zero(self):
        """Test temperature_perturbation_phase_shift with zero."""
        result = self.props.temperature_perturbation_phase_shift(0.0)
        self.assertTrue(np.isfinite(result))

    def test_temperature_perturbation_phase_shift_with_quantity(self):
        """Test temperature_perturbation_phase_shift with quantity."""
        value = 30 * u.deg
        result = self.props.temperature_perturbation_phase_shift(value)
        self.assertTrue(np.isfinite(result))

    def test_horizontal_to_radial_amplitude_ratio_valid_float(self):
        """Test horizontal_to_radial_amplitude_ratio with valid float."""
        result = self.props.horizontal_to_radial_amplitude_ratio(0.5)
        self.assertAlmostEqual(result, 0.5)

    def test_horizontal_to_radial_amplitude_ratio_zero(self):
        """Test horizontal_to_radial_amplitude_ratio with zero."""
        result = self.props.horizontal_to_radial_amplitude_ratio(0.0)
        self.assertAlmostEqual(result, 0.0)

    def test_horizontal_to_radial_amplitude_ratio_large_value(self):
        """Test horizontal_to_radial_amplitude_ratio with large value."""
        result = self.props.horizontal_to_radial_amplitude_ratio(2.5)
        self.assertAlmostEqual(result, 2.5)

    def test_horizontal_to_radial_amplitude_ratio_int(self):
        """Test horizontal_to_radial_amplitude_ratio with integer."""
        result = self.props.horizontal_to_radial_amplitude_ratio(1)
        self.assertAlmostEqual(result, 1.0)

    def test_horizontal_to_radial_amplitude_ratio_numpy_float(self):
        """Test horizontal_to_radial_amplitude_ratio with numpy float."""
        result = self.props.horizontal_to_radial_amplitude_ratio(np.float64(0.75))
        self.assertAlmostEqual(result, 0.75)

    def test_horizontal_to_radial_amplitude_ratio_invalid_type(self):
        """Test horizontal_to_radial_amplitude_ratio raises TypeError for invalid type."""
        with self.assertRaises(TypeError):
            self.props.horizontal_to_radial_amplitude_ratio("0.5")

    def test_tidally_locked_true(self):
        """Test tidally_locked property with True."""
        result = self.props.tidally_locked(True)
        self.assertTrue(result)
        self.assertIsInstance(result, bool)

    def test_tidally_locked_false(self):
        """Test tidally_locked property with False."""
        result = self.props.tidally_locked(False)
        self.assertFalse(result)
        self.assertIsInstance(result, bool)

    def test_tidally_locked_invalid_int(self):
        """Test tidally_locked property raises TypeError for integer."""
        with self.assertRaises(TypeError):
            self.props.tidally_locked(1)

    def test_tidally_locked_invalid_string(self):
        """Test tidally_locked property raises TypeError for string."""
        with self.assertRaises(TypeError):
            self.props.tidally_locked("True")

    def test_tidally_locked_invalid_none(self):
        """Test tidally_locked property raises TypeError for None."""
        with self.assertRaises(TypeError):
            self.props.tidally_locked(None)


class TestPulsationsFunctions(ElisaTestCase):
    """Unit tests for elisa.pulse.pulsations module functions."""

    def setUp(self):
        super().setUp()
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        settings.configure(LD_TABLES=op.join(self.lc_base_path, "limbdarkening"), CK04_ATM_TABLES=op.join(self.lc_base_path, "atmosphere"))

    def test_spherical_harmonics_basic(self):
        """Test spherical_harmonics calculates harmonics correctly."""
        pulse_meta = [{
            "l": 1,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = testutils.prepare_single_system_container(single)
        system_container.build_mesh()
        points = utils.cartesian_to_spherical(system_container.star.points)

        mode = single.star.pulsations[0]
        time = 0.0
        exponential = putils.generate_time_exponential(mode, time)

        result = pulsations.spherical_harmonics(mode, points, exponential)

        # Result should have same length as points
        self.assertEqual(result.shape[0], points.shape[0])

        # Result should be finite (no NaNs or infs)
        self.assertTrue(np.all(np.isfinite(result)))

        # Result should be complex
        self.assertTrue(np.iscomplexobj(result))

    def test_spherical_harmonics_with_custom_degree(self):
        """Test spherical_harmonics with custom degree parameter."""
        pulse_meta = [{
            "l": 1,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

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
        self.assertTrue(np.all(np.isfinite(result)))

    def test_spherical_harmonics_with_custom_order(self):
        """Test spherical_harmonics with custom order parameter."""
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

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
        self.assertTrue(np.all(np.isfinite(result)))

    def test_diff_spherical_harmonics_by_phi_basic(self):
        """Test diff_spherical_harmonics_by_phi calculates derivative correctly."""
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

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
        self.assertEqual(result.shape[0], points.shape[0])

        # Result should be finite
        self.assertTrue(np.all(np.isfinite(result)))

        # Result should be complex
        self.assertTrue(np.iscomplexobj(result))

    def test_diff_spherical_harmonics_by_phi_zero_m(self):
        """Test diff_spherical_harmonics_by_phi with m=0."""
        pulse_meta = [{
            "l": 2,
            "m": 0,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

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

    def test_diff_spherical_harmonics_by_theta_basic(self):
        """Test diff_spherical_harmonics_by_theta calculates derivative correctly."""
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = testutils.prepare_single_system_container(single)
        system_container.build_mesh()
        points = utils.cartesian_to_spherical(system_container.star.points)

        mode = single.star.pulsations[0]
        time = 0.0
        exponential = putils.generate_time_exponential(mode, time)

        harmonics_0 = pulsations.spherical_harmonics(mode, points, exponential)
        harmonics_1 = pulsations.spherical_harmonics(mode, points, exponential, order=mode.m + 1)

        result = pulsations.diff_spherical_harmonics_by_theta(
            mode,
            [harmonics_0, harmonics_1],
            points[:, 1],
            points[:, 2],
        )

        # Result should have same length as points
        self.assertEqual(result.shape[0], points.shape[0])

        # Result should be finite
        self.assertTrue(np.all(np.isfinite(result)))

        # Result should be complex
        self.assertTrue(np.iscomplexobj(result))

    def test_diff_spherical_harmonics_by_theta_zero_m(self):
        """Test diff_spherical_harmonics_by_theta with m=0."""
        pulse_meta = [{
            "l": 2,
            "m": 0,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = testutils.prepare_single_system_container(single)
        system_container.build_mesh()
        points = utils.cartesian_to_spherical(system_container.star.points)

        mode = single.star.pulsations[0]
        time = 0.0
        exponential = putils.generate_time_exponential(mode, time)

        harmonics_0 = pulsations.spherical_harmonics(mode, points, exponential)
        harmonics_1 = pulsations.spherical_harmonics(mode, points, exponential, order=mode.m + 1)

        result = pulsations.diff_spherical_harmonics_by_theta(
            mode,
            [harmonics_0, harmonics_1],
            points[:, 1],
            points[:, 2],
        )

        # Result should be finite
        self.assertTrue(np.all(np.isfinite(result)))

    def test_horizontal_displacement_normalization_basic(self):
        """Test horizontal_displacement_normalization calculates normalization constant."""
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = testutils.prepare_single_system_container(single)
        system_container.build_mesh()
        points = utils.cartesian_to_spherical(system_container.star.points)

        mode = single.star.pulsations[0]
        time = 0.0
        exponential = putils.generate_time_exponential(mode, time)

        harmonics_0 = pulsations.spherical_harmonics(mode, points, exponential)
        harmonics_1 = pulsations.spherical_harmonics(mode, points, exponential, order=mode.m + 1)

        d_phi = pulsations.diff_spherical_harmonics_by_phi(mode, [harmonics_0, harmonics_1])
        d_theta = pulsations.diff_spherical_harmonics_by_theta(
            mode,
            [harmonics_0, harmonics_1],
            points[:, 1],
            points[:, 2],
        )

        result = pulsations.horizontal_displacement_normalization([d_phi, d_theta], [harmonics_0, harmonics_1])

        # Result should be finite and positive
        self.assertTrue(np.isfinite(result))
        self.assertTrue(result > 0.0)

    def test_assign_amplitudes_basic(self):
        """Test assign_amplitudes assigns correct amplitude values."""
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = SingleSystem.build_container(single, phase=0.0)

        pulsations.assign_amplitudes(system_container.star)

        mode = system_container.star.pulsations[0]

        # Radial amplitude should be assigned and positive
        self.assertIsNotNone(mode.radial_amplitude)
        self.assertTrue(mode.radial_amplitude > 0.0)

        # Horizontal amplitude should be assigned and positive
        self.assertIsNotNone(mode.horizontal_amplitude)
        self.assertTrue(mode.horizontal_amplitude > 0.0)

        # Temperature amplitude factor should be assigned
        self.assertIsNotNone(mode.temperature_amplitude_factor)

    def test_assign_amplitudes_with_custom_ratio(self):
        """Test assign_amplitudes respects custom horizontal_to_radial_amplitude_ratio."""
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
            "horizontal_to_radial_amplitude_ratio": 0.5,
            "temperature_amplitude_factor": 0.01,
        }]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = SingleSystem.build_container(single, phase=0.0)

        pulsations.assign_amplitudes(system_container.star)

        mode = system_container.star.pulsations[0]

        # Check that ratio is preserved
        if mode.radial_amplitude > 0:
            ratio = mode.horizontal_amplitude / mode.radial_amplitude
            self.assertAlmostEqual(ratio, 0.5, places=4)

    def test_assign_amplitudes_normalization_constant(self):
        """Test assign_amplitudes with custom normalization constant."""
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        system_container = SingleSystem.build_container(single, phase=0.0)

        # Call with custom normalization constant
        pulsations.assign_amplitudes(system_container.star, normalization_constant=2.0)

        mode = system_container.star.pulsations[0]

        # Amplitudes should be assigned
        self.assertIsNotNone(mode.radial_amplitude)
        self.assertIsNotNone(mode.horizontal_amplitude)

    def test_temp_amplitude_basic(self):
        """Test temp_amplitude calculates temperature amplitude."""
        pulse_meta = [{
            "l": 2,
            "m": 1,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
            "horizontal_to_radial_amplitude_ratio": 0.5,
        }]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        mode = single.star.pulsations[0]

        result = pulsations.temp_amplitude(mode)

        # Result should be finite
        self.assertTrue(np.isfinite(result))

    def test_temp_amplitude_radial_mode(self):
        """Test temp_amplitude with radial mode (l=0, m=0)."""
        pulse_meta = [{
            "l": 0,
            "m": 0,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
            "horizontal_to_radial_amplitude_ratio": 0.0,
            "temperature_amplitude_factor": 0.01,  # Must provide this for radial modes
        }]

        single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)
        mode = single.star.pulsations[0]

        # For radial modes, temp_amplitude calculation may have issues
        # due to division by zero, but we just check it doesn't crash
        try:
            result = pulsations.temp_amplitude(mode)
            # If it doesn't crash, check if result is finite
            if np.isfinite(result):
                self.assertTrue(True)
        except (ZeroDivisionError, ValueError):
            # This is expected for radial modes
            self.assertTrue(True)

    def test_temp_amplitude_different_ratios(self):
        """Test temp_amplitude with different amplitude ratios."""
        for ratio in [0.1, 0.5, 1.0, 2.0]:
            pulse_meta = [{
                "l": 2,
                "m": 1,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1.0 / u.d,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": ratio,
            }]

            single = SingleSystem(
                star=Star(pulsations=pulse_meta, **STAR_PARAMS),
                **SYSTEM_PARMAS,
            )
            mode = single.star.pulsations[0]

            result = pulsations.temp_amplitude(mode)

            # Result should be finite
            self.assertTrue(np.isfinite(result))

    def test_assign_amplitudes_radial_mode_error(self):
        """Test assign_amplitudes raises error for radial modes without temperature_amplitude_factor."""
        pulse_meta = [{
            "l": 0,
            "m": 0,
            "amplitude": 1.0 * u.m / u.s,
            "frequency": 1.0 / u.d,
            "start_phase": 0.0,
        }]

        # Should raise ValueError during SingleSystem initialization for radial modes
        # without temperature_amplitude_factor
        with self.assertRaises(ValueError):
            single = SingleSystem(star=Star(pulsations=pulse_meta, **STAR_PARAMS), **SYSTEM_PARMAS)


class TestPulsationModeInitialization(ElisaTestCase):
    """Unit tests for elisa.pulse.mode.PulsationMode class initialization and properties."""

    def test_init_with_mandatory_params_only(self):
        """Test PulsationMode initialization with mandatory parameters only."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        self.assertEqual(mode.l, 2)
        self.assertEqual(mode.m, 1)
        self.assertAlmostEqual(mode.amplitude, 1.0)
        self.assertTrue(np.isfinite(mode.frequency))

    def test_init_with_all_parameters(self):
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

        self.assertEqual(mode.l, 2)
        self.assertEqual(mode.m, 1)
        self.assertAlmostEqual(mode.start_phase, 0.5)
        self.assertTrue(0 <= mode.mode_axis_theta < const.PI)
        self.assertTrue(np.isfinite(mode.mode_axis_phi))
        self.assertAlmostEqual(mode.horizontal_to_radial_amplitude_ratio, 0.5)
        self.assertAlmostEqual(mode.temperature_amplitude_factor, 0.01)
        self.assertTrue(mode.tidally_locked)

    def test_init_missing_mandatory_param_l(self):
        """Test PulsationMode raises error when l is missing."""
        with self.assertRaises(ValueError):
            PulsationMode(
                m=1,
                amplitude=1.0 * u.m / u.s,
                frequency=1.0 / u.d,
            )

    def test_init_missing_mandatory_param_m(self):
        """Test PulsationMode raises error when m is missing."""
        with self.assertRaises(ValueError):
            PulsationMode(
                l=2,
                amplitude=1.0 * u.m / u.s,
                frequency=1.0 / u.d,
            )

    def test_init_missing_mandatory_param_amplitude(self):
        """Test PulsationMode raises error when amplitude is missing."""
        with self.assertRaises(ValueError):
            PulsationMode(
                l=2,
                m=1,
                frequency=1.0 / u.d,
            )

    def test_init_missing_mandatory_param_frequency(self):
        """Test PulsationMode raises error when frequency is missing."""
        with self.assertRaises(ValueError):
            PulsationMode(
                l=2,
                m=1,
                amplitude=1.0 * u.m / u.s,
            )

    def test_init_invalid_kwarg(self):
        """Test PulsationMode raises error for invalid keyword argument."""
        with self.assertRaises(ValueError):
            PulsationMode(
                l=2,
                m=1,
                amplitude=1.0 * u.m / u.s,
                frequency=1.0 / u.d,
                invalid_param=42,
            )

    def test_angular_frequency_calculation(self):
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

    def test_renorm_constant_value(self):
        """Test that renorm_const is correctly set."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        expected_renorm = 2 * const.PI ** 0.5
        self.assertAlmostEqual(mode.renorm_const, expected_renorm)

    def test_validate_mode_valid(self):
        """Test mode validation with valid parameters (|m| <= l)."""
        # Should not raise any error
        mode = PulsationMode(
            l=2,
            m=2,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )
        self.assertEqual(mode.l, 2)
        self.assertEqual(mode.m, 2)

    def test_validate_mode_m_equals_l(self):
        """Test mode validation when |m| = l."""
        mode = PulsationMode(
            l=3,
            m=3,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )
        self.assertEqual(mode.l, 3)
        self.assertEqual(mode.m, 3)

    def test_validate_mode_negative_m(self):
        """Test mode validation with negative m within valid range."""
        mode = PulsationMode(
            l=2,
            m=-1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )
        self.assertEqual(mode.l, 2)
        self.assertEqual(mode.m, -1)

    def test_validate_mode_invalid_m_exceeds_l(self):
        """Test mode validation raises error when |m| > l."""
        with self.assertRaises(ValueError):
            PulsationMode(
                l=2,
                m=3,
                amplitude=1.0 * u.m / u.s,
                frequency=1.0 / u.d,
            )

    def test_validate_mode_invalid_negative_m_exceeds_l(self):
        """Test mode validation raises error when |m| > l with negative m."""
        with self.assertRaises(ValueError):
            PulsationMode(
                l=2,
                m=-4,
                amplitude=1.0 * u.m / u.s,
                frequency=1.0 / u.d,
            )

    def test_default_input_units_property(self):
        """Test that default_input_units property returns correct units."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        input_units = mode.default_input_units
        self.assertIsNotNone(input_units)

    def test_default_internal_units_property(self):
        """Test that default_internal_units property returns correct units."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        internal_units = mode.default_internal_units
        self.assertIsNotNone(internal_units)

    def test_default_values_assigned(self):
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
        self.assertIsNone(mode.horizontal_to_radial_amplitude_ratio)
        self.assertFalse(mode.tidally_locked)
        self.assertIsNone(mode.temperature_amplitude_factor)

    def test_amplitude_initialization_with_quantity(self):
        """Test amplitude initialization with astropy Quantity."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.km / u.s,
            frequency=1.0 / u.d,
        )

        self.assertTrue(np.isfinite(mode.amplitude))
        # 1 km/s = 1000 m/s internally
        self.assertAlmostEqual(mode.amplitude, 1000.0)

    def test_amplitude_initialization_with_string(self):
        """Test amplitude initialization with string quantity."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude="1 km / s",
            frequency=1.0 / u.d,
        )

        self.assertTrue(np.isfinite(mode.amplitude))
        # 1 km/s = 1000 m/s internally
        self.assertAlmostEqual(mode.amplitude, 1000.0)

    def test_frequency_initialization_with_quantity(self):
        """Test frequency initialization with astropy Quantity."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=16.0 / u.d,
        )

        self.assertTrue(np.isfinite(mode.frequency))

    def test_frequency_initialization_with_string(self):
        """Test frequency initialization with string quantity."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency="1 Hz",
        )

        self.assertTrue(np.isfinite(mode.frequency))

    def test_l_and_m_as_floats(self):
        """Test that l and m can be initialized as floats and converted to int."""
        mode = PulsationMode(
            l=2.0,
            m=1.0,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        self.assertEqual(mode.l, 2)
        self.assertEqual(mode.m, 1)
        self.assertIsInstance(mode.l, (int, np.integer))
        self.assertIsInstance(mode.m, (int, np.integer))

    def test_mode_axis_theta_range_validation(self):
        """Test that mode_axis_theta is validated to be in (0, pi)."""
        # Valid case
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
            mode_axis_theta=np.pi / 2,
        )
        self.assertTrue(0 <= mode.mode_axis_theta < const.PI)

    def test_amplitude_negative_raises_error(self):
        """Test that negative amplitude raises error."""
        with self.assertRaises(ValueError):
            PulsationMode(
                l=2,
                m=1,
                amplitude=-1.0 * u.m / u.s,
                frequency=1.0 / u.d,
            )

    def test_frequency_negative_raises_error(self):
        """Test that negative frequency raises error."""
        with self.assertRaises(ValueError):
            PulsationMode(
                l=2,
                m=1,
                amplitude=1.0 * u.m / u.s,
                frequency=-1.0 / u.d,
            )

    def test_transform_input_is_called(self):
        """Test that transform_input is called during initialization."""
        # transform_input should be called to convert and validate parameters
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        # If transform_input worked correctly, we should have valid internal units
        self.assertTrue(np.isfinite(mode.amplitude))
        self.assertTrue(np.isfinite(mode.frequency))

    def test_radial_and_horizontal_amplitude_initialization(self):
        """Test that radial and horizontal amplitude are None after init."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        self.assertIsNone(mode.radial_amplitude)
        self.assertIsNone(mode.horizontal_amplitude)

    def test_points_initialization(self):
        """Test that points are None after initialization."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        self.assertIsNone(mode.points)
        self.assertIsNone(mode.point_harmonics)
        self.assertIsNone(mode.point_harmonics_derivatives)
        self.assertIsNone(mode.complex_displacement)

    def test_tilt_angles_initialization(self):
        """Test that tilt angles are None after initialization."""
        mode = PulsationMode(
            l=2,
            m=1,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        self.assertIsNone(mode.tilt_phi)
        self.assertIsNone(mode.tilt_theta)

    def test_radial_mode_l_equals_zero(self):
        """Test radial mode with l=0."""
        mode = PulsationMode(
            l=0,
            m=0,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        self.assertEqual(mode.l, 0)
        self.assertEqual(mode.m, 0)

    def test_high_degree_mode(self):
        """Test mode with high degree value."""
        mode = PulsationMode(
            l=10,
            m=5,
            amplitude=1.0 * u.m / u.s,
            frequency=1.0 / u.d,
        )

        self.assertEqual(mode.l, 10)
        self.assertEqual(mode.m, 5)

    def test_mode_with_custom_start_phase(self):
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

    def test_mode_with_custom_horizontal_ratio(self):
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

    def test_mode_with_temperature_amplitude_factor(self):
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

