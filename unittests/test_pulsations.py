import os
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal, assert_array_less

from elisa import units as u
from elisa import utils
from elisa import const
from elisa import Observer

from elisa.base.star import Star
from elisa.single_system.system import SingleSystem
from elisa.binary_system.system import BinarySystem
from elisa.pulse import pulsations, container_ops, utils as putils
from unittests.utils import ElisaTestCase
from unittests import utils as testutils

STAR_PARAMS = {
    'mass': 2.0 * u.solMass,
    't_eff': 10000 * u.K,
    'gravity_darkening': 1.0,
    'discretization_factor': 5,
    'albedo': 1.0,
    'metallicity': 0.0,
    'polar_log_g': 4.0 * u.dex(u.cm / u.s ** 2)
}

SYSTEM_PARMAS = {
    'gamma': 0 * u.km / u.s,
    'inclination': 80 * u.deg,
    'rotation_period': 30 * u.d,
}

BINARY_STAR = {
    'mass': 2.0 * u.solMass,
    't_eff': 10000 * u.K,
    'gravity_darkening': 1.0,
    'discretization_factor': 5,
    'albedo': 1.0,
    'metallicity': 0.0,
    'synchronicity': 1.0,
    'surface_potential': 10
}

BINARY_SYSTEM = {
    'argument_of_periastron': 0 * u.deg,
    'gamma': 0 * u.km / u.s,
    'period': 2.0 * u.d,
    'eccentricity': 0.1,
    'inclination': 87 * u.deg,
    'primary_minimum_time': 0 * u.d,
    'phase_shift': 0.0,
}

TOL = 1e-3


class PulsatingStarInitTestCase(ElisaTestCase):
    def setUp(self):
        self.pulsation_modes = [
            {
                'l': 1,
                'm': 1,
                'amplitude': 0.050 * u.km / u.s,
                'frequency': 16 / u.d,
                'start_phase': 0.75,
            },
            {
                'l': 1,
                'm': -1,
                'amplitude': 50,
                'frequency': 16,
                'start_phase': 1.5,
            }
        ]

        self.star_params = {
            'mass': 2.15 * u.solMass,
            't_eff': 10000 * u.K,
            'gravity_darkening': 1.0,
            'discretization_factor': 3,
            'albedo': 0.6,
            'metallicity': 0.0,
            'polar_log_g': 4.4 * u.dex(u.cm / u.s ** 2)
        }

        self.system_params = {'gamma': 0 * u.km / u.s,
                              'inclination': 80 * u.deg,
                              'rotation_period': 30 * u.d,
                              }

    def prepare_system(self, pulsations=None):
        pulsations = self.pulsation_modes if pulsations is None else pulsations
        star = Star(pulsations=pulsations, **self.star_params)
        return SingleSystem(star=star, **self.system_params)

    def test_mode_initialization_of_parameters(self):
        """
        testing whether pulsation parameters are initialised to correct values
        :return:
        """
        rounding_prec = 6
        expected_values = [
            [1, 1, 50, 0.000185, 0.75],
            [1, -1, 50, 16, 1.5]
        ]

        star = self.prepare_system().star

        for ii, mode in star.pulsations.items():
            list_to_check = [mode.l, mode.m, np.round(mode.amplitude, rounding_prec),
                             np.round(mode.frequency, rounding_prec), np.round(mode.start_phase, rounding_prec)]
            assert_array_equal(expected_values[ii], list_to_check)

    def test_renorm_constant(self):
        """
        Testing if RMS of pulsation mode gives 1
        :return:
        """
        puls_meta = [{
            'l': 1,
            'm': 1,
            'amplitude': 1 * u.m / u.s,
            'frequency': 1 / u.d,
            'start_phase': 0.0,
            'temperature_amplitude_factor': 1.0
        }]

        time = 0

        single = self.prepare_system()

        system_container = testutils.prepare_single_system_container(single)
        system_container.build_mesh()
        points = utils.cartesian_to_spherical(system_container.star.points)

        for ll in range(0, 10):
            for mm in range(ll + 1):
                puls_meta[0]['l'] = ll
                puls_meta[0]['m'] = mm
                single.star.pulsations = puls_meta
                single.init()

                mode = single.star.pulsations[0]
                exponential = putils.generate_time_exponential(mode, time)
                sph_harm = pulsations.spherical_harmonics(mode, points, exponential)

                test_val = np.sqrt(np.sum(np.abs(sph_harm) ** 2) / points.shape[0])
                assert_almost_equal(test_val, 1.0, 2)


class TestPulsationModule(ElisaTestCase):

    def setUp(self):
        super(TestPulsationModule, self).setUp()
        self.base_path = os.path.dirname(os.path.abspath(__file__))

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
            'l': 10,
            'm': 2,
            'amplitude': 1 * u.m / u.s,
            'frequency': 1 / u.d,
            'start_phase': 0.0,
            'horizontal_to_radial_amplitude_ratio': in_ratio
        }]

        single = self.prepare_system(pulse_meta)
        system_container = SingleSystem.build_container(single, phase=0)

        r_eq = single.star.equivalent_radius

        mode = system_container.star.pulsations[0]
        theta = mode.points[:, 2]

        radial = np.mean(np.abs(mode.complex_displacement[:, 0])**2)**0.5
        dphi = np.abs(mode.complex_displacement[:, 1])
        dtheta = np.abs(mode.complex_displacement[:, 2])

        horizontal = r_eq * np.mean((dtheta**2 + (np.sin(theta)*dphi)**2))**0.5
        ratio = horizontal / radial
        self.assertTrue(abs(ratio - in_ratio) < 0.1)

    def test_kinematics_single(self):
        amplitude = 1.0
        freq = 1
        omega = const.FULL_ARC * freq
        pulse_meta = [{
            'l': 2,
            'm': 1,
            'amplitude': amplitude * u.m / u.s,
            'frequency': freq / u.s,
            'start_phase': 0.0,
            'horizontal_to_radial_amplitude_ratio': 0.0,
            'temperature_amplitude_factor': 0.01
        }]
        overshoot = 2.0  # this value may change for different modes

        single = self.prepare_system(pulse_meta)
        system_container = SingleSystem.build_container(single, phase=0.2854)

        # displacement
        r_ampl = amplitude / omega
        args = (system_container.star, 0.0, False, True, True)
        radius = np.abs(container_ops.position_perturbation(*args)[:, 0])
        assert_array_less(radius, overshoot * r_ampl)
        assert_array_less(r_ampl, radius.max())

        # velocity
        args = (system_container.star, 1, False, True, True)
        velocities = np.abs(container_ops.velocity_perturbation(*args)[:, 0])
        assert_array_less(velocities, overshoot * amplitude)
        assert_array_less(amplitude, velocities.max())

        # acceleration
        a_amp = amplitude * omega
        args = (system_container.star, 1.0, False, True, True)
        acc = np.abs(container_ops.gravity_acc_perturbation(*args)[:, 0])
        assert_array_less(acc, overshoot * a_amp)
        assert_array_less(a_amp, acc.max())

        # teff
        t_ampl = pulse_meta[0]['temperature_amplitude_factor'] * system_container.star.t_eff
        args = (system_container.star, 1.0, False, True)
        ts = np.abs(container_ops.temp_perturbation(*args))
        assert_array_less(ts, overshoot * t_ampl)
        assert_array_less(t_ampl, ts.max())

    def test_single_pulsating_lc(self):
        freq = 15
        pulse_meta = [{
            'l': 2,
            'm': 1,
            'amplitude': 100 * u.m / u.s,
            'frequency': freq / u.d,
            'start_phase': 0.0,
            'horizontal_to_radial_amplitude_ratio': 0.0,
            'temperature_amplitude_factor': 0.01
        }]

        single = self.prepare_system(pulse_meta)
        o = Observer(passband=['Generic.Bessell.V', ], system=single)

        phases = np.linspace(0.0, 1 / (freq * single.rotation_period))

        expected = testutils.load_light_curve("single.pulsating.v.json")
        expected_phases = expected[0]
        expected_flux = testutils.normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        obtained = o.lc(phases=phases)
        obtained_phases = obtained[0]
        obtained_flux = testutils.normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        self.assertTrue(np.all(np.abs(np.round(obtained_phases, 3) - np.round(expected_phases, 3)) < TOL))
        self.assertTrue(np.all(np.abs(np.round(obtained_flux, 3) - np.round(expected_flux, 3)) < TOL))

        # o.plot.phase_curve()

    def test_single_pulsating_rv(self):
        freq = 15
        pulse_meta = [{
            'l': 1,
            'm': 1,
            'amplitude': 1 * u.km / u.s,
            'frequency': freq / u.d,
            'start_phase': 0.0,
            'horizontal_to_radial_amplitude_ratio': 0.0,
            'temperature_amplitude_factor': 0.01
        }]

        single = self.prepare_system(pulse_meta)
        o = Observer(system=single)

        phases = np.linspace(0.0, 1 / (freq * single.period), 50)

        expected = testutils.load_radial_curve("single.pulsating.json")
        expected_rv = testutils.normalize_single_rv_for_unittests(expected["star"])

        obtained = o.rv(phases=phases, method='radiometric')
        obtained_rv = testutils.normalize_single_rv_for_unittests(obtained[1]['star'])

        assert_array_equal(np.round(expected_rv, 3), np.round(obtained_rv, 3))

        # o.plot.rv_curve()

    def test_kinematics_binary(self):
        amplitude = 1.0
        freq = 1
        omega = const.FULL_ARC * freq
        pulse_meta = [{
            'l': 4,
            'm': 2,
            'amplitude': amplitude * u.m / u.s,
            'frequency': freq / u.s,
            'start_phase': 0.0,
            'horizontal_to_radial_amplitude_ratio': 1.0,
            'temperature_amplitude_factor': 0.01
        }]
        overshoot = 2.0  # this value may change for different modes

        binary = self.prepare_binary(pulse_meta)
        system_container = BinarySystem.build_container(binary, phase=0.6741)

        # displacement
        r_ampl = amplitude / omega
        args = (system_container.secondary, system_container.position.distance, False, True, True)
        radius = np.abs(container_ops.position_perturbation(*args)[:, 0]) * binary.semi_major_axis
        assert_array_less(radius, overshoot * r_ampl)
        assert_array_less(r_ampl, radius.max())

        # velocity
        args = (system_container.secondary, binary.semi_major_axis, False, True, True)
        velocities = np.abs(container_ops.velocity_perturbation(*args)[:, 0])
        assert_array_less(velocities, overshoot * amplitude)
        assert_array_less(amplitude, velocities.max())

        # acceleration
        a_amp = amplitude * omega
        args = (system_container.secondary, binary.semi_major_axis, False, True, True)
        acc = np.abs(container_ops.gravity_acc_perturbation(*args)[:, 0])
        assert_array_less(acc, overshoot * a_amp)
        assert_array_less(a_amp, acc.max())

        # teff
        t_ampl = pulse_meta[0]['temperature_amplitude_factor'] * system_container.secondary.t_eff
        args = (system_container.secondary, binary.semi_major_axis, False, True)
        ts = np.abs(container_ops.temp_perturbation(*args))
        assert_array_less(ts, overshoot * t_ampl)
        assert_array_less(t_ampl, ts.max())

    def test_binary_pulsating_lc(self):
        freq = 15
        pulse_meta = [{
            'l': 2,
            'm': 1,
            'amplitude': 100 * u.m / u.s,
            'frequency': freq / u.d,
            'start_phase': 0.0,
            'horizontal_to_radial_amplitude_ratio': 0.0,
            'temperature_amplitude_factor': 0.01
        }]

        binary = self.prepare_binary(pulse_meta)
        o = Observer(passband=['Generic.Bessell.V', ], system=binary)

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

    def test_binary_pulsating_rv(self):
        freq = 15
        pulse_meta = [{
            'l': 1,
            'm': 1,
            'amplitude': 5 * u.km / u.s,
            'frequency': freq / u.d,
            'start_phase': 0.0,
            'horizontal_to_radial_amplitude_ratio': 0.0,
            'temperature_amplitude_factor': 0.01
        }]

        binary = self.prepare_binary(pulse_meta)
        o = Observer(system=binary)

        start_phase = 0.05
        phases = np.linspace(start_phase, start_phase + 2 / (freq * binary.period), 10)

        expected = testutils.load_radial_curve("binary.pulsating.json")
        expected_rvp, expected_rvs = testutils.normalize_rv_for_unittests(expected["primary"], expected["secondary"])

        obtained = o.rv(phases=phases, method='radiometric')
        obtained_rvp, obtained_rvs = testutils.normalize_rv_for_unittests(obtained[1]['primary'], obtained[1]['secondary'])

        assert_array_equal(np.round(expected_rvp, 3), np.round(obtained_rvp, 3))
        assert_array_equal(np.round(expected_rvs, 3), np.round(obtained_rvs, 3))

        # o.plot.rv_curve()
