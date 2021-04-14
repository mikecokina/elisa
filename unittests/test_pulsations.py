import os
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from elisa import units as u
from elisa import utils
from elisa import const

from elisa.base.star import Star
from elisa.single_system.system import SingleSystem
from elisa.pulse import pulsations, utils as putils
from unittests.utils import ElisaTestCase
from unittests import utils as testutils

STAR_PARAMS = {
    'mass': 2.15 * u.solMass,
    't_eff': 10000 * u.K,
    'gravity_darkening': 1.0,
    'discretization_factor': 3,
    'albedo': 0.6,
    'metallicity': 0.0,
    'polar_log_g': 4.4 * u.dex(u.cm / u.s ** 2)
}

SYSTEM_PARMAS = {'gamma': 0 * u.km / u.s,
                 'inclination': 80 * u.deg,
                 'rotation_period': 30 * u.d,
                 }


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

    def prepare_system(self, pulsations):
        star = Star(pulsations=pulsations, **STAR_PARAMS)
        return SingleSystem(star=star, **SYSTEM_PARMAS)

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

        single = self.prepare_system(pulsations=pulse_meta)
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


#     def test_displacement(self):
#         """Test if mode displacement is within range."""
#         pulse_meta = [{
#             'l': 1,
#             'm': 1,
#             'amplitude': 1 * u.m / u.s,
#             'frequency': 1 / u.d,
#             'start_phase': 0.0,
#         }]
#
#         time = 0
#         single = self.prepare_system(pulsations=pulse_meta)
#         system_container = testutils.prepare_single_system_container(single)
#         system_container.build_mesh()
#         pulsations.generate_harmonics(system_container.star, com_x=0.0, phase=0.0, time=time)
#         points = utils.cartesian_to_spherical(system_container.star.points)
#
#         mode = system_container.star.pulsations[0]
#
#         displacement = pulsations.calculate_displacement_coordinates(mode, points, mode.point_harmonics,
#                                                                      mode.point_harmonics_derivatives, scale=1)
#         radial_disp = displacement[:, 0]
#         assert_array_less(np.abs(radial_disp), np.full(radial_disp.shape, 17000))
#         self.assertGreater(np.max(radial_disp), 16000)
#         phi_disp = displacement[:, 1]
#         assert_array_less(np.abs(phi_disp), np.full(phi_disp.shape, 0.017))
#         self.assertGreater(np.max(phi_disp), 0.016)
#         theta_disp = displacement[:, 2]
#         assert_array_less(np.abs(theta_disp), np.full(theta_disp.shape, 0.001))
#         self.assertGreater(np.max(phi_disp), 0.00095)
