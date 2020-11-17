import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from elisa import units as u
from elisa import utils

from elisa.base.star import Star
from elisa.single_system.system import SingleSystem
from elisa.pulse import pulsations, utils as putils
from unittests.utils import ElisaTestCase
from unittests import utils as testutils


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
                'polar_log_g': 4.4 * u.dex(u.cm/u.s**2)
            }

        self.system_params = {'gamma': 0 * u.km / u.s,
                              'inclination': 80 * u.deg,
                              'rotation_period': 30 * u.d,
                              }

    def prepare_system(self):
        # self.star_params['pulsations'] = self.pulsation_modes
        star = Star(pulsations=self.pulsation_modes, **self.star_params)
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
            'start_phase': 0.75,
        }]

        time = 0

        single = self.prepare_system()

        system_container = testutils.prepare_single_system_container(single)
        system_container.build_mesh()
        points = utils.cartesian_to_spherical(system_container.star.points)

        for ll in range(0, 10):
            for mm in range(ll+1):
                puls_meta[0]['l'] = ll
                puls_meta[0]['m'] = mm
                single.star.pulsations = puls_meta
                single.init()

                mode = single.star.pulsations[0]
                exponential = putils.generate_time_exponential(mode, time)
                sph_harm = pulsations.spherical_harmonics(mode, points, exponential)

                test_val = np.sqrt(np.sum(np.abs(sph_harm)**2)/points.shape[0])
                assert_almost_equal(test_val, 1.0, 2)
