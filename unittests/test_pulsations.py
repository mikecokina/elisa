import numpy as np
from numpy.testing import assert_array_equal

from elisa import units as u

from elisa.base.star import Star
from unittests.utils import ElisaTestCase


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
                'discretization_factor': 5,
                'albedo': 0.6,
                'metallicity': 0.0,
            }

    def prepare_system(self):
        # self.star_params['pulsations'] = self.pulsation_modes
        return Star(pulsations=self.pulsation_modes, **self.star_params)

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

        star = self.prepare_system()

        for ii, mode in star.pulsations.items():
            list_to_check = [mode.l, mode.m, np.round(mode.amplitude, rounding_prec),
                             np.round(mode.frequency, rounding_prec), np.round(mode.start_phase, rounding_prec)]
            assert_array_equal(expected_values[ii], list_to_check)
