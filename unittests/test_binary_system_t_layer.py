import numpy as np
from numpy.testing import assert_array_equal

from elisa.binary_system import t_layer
from unittests.utils import ElisaTestCase


class SupportMethodTestCase(ElisaTestCase):

    @staticmethod
    def test_jd_to_phase():
        p, t0 = 5.0, 3.0
        jd = np.array([1, 2, 3.1, 4, 5, 6, 7, 8, 9, 10, 11])
        expected = np.round([0.6, 0.8, 0.02, 0.2, 0.4, 0.6, 0.8, 0., 0.2, 0.4, 0.6], 5)
        obtained = np.round(t_layer.jd_to_phase(t0, p, jd), 5)
        assert_array_equal(expected, obtained)

    @staticmethod
    def test_phase_to_jd():
        p, t0 = 8.5, 3.0
        jd = np.array([3.1, 4, 5, 6, 7, 8, 9, 10, 11])
        phases = t_layer.jd_to_phase(t0, p, jd)
        obtained = np.round(t_layer.phase_to_jd(t0, p, phases), 5)
        assert_array_equal(obtained, jd)
