import numpy as np
import elisa.const as c

from numpy.testing import assert_array_equal
from elisa.single_system.orbit import orbit
from unittests.utils import ElisaTestCase


class RotationalMotionTestCase(ElisaTestCase):
    def setUp(self):
        self.params_combination = np.array(
            [{"phase_shift": 0, "rotation_period": 0.9, "inclination": c.HALF_PI},

             {"phase_shift": np.pi / 2.0, "rotation_period": 0.9, "inclination": c.HALF_PI},

             {"rotation_period": 0.9, "inclination": c.HALF_PI},

             {"phase_shift": np.pi / 2.0, "rotation_period": 0.9, "inclination": np.radians(86.4)},
             ])

    def test_rotational_motion(self):
        phases = np.array([-0.1, 0.0, 0.1, 0.5, 1.0, 1.1])
        obtained = []
        expected = [np.array([[-0.6283, -0.1],
                              [0., 0.],
                              [0.6283, 0.1],
                              [3.1416, 0.5],
                              [6.2832, 1.],
                              [6.9115, 1.1]]),
                    np.array([[-10.4979, -0.1],
                              [-9.8696, 0.],
                              [-9.2413, 0.1],
                              [-6.728, 0.5],
                              [-3.5864, 1.],
                              [-2.9581, 1.1]]),
                    np.array([[-10.4979, -0.1],
                              [-9.8696, 0.],
                              [-9.2413, 0.1],
                              [-6.728, 0.5],
                              [-3.5864, 1.],
                              [-2.9581, 1.1]])]
        for i, combo in enumerate(self.params_combination[np.array([0, 1, -1])]):
            o = orbit.Orbit(**combo)
            obtained.append(np.round(o.rotational_motion(phases), 4))
        assert_array_equal(expected, obtained)

    def test_rotational_motion_from_azimuths(self):
        azimuths = np.array([-0.1, 0.0, 0.1, 0.5, 1.0, 1.1]) * c.FULL_ARC
        obtained = []
        expected = [np.array([[-0.6283, -0.1],
                              [0., 0.],
                              [0.6283, 0.1],
                              [3.1416, 0.5],
                              [6.2832, 1.],
                              [6.9115, 1.1]]),
                    np.array([[-0.6283, 1.4708],
                              [0., 1.5708],
                              [0.6283, 1.6708],
                              [3.1416, 2.0708],
                              [6.2832, 2.5708],
                              [6.9115, 2.6708]]),
                    np.array([[-0.6283, 1.4708],
                              [0., 1.5708],
                              [0.6283, 1.6708],
                              [3.1416, 2.0708],
                              [6.2832, 2.5708],
                              [6.9115, 2.6708]])]
        for i, combo in enumerate(self.params_combination[np.array([0, 1, -1])]):
            o = orbit.Orbit(**combo)
            obtained.append(np.round(o.rotational_motion_from_azimuths(azimuths), 4))
        assert_array_equal(expected, obtained)
