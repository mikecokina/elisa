import unittest

import numpy as np
from astropy import units as u
from numpy.testing import assert_array_equal

import elisa.const as c
from elisa.orbit import Orbit


class TestOrbit(unittest.TestCase):

    def setUp(self):
        self.params_combination = np.array(
            [{"argument_of_periastron": np.pi / 2.0, "period": 0.9,
              "eccentricity": 0.0, "inclination": c.HALF_PI},

             {"argument_of_periastron": np.pi / 2.0, "period": 0.9,
              "eccentricity": 0.8, "inclination": c.HALF_PI},

             {"argument_of_periastron": np.pi / 2.0, "period": 0.9,
              "eccentricity": 0.8, "inclination": np.radians(89)},

             {"argument_of_periastron": np.pi / 2.0, "period": 0.9,
              "eccentricity": 0.8, "inclination": np.radians(91)},

             {"argument_of_periastron": 315 * u.deg, "period": 0.8,
              "eccentricity": 0.3, "inclination": c.HALF_PI},

             {"argument_of_periastron": 315 * u.deg, "period": 0.8,
              "eccentricity": 0.6, "inclination": 120 * u.deg},

             {"argument_of_periastron": 15 * u.deg, "period": 0.8,
              "eccentricity": 0.2, "inclination": 15 * u.deg}
             ])

    def test_periastron_distance(self):

        expected_distances = [1.0, 0.2, 0.2, 0.2, 0.7, 0.4, 0.8]
        obtained_distances = []
        for i, combo in enumerate(self.params_combination):
            o = Orbit(**combo)
            obtained_distances.append(round(o.periastron_distance, 10))
        self.assertEquals(expected_distances, obtained_distances)

    def test_periastron_phase(self):
        expected_distances, obtained_distances = [], []
        expected_hardcoded = [1., .2, 0.2, 0.2, 0.7, 0.4, 0.8]

        for i, combo in enumerate(self.params_combination):
            o = Orbit(**combo)

            expected_distances.append(round(o.periastron_distance, 6))
            obtained_distances.append(round(o.orbital_motion(phase=o.periastron_phase)[0][0], 6))

        self.assertEqual(expected_distances, expected_hardcoded)
        self.assertEquals(expected_distances, obtained_distances)

    def test_true_anomaly_to_azimuth(self):
        true_anomalies = np.array([np.radians(0.0), np.radians(45.0)])
        expected = [[0.027416, 0.812814],
                    [0.027416, 0.812814],
                    [0.027416, 0.812814],
                    [0.027416, 0.812814],
                    [5.497787, 0.000000],
                    [5.497787, 0.000000],
                    [0.261799, 1.047198]]
        obtained = []
        for i, combo in enumerate(self.params_combination):
            o = Orbit(**combo)

            obtained.append(np.round(o.true_anomaly_to_azimuth(true_anomalies), 6))

        assert_array_equal(obtained, expected)

    def test_relative_radius(self):
        true_anomalies = np.array([np.radians(0.0), np.radians(45.0), np.radians(-45.0)])
        expected = [[1., 1., 1.],
                    [0.2, 0.229931, 0.229931],
                    [0.2, 0.229931, 0.229931],
                    [0.2, 0.229931, 0.229931],
                    [0.7, 0.750743, 0.750743],
                    [0.4, 0.449355, 0.449355],
                    [0.8, 0.841057, 0.841057]]
        obtained = []

        for i, combo in enumerate(self.params_combination):
            o = Orbit(**combo)
            obtained.append(np.round(o.relative_radius(true_anomalies), 6))
        assert_array_equal(expected, obtained)

    def test_get_conjuction(self):
        expected = [
            {
                'primary_eclipse': {'true_anomaly': 1.5434, 'true_phase': 0.2456},
                'secondary_eclipse': {'true_anomaly': 4.6850, 'true_phase': 0.7456},
            },
            {
                'primary_eclipse': {'true_anomaly': 1.5434, 'true_phase': 0.0251},
                'secondary_eclipse': {'true_anomaly': 4.6850, 'true_phase': 0.9730},
            },

            {
                'primary_eclipse': {'true_anomaly': 1.3090, 'true_phase': 0.1495},
                'secondary_eclipse': {'true_anomaly': 4.4506, 'true_phase': 0.7719},
            }
        ]
        obtained = list()

        for i, combo in enumerate(self.params_combination[np.array([0, 1, -1])]):
            o = Orbit(**combo)
            con = o.get_conjuction()
            obtained.append({
                eclipse: {q: round(con[eclipse][q], 4) for q in ['true_anomaly', 'true_phase']}
                for eclipse in ['primary_eclipse', 'secondary_eclipse']
            })

        for idx, _ in enumerate(obtained):
            self.assertDictEqual(expected[idx], obtained[idx])

    def test_orbital_motion(self):
        phases = [-0.1, 0.0, 0.1, 0.5, 1.0, 1.1]
        obtained = []
        expected = np.array([
            [[1., 0.9425, 0.9151, -0.1],
             [1., 1.5708, 1.5434, 0.],
             [1., 2.1991, 2.1717, 0.1],
             [1., 4.7124, 4.685, 0.5],
             [1., 1.5708, 1.5434, 1.],
             [1., 2.1991, 2.1717, 1.1]],
            [[0.727, 4.0569, 4.0295, -0.1],
             [0.3523, 1.5708, 1.5434, 0.],
             [1.0121, 2.5345, 2.5071, 0.1],
             [1.7969, 3.1982, 3.1708, 0.5],
             [0.3523, 1.5708, 1.5434, 1.],
             [1.0121, 2.5345, 2.5071, 1.1]],
            [[0.8148, 0.7323, 0.4705, -0.1],
             [0.9128, 1.5708, 1.309, 0.],
             [1.0384, 2.2197, 1.9579, 0.1],
             [1.1399, 4.0651, 3.8033, 0.5],
             [0.9128, 1.5708, 1.309, 1.],
             [1.0384, 2.2197, 1.9579, 1.1]]])
        for i, combo in enumerate(self.params_combination[np.array([0, 1, -1])]):
            o = Orbit(**combo)
            obtained.append(np.round(o.orbital_motion(phases), 4))
        assert_array_equal(expected, obtained)
