import numpy as np
import elisa.const as c

from numpy.testing import assert_array_equal
from elisa.binary_system.orbit import orbit
from elisa import units as u
from unittests.utils import ElisaTestCase


class OrbitTestCase(ElisaTestCase):

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
            o = orbit.Orbit(**combo)
            obtained_distances.append(round(o.periastron_distance, 10))
        self.assertEqual(expected_distances, obtained_distances)

    def test_periastron_phase(self):
        expected_distances, obtained_distances = [], []
        expected_hardcoded = [1., .2, 0.2, 0.2, 0.7, 0.4, 0.8]

        for i, combo in enumerate(self.params_combination):
            o = orbit.Orbit(**combo)

            expected_distances.append(round(o.periastron_distance, 6))
            obtained_distances.append(round(o.orbital_motion(phase=o.periastron_phase)[0][0], 6))

        self.assertEqual(expected_distances, expected_hardcoded)
        self.assertEqual(expected_distances, obtained_distances)

    def test_true_anomaly_to_azimuth(self):
        true_anomalies = np.array([np.radians(0.0), np.radians(45.0)])

        expected = [[1.570796, 2.356194],
                    [1.570796, 2.356194],
                    [1.570796, 2.356194],
                    [1.570796, 2.356194],
                    [5.497787, 0.],
                    [5.497787, 0.],
                    [0.261799, 1.047198]]
        obtained = []
        for i, combo in enumerate(self.params_combination):
            o = orbit.Orbit(**combo)
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
            o = orbit.Orbit(**combo)
            obtained.append(np.round(o.relative_radius(true_anomalies), 6))
        assert_array_equal(expected, obtained)

    def test_get_conjuction(self):
        expected = [
            {
                "primary_eclipse": {
                    "true_anomaly": 0.0,
                    "true_phase": 0.0
                },
                "secondary_eclipse": {
                    "true_anomaly": 3.1416,
                    "true_phase": 0.5
                }
            },
            {
                "primary_eclipse": {
                    "true_anomaly": 0.0,
                    "true_phase": 0.0
                },
                "secondary_eclipse": {
                    "true_anomaly": 3.1416,
                    "true_phase": 0.5
                }
            },
            {
                "primary_eclipse": {
                    "true_anomaly": 1.309,
                    "true_phase": 0.1495
                },
                "secondary_eclipse": {
                    "true_anomaly": 4.4506,
                    "true_phase": 0.7719
                }
            }
        ]
        obtained = list()

        for i, combo in enumerate(self.params_combination[np.array([0, 1, -1])]):
            o = orbit.Orbit(**combo)
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
        expected = np.array([np.array([[1., 0.9425, 5.6549, -0.1],
                                       [1., 1.5708, 0., 0.],
                                       [1., 2.1991, 0.6283, 0.1],
                                       [1., 4.7124, 3.1416, 0.5],
                                       [1., 1.5708, 6.2832, 1.],
                                       [1., 2.1991, 0.6283, 1.1]]),
                             np.array([[0.8791, 5.4529, 3.8821, -0.1],
                                       [0.2, 1.5708, 0., 0.],
                                       [0.8791, 3.9719, 2.4011, 0.1],
                                       [1.8, 4.7124, 3.1416, 0.5],
                                       [0.2, 1.5708, 6.2832, 1.],
                                       [0.8791, 3.9719, 2.4011, 1.1]]),
                             np.array([[0.8148, 0.7323, 0.4705, -0.1],
                                       [0.9128, 1.5708, 1.309, 0.],
                                       [1.0384, 2.2197, 1.9579, 0.1],
                                       [1.1399, 4.0651, 3.8033, 0.5],
                                       [0.9128, 1.5708, 1.309, 1.],
                                       [1.0384, 2.2197, 1.9579, 1.1]])])

        for i, combo in enumerate(self.params_combination[np.array([0, 1, -1])]):
            o = orbit.Orbit(**combo)
            obtained.append(np.round(o.orbital_motion(phases), 4))

        self.assertTrue(np.all(np.cos(expected[0]) - np.cos(obtained[0])) < 1e-4)
        self.assertTrue(np.all(expected[:2] - obtained[:2]) < 1e-4)

    def test_azimuth_to_true_anomaly(self):
        o = orbit.Orbit(**self.params_combination[0])
        o.argument_of_periastron = (139 * u.deg).to(u.rad).value

        azimuths = np.array([1.56, 0.25, 3.14, 6.0, 156])
        expected = [5.4172, 4.1072, 0.714, 3.574, 2.7775]
        obtained = np.round(o.azimuth_to_true_anomaly(azimuths), 4)
        assert_array_equal(obtained, expected)


class OrbitStaticMethodTestCase(ElisaTestCase):
    def test_angular_velocity(self):
        expected = 7.349e-05
        obtained = round(orbit.angular_velocity(1.25, 0.3, 0.869), 8)
        self.assertEqual(expected, obtained)
