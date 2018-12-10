import unittest
import numpy as np
from engine.orbit import Orbit
import engine.const as c
from astropy import units as u


class TestOrbit(unittest.TestCase):

    def setUp(self):
        self.params_combination = [{"argument_of_periastron": np.pi / 2.0, "period": 0.9,
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
                                   ]

    def test_periastron_distance(self):

        expected_distances = [1.0, 0.2, 0.2, 0.2, 0.7, 0.4, 0.8]
        obtained_distances = []
        for i, combo in enumerate(self.params_combination):
            o = Orbit(period=combo["period"], inclination=combo["inclination"],
                      argument_of_periastron=combo["argument_of_periastron"], eccentricity=combo["eccentricity"])
            obtained_distances.append(round(o.periastron_distance, 10))

        self.assertEquals(expected_distances, obtained_distances)

    def test_periastron_phase(self):
        expected_distances, obtained_distances = [], []

        for i, combo in enumerate(self.params_combination):
            o = Orbit(period=combo["period"], inclination=combo["inclination"],
                      argument_of_periastron=combo["argument_of_periastron"], eccentricity=combo["eccentricity"])

            expected_distances.append(round(o.periastron_distance, 6))
            obtained_distances.append(round(o.orbital_motion(phase=o.periastron_phase)[0][0], 6))

        self.assertEquals(expected_distances, obtained_distances)



