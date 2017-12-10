import unittest
import numpy as np
from engine.orbit import Orbit
from engine.star import Star
import engine.const as c
from astropy import units as u


class TestOrbit(unittest.TestCase):

    def setUp(self):
        self.params_combination = [{"argument_of_periastron": np.pi / 2.0, "period": 0.9,
                                    "eccentricity": 0.0, "inclination": c.HALF_PI},

                                   {"argument_of_periastron": 315 * u.deg, "period": 0.8,
                                    "eccentricity": 0.3, "inclination": c.HALF_PI}]

    def test_periastron_distance(self):

        for i, combo in enumerate(self.params_combination):
            o = Orbit(period=combo["period"], inclination=combo["inclination"],
                      argument_of_periastron=combo["argument_of_periastron"], eccentricity=combo["eccentricity"])
            # print(o.periastron_distance)

t = TestOrbit()
t.setUp()
t.test_periastron_distance()