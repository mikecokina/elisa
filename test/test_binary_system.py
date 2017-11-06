import unittest
from engine.binary_system import BinarySystem
from astropy import units as u
import numpy as np

class TestBinarySystem(unittest.TestCase):

    def test_orbit_change(self):
        bs = BinarySystem()

        bs.period = 10.0
        self.assertEqual(bs.orbit.period, 10)
        self.assertEqual(bs.period, 10)

        bs.period = 864000 * u.s
        self.assertEqual(bs.orbit.period, 10)
        self.assertEqual(bs.period, 10)

        bs.eccentricity = 0.6
        self.assertEqual(bs.orbit.eccentricity, 0.6)
        self.assertEqual(bs.eccentricity, 0.6)

        bs.inclination = 0.25
        self.assertEqual(bs.orbit.inclination, 0.25)
        self.assertEqual(bs.inclination, 0.25)

        bs.inclination = 90 * u.deg
        self.assertEqual(bs.orbit.inclination, 0.5 * np.pi)
        self.assertEqual(bs.inclination, 0.5 * np.pi)

        bs.periastron = 1.289
        self.assertEqual(bs.orbit.periastron, 1.289)
        self.assertEqual(bs.periastron, 1.289)

        bs.periastron = 180 * u.deg
        self.assertEqual(bs.orbit.periastron, np.pi)
        self.assertEqual(bs.periastron, np.pi)

    def test_binary_change(self):
        bs = BinarySystem()

        bs.gamma = 2.5
        self.assertEqual(bs.gamma, 2.5)

        bs.gamma = 2.5 * u.km / u.s
        self.assertEqual(bs.gamma, 2500)

