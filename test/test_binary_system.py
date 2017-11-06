import unittest
import numpy as np
from engine.binary_system import BinarySystem
from engine.system import System
from astropy import units as u


class TestBinarySystemProperties(unittest.TestCase):

    def test_orbit_change(self):
        bs = BinarySystem()

        bs.period = 10.0
        self.assertEqual(bs.orbit.period, 10)

        bs.eccentricity = 0.6
        self.assertEqual(bs.orbit.eccentricity, 0.6)

        bs.inclination = 0.25
        self.assertEqual(bs.orbit.inclination, 0.25)

        bs.periastron = 0.2345
        self.assertEqual(bs.orbit.periastron, 0.2345)

    def test_binary_system_unit(self):
        bs = BinarySystem()

        bs.period = 432000 * u.s
        self.assertEqual(bs.period, 5)
        with self.assertRaises(TypeError):
            bs.period = '0000'

        bs.inclination = 90 * u.deg
        self.assertEqual(bs.inclination, 0.5 * np.pi)
        with self.assertRaises(TypeError):
            bs.inclination = '0000'

        bs.periastron = 45 * u.deg
        self.assertEqual(bs.periastron, np.pi * 0.25)
        with self.assertRaises(TypeError):
            bs.inclination = '0000'


        # add all

    def test_binary_change(self):
        bs = BinarySystem()

        bs.period = 12.0
        self.assertEqual(bs.period, 12)

        bs.gamma = 2.556
        self.assertEqual(bs.gamma, 2.556)
