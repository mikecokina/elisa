import unittest
from engine.binary_system import BinarySystem

class TestBinarySystem(unittest.TestCase):

    def test_orbit_change(self):
        bs = BinarySystem()

        bs.period = 10.0
        self.assertEqual(bs.orbit.period, 10)

        bs.eccentricity = 0.6
        self.assertEqual(bs.orbit.eccentricity, 0.6)

        bs.inclination = 0.25
        self.assertEqual(bs.orbit.inclination, 0.25)

        bs.periastron = 1.289
        self.assertEqual(bs.orbit.periastron, 1.289)
