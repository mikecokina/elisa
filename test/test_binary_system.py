import unittest
from engine.binary_system import BinarySystem

class TestBinarySystem(unittest.TestCase):

    def test_orbit(self):
        bs = BinarySystem(period=15)
        bs.period = 10.0
        self.assertEqual(bs.orbit.period, 10)

        bs.eccentricity = 0.6
        self.assertEqual(bs.orbit.eccentricity, 0.6)
