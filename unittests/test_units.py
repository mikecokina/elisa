from elisa.units import *
from unittests.utils import ElisaTestCase


class TestUnits(ElisaTestCase):
    def test_MASS_UNIT(self):
        self.assertEqual(MASS_UNIT, u.kg)

    def test_TEMPERATURE_UNIT(self):
        self.assertEqual(TEMPERATURE_UNIT, u.K)

    def test_DISTANCE_UNIT(self):
        self.assertEqual(DISTANCE_UNIT, u.m)

    def test_TIME_UNIT(self):
        self.assertEqual(TIME_UNIT, u.s)

    def test_ARC_UNIT(self):
        self.assertEqual(ARC_UNIT, u.rad)

    def test_PERIOD_UNIT(self):
        self.assertEqual(PERIOD_UNIT, u.d)

    def test_VELOCITY_UNIT(self):
        self.assertEqual(VELOCITY_UNIT, u.m / u.s)

    def test_ACCELERATION_UNIT(self):
        self.assertEqual(ACCELERATION_UNIT, u.m / (u.s**2))

    def test_LOG_ACCELERATION_UNIT(self):
        self.assertEqual(LOG_ACCELERATION_UNIT, u.dex(u.m / (u.s**2)))

    def test_FREQUENCY_UNIT(self):
        self.assertEqual(FREQUENCY_UNIT, u.Hz)
