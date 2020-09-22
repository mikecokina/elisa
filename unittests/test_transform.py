import numpy as np
from numpy.testing import assert_array_equal

from elisa import const, units as u
from elisa.base.transform import SystemProperties, BodyProperties, StarProperties, SpotProperties
from elisa.binary_system.transform import BinarySystemProperties
from elisa.binary_system.orbit.transform import OrbitProperties
from unittests.utils import ElisaTestCase


def generate_test(values, transform, expected, _round=None):
    obtained = [transform(val) for val in values]
    if _round is not None:
        obtained = np.round(obtained, _round)
    assert_array_equal(expected, obtained)


def generate_raise_test(self, value, transofrm, in_exception):
    with self.assertRaises(Exception) as context:
        transofrm(value)
    self.assertTrue(in_exception in str(context.exception))


class TransformBinarySystemPropertiesTestCase(ElisaTestCase):
    @staticmethod
    def test_eccentricity():
        valid_values = [0.0, 0.1, 0.95]
        generate_test(valid_values, BinarySystemProperties.eccentricity, valid_values)

    def test_eccentricity_raise(self):
        generate_raise_test(self, 1.1, BinarySystemProperties.eccentricity, "out of")

    @staticmethod
    def test_argument_of_periastron():
        valid_values = [0.0, 180., const.PI * u.rad, 180.0 * u.deg]
        expected = [0., 3.1416, 3.1416, 3.1416]
        generate_test(valid_values, BinarySystemProperties.argument_of_periastron, expected, 4)

    def test_argument_of_periastron_raise(self):
        generate_raise_test(self, "98.2", BinarySystemProperties.argument_of_periastron, "is not")

    def test_phase_shift(self):
        self.assertTrue(BinarySystemProperties.phase_shift(9.1234) == 9.1234)

    @staticmethod
    def test_primary_minimum_time():
        valid_values = [1.0, 180. * u.PERIOD_UNIT, 86400.0 * u.s]
        expected = [1.0, 180.0, 1.0]
        generate_test(valid_values, BinarySystemProperties.primary_minimum_time, expected)

    def test_primary_minimum_time_raise(self):
        generate_raise_test(self, "98.2", BinarySystemProperties.primary_minimum_time, "is not")


class TransformSystemPropertiesTestCase(ElisaTestCase):
    @staticmethod
    def test_inclination():
        valid_values = [0.0, 180., const.PI * u.rad, 180.0 * u.deg]
        expected = [0., 3.1416, 3.1416, 3.1416]
        generate_test(valid_values, SystemProperties.inclination, expected, 4)

    def test_inclination_raise(self):
        generate_raise_test(self, const.FULL_ARC * u.rad, SystemProperties.inclination, "is out of bounds")

    @staticmethod
    def test_period():
        valid_values = [1.0, 180. * u.PERIOD_UNIT, 86400.0 * u.s]
        expected = [1.0, 180.0, 1.0]
        generate_test(valid_values, SystemProperties.period, expected)

    def test_period_raise(self):
        generate_raise_test(self, "98.2", SystemProperties.period, "is not")

    @staticmethod
    def test_gamma():
        valid_values = [1.0, 180. * u.VELOCITY_UNIT, 1.0 * u.km / u.s]
        expected = [1.0, 180.0, 1000.0]
        generate_test(valid_values, SystemProperties.gamma, expected)

    def test_gamma_raise(self):
        generate_raise_test(self, "98.2", SystemProperties.gamma, "is not")
        generate_raise_test(self, 1.0 * u.km * u.s, SystemProperties.gamma, "are not convertible")

    @staticmethod
    def test_additional_light():
        valid_values = [0.0, 0.55, 1.0]
        generate_test(valid_values, SystemProperties.additional_light, valid_values)

    def test_additional_light_raise(self):
        generate_raise_test(self, 1.5, SystemProperties.additional_light, "Invalid value")


class TransformBodyPropertiesTestCase(ElisaTestCase):
    @staticmethod
    def test_synchronicity():
        valid_values = [1.0, 1.1, 0.5]
        obtained = [BodyProperties.synchronicity(val) for val in valid_values]
        assert_array_equal(valid_values, obtained)

    def test_synchronicity_raise(self):
        generate_raise_test(self, -0.14, BodyProperties.synchronicity, "Invalid synchronicity")

    def test_mass(self):
        valid_values = [1.0, 1.0 * u.solMass, 30. * u.MASS_UNIT]
        obtained = np.array([BodyProperties.mass(val) for val in valid_values])
        expected = np.array([1.98847542e+30, 1.98847542e+30, 3.00000000e+01])
        print(obtained)
        print(expected)
        print(abs((expected - obtained) / expected))
        self.assertTrue(np.all(abs((expected - obtained) / expected) < 1e-8))

    def test_mass_raise(self):
        generate_raise_test(self, -0.14, BodyProperties.mass, "Invalid mass")

    @staticmethod
    def test_albedo():
        valid_values = [0.0, 1.0, 0.5]
        generate_test(valid_values, BodyProperties.albedo, valid_values)

    def test_albedo_raise(self):
        generate_raise_test(self, -0.14, BodyProperties.albedo, "is out of range")
        generate_raise_test(self, 15239, BodyProperties.albedo, "is out of range")

    @staticmethod
    def test_discretization_factor():
        valid_values = [0.0, 45., 0.25 * u.rad, 45.0 * u.deg]
        expected = [0., 0.7854, 0.25, 0.7854]
        generate_test(valid_values, BodyProperties.discretization_factor, expected, 4)

    @staticmethod
    def test_t_eff():
        valid_values = [10.0, 180.0 * u.TEMPERATURE_UNIT]
        expected = [10., 180.]
        generate_test(valid_values, BodyProperties.t_eff, expected, 4)

    def test_t_eff_raise(self):
        generate_raise_test(self, "10", BodyProperties.t_eff, "is not")

    @staticmethod
    def test_polar_radius():
        valid_values = [10.0, 180.0 * u.DISTANCE_UNIT, 1000 * u.m, 1 * u.km]
        expected = [10., 180., 1000., 1000.]
        generate_test(valid_values, BodyProperties.polar_radius, expected, 4)


class TransformStarPropertiesTestCase(ElisaTestCase):
    @staticmethod
    def test_metallicity():
        valid_values = [-0.1, 0.0, 1.0, 2.1]
        generate_test(valid_values, StarProperties.metallicity, valid_values, 4)

    @staticmethod
    def test_surface_potential():
        valid_values = [0.1, 1.1, 10001.123]
        generate_test(valid_values, StarProperties.surface_potential, valid_values, 4)

    def test_surface_potential_raise(self):
        generate_raise_test(self, -0.1, StarProperties.surface_potential, "Invalid surface potential")

    @staticmethod
    def test_gravity_darkening():
        valid_values = [0.1, 0.0, 1]
        generate_test(valid_values, StarProperties.gravity_darkening, valid_values, 4)

    def test_gravity_darkening_raise(self):
        generate_raise_test(self, -0.1, StarProperties.gravity_darkening, "is out of range")

    @staticmethod
    def test_polar_log_g():
        valid_values = [0.0, 1.0, 4.0 * u.LOG_ACCELERATION_UNIT]
        expected = [0.0, 1.0, 4.0]
        generate_test(valid_values, StarProperties.polar_log_g, expected, 4)


class TransformSpotPropertiesTestCase(ElisaTestCase):
    @staticmethod
    def test_latitude():
        valid_values = [0.0, 180., const.PI * u.rad, 180.0 * u.deg]
        expected = [0., 3.1416, 3.1416, 3.1416]
        generate_test(valid_values, SpotProperties.latitude, expected, 4)

    @staticmethod
    def test_longitude():
        valid_values = [0.0, 180., const.PI * u.rad, 180.0 * u.deg]
        expected = [0., 3.1416, 3.1416, 3.1416]
        generate_test(valid_values, SpotProperties.longitude, expected, 4)

    @staticmethod
    def test_angular_radius():
        valid_values = [0.0, 180., const.PI * u.rad, 180.0 * u.deg]
        expected = [0., 3.1416, 3.1416, 3.1416]
        generate_test(valid_values, SpotProperties.angular_radius, expected, 4)

    @staticmethod
    def test_temperature_factor():
        valid_values = [0.1, 0.5, 0.9999]
        generate_test(valid_values, SpotProperties.temperature_factor, valid_values)


class TransformOrbitProperties(ElisaTestCase):
    @staticmethod
    def test_period():
        periods = [0.25 * u.d, 0.65, 86400 * u.s]
        expected = [0.25, 0.65, 1.0]
        generate_test(periods, OrbitProperties.period, expected)

    @staticmethod
    def test_eccentricity():
        values = [0.0, 0.1, 0.2, 0.999999]
        generate_test(values, OrbitProperties.eccentricity, values)

    @staticmethod
    def test_argument_of_periastron():
        periastrons = [135 * u.deg, 0.65, 1.56 * u.rad]
        expected = [2.356, 0.65, 1.56]
        generate_test(periastrons, OrbitProperties.argument_of_periastron, expected, 3)

    @staticmethod
    def test_inclination():
        inclinations = [135 * u.deg, 0.65, 1.56 * u.rad]
        expected = [2.356, 0.65, 1.56]
        generate_test(inclinations, OrbitProperties.inclination, expected, 3)
