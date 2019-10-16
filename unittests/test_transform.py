import numpy as np
from numpy.testing import assert_array_equal

from elisa.base.transform import SystemParameters, BodyParameters, StarParameters
from elisa.binary_system.transform import BinarySystemParameters
from unittests.utils import ElisaTestCase
from elisa import const, units
from astropy import units as au


class TransformBinarySystemParametersTestCase(ElisaTestCase):
    @staticmethod
    def test_eccentricity():
        valid_values = [0.0, 0.1, 0.95]
        obtained = [BinarySystemParameters.eccentricity(val) for val in valid_values]
        assert_array_equal(valid_values, obtained)

    def test_eccentricity_raise(self):
        with self.assertRaises(Exception) as context:
            BinarySystemParameters.eccentricity(1.1)
        self.assertTrue("out of" in str(context.exception))

    @staticmethod
    def test_argument_of_periastron():
        valid_values = [0.0, 180., const.PI * units.rad, 180.0 * units.deg]
        obtained = np.round([BinarySystemParameters.argument_of_periastron(val) for val in valid_values], 4)
        expected = [0., 3.1416, 3.1416, 3.1416]
        assert_array_equal(expected, obtained)

    def test_argument_of_periastron_raise(self):
        with self.assertRaises(Exception) as context:
            BinarySystemParameters.argument_of_periastron("98.2")
        self.assertTrue("is not" in str(context.exception))

    def test_phase_shift(self):
        self.assertTrue(BinarySystemParameters.phase_shift(9.1234) == 9.1234)

    @staticmethod
    def test_primary_minimum_time():
        valid_values = [1.0, 180. * units.PERIOD_UNIT, 86400.0 * au.s]
        obtained = [BinarySystemParameters.primary_minimum_time(val) for val in valid_values]
        expected = [1.0, 180.0, 1.0]
        assert_array_equal(expected, obtained)

    def test_primary_minimum_time_raise(self):
        with self.assertRaises(Exception) as context:
            BinarySystemParameters.primary_minimum_time("98.2")
        self.assertTrue("is not" in str(context.exception))


class TransformSystemParametersTestCase(ElisaTestCase):
    @staticmethod
    def test_inclination():
        valid_values = [0.0, 180., const.PI * units.rad, 180.0 * units.deg]
        obtained = np.round([SystemParameters.inclination(val) for val in valid_values], 4)
        expected = [0., 3.1416, 3.1416, 3.1416]
        assert_array_equal(expected, obtained)
    
    def test_inclination_raise(self):
        with self.assertRaises(Exception) as context:
            SystemParameters.inclination(const.FULL_ARC * units.rad)
        self.assertTrue("is out of bounds" in str(context.exception))

    @staticmethod
    def test_period():
        valid_values = [1.0, 180. * units.PERIOD_UNIT, 86400.0 * au.s]
        obtained = [SystemParameters.period(val) for val in valid_values]
        expected = [1.0, 180.0, 1.0]
        assert_array_equal(expected, obtained)
    
    def test_period_raise(self):
        with self.assertRaises(Exception) as context:
            SystemParameters.period("98.2")
        self.assertTrue("is not" in str(context.exception))

    @staticmethod
    def test_gamma():
        valid_values = [1.0, 180. * units.VELOCITY_UNIT, 1.0 * units.km / au.s]
        obtained = [SystemParameters.gamma(val) for val in valid_values]
        expected = [1.0, 180.0, 1000.0]
        assert_array_equal(expected, obtained)

    def test_gamma_raise(self):
        with self.assertRaises(Exception) as context:
            SystemParameters.gamma("98.2")
        self.assertTrue("is not" in str(context.exception))

        with self.assertRaises(Exception) as context:
            SystemParameters.gamma(1.0 * units.km * au.s)
        self.assertTrue("are not convertible" in str(context.exception))

    @staticmethod
    def test_additional_light():
        valid_values = [0.0, 0.55, 1.0]
        obtained = [SystemParameters.additional_light(val) for val in valid_values]
        assert_array_equal(valid_values, obtained)

    def test_additional_light_raise(self):
        with self.assertRaises(Exception) as context:
            SystemParameters.additional_light(1.5)
        self.assertTrue("Invalid value" in str(context.exception))


class TransformBodyParametersTestCase(ElisaTestCase):
    @staticmethod
    def test_synchronicity():
        valid_values = [1.0, 1.1, 0.5]
        obtained = [BodyParameters.synchronicity(val) for val in valid_values]
        assert_array_equal(valid_values, obtained)

    def test_synchronicity_raise(self):
        with self.assertRaises(Exception) as context:
            BodyParameters.synchronicity(-0.14)
        self.assertTrue("Invalid synchronicity" in str(context.exception))

    def test_mass(self):
        valid_values = [1.0, 1.0 * units.solMass, 30. * units.MASS_UNIT]
        obtained = np.array([BodyParameters.mass(val) for val in valid_values])
        expected = np.array([1.98847542e+30, 1.98847542e+30, 3.00000000e+01])
        self.assertTrue(np.all(expected - obtained < 1e22))

    def test_mass_raise(self):
        with self.assertRaises(Exception) as context:
            BodyParameters.mass(-0.14)
        self.assertTrue("Invalid mass" in str(context.exception))

    @staticmethod
    def test_albedo():
        valid_values = [0.0, 1.0, 0.5]
        obtained = [BodyParameters.albedo(val) for val in valid_values]
        assert_array_equal(obtained, valid_values)

    def test_albedo_raise(self):
        with self.assertRaises(Exception) as context:
            BodyParameters.albedo(-0.14)
        self.assertTrue("is out of range" in str(context.exception))

        with self.assertRaises(Exception) as context:
            BodyParameters.albedo(15239)
        self.assertTrue("is out of range" in str(context.exception))

    @staticmethod
    def test_discretization_factor():
        valid_values = [0.0, 180., const.PI * units.rad, 180.0 * units.deg]
        obtained = np.round([BodyParameters.discretization_factor(val) for val in valid_values], 4)
        expected = [0., 3.1416, 3.1416, 3.1416]
        assert_array_equal(expected, obtained)

    @staticmethod
    def test_t_eff():
        valid_values = [10.0, 180.0 * units.TEMPERATURE_UNIT]
        obtained = np.round([BodyParameters.t_eff(val) for val in valid_values], 4)
        expected = [10., 180.]
        assert_array_equal(expected, obtained)
        
    def test_t_eff_raise(self):
        with self.assertRaises(Exception) as context:
            BodyParameters.t_eff("10")
        self.assertTrue("is not" in str(context.exception))

    @staticmethod
    def test_polar_radius():
        valid_values = [10.0, 180.0 * units.DISTANCE_UNIT, 1000 * units.m, 1 * units.km]
        obtained = np.round([BodyParameters.polar_radius(val) for val in valid_values], 4)
        expected = [10., 180., 1000., 1000.]
        assert_array_equal(expected, obtained)


class TransformStarParametersTestCase(ElisaTestCase):
    @staticmethod
    def test_metallicity():
        valid_values = [-0.1, 0.0, 1.0, 2.1]
        obtained = np.round([StarParameters.metallicity(val) for val in valid_values], 4)
        assert_array_equal(valid_values, obtained)

    @staticmethod
    def test_surface_potential():
        valid_values = [0.1, 1.1, 10001.123]
        obtained = np.round([StarParameters.surface_potential(val) for val in valid_values], 4)
        assert_array_equal(valid_values, obtained)

    def test_surface_potential_raise(self):
        with self.assertRaises(Exception) as context:
            StarParameters.surface_potential(-0.1)
        self.assertTrue("Invalid surface potential" in str(context.exception))

    @staticmethod
    def test_gravity_darkening():
        valid_values = [0.1, 0.0, 1]
        obtained = np.round([StarParameters.gravity_darkening(val) for val in valid_values], 4)
        assert_array_equal(valid_values, obtained)

    def test_gravity_darkening_raise(self):
        with self.assertRaises(Exception) as context:
            StarParameters.gravity_darkening(-0.1)
        self.assertTrue("is out of range" in str(context.exception))

    @staticmethod
    def test_polar_log_g():
        pass
