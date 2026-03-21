# keep it first
# due to stupid astropy units/constants implementation
from unittests import set_astropy_units

from elisa.units import *
from unittests.utils import ElisaTestCase

set_astropy_units()


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


class TestAstropyUnitAliases(ElisaTestCase):
    """Test that astropy unit aliases are accessible and correct."""

    def test_degree_alias(self):
        """Test that degree unit alias is accessible."""
        self.assertEqual(deg, u.deg)

    def test_radian_alias(self):
        """Test that radian unit alias is accessible."""
        self.assertEqual(rad, u.rad)

    def test_kilometer_alias(self):
        """Test that kilometer unit alias is accessible."""
        self.assertEqual(km, u.km)

    def test_solar_mass_alias(self):
        """Test that solar mass unit alias is accessible."""
        self.assertEqual(solMass, u.solMass)

    def test_solar_radius_alias(self):
        """Test that solar radius unit alias is accessible."""
        self.assertEqual(solRad, u.solRad)


class TestDefaultInputUnits(ElisaTestCase):
    """Test default input unit definitions."""

    def test_default_inclination_input_unit(self):
        """Test that default inclination input unit is in degrees."""
        self.assertEqual(DEFAULT_INCLINATION_INPUT_UNIT, deg)

    def test_default_period_input_unit(self):
        """Test that default period input unit is in days."""
        self.assertEqual(DEFAULT_PERIOD_INPUT_UNIT, d)

    def test_default_gamma_input_unit(self):
        """Test that default gamma input unit is in m/s."""
        self.assertEqual(DEFAULT_GAMMA_INPUT_UNIT, m / s)

    def test_distance_to_obs_input_unit(self):
        """Test that default distance to observer unit is in parsecs."""
        self.assertEqual(DISTANCE_TO_OBS_INPUT_UNIT, pc)


class TestBaseUnitsClass(ElisaTestCase):
    """Test BaseUnits class functionality."""

    def test_default_spot_units_as_dict(self):
        """Test that DefaultSpotUnits can be converted to dictionary."""
        spot_dict = DefaultSpotUnits.as_dict()
        self.assertIsInstance(spot_dict, dict)
        self.assertIn("longitude", spot_dict)
        self.assertIn("latitude", spot_dict)
        self.assertIn("angular_radius", spot_dict)
        # Verify units are strings
        self.assertEqual(spot_dict["longitude"], "rad")
        self.assertEqual(spot_dict["latitude"], "rad")

    def test_default_star_units_as_dict(self):
        """Test that DefaultStarUnits can be converted to dictionary."""
        star_dict = DefaultStarUnits.as_dict()
        self.assertIsInstance(star_dict, dict)
        self.assertIn("mass", star_dict)
        self.assertIn("t_eff", star_dict)
        self.assertIn("metallicity", star_dict)
        self.assertIn("spots", star_dict)
        # Spots should be nested dictionary
        self.assertIsInstance(star_dict["spots"], dict)
        self.assertIn("longitude", star_dict["spots"])

    def test_default_binary_system_units_structure(self):
        """Test structure of DefaultBinarySystemUnits."""
        binary_dict = DefaultBinarySystemUnits.as_dict()
        # Should have system, primary, secondary, component sections
        self.assertIn("system", binary_dict)
        self.assertIn("primary", binary_dict)
        self.assertIn("secondary", binary_dict)
        self.assertIn("component", binary_dict)
        # Verify system has expected parameters
        system_dict = binary_dict["system"]
        self.assertIn("inclination", system_dict)
        self.assertIn("period", system_dict)
        self.assertIn("eccentricity", system_dict)

    def test_default_single_system_units_structure(self):
        """Test structure of DefaultSingleSystemUnits."""
        single_dict = DefaultSingleSystemUnits.as_dict()
        # Should have system and star sections
        self.assertIn("system", single_dict)
        self.assertIn("star", single_dict)
        # Star section should have standard stellar parameters
        star_dict = single_dict["star"]
        self.assertIn("mass", star_dict)
        self.assertIn("t_eff", star_dict)
        self.assertIn("metallicity", star_dict)
        self.assertIn("discretization_factor", star_dict)


class TestDefaultUnitMap(ElisaTestCase):
    """Test default unit map functionality."""

    def test_default_unit_map_keys(self):
        """Test that default_unit_map has expected keys."""
        self.assertIn("SingleSystem", default_unit_map)
        self.assertIn("BinarySystem", default_unit_map)
        self.assertIn("Star", default_unit_map)
        self.assertIn("Spot", default_unit_map)

    def test_default_unit_map_single_system_entry(self):
        """Test that SingleSystem entry in default_unit_map is valid."""
        self.assertIsNotNone(default_unit_map["SingleSystem"])

    def test_default_unit_map_binary_system_entry(self):
        """Test that BinarySystem entry in default_unit_map is valid."""
        self.assertIsNotNone(default_unit_map["BinarySystem"])

    def test_default_unit_map_star_entry(self):
        """Test that Star entry in default_unit_map is valid."""
        self.assertIsNotNone(default_unit_map["Star"])

    def test_default_unit_map_spot_entry(self):
        """Test that Spot entry in default_unit_map is valid."""
        self.assertIsNotNone(default_unit_map["Spot"])


class TestInputOutputUnitConsistency(ElisaTestCase):
    """Test consistency between internal and input units."""

    def test_spot_units_consistency(self):
        """Test that spot internal and input units have same keys."""
        default_spot = DefaultSpotUnits.as_dict()
        input_spot = DefaultSpotInputUnits.as_dict()
        # Both should have same keys
        self.assertEqual(set(default_spot.keys()), set(input_spot.keys()))

    def test_star_units_consistency(self):
        """Test that star internal and input units have same keys."""
        default_star = DefaultStarUnits.as_dict()
        input_star = DefaultStarInputUnits.as_dict()
        # Both should have same keys
        self.assertEqual(set(default_star.keys()), set(input_star.keys()))

    def test_binary_system_units_consistency(self):
        """Test that binary system internal and input units are consistent."""
        default_binary = DefaultBinarySystemUnits.as_dict()
        input_binary = DefaultBinarySystemInputUnits.as_dict()
        # Both should have same top-level keys
        self.assertEqual(set(default_binary.keys()), set(input_binary.keys()))

    def test_single_system_units_consistency(self):
        """Test that single system internal and input units are consistent."""
        default_single = DefaultSingleSystemUnits.as_dict()
        input_single = DefaultSingleSystemInputUnits.as_dict()
        # Both should have same top-level keys
        self.assertEqual(set(default_single.keys()), set(input_single.keys()))

    def test_system_parameter_units(self):
        """Test that system parameter units are properly defined."""
        system_units = DefaultSystemUnits.as_dict()
        # Should contain key system parameters
        self.assertIn("inclination", system_units)
        self.assertIn("period", system_units)
        self.assertIn("gamma", system_units)
        self.assertIn("distance", system_units)
        # Verify they use correct units
        self.assertEqual(system_units["inclination"], "rad")


class TestSpecialUnitDefinitions(ElisaTestCase):
    """Test special unit definitions and constants."""

    def test_dimensionless_unit(self):
        """Test that dimensionless unit is properly defined."""
        self.assertEqual(dimensionless_unscaled, u.dimensionless_unscaled)

    def test_log_acceleration_unit_exists(self):
        """Test that log acceleration unit is properly defined."""
        # Should be dex(acceleration_unit)
        self.assertIsNotNone(LOG_ACCELERATION_UNIT)

    def test_luminosity_unit_defined(self):
        """Test that luminosity unit is properly defined."""
        self.assertEqual(LUMINOSITY_UNIT, u.W)

    def test_radiance_unit_defined(self):
        """Test that radiance unit is properly defined."""
        expected = u.W / (u.m ** 2 * u.sr)
        self.assertEqual(RADIANCE_UNIT, expected)

    def test_angular_frequency_unit_defined(self):
        """Test that angular frequency unit is properly defined."""
        expected = u.rad / u.s
        self.assertEqual(ANGULAR_FREQUENCY_UNIT, expected)

    def test_pulsation_units_structure(self):
        """Test that pulsation units have correct structure."""
        pulsation_dict = DefaultPulsationsUnits.as_dict()
        self.assertIn("amplitude", pulsation_dict)
        self.assertIn("frequency", pulsation_dict)
        self.assertIn("l", pulsation_dict)
        self.assertIn("m", pulsation_dict)
        self.assertIn("mode_axis_theta", pulsation_dict)
        self.assertIn("mode_axis_phi", pulsation_dict)
