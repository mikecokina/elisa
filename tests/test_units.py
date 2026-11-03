# The tests package selects Astropy constants before test-module imports.
from elisa.units import (
    ACCELERATION_UNIT,
    ANGULAR_FREQUENCY_UNIT,
    ARC_UNIT,
    DEFAULT_GAMMA_INPUT_UNIT,
    DEFAULT_INCLINATION_INPUT_UNIT,
    DEFAULT_PERIOD_INPUT_UNIT,
    DISTANCE_TO_OBS_INPUT_UNIT,
    DISTANCE_UNIT,
    FREQUENCY_UNIT,
    LOG_ACCELERATION_UNIT,
    LUMINOSITY_UNIT,
    MASS_UNIT,
    PERIOD_UNIT,
    RADIANCE_UNIT,
    TEMPERATURE_UNIT,
    TIME_UNIT,
    VELOCITY_UNIT,
    DefaultBinarySystemInputUnits,
    DefaultBinarySystemUnits,
    DefaultPulsationsUnits,
    DefaultSingleSystemInputUnits,
    DefaultSingleSystemUnits,
    DefaultSpotInputUnits,
    DefaultSpotUnits,
    DefaultStarInputUnits,
    DefaultStarUnits,
    DefaultSystemUnits,
    d,
    default_unit_map,
    deg,
    dimensionless_unscaled,
    km,
    m,
    pc,
    rad,
    s,
    solMass,
    solRad,
    u,
)
from tests import set_astropy_units
from tests.utils import ElisaTestCase

set_astropy_units()


class TestUnits(ElisaTestCase):
    def test_MASS_UNIT(self) -> None:
        assert (u.kg) == (MASS_UNIT)

    def test_TEMPERATURE_UNIT(self) -> None:
        assert (TEMPERATURE_UNIT) == (u.K)

    def test_DISTANCE_UNIT(self) -> None:
        assert (u.m) == (DISTANCE_UNIT)

    def test_TIME_UNIT(self) -> None:
        assert (u.s) == (TIME_UNIT)

    def test_ARC_UNIT(self) -> None:
        assert (u.rad) == (ARC_UNIT)

    def test_PERIOD_UNIT(self) -> None:
        assert (u.d) == (PERIOD_UNIT)

    def test_VELOCITY_UNIT(self) -> None:
        assert (u.m / u.s) == (VELOCITY_UNIT)

    def test_ACCELERATION_UNIT(self) -> None:
        assert (u.m / (u.s**2)) == (ACCELERATION_UNIT)

    def test_LOG_ACCELERATION_UNIT(self) -> None:
        assert (u.dex(u.m / (u.s**2))) == (LOG_ACCELERATION_UNIT)

    def test_FREQUENCY_UNIT(self) -> None:
        assert (u.Hz) == (FREQUENCY_UNIT)


class TestAstropyUnitAliases(ElisaTestCase):
    """Test that astropy unit aliases are accessible and correct."""

    def test_degree_alias(self) -> None:
        """Test that degree unit alias is accessible."""
        assert (deg) == (u.deg)

    def test_radian_alias(self) -> None:
        """Test that radian unit alias is accessible."""
        assert (rad) == (u.rad)

    def test_kilometer_alias(self) -> None:
        """Test that kilometer unit alias is accessible."""
        assert (km) == (u.km)

    def test_solar_mass_alias(self) -> None:
        """Test that solar mass unit alias is accessible."""
        assert (solMass) == (u.solMass)

    def test_solar_radius_alias(self) -> None:
        """Test that solar radius unit alias is accessible."""
        assert (solRad) == (u.solRad)


class TestDefaultInputUnits(ElisaTestCase):
    """Test default input unit definitions."""

    def test_default_inclination_input_unit(self) -> None:
        """Test that default inclination input unit is in degrees."""
        assert (deg) == (DEFAULT_INCLINATION_INPUT_UNIT)

    def test_default_period_input_unit(self) -> None:
        """Test that default period input unit is in days."""
        assert (d) == (DEFAULT_PERIOD_INPUT_UNIT)

    def test_default_gamma_input_unit(self) -> None:
        """Test that default gamma input unit is in m/s."""
        assert (m / s) == (DEFAULT_GAMMA_INPUT_UNIT)

    def test_distance_to_obs_input_unit(self) -> None:
        """Test that default distance to observer unit is in parsecs."""
        assert (pc) == (DISTANCE_TO_OBS_INPUT_UNIT)


class TestBaseUnitsClass(ElisaTestCase):
    """Test BaseUnits class functionality."""

    def test_default_spot_units_as_dict(self) -> None:
        """Test that DefaultSpotUnits can be converted to dictionary."""
        spot_dict = DefaultSpotUnits.as_dict()
        assert isinstance(spot_dict, dict)
        assert ("longitude") in (spot_dict)
        assert ("latitude") in (spot_dict)
        assert ("angular_radius") in (spot_dict)
        # Verify units are strings
        assert (spot_dict["longitude"]) == ("rad")
        assert (spot_dict["latitude"]) == ("rad")

    def test_default_star_units_as_dict(self) -> None:
        """Test that DefaultStarUnits can be converted to dictionary."""
        star_dict = DefaultStarUnits.as_dict()
        assert isinstance(star_dict, dict)
        assert ("mass") in (star_dict)
        assert ("t_eff") in (star_dict)
        assert ("metallicity") in (star_dict)
        assert ("spots") in (star_dict)
        # Spots should be nested dictionary
        assert isinstance(star_dict["spots"], dict)
        assert ("longitude") in (star_dict["spots"])

    def test_default_binary_system_units_structure(self) -> None:
        """Test structure of DefaultBinarySystemUnits."""
        binary_dict = DefaultBinarySystemUnits.as_dict()
        # Should have system, primary, secondary, component sections
        assert ("system") in (binary_dict)
        assert ("primary") in (binary_dict)
        assert ("secondary") in (binary_dict)
        assert ("component") in (binary_dict)
        # Verify system has expected parameters
        system_dict = binary_dict["system"]
        assert ("inclination") in (system_dict)
        assert ("period") in (system_dict)
        assert ("eccentricity") in (system_dict)

    def test_default_single_system_units_structure(self) -> None:
        """Test structure of DefaultSingleSystemUnits."""
        single_dict = DefaultSingleSystemUnits.as_dict()
        # Should have system and star sections
        assert ("system") in (single_dict)
        assert ("star") in (single_dict)
        # Star section should have standard stellar parameters
        star_dict = single_dict["star"]
        assert ("mass") in (star_dict)
        assert ("t_eff") in (star_dict)
        assert ("metallicity") in (star_dict)
        assert ("discretization_factor") in (star_dict)


class TestDefaultUnitMap(ElisaTestCase):
    """Test default unit map functionality."""

    def test_default_unit_map_keys(self) -> None:
        """Test that default_unit_map has expected keys."""
        assert ("SingleSystem") in (default_unit_map)
        assert ("BinarySystem") in (default_unit_map)
        assert ("Star") in (default_unit_map)
        assert ("Spot") in (default_unit_map)

    def test_default_unit_map_single_system_entry(self) -> None:
        """Test that SingleSystem entry in default_unit_map is valid."""
        assert (default_unit_map["SingleSystem"]) is not None

    def test_default_unit_map_binary_system_entry(self) -> None:
        """Test that BinarySystem entry in default_unit_map is valid."""
        assert (default_unit_map["BinarySystem"]) is not None

    def test_default_unit_map_star_entry(self) -> None:
        """Test that Star entry in default_unit_map is valid."""
        assert (default_unit_map["Star"]) is not None

    def test_default_unit_map_spot_entry(self) -> None:
        """Test that Spot entry in default_unit_map is valid."""
        assert (default_unit_map["Spot"]) is not None


class TestInputOutputUnitConsistency(ElisaTestCase):
    """Test consistency between internal and input units."""

    def test_spot_units_consistency(self) -> None:
        """Test that spot internal and input units have same keys."""
        default_spot = DefaultSpotUnits.as_dict()
        input_spot = DefaultSpotInputUnits.as_dict()
        # Both should have same keys
        assert (set(default_spot.keys())) == (set(input_spot.keys()))

    def test_star_units_consistency(self) -> None:
        """Test that star internal and input units have same keys."""
        default_star = DefaultStarUnits.as_dict()
        input_star = DefaultStarInputUnits.as_dict()
        # Both should have same keys
        assert (set(default_star.keys())) == (set(input_star.keys()))

    def test_binary_system_units_consistency(self) -> None:
        """Test that binary system internal and input units are consistent."""
        default_binary = DefaultBinarySystemUnits.as_dict()
        input_binary = DefaultBinarySystemInputUnits.as_dict()
        # Both should have same top-level keys
        assert (set(default_binary.keys())) == (set(input_binary.keys()))

    def test_single_system_units_consistency(self) -> None:
        """Test that single system internal and input units are consistent."""
        default_single = DefaultSingleSystemUnits.as_dict()
        input_single = DefaultSingleSystemInputUnits.as_dict()
        # Both should have same top-level keys
        assert (set(default_single.keys())) == (set(input_single.keys()))

    def test_system_parameter_units(self) -> None:
        """Test that system parameter units are properly defined."""
        system_units = DefaultSystemUnits.as_dict()
        # Should contain key system parameters
        assert ("inclination") in (system_units)
        assert ("period") in (system_units)
        assert ("gamma") in (system_units)
        assert ("distance") in (system_units)
        # Verify they use correct units
        assert (system_units["inclination"]) == ("rad")


class TestSpecialUnitDefinitions(ElisaTestCase):
    """Test special unit definitions and constants."""

    def test_dimensionless_unit(self) -> None:
        """Test that dimensionless unit is properly defined."""
        assert (dimensionless_unscaled) == (u.dimensionless_unscaled)

    def test_log_acceleration_unit_exists(self) -> None:
        """Test that log acceleration unit is properly defined."""
        # Should be dex(acceleration_unit)
        assert (LOG_ACCELERATION_UNIT) is not None

    def test_luminosity_unit_defined(self) -> None:
        """Test that luminosity unit is properly defined."""
        assert (LUMINOSITY_UNIT) == (u.W)

    def test_radiance_unit_defined(self) -> None:
        """Test that radiance unit is properly defined."""
        expected = u.W / (u.m**2 * u.sr)
        assert (expected) == (RADIANCE_UNIT)

    def test_angular_frequency_unit_defined(self) -> None:
        """Test that angular frequency unit is properly defined."""
        expected = u.rad / u.s
        assert (expected) == (ANGULAR_FREQUENCY_UNIT)

    def test_pulsation_units_structure(self) -> None:
        """Test that pulsation units have correct structure."""
        pulsation_dict = DefaultPulsationsUnits.as_dict()
        assert ("amplitude") in (pulsation_dict)
        assert ("frequency") in (pulsation_dict)
        assert ("l") in (pulsation_dict)
        assert ("m") in (pulsation_dict)
        assert ("mode_axis_theta") in (pulsation_dict)
        assert ("mode_axis_phi") in (pulsation_dict)
