from __future__ import annotations

from typing import TYPE_CHECKING

# The tests package selects Astropy constants before test-module imports.
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from elisa import const
from elisa import units as u
from elisa.base.error import ElisaParametersError
from elisa.base.transform import BodyProperties, SpotProperties, StarProperties, SystemProperties
from elisa.binary_system.orbit.transform import OrbitProperties
from elisa.binary_system.transform import BinarySystemProperties
from tests import set_astropy_units
from tests.utils import ElisaTestCase

if TYPE_CHECKING:
    from collections.abc import Callable

    import astropy.units.function.logarithmic
    import astropy.units.quantity


set_astropy_units()


def generate_test(
    values: list[astropy.units.function.logarithmic.Dex | float | int]
    | list[astropy.units.quantity.Quantity | float]
    | list[float | int]
    | list[float],
    transform: Callable[..., float | None] | Callable[..., np.float64 | None],
    expected: list[float | int] | list[float | np.float64] | list[float],
    _round: int | None = None,
) -> None:
    obtained = [transform(val) for val in values]
    if _round is not None:
        obtained = np.round(obtained, _round)
    assert_array_equal(expected, obtained)


def generate_raise_test(
    _self: TransformBinarySystemPropertiesTestCase
    | TransformBodyPropertiesTestCase
    | TransformStarPropertiesTestCase
    | TransformSystemPropertiesTestCase,
    value: astropy.units.quantity.Quantity | dict[str, str] | float | str,
    transform: Callable[..., float | None],
    in_exception: str,
) -> None:
    with pytest.raises(ElisaParametersError) as context:
        transform(value)
    assert in_exception in str(context.value)


class TransformBinarySystemPropertiesTestCase(ElisaTestCase):
    @staticmethod
    def test_eccentricity() -> None:
        valid_values = [0.0, 0.1, 0.95]
        generate_test(valid_values, BinarySystemProperties.eccentricity, valid_values)

    def test_eccentricity_raise(self) -> None:
        generate_raise_test(self, 1.1, BinarySystemProperties.eccentricity, "out of")

    @staticmethod
    def test_argument_of_periastron() -> None:
        valid_values = [0.0, 180.0, const.PI * u.rad, 180.0 * u.deg]
        expected = [0.0, 3.1416, 3.1416, 3.1416]
        generate_test(valid_values, BinarySystemProperties.argument_of_periastron, expected, 4)

    def test_argument_of_periastron_raise(self) -> None:
        generate_raise_test(self, "9dsg.2", BinarySystemProperties.argument_of_periastron, "argument_of_periastron")

    def test_phase_shift(self) -> None:
        assert BinarySystemProperties.phase_shift(9.1234) == 9.1234

    @staticmethod
    def test_primary_minimum_time() -> None:
        valid_values = [1.0, 180.0 * u.PERIOD_UNIT, 86400.0 * u.s]
        expected = [1.0, 180.0, 1.0]
        generate_test(valid_values, BinarySystemProperties.primary_minimum_time, expected)

    def test_primary_minimum_time_raise(self) -> None:
        generate_raise_test(self, "92", BinarySystemProperties.primary_minimum_time, "primary_minimum_time")

    @staticmethod
    def test_argument_of_periastron_wrap() -> None:
        """Ensure values larger than a full arc are wrapped into [0, FULL_ARC)."""
        # 720 degrees should wrap to 0 radians
        wrap_values = [720.0, 1080.0 * 1.0]  # 720 deg and 1080 deg
        expected = [0.0, 0.0]
        generate_test(wrap_values, BinarySystemProperties.argument_of_periastron, expected, 4)

    @staticmethod
    def test_t0_alias() -> None:
        """Verify that t0 delegates to primary_minimum_time."""
        values = [1.0, 180.0 * u.PERIOD_UNIT]
        expected = [1.0, 180.0]
        obtained_t0 = [BinarySystemProperties.t0(v) for v in values]
        obtained_pmt = [BinarySystemProperties.primary_minimum_time(v) for v in values]
        assert_array_equal(np.array(obtained_t0), np.array(obtained_pmt))
        assert_array_equal(np.array(obtained_t0), np.array(expected))

    def test_eccentricity_boundary_values(self) -> None:
        """Test eccentricity at boundary values."""
        # Test at lower boundary (0.0)
        result = BinarySystemProperties.eccentricity(0.0)
        assert result == 0.0

        # Test near upper boundary (just below 1.0)
        result = BinarySystemProperties.eccentricity(0.9999)
        self.assertAlmostEqual(result, 0.9999, places=4)

    def test_eccentricity_negative_raise(self) -> None:
        """Test eccentricity raises for negative values."""
        generate_raise_test(self, -0.1, BinarySystemProperties.eccentricity, "out of boundaries")

    def test_argument_of_periastron_with_astropy_quantity(self) -> None:
        """Test argument_of_periastron with astropy Quantity input."""
        # Test with radians
        result = BinarySystemProperties.argument_of_periastron(const.PI * u.rad)
        self.assertAlmostEqual(result, const.PI, places=4)

        # Test with degrees
        result = BinarySystemProperties.argument_of_periastron(90.0 * u.deg)
        expected = (90.0 * u.deg).to(u.rad).value
        self.assertAlmostEqual(result, expected, places=4)

    def test_argument_of_periastron_type_raise(self) -> None:
        """Test argument_of_periastron raises for invalid types."""
        generate_raise_test(self, {"invalid": "dict"}, BinarySystemProperties.argument_of_periastron, "not")

    def test_phase_shift_various_values(self) -> None:
        """Test phase_shift with various input values."""
        test_values = [0.0, 0.5, 1.0, -0.5, 10.5]
        for val in test_values:
            result = BinarySystemProperties.phase_shift(val)
            assert float(result) == float(val)

    def test_primary_minimum_time_with_various_units(self) -> None:
        """Test primary_minimum_time with different unit representations."""
        # Test with days (default)
        result = BinarySystemProperties.primary_minimum_time(2.5)
        self.assertAlmostEqual(result, 2.5, places=4)

        # Test with seconds
        result = BinarySystemProperties.primary_minimum_time(86400.0 * u.s)
        self.assertAlmostEqual(result, 1.0, places=4)  # 1 day in default units

    def test_primary_minimum_time_invalid_unit_raise(self) -> None:
        """Test primary_minimum_time raises for incompatible units."""
        generate_raise_test(self, 1.0 * u.kg, BinarySystemProperties.primary_minimum_time, "primary_minimum_time")


class TransformSystemPropertiesTestCase(ElisaTestCase):
    @staticmethod
    def test_inclination() -> None:
        valid_values = [0.0, 180.0, const.PI * u.rad, 180.0 * u.deg]
        expected = [0.0, 3.1416, 3.1416, 3.1416]
        generate_test(valid_values, SystemProperties.inclination, expected, 4)

    def test_inclination_raise(self) -> None:
        generate_raise_test(self, const.FULL_ARC * u.rad, SystemProperties.inclination, "is out of bounds")

    @staticmethod
    def test_period() -> None:
        valid_values = [1.0, 180.0 * u.PERIOD_UNIT, 86400.0 * u.s]
        expected = [1.0, 180.0, 1.0]
        generate_test(valid_values, SystemProperties.period, expected)

    def test_period_raise(self) -> None:
        generate_raise_test(self, "98.2", SystemProperties.period, "period")

    @staticmethod
    def test_gamma() -> None:
        valid_values = [1.0, 180.0 * u.VELOCITY_UNIT, 1.0 * u.km / u.s]
        expected = [1.0, 180.0, 1000.0]
        generate_test(valid_values, SystemProperties.gamma, expected)

    def test_gamma_raise(self) -> None:
        generate_raise_test(self, "98.2", SystemProperties.gamma, "gamma")
        generate_raise_test(self, 1.0 * u.km * u.s, SystemProperties.gamma, "gamma")

    @staticmethod
    def test_additional_light() -> None:
        valid_values = [0.0, 0.55, 1.0]
        generate_test(valid_values, SystemProperties.additional_light, valid_values)

    def test_additional_light_raise(self) -> None:
        generate_raise_test(self, 1.5, SystemProperties.additional_light, "Invalid value")


class TransformBodyPropertiesTestCase(ElisaTestCase):
    @staticmethod
    def test_synchronicity() -> None:
        valid_values = [1.0, 1.1, 0.5]
        obtained = [BodyProperties.synchronicity(val) for val in valid_values]
        assert_array_equal(valid_values, obtained)

    def test_synchronicity_raise(self) -> None:
        generate_raise_test(self, -0.14, BodyProperties.synchronicity, "Invalid synchronicity")

    @staticmethod
    def test_albedo() -> None:
        valid_values = [0.0, 1.0, 0.5]
        generate_test(valid_values, BodyProperties.albedo, valid_values)

    def test_albedo_raise(self) -> None:
        generate_raise_test(self, -0.14, BodyProperties.albedo, "is out of range")
        generate_raise_test(self, 15239, BodyProperties.albedo, "is out of range")

    @staticmethod
    def test_discretization_factor() -> None:
        valid_values = [0.0, 45.0, 0.25 * u.rad, 45.0 * u.deg]
        expected = [0.0, 0.7854, 0.25, 0.7854]
        generate_test(valid_values, BodyProperties.discretization_factor, expected, 4)

    @staticmethod
    def test_t_eff() -> None:
        valid_values = [10.0, 180.0 * u.TEMPERATURE_UNIT]
        expected = [10.0, 180.0]
        generate_test(valid_values, BodyProperties.t_eff, expected, 4)

    def test_t_eff_raise(self) -> None:
        generate_raise_test(self, "hakuna matata", BodyProperties.t_eff, "t_eff")

    @staticmethod
    def test_radius() -> None:
        valid_values = [10.0, 180.0 * u.DISTANCE_UNIT, 1000 * u.m, 1 * u.km]
        expected = [(10.0 * u.solRad).to(u.m).value, 180.0, 1000.0, 1000.0]
        generate_test(valid_values, StarProperties.equivalent_radius, expected, 4)


class TransformStarPropertiesTestCase(ElisaTestCase):
    def test_mass(self) -> None:
        valid_values = [1.0, 1.0 * u.solMass, 30.0 * u.MASS_UNIT]
        obtained = np.array([StarProperties.mass(val) for val in valid_values])
        expected = np.array([1.98847542e30, 1.98847542e30, 3.00000000e01])
        assert np.all(abs((expected - obtained) / expected) < 1e-8)

    def test_mass_raise(self) -> None:
        generate_raise_test(self, -0.14, StarProperties.mass, "Invalid mass")

    @staticmethod
    def test_metallicity() -> None:
        valid_values = [-0.1, 0.0, 1.0, 2.1]
        generate_test(valid_values, StarProperties.metallicity, valid_values, 4)

    @staticmethod
    def test_surface_potential() -> None:
        valid_values = [0.1, 1.1, 10001.123]
        generate_test(valid_values, StarProperties.surface_potential, valid_values, 4)

    def test_surface_potential_raise(self) -> None:
        generate_raise_test(self, -0.1, StarProperties.surface_potential, "Invalid surface potential")

    @staticmethod
    def test_gravity_darkening() -> None:
        valid_values = [0.1, 0.0, 1]
        generate_test(valid_values, StarProperties.gravity_darkening, valid_values, 4)

    def test_gravity_darkening_raise(self) -> None:
        generate_raise_test(self, -0.1, StarProperties.gravity_darkening, "is out of range")

    @staticmethod
    def test_polar_log_g() -> None:
        valid_values = [2, 3.0, 4.0 * u.LOG_ACCELERATION_UNIT]
        expected = [0.0, 1.0, 4.0]
        generate_test(valid_values, StarProperties.polar_log_g, expected, 4)


class TransformSpotPropertiesTestCase(ElisaTestCase):
    @staticmethod
    def test_latitude() -> None:
        valid_values = [0.0, 180.0, const.PI * u.rad, 180.0 * u.deg]
        expected = [0.0, 3.1416, 3.1416, 3.1416]
        generate_test(valid_values, SpotProperties.latitude, expected, 4)

    @staticmethod
    def test_longitude() -> None:
        valid_values = [0.0, 180.0, const.PI * u.rad, 180.0 * u.deg]
        expected = [0.0, 3.1416, 3.1416, 3.1416]
        generate_test(valid_values, SpotProperties.longitude, expected, 4)

    def test_angular_radius(self) -> None:
        valid_values = [90.0, 180.0, const.PI * u.rad, 180.0 * u.deg]
        expected = [1.5708, 3.1416, 3.1416, 3.1416]
        generate_test(valid_values, SpotProperties.angular_radius, expected, 4)
        with pytest.raises(ElisaParametersError):
            SpotProperties.angular_radius(0.0)

    @staticmethod
    def test_temperature_factor() -> None:
        valid_values = [0.1, 0.5, 0.9999]
        generate_test(valid_values, SpotProperties.temperature_factor, valid_values)


class TransformOrbitProperties(ElisaTestCase):
    @staticmethod
    def test_period() -> None:
        periods = [0.25 * u.d, 0.65, 86400 * u.s]
        expected = [0.25, 0.65, 1.0]
        generate_test(periods, OrbitProperties.period, expected)

    @staticmethod
    def test_eccentricity() -> None:
        values = [0.0, 0.1, 0.2, 0.999999]
        generate_test(values, OrbitProperties.eccentricity, values)

    @staticmethod
    def test_argument_of_periastron() -> None:
        periastrons = [135 * u.deg, 0.65, 1.56 * u.rad]
        expected = [2.356, 0.65, 1.56]
        generate_test(periastrons, OrbitProperties.argument_of_periastron, expected, 3)

    @staticmethod
    def test_inclination() -> None:
        inclinations = [135 * u.deg, 0.65, 1.56 * u.rad]
        expected = [2.356, 0.65, 1.56]
        generate_test(inclinations, OrbitProperties.inclination, expected, 3)
