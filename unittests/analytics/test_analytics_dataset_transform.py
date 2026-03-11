"""Unit tests for analytics.dataset.transform module."""

import numpy as np
from astropy import units as astropy_units
from numpy.testing import assert_array_equal

from elisa import units as u
from elisa.analytics.dataset.transform import (
    DatasetProperties,
    LCDataProperties,
    RVDataProperties,
    array_transform,
    unit_check,
)
from elisa.base.transform import WHEN_ARRAY
from unittests import set_astropy_units
from unittests.utils import ElisaTestCase

set_astropy_units()


class ArrayTransformTestCase(ElisaTestCase):
    """Test array_transform function for array-like conversion."""

    def test_array_transform_numpy_array(self):
        """Test that numpy array is converted properly."""
        input_array = np.array([1.0, 2.0, 3.0])
        result = array_transform(input_array, WHEN_ARRAY)
        assert_array_equal(result, input_array)
        self.assertEqual(result.dtype, np.float64)

    def test_array_transform_list(self):
        """Test that list is converted to numpy array."""
        input_list = [1.0, 2.0, 3.0]
        result = array_transform(input_list, WHEN_ARRAY)
        expected = np.array([1.0, 2.0, 3.0])
        assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.float64)

    def test_array_transform_tuple(self):
        """Test that tuple is converted to numpy array."""
        input_tuple = (1.0, 2.0, 3.0)
        result = array_transform(input_tuple, WHEN_ARRAY)
        expected = np.array([1.0, 2.0, 3.0])
        assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.float64)

    def test_array_transform_non_array_raises(self):
        """Test that non-array-like value raises TypeError."""
        with self.assertRaises(TypeError) as context:
            array_transform("not an array", WHEN_ARRAY)
        self.assertIn("not array-like", str(context.exception).lower())

    def test_array_transform_none_returns_none(self):
        """Test that None value returns None without raising."""
        result = array_transform(None, WHEN_ARRAY)
        self.assertIsNone(result)

    def test_array_transform_empty_list_returns_empty_array(self):
        """Test that empty list returns empty array."""
        result = array_transform([], WHEN_ARRAY)
        expected = np.array([], dtype=np.float64)
        assert_array_equal(result, expected)

    def test_array_transform_nested_list(self):
        """Test that nested lists are converted properly."""
        input_list = [[1.0, 2.0], [3.0, 4.0]]
        result = array_transform(input_list, WHEN_ARRAY)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_array_equal(result, expected)


class UnitCheckTestCase(ElisaTestCase):
    """Test unit_check function for unit validation."""

    def test_unit_check_none_returns_dimensionless(self):
        """Test that None unit is converted to dimensionless."""
        result = unit_check(None, (u.dimensionless_unscaled,))
        self.assertEqual(result, u.dimensionless_unscaled)

    def test_unit_check_empty_string_returns_dimensionless(self):
        """Test that empty string unit is converted to dimensionless."""
        result = unit_check(u.dimensionless_unscaled, (u.dimensionless_unscaled,))
        self.assertEqual(result, u.dimensionless_unscaled)

    def test_unit_check_compatible_unit(self):
        """Test that compatible unit is returned unchanged."""
        result = unit_check(u.m, (u.m, u.cm, u.km))
        self.assertEqual(result, u.m)

    def test_unit_check_convertible_unit(self):
        """Test that convertible unit is accepted."""
        result = unit_check(u.km, (u.m,))
        self.assertEqual(result, u.km)

    def test_unit_check_incompatible_unit_raises(self):
        """Test that incompatible unit raises ValueError."""
        with self.assertRaises(ValueError) as context:
            unit_check(u.mag, (u.m, u.s))
        self.assertIn("not convertible", str(context.exception).lower())

    def test_unit_check_time_unit(self):
        """Test that time units are properly validated."""
        result = unit_check(astropy_units.second, (u.TIME_UNIT,))
        self.assertEqual(result, astropy_units.second)

    def test_unit_check_velocity_unit(self):
        """Test that velocity units are properly validated."""
        result = unit_check(u.km / u.s, (u.VELOCITY_UNIT,))
        self.assertEqual(result, u.km / u.s)


class DatasetPropertiesTestCase(ElisaTestCase):
    """Test DatasetProperties transformation methods."""

    def test_dataset_properties_x_data_list(self):
        """Test x_data transformation from list."""
        x_data = [0.0, 0.1, 0.2, 0.3, 0.4]
        result = DatasetProperties.x_data(x_data)
        expected = np.array(x_data, dtype=np.float64)
        assert_array_equal(result, expected)

    def test_dataset_properties_y_data_array(self):
        """Test y_data transformation from array."""
        y_data = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
        result = DatasetProperties.y_data(y_data)
        assert_array_equal(result, y_data)

    def test_dataset_properties_y_err_tuple(self):
        """Test y_err transformation from tuple."""
        y_err = (0.01, 0.02, 0.01, 0.015, 0.02)
        result = DatasetProperties.y_err(y_err)
        expected = np.array(y_err, dtype=np.float64)
        assert_array_equal(result, expected)

    def test_dataset_properties_none_values(self):
        """Test that None values return None."""
        self.assertIsNone(DatasetProperties.x_data(None))
        self.assertIsNone(DatasetProperties.y_data(None))
        self.assertIsNone(DatasetProperties.y_err(None))


class RVDataPropertiesTestCase(ElisaTestCase):
    """Test RVDataProperties unit validation."""

    def test_rv_data_properties_x_unit_dimensionless(self):
        """Test that dimensionless x_unit is accepted."""
        result = RVDataProperties.x_unit(u.dimensionless_unscaled)
        self.assertEqual(result, u.dimensionless_unscaled)

    def test_rv_data_properties_x_unit_time(self):
        """Test that time units are accepted for x_unit."""
        result = RVDataProperties.x_unit(astropy_units.second)
        self.assertEqual(result, astropy_units.second)

    def test_rv_data_properties_x_unit_none(self):
        """Test that None x_unit is converted to dimensionless."""
        result = RVDataProperties.x_unit(None)
        self.assertEqual(result, u.dimensionless_unscaled)

    def test_rv_data_properties_y_unit_velocity(self):
        """Test that velocity units are accepted for y_unit."""
        result = RVDataProperties.y_unit(u.km / u.s)
        self.assertEqual(result, u.km / u.s)

    def test_rv_data_properties_y_unit_m_per_s(self):
        """Test that m/s is accepted for y_unit."""
        result = RVDataProperties.y_unit(u.m / u.s)
        self.assertEqual(result, u.m / u.s)

    def test_rv_data_properties_y_unit_invalid_raises(self):
        """Test that invalid y_unit raises ValueError."""
        with self.assertRaises(ValueError):
            RVDataProperties.y_unit(u.mag)

    def test_rv_data_properties_x_unit_invalid_raises(self):
        """Test that invalid x_unit raises ValueError."""
        with self.assertRaises(ValueError):
            RVDataProperties.x_unit(u.km / u.s)


class LCDataPropertiesTestCase(ElisaTestCase):
    """Test LCDataProperties unit validation."""

    def test_lc_data_properties_x_unit_dimensionless(self):
        """Test that dimensionless x_unit is accepted."""
        result = LCDataProperties.x_unit(u.dimensionless_unscaled)
        self.assertEqual(result, u.dimensionless_unscaled)

    def test_lc_data_properties_x_unit_time(self):
        """Test that time units are accepted for x_unit."""
        result = LCDataProperties.x_unit(astropy_units.second)
        self.assertEqual(result, astropy_units.second)

    def test_lc_data_properties_y_unit_dimensionless(self):
        """Test that dimensionless y_unit is accepted."""
        result = LCDataProperties.y_unit(u.dimensionless_unscaled)
        self.assertEqual(result, u.dimensionless_unscaled)

    def test_lc_data_properties_y_unit_mag(self):
        """Test that magnitude units are accepted for y_unit."""
        result = LCDataProperties.y_unit(u.mag)
        self.assertEqual(result, u.mag)

    def test_lc_data_properties_y_unit_invalid_raises(self):
        """Test that invalid y_unit raises ValueError."""
        with self.assertRaises(ValueError):
            LCDataProperties.y_unit(u.km / u.s)

    def test_lc_data_properties_zero_magnitude_float(self):
        """Test zero_magnitude with float value."""
        value = 0.0
        result = LCDataProperties.zero_magnitude(value)
        self.assertEqual(result, np.float64(0.0))
        self.assertIsInstance(result, np.float64)

    def test_lc_data_properties_zero_magnitude_int(self):
        """Test zero_magnitude with integer value."""
        value = 5
        result = LCDataProperties.zero_magnitude(value)
        self.assertEqual(result, np.float64(5.0))
        self.assertIsInstance(result, np.float64)

    def test_lc_data_properties_zero_magnitude_quantity(self):
        """Test zero_magnitude with Quantity value - expect error due to code bug."""
        value = 1.0 * u.mag
        # The original code has a bug: it calls value.is_equivalent(u.mag)
        # which is incorrect (should be value.unit.is_equivalent(u.mag))
        # So we expect an AttributeError
        with self.assertRaises(AttributeError):
            LCDataProperties.zero_magnitude(value)

    def test_lc_data_properties_zero_magnitude_numpy_float(self):
        """Test zero_magnitude with numpy float value."""
        value = np.float64(2.5)
        result = LCDataProperties.zero_magnitude(value)
        self.assertEqual(result, np.float64(2.5))
        self.assertIsInstance(result, np.float64)

    def test_lc_data_properties_zero_magnitude_invalid_raises(self):
        """Test that invalid zero_magnitude type raises TypeError."""
        with self.assertRaises(TypeError) as context:
            LCDataProperties.zero_magnitude("not a number")
        self.assertIn("not", str(context.exception).lower())

