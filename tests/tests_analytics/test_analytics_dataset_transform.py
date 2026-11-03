"""Unit tests for analytics.dataset.transform module."""

import numpy as np
import pytest
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
from elisa.base.error import ElisaAnalyticsDataError
from elisa.base.transform import WHEN_ARRAY
from tests import set_astropy_units
from tests.utils import ElisaTestCase

set_astropy_units()


class ArrayTransformTestCase(ElisaTestCase):
    """Test array_transform function for array-like conversion."""

    def test_array_transform_numpy_array(self) -> None:
        """Test that numpy array is converted properly."""
        input_array = np.array([1.0, 2.0, 3.0])
        result = array_transform(input_array, WHEN_ARRAY)
        assert_array_equal(result, input_array)
        assert (result.dtype) == (np.float64)

    def test_array_transform_list(self) -> None:
        """Test that list is converted to numpy array."""
        input_list = [1.0, 2.0, 3.0]
        result = array_transform(input_list, WHEN_ARRAY)
        expected = np.array([1.0, 2.0, 3.0])
        assert_array_equal(result, expected)
        assert (result.dtype) == (np.float64)

    def test_array_transform_tuple(self) -> None:
        """Test that tuple is converted to numpy array."""
        input_tuple = (1.0, 2.0, 3.0)
        result = array_transform(input_tuple, WHEN_ARRAY)
        expected = np.array([1.0, 2.0, 3.0])
        assert_array_equal(result, expected)
        assert (result.dtype) == (np.float64)

    def test_array_transform_non_array_raises(self) -> None:
        """Test that non-array-like value raises ElisaAnalyticsDataError."""
        with pytest.raises(ElisaAnalyticsDataError) as context:
            array_transform("not an array", WHEN_ARRAY)
        assert ("not array-like") in (str(context.value).lower())

    def test_array_transform_none_returns_none(self) -> None:
        """Test that None value returns None without raising."""
        result = array_transform(None, WHEN_ARRAY)
        assert (result) is None

    def test_array_transform_empty_list_returns_empty_array(self) -> None:
        """Test that empty list returns empty array."""
        result = array_transform([], WHEN_ARRAY)
        expected = np.array([], dtype=np.float64)
        assert_array_equal(result, expected)

    def test_array_transform_nested_list(self) -> None:
        """Test that nested lists are converted properly."""
        input_list = [[1.0, 2.0], [3.0, 4.0]]
        result = array_transform(input_list, WHEN_ARRAY)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_array_equal(result, expected)


class UnitCheckTestCase(ElisaTestCase):
    """Test unit_check function for unit validation."""

    def test_unit_check_none_returns_dimensionless(self) -> None:
        """Test that None unit is converted to dimensionless."""
        result = unit_check(None, (u.dimensionless_unscaled,))
        assert (result) == (u.dimensionless_unscaled)

    def test_unit_check_empty_string_returns_dimensionless(self) -> None:
        """Test that empty string unit is converted to dimensionless."""
        result = unit_check("", (u.dimensionless_unscaled,))
        assert (result) == (u.dimensionless_unscaled)

    def test_unit_check_compatible_unit(self) -> None:
        """Test that compatible unit is returned unchanged."""
        result = unit_check(u.m, (u.m, u.cm, u.km))
        assert (result) == (u.m)

    def test_unit_check_convertible_unit(self) -> None:
        """Test that convertible unit is accepted."""
        result = unit_check(u.km, (u.m,))
        assert (result) == (u.km)

    def test_unit_check_incompatible_unit_raises(self) -> None:
        """Test that incompatible unit raises ElisaAnalyticsDataError."""
        with pytest.raises(ElisaAnalyticsDataError) as context:
            unit_check(u.mag, (u.m, u.s))
        assert ("not convertible") in (str(context.value).lower())

    def test_unit_check_time_unit(self) -> None:
        """Test that time units are properly validated."""
        result = unit_check(astropy_units.second, (u.TIME_UNIT,))
        assert (result) == (astropy_units.second)

    def test_unit_check_velocity_unit(self) -> None:
        """Test that velocity units are properly validated."""
        result = unit_check(u.km / u.s, (u.VELOCITY_UNIT,))
        assert (result) == (u.km / u.s)


class DatasetPropertiesTestCase(ElisaTestCase):
    """Test DatasetProperties transformation methods."""

    def test_dataset_properties_x_data_list(self) -> None:
        """Test x_data transformation from list."""
        x_data = [0.0, 0.1, 0.2, 0.3, 0.4]
        result = DatasetProperties.x_data(x_data)
        expected = np.array(x_data, dtype=np.float64)
        assert_array_equal(result, expected)

    def test_dataset_properties_y_data_array(self) -> None:
        """Test y_data transformation from array."""
        y_data = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
        result = DatasetProperties.y_data(y_data)
        assert_array_equal(result, y_data)

    def test_dataset_properties_y_err_tuple(self) -> None:
        """Test y_err transformation from tuple."""
        y_err = (0.01, 0.02, 0.01, 0.015, 0.02)
        result = DatasetProperties.y_err(y_err)
        expected = np.array(y_err, dtype=np.float64)
        assert_array_equal(result, expected)

    def test_dataset_properties_none_values(self) -> None:
        """Test that None values return None."""
        assert (DatasetProperties.x_data(None)) is None
        assert (DatasetProperties.y_data(None)) is None
        assert (DatasetProperties.y_err(None)) is None


class RVDataPropertiesTestCase(ElisaTestCase):
    """Test RVDataProperties unit validation."""

    def test_rv_data_properties_x_unit_dimensionless(self) -> None:
        """Test that dimensionless x_unit is accepted."""
        result = RVDataProperties.x_unit(u.dimensionless_unscaled)
        assert (result) == (u.dimensionless_unscaled)

    def test_rv_data_properties_x_unit_time(self) -> None:
        """Test that time units are accepted for x_unit."""
        result = RVDataProperties.x_unit(astropy_units.second)
        assert (result) == (astropy_units.second)

    def test_rv_data_properties_x_unit_none(self) -> None:
        """Test that None x_unit is converted to dimensionless."""
        result = RVDataProperties.x_unit(None)
        assert (result) == (u.dimensionless_unscaled)

    def test_rv_data_properties_y_unit_velocity(self) -> None:
        """Test that velocity units are accepted for y_unit."""
        result = RVDataProperties.y_unit(u.km / u.s)
        assert (result) == (u.km / u.s)

    def test_rv_data_properties_y_unit_m_per_s(self) -> None:
        """Test that m/s is accepted for y_unit."""
        result = RVDataProperties.y_unit(u.m / u.s)
        assert (result) == (u.m / u.s)

    def test_rv_data_properties_y_unit_invalid_raises(self) -> None:
        """Test that invalid y_unit raises ElisaAnalyticsDataError."""
        with pytest.raises(ElisaAnalyticsDataError):
            RVDataProperties.y_unit(u.mag)

    def test_rv_data_properties_x_unit_invalid_raises(self) -> None:
        """Test that invalid x_unit raises ElisaAnalyticsDataError."""
        with pytest.raises(ElisaAnalyticsDataError):
            RVDataProperties.x_unit(u.km / u.s)


class LCDataPropertiesTestCase(ElisaTestCase):
    """Test LCDataProperties unit validation."""

    def test_lc_data_properties_x_unit_dimensionless(self) -> None:
        """Test that dimensionless x_unit is accepted."""
        result = LCDataProperties.x_unit(u.dimensionless_unscaled)
        assert (result) == (u.dimensionless_unscaled)

    def test_lc_data_properties_x_unit_time(self) -> None:
        """Test that time units are accepted for x_unit."""
        result = LCDataProperties.x_unit(astropy_units.second)
        assert (result) == (astropy_units.second)

    def test_lc_data_properties_y_unit_dimensionless(self) -> None:
        """Test that dimensionless y_unit is accepted."""
        result = LCDataProperties.y_unit(u.dimensionless_unscaled)
        assert (result) == (u.dimensionless_unscaled)

    def test_lc_data_properties_y_unit_mag(self) -> None:
        """Test that magnitude units are accepted for y_unit."""
        result = LCDataProperties.y_unit(u.mag)
        assert (result) == (u.mag)

    def test_lc_data_properties_y_unit_invalid_raises(self) -> None:
        """Test that invalid y_unit raises ElisaAnalyticsDataError."""
        with pytest.raises(ElisaAnalyticsDataError):
            LCDataProperties.y_unit(u.km / u.s)

    def test_lc_data_properties_zero_magnitude_float(self) -> None:
        """Test zero_magnitude with float value."""
        value = 0.0
        result = LCDataProperties.zero_magnitude(value)
        assert (result) == (np.float64(0.0))
        assert isinstance(result, np.float64)

    def test_lc_data_properties_zero_magnitude_int(self) -> None:
        """Test zero_magnitude with integer value."""
        value = 5
        result = LCDataProperties.zero_magnitude(value)
        assert (result) == (np.float64(5.0))
        assert isinstance(result, np.float64)

    def test_lc_data_properties_zero_magnitude_quantity(self) -> None:
        """A valid scalar magnitude Quantity converts to float64."""
        result = LCDataProperties.zero_magnitude(1.0 * u.mag)
        assert (result) == (np.float64(1.0))
        assert isinstance(result, np.float64)

    def test_lc_data_properties_zero_magnitude_numpy_float(self) -> None:
        """Test zero_magnitude with numpy float value."""
        value = np.float64(2.5)
        result = LCDataProperties.zero_magnitude(value)
        assert (result) == (np.float64(2.5))
        assert isinstance(result, np.float64)

    def test_lc_data_properties_zero_magnitude_invalid_raises(self) -> None:
        """Test that invalid zero_magnitude type raises ElisaAnalyticsDataError."""
        with pytest.raises(ElisaAnalyticsDataError) as context:
            LCDataProperties.zero_magnitude("not a number")
        assert ("zero_magnitude") in (str(context.value))
