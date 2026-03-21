# keep it first
# due to stupid astropy units/constants implementation
from unittests import set_astropy_units

import os.path as op
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal
from elisa.analytics.dataset.utils import (
    read_data_file,
    convert_data,
    convert_flux,
    convert_flux_error,
    convert_unit,
    central_moving_average,
)
from unittests.utils import ElisaTestCase
from elisa.analytics import RVData, LCData
from elisa import units as u

set_astropy_units()


class DataTestCase(ElisaTestCase):
    DATA = op.join(op.abspath(op.dirname(__file__)), "../data", "dataset")


class DataReadTestCase(DataTestCase):
    def test_read_data_file(self):
        fpath = op.join(self.DATA, "dummy.csv")
        data = read_data_file(fpath, data_columns=(0, 1, 2)).T

        self.assertEqual(len(data), 3)
        assert_array_equal([0, 1, 2, 3, 4], data[0])
        assert_array_equal([1e-1] * 5, data[1])
        assert_array_equal([1e-3, 1e-3, 1e-3, 2e-3, 2e-3], data[2])


class ConvertDataTestCase(ElisaTestCase):
    """Test convert_data function for unit conversions."""

    def test_convert_data_dimensionless(self):
        """Test conversion with dimensionless unit returns data unchanged."""
        data = np.array([1.0, 2.0, 3.0])
        result = convert_data(data, u.dimensionless_unscaled, u.dimensionless_unscaled)
        assert_array_equal(result, data)

    def test_convert_data_with_units(self):
        """Test conversion from km/s to m/s."""
        data = np.array([100.0, 200.0, 300.0])
        result = convert_data(data, u.km / u.s, u.m / u.s)
        expected = np.array([100000.0, 200000.0, 300000.0])
        assert_array_almost_equal(result, expected)

    def test_convert_data_mag_to_mag(self):
        """Test conversion with same units returns data unchanged."""
        data = np.array([1.0, 2.0, 3.0])
        result = convert_data(data, u.mag, u.mag)
        assert_array_equal(result, data)


class ConvertFluxTestCase(ElisaTestCase):
    """Test convert_flux function for flux conversions."""

    def test_convert_flux_dimensionless_unchanged(self):
        """Test that dimensionless flux is returned unchanged."""
        data = np.array([0.5, 0.8, 1.0])
        result = convert_flux(data, u.dimensionless_unscaled)
        assert_array_equal(result, data)

    def test_convert_flux_mag_without_zero_point_raises(self):
        """Test that converting magnitudes without zero_point raises ValueError."""
        data = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError) as context:
            convert_flux(data, u.mag)
        self.assertIn("zero point", str(context.exception).lower())

    def test_convert_flux_mmag_without_zero_point_raises(self):
        """Test that converting millimagnitudes without zero_point raises ValueError."""
        data = np.array([100.0, 200.0, 300.0])
        with self.assertRaises(ValueError) as context:
            convert_flux(data, u.mmag)
        self.assertIn("zero point", str(context.exception).lower())

    def test_convert_flux_mag_with_zero_point(self):
        """Test conversion from magnitudes to flux with zero point."""
        data = np.array([1.0])
        result = convert_flux(data, u.mag, zero_point=0.0)
        # Magnitude 1.0 corresponds to flux = 10^(-1/2.5) ≈ 0.3981
        self.assertAlmostEqual(result[0], 0.3981, places=3)

    def test_convert_flux_mmag_with_zero_point(self):
        """Test conversion from millimagnitudes to flux with zero point."""
        data = np.array([1000.0])  # 1000 mmag = 1 mag
        result = convert_flux(data, u.mmag, zero_point=0.0)
        # Should give same result as 1 mag
        self.assertAlmostEqual(result[0], 0.3981, places=3)


class ConvertFluxErrorTestCase(ElisaTestCase):
    """Test convert_flux_error function for error conversions."""

    def test_convert_flux_error_dimensionless_unchanged(self):
        """Test that dimensionless error is returned unchanged."""
        error = np.array([0.01, 0.02, 0.03])
        result = convert_flux_error(error, u.dimensionless_unscaled)
        assert_array_equal(result, error)

    def test_convert_flux_error_mag_without_zero_point_raises(self):
        """Test that converting magnitude errors without zero_point raises ValueError."""
        error = np.array([0.01, 0.02, 0.03])
        with self.assertRaises(ValueError) as context:
            convert_flux_error(error, u.mag)
        self.assertIn("zero point", str(context.exception).lower())

    def test_convert_flux_error_mmag_without_zero_point_raises(self):
        """Test that converting millimagnitude errors without zero_point raises ValueError."""
        error = np.array([10.0, 20.0, 30.0])
        with self.assertRaises(ValueError) as context:
            convert_flux_error(error, u.mmag)
        self.assertIn("zero point", str(context.exception).lower())

    def test_convert_flux_error_mag_with_zero_point(self):
        """Test conversion from magnitude errors to flux errors."""
        error = np.array([0.05])
        result = convert_flux_error(error, u.mag, zero_point=0.0)
        # Error should be positive and converted properly
        self.assertGreater(result[0], 0)

    def test_convert_flux_error_mmag_with_zero_point(self):
        """Test conversion from millimagnitude errors to flux errors."""
        error = np.array([50.0])  # 50 mmag = 0.05 mag
        result = convert_flux_error(error, u.mmag, zero_point=0.0)
        # Should give same result as 0.05 mag
        expected = convert_flux_error(np.array([0.05]), u.mag, zero_point=0.0)
        assert_array_almost_equal(result, expected)


class ConvertUnitTestCase(ElisaTestCase):
    """Test convert_unit function for unit conversions."""

    def test_convert_unit_dimensionless_unchanged(self):
        """Test that dimensionless unit is returned unchanged."""
        result = convert_unit(u.dimensionless_unscaled, u.mag)
        self.assertEqual(result, u.dimensionless_unscaled)

    def test_convert_unit_to_new_unit(self):
        """Test that non-dimensionless units are converted."""
        result = convert_unit(u.km / u.s, u.m / u.s)
        self.assertEqual(result, u.m / u.s)

    def test_convert_unit_same_unit(self):
        """Test conversion with same unit."""
        result = convert_unit(u.mag, u.mag)
        self.assertEqual(result, u.mag)


class CentralMovingAverageTestCase(ElisaTestCase):
    """Test central_moving_average function for smoothing data."""

    def setUp(self):
        """Set up test data."""
        # Create simple test data
        self.phases = np.linspace(0, 1, 100)
        self.flux = 1.0 + 0.1 * np.sin(2 * np.pi * self.phases)
        self.errors = 0.01 * np.ones_like(self.flux)

    def test_central_moving_average_basic(self):
        """Test basic averaging functionality."""
        from elisa.analytics import LCData

        lc_data = LCData(
            x_data=self.phases,
            y_data=self.flux,
            y_err=self.errors,
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled,
        )

        original_length = len(lc_data.x_data)
        central_moving_average(lc_data, n_bins=20, radius=1)

        # Output should have fewer points (binned)
        self.assertLess(len(lc_data.x_data), original_length)
        self.assertEqual(len(lc_data.y_data), len(lc_data.x_data))
        self.assertEqual(len(lc_data.y_err), len(lc_data.x_data))

    def test_central_moving_average_with_errors(self):
        """Test averaging with error weighting."""
        from elisa.analytics import LCData

        lc_data = LCData(
            x_data=self.phases,
            y_data=self.flux,
            y_err=self.errors,
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled,
        )

        central_moving_average(lc_data, n_bins=10, radius=1, cyclic_boundaries=True)

        # Check that flux values are reasonable (should be close to original)
        self.assertTrue(np.all(lc_data.y_data >= 0.9))
        self.assertTrue(np.all(lc_data.y_data <= 1.1))

    def test_central_moving_average_without_errors(self):
        """Test averaging without error information."""
        from elisa.analytics import LCData

        lc_data = LCData(
            x_data=self.phases,
            y_data=self.flux,
            y_err=None,
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled,
        )

        central_moving_average(lc_data, n_bins=15, radius=1, cyclic_boundaries=True)

        # Check that output has expected length
        self.assertLessEqual(len(lc_data.x_data), 15)
        self.assertEqual(len(lc_data.y_data), len(lc_data.x_data))
        self.assertEqual(len(lc_data.y_err), len(lc_data.x_data))

    def test_central_moving_average_non_cyclic(self):
        """Test averaging with non-cyclic boundaries."""
        from elisa.analytics import LCData

        lc_data = LCData(
            x_data=self.phases,
            y_data=self.flux,
            y_err=self.errors,
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled,
        )

        central_moving_average(lc_data, n_bins=20, radius=2, cyclic_boundaries=False)

        # Check that averaging was performed
        self.assertGreater(len(lc_data.x_data), 0)
        self.assertEqual(len(lc_data.y_data), len(lc_data.x_data))

    def test_central_moving_average_output_phase_range(self):
        """Test that output phases are within expected range."""
        from elisa.analytics import LCData

        lc_data = LCData(
            x_data=self.phases,
            y_data=self.flux,
            y_err=self.errors,
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled,
        )

        central_moving_average(lc_data, n_bins=10, radius=1, cyclic_boundaries=True)

        # Output phases should be within original range
        self.assertGreaterEqual(np.min(lc_data.x_data), np.min(self.phases))
        self.assertLessEqual(np.max(lc_data.x_data), np.max(self.phases))


class LVDataTestCase(DataTestCase):
    def test_from_file(self):
        fpath = op.join(self.DATA, "dummy.csv")
        x_unit = u.dimensionless_unscaled
        y_unit = u.km/u.s
        rv_data = RVData.from_file(fpath, x_unit, y_unit=y_unit)

        self.assertEqual(rv_data.y_unit, u.m / u.s)
        self.assertEqual(rv_data.x_unit, u.dimensionless_unscaled)

        assert_array_equal(rv_data.x_data, np.arange(0, 5, 1))
        assert_array_equal(rv_data.y_data, [100] * 5)


class RVDataTestCase(DataTestCase):
    def test_from_file(self):
        fpath = op.join(self.DATA, "dummy.csv")
        x_unit = u.dimensionless_unscaled
        y_unit = u.dimensionless_unscaled
        lc_data = LCData.from_file(fpath, x_unit, y_unit=y_unit)

        self.assertEqual(lc_data.y_unit, u.dimensionless_unscaled)
        self.assertEqual(lc_data.x_unit, u.dimensionless_unscaled)

        assert_array_equal(lc_data.x_data, np.arange(0, 5, 1))
        assert_array_equal(lc_data.y_data, [0.1] * 5)

    def test_from_file_in_mag(self):
        fpath = op.join(self.DATA, "dummy.csv")
        x_unit = u.dimensionless_unscaled
        y_unit = u.mag
        lc_data = LCData.from_file(fpath, x_unit, y_unit=y_unit, reference_magnitude=1.0)

        assert_array_equal(lc_data.x_data, np.arange(0, 5, 1))
        assert_array_equal(np.round(lc_data.y_data, 2), [2.29] * 5)
