"""Unit tests for elisa.analytics.tools.utils module."""
from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import numpy as np

from elisa.analytics.tools import utils
from unittests import set_astropy_units
from unittests.utils import ElisaTestCase

if TYPE_CHECKING:
    from numpy.typing import NDArray

set_astropy_units()


class TestLightcurvesMeanError(ElisaTestCase):
    """Test cases for :func:`lightcurves_mean_error` function."""

    def test_lightcurves_mean_error_basic(self) -> None:
        """Test basic light curve error calculation.

        Verifies that the function returns 5% of the mean value of the
        light curve, as expected.

        :return: None
        """
        lc: NDArray = np.array([0.9, 0.95, 1.0, 0.95, 0.9])
        expected_error: float = np.mean(lc) * 0.05

        result: float = utils.lightcurves_mean_error(lc)

        self.assertAlmostEqual(result, expected_error, places=10)

    def test_lightcurves_mean_error_constant_values(self) -> None:
        """Test error calculation with constant light curve values.

        Verifies that the function correctly calculates 5% error for
        a light curve with all identical values.

        :return: None
        """
        lc: NDArray = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        expected_error: float = 1.0 * 0.05

        result: float = utils.lightcurves_mean_error(lc)

        self.assertAlmostEqual(result, expected_error, places=10)

    def test_lightcurves_mean_error_zero_mean(self) -> None:
        """Test error calculation with zero mean light curve.

        Verifies that the function returns zero error when the mean of
        the light curve is zero.

        :return: None
        """
        lc: NDArray = np.array([-1.0, 0.0, 1.0, -1.0, 1.0])
        expected_error: float = np.mean(lc) * 0.05

        result: float = utils.lightcurves_mean_error(lc)

        self.assertAlmostEqual(result, expected_error, places=10)

    def test_lightcurves_mean_error_single_value(self) -> None:
        """Test error calculation with single-element light curve.

        Verifies that the function correctly handles a single-element array.

        :return: None
        """
        lc: NDArray = np.array([0.8])
        expected_error: float = 0.8 * 0.05

        result: float = utils.lightcurves_mean_error(lc)

        self.assertAlmostEqual(result, expected_error, places=10)

    def test_lightcurves_mean_error_large_values(self) -> None:
        """Test error calculation with large light curve values.

        Verifies that the function scales correctly with large magnitudes.

        :return: None
        """
        lc: NDArray = np.array([1000.0, 2000.0, 3000.0])
        expected_error: float = np.mean(lc) * 0.05

        result: float = utils.lightcurves_mean_error(lc)

        self.assertAlmostEqual(result, expected_error, places=10)


class TestRadialcurvesMeanError(ElisaTestCase):
    """Test cases for :func:`radialcurves_mean_error` function."""

    def test_radialcurves_mean_error_basic(self) -> None:
        """Test basic radial velocity error calculation.

        Verifies that the function returns 5% of the mean absolute value
        of the radial velocity curve.

        :return: None
        """
        rv: NDArray = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        expected_error: float = np.mean(rv) * 0.05

        result: float = utils.radialcurves_mean_error(rv)

        self.assertAlmostEqual(result, expected_error, places=10)

    def test_radialcurves_mean_error_constant_values(self) -> None:
        """Test error calculation with constant radial velocity values.

        Verifies that the function correctly calculates 5% error for
        a radial velocity curve with all identical values.

        :return: None
        """
        rv: NDArray = np.array([50.0, 50.0, 50.0, 50.0])
        expected_error: float = 50.0 * 0.05

        result: float = utils.radialcurves_mean_error(rv)

        self.assertAlmostEqual(result, expected_error, places=10)

    def test_radialcurves_mean_error_zero_mean(self) -> None:
        """Test error calculation with zero mean radial velocity.

        Verifies that the function returns zero error when the mean of
        the radial velocities is zero.

        :return: None
        """
        rv: NDArray = np.array([-50.0, -25.0, 25.0, 50.0])
        expected_error: float = np.mean(rv) * 0.05

        result: float = utils.radialcurves_mean_error(rv)

        self.assertAlmostEqual(result, expected_error, places=10)

    def test_radialcurves_mean_error_negative_values(self) -> None:
        """Test error calculation with negative radial velocities.

        Verifies that the function correctly handles negative velocity values.

        :return: None
        """
        rv: NDArray = np.array([-100.0, -200.0, -300.0])
        expected_error: float = np.mean(rv) * 0.05

        result: float = utils.radialcurves_mean_error(rv)

        self.assertAlmostEqual(result, expected_error, places=10)

    def test_radialcurves_mean_error_single_value(self) -> None:
        """Test error calculation with single-element radial velocity.

        Verifies that the function correctly handles a single-element array.

        :return: None
        """
        rv: NDArray = np.array([25.0])
        expected_error: float = 25.0 * 0.05

        result: float = utils.radialcurves_mean_error(rv)

        self.assertAlmostEqual(result, expected_error, places=10)


class TestIsTimeDependent(ElisaTestCase):
    """Test cases for :func:`is_time_dependent` function."""

    def test_is_time_dependent_true(self) -> None:
        """Test detection of time-dependent parameters.

        Verifies that the function returns True when both period and
        primary_minimum_time are present in the labels.

        :return: None
        """
        labels: list[str] = [
            "system@period",
            "system@primary_minimum_time",
            "primary@t_eff",
        ]

        result: bool = utils.is_time_dependent(labels)

        self.assertTrue(result)

    def test_is_time_dependent_missing_period(self) -> None:
        """Test with missing period parameter.

        Verifies that the function returns False when period is absent
        even if primary_minimum_time is present.

        :return: None
        """
        labels: list[str] = [
            "system@primary_minimum_time",
            "primary@t_eff",
        ]

        result: bool = utils.is_time_dependent(labels)

        self.assertFalse(result)

    def test_is_time_dependent_missing_minimum_time(self) -> None:
        """Test with missing primary_minimum_time parameter.

        Verifies that the function returns False when primary_minimum_time
        is absent even if period is present.

        :return: None
        """
        labels: list[str] = [
            "system@period",
            "primary@t_eff",
        ]

        result: bool = utils.is_time_dependent(labels)

        self.assertFalse(result)

    def test_is_time_dependent_empty_labels(self) -> None:
        """Test with empty labels list.

        Verifies that the function returns False when no labels are provided.

        :return: None
        """
        labels: list[str] = []

        result: bool = utils.is_time_dependent(labels)

        self.assertFalse(result)

    def test_is_time_dependent_both_missing(self) -> None:
        """Test with both required parameters missing.

        Verifies that the function returns False when neither period nor
        primary_minimum_time are present.

        :return: None
        """
        labels: list[str] = [
            "primary@t_eff",
            "secondary@t_eff",
        ]

        result: bool = utils.is_time_dependent(labels)

        self.assertFalse(result)


class TestTimeLayerResolver(ElisaTestCase):
    """Test cases for :func:`time_layer_resolver` function."""

    def test_time_layer_resolver_time_dependent_no_pop(self) -> None:
        """Test time-dependent data conversion without pop.

        Verifies that JD times are converted to phases when time-dependent
        parameters are present, and primary_minimum_time is preserved.

        :return: None
        """
        x_data: NDArray = np.array([2450000.0, 2450001.0, 2450002.0])
        kwargs: dict = {
            "system@period": 1.0,
            "system@primary_minimum_time": 2450000.5,
            "primary@t_eff": 5000.0,
        }

        x_data_new, kwargs_new = utils.time_layer_resolver(x_data, pop=False, **kwargs)

        # Check that x_data was converted to phases
        self.assertIsInstance(x_data_new, np.ndarray)
        self.assertEqual(len(x_data_new), len(x_data))
        # Check that primary_minimum_time is still in kwargs
        self.assertIn("system@primary_minimum_time", kwargs_new)

    def test_time_layer_resolver_time_dependent_with_pop(self) -> None:
        """Test time-dependent data conversion with pop.

        Verifies that JD times are converted to phases and
        primary_minimum_time is removed from kwargs when pop=True.

        :return: None
        """
        x_data: NDArray = np.array([2450000.0, 2450001.0, 2450002.0])
        kwargs: dict = {
            "system@period": 1.0,
            "system@primary_minimum_time": 2450000.5,
            "primary@t_eff": 5000.0,
        }

        x_data_new, kwargs_new = utils.time_layer_resolver(x_data, pop=True, **kwargs)

        # Check that x_data was converted
        self.assertIsInstance(x_data_new, np.ndarray)
        # Check that primary_minimum_time was removed
        self.assertNotIn("system@primary_minimum_time", kwargs_new)
        # Check that other parameters remain
        self.assertIn("primary@t_eff", kwargs_new)

    def test_time_layer_resolver_phase_data(self) -> None:
        """Test with phase data (non-time-dependent).

        Verifies that phase data (0-1 range) is converted to modulo 1.0
        when time-dependent parameters are absent.

        :return: None
        """
        x_data: NDArray = np.array([0.2, 0.7, 1.2, 1.8])
        kwargs: dict = {
            "primary@t_eff": 5000.0,
            "secondary@t_eff": 6000.0,
        }

        x_data_new, kwargs_new = utils.time_layer_resolver(x_data, pop=False, **kwargs)

        # Check that x_data was converted using modulo 1.0
        expected: NDArray = x_data % 1.0
        np.testing.assert_array_almost_equal(x_data_new, expected)
        # Check that kwargs remain unchanged
        self.assertEqual(kwargs_new, kwargs)

    def test_time_layer_resolver_phase_data_with_pop(self) -> None:
        """Test phase data with pop=True.

        Verifies that pop parameter is ignored when data is not time-dependent.

        :return: None
        """
        x_data: NDArray = np.array([0.1, 0.5, 0.9])
        kwargs: dict = {
            "primary@t_eff": 5000.0,
        }

        x_data_new, kwargs_new = utils.time_layer_resolver(x_data, pop=True, **kwargs)

        # Check that x_data was converted using modulo 1.0
        expected: NDArray = x_data % 1.0
        np.testing.assert_array_almost_equal(x_data_new, expected)

    def test_time_layer_resolver_empty_kwargs(self) -> None:
        """Test with empty kwargs dictionary.

        Verifies that the function handles empty kwargs gracefully,
        treating data as phase data.

        :return: None
        """
        x_data: NDArray = np.array([0.2, 0.8, 1.3])
        kwargs: dict = {}

        x_data_new, kwargs_new = utils.time_layer_resolver(x_data, pop=False, **kwargs)

        # Check that x_data was converted using modulo 1.0
        expected: NDArray = x_data % 1.0
        np.testing.assert_array_almost_equal(x_data_new, expected)
        # Check that kwargs remain empty
        self.assertEqual(kwargs_new, {})


if __name__ == "__main__":
    unittest.main()

