"""Unit tests for elisa.base.graphics.plot module."""
from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import numpy as np
from numpy.testing import assert_array_almost_equal

from elisa import const
from elisa.base.graphics.plot import (
    add_colormap_to_plt_kwargs,
    horizonatal_displacement_cmap,
    horizontal_g_pert_cmap,
    to_log,
    v_horizontal_pert_cmap,
    v_rad_pert_cmap,
)
from unittests import set_astropy_units
from unittests.utils import ElisaTestCase

if TYPE_CHECKING:
    from numpy.typing import NDArray

set_astropy_units()


class TestToLog(ElisaTestCase):
    """Test cases for :func:`to_log` function."""

    def test_to_log_logarithmic_scale(self) -> None:
        """
        Test conversion to logarithmic scale with 'log' string.

        Verifies that the :func:`to_log` function correctly applies log10
        transformation when scale is 'log'.

        :return: None
        """
        value: NDArray = np.array([1.0, 10.0, 100.0, 1000.0])
        result: NDArray = to_log(value, 'log')
        expected: NDArray = np.log10(value)
        assert_array_almost_equal(result, expected)

    def test_to_log_logarithmic_string(self) -> None:
        """
        Test conversion to logarithmic scale with 'logarithmic' string.

        Verifies that the :func:`to_log` function correctly applies log10
        transformation when scale is 'logarithmic'.

        :return: None
        """
        value: NDArray = np.array([1.0, 10.0, 100.0, 1000.0])
        result: NDArray = to_log(value, 'logarithmic')
        expected: NDArray = np.log10(value)
        assert_array_almost_equal(result, expected)

    def test_to_log_linear_scale(self) -> None:
        """
        Test that linear scale returns values unchanged.

        Verifies that the :func:`to_log` function returns the original
        values when scale is 'linear' without any transformation.

        :return: None
        """
        value: NDArray = np.array([1.0, 10.0, 100.0, 1000.0])
        result: NDArray = to_log(value, 'linear')
        assert_array_almost_equal(result, value)

    def test_to_log_scalar_value(self) -> None:
        """
        Test :func:`to_log` with scalar values.

        Verifies that the :func:`to_log` function correctly handles scalar
        inputs for both logarithmic and linear scales.

        :return: None
        """
        result_log: float = to_log(100.0, 'log')
        result_linear: float = to_log(100.0, 'linear')
        self.assertAlmostEqual(result_log, 2.0)
        self.assertAlmostEqual(result_linear, 100.0)

    def test_to_log_invalid_scale(self) -> None:
        """
        Test :func:`to_log` with invalid scale string returns linear values.

        Verifies that the :func:`to_log` function returns the original
        values unchanged when given an invalid scale string, treating it
        as a linear scale.

        :return: None
        """
        value: NDArray = np.array([1.0, 10.0, 100.0])
        result: NDArray = to_log(value, 'invalid')
        assert_array_almost_equal(result, value)


class TestAddColormapToPltKwargs(ElisaTestCase):
    """Test cases for :func:`add_colormap_to_plt_kwargs` function."""

    def test_unknown_colormap_raises_key_error(self) -> None:
        """
        Test that unknown colormap raises KeyError.

        Verifies that the :func:`add_colormap_to_plt_kwargs` function raises
        a KeyError when provided with an invalid colormap name.

        :return: None
        :raises KeyError: When colormap name is not recognized
        """
        args: tuple = (
            'unknown_colormap',
            None,
            0.0,
            0.0,
            1.0,
            0.0,
            const.Position(0, np.nan, 0.0, np.nan, 0.0),
        )
        error_msg: str = (
            'Attempting to use unknown colormap should raise KeyError'
        )
        with self.assertRaises(KeyError, msg=error_msg):
            add_colormap_to_plt_kwargs(*args)

    def test_colormap_none_returns_none(self) -> None:
        """
        Test that None colormap returns None.

        Verifies that the :func:`add_colormap_to_plt_kwargs` function returns
        None when passed None as the colormap argument, indicating no colormap
        should be applied.

        :return: None
        """
        args: tuple = (
            None,
            None,
            0.0,
            0.0,
            1.0,
            0.0,
            const.Position(0, np.nan, 0.0, np.nan, 0.0),
        )
        result = add_colormap_to_plt_kwargs(*args)
        self.assertIsNone(result)





class TestHorizonatalDisplacementCmap(ElisaTestCase):
    """Test cases for :func:`horizonatal_displacement_cmap` function."""

    def test_horizonatal_displacement_cmap_no_pulsations_raises_error(
        self,
    ) -> None:
        """
        Test that :func:`horizonatal_displacement_cmap` raises ValueError.

        Verifies that the :func:`horizonatal_displacement_cmap` function
        raises a ValueError when called on a star container that does not
        have pulsations, since horizontal displacement colormaps are only
        relevant for pulsating stars.

        :return: None
        :raises ValueError: When star has no pulsations
        """
        class MockStar:
            """Mock star container without pulsations."""

            def has_pulsations(self) -> bool:
                """Return False to indicate no pulsations."""
                return False

        mock_star: MockStar = MockStar()
        error_msg: str = (
            'Horizontal displacement colormap should raise ValueError '
            'for stars without pulsations'
        )
        with self.assertRaises(ValueError, msg=error_msg):
            horizonatal_displacement_cmap(mock_star, 'linear', 'default', False, 1.0)


class TestVRadPertCmap(ElisaTestCase):
    """Test cases for :func:`v_rad_pert_cmap` function."""

    def test_v_rad_pert_cmap_no_pulsations_raises_error(self) -> None:
        """
        Test that :func:`v_rad_pert_cmap` raises ValueError.

        Verifies that the :func:`v_rad_pert_cmap` function raises a
        ValueError when called on a star container that does not have
        pulsations, since radial velocity perturbation colormaps are only
        relevant for pulsating stars.

        :return: None
        :raises ValueError: When star has no pulsations
        """
        class MockStar:
            """Mock star container without pulsations."""

            def has_pulsations(self) -> bool:
                """Return False to indicate no pulsations."""
                return False

        mock_star: MockStar = MockStar()
        error_msg: str = (
            'Radial velocity perturbation colormap should raise ValueError '
            'for stars without pulsations'
        )
        with self.assertRaises(ValueError, msg=error_msg):
            v_rad_pert_cmap(mock_star, 'linear', 'default', False, 1.0)


class TestVHorizontalPertCmap(ElisaTestCase):
    """Test cases for :func:`v_horizontal_pert_cmap` function."""

    def test_v_horizontal_pert_cmap_no_pulsations_raises_error(self) -> None:
        """
        Test that :func:`v_horizontal_pert_cmap` raises ValueError.

        Verifies that the :func:`v_horizontal_pert_cmap` function raises a
        ValueError when called on a star container that does not have
        pulsations, since horizontal velocity perturbation colormaps are only
        relevant for pulsating stars.

        :return: None
        :raises ValueError: When star has no pulsations
        """
        class MockStar:
            """Mock star container without pulsations."""

            def has_pulsations(self) -> bool:
                """Return False to indicate no pulsations."""
                return False

        mock_star: MockStar = MockStar()
        error_msg: str = (
            'Horizontal velocity perturbation colormap should raise ValueError '
            'for stars without pulsations'
        )
        with self.assertRaises(ValueError, msg=error_msg):
            v_horizontal_pert_cmap(mock_star, 'linear', 'default', False, 1.0)


class TestHorizontalGPertCmap(ElisaTestCase):
    """Test cases for :func:`horizontal_g_pert_cmap` function."""

    def test_horizontal_g_pert_cmap_no_pulsations_raises_error(self) -> None:
        """
        Test that :func:`horizontal_g_pert_cmap` raises ValueError.

        Verifies that the :func:`horizontal_g_pert_cmap` function raises a
        ValueError when called on a star container that does not have
        pulsations, since horizontal gravity acceleration perturbation
        colormaps are only relevant for pulsating stars.

        :return: None
        :raises ValueError: When star has no pulsations
        """
        class MockStar:
            """Mock star container without pulsations."""

            def has_pulsations(self) -> bool:
                """Return False to indicate no pulsations."""
                return False

        mock_star: MockStar = MockStar()
        error_msg: str = (
            'Horizontal gravity acceleration perturbation colormap should '
            'raise ValueError for stars without pulsations'
        )
        with self.assertRaises(ValueError, msg=error_msg):
            horizontal_g_pert_cmap(mock_star, 'linear', 'default', False, 1.0)


if __name__ == '__main__':
    unittest.main()

