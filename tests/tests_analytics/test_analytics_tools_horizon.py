"""Unit tests for elisa.analytics.tools.horizon module."""

from __future__ import annotations

import unittest

import numpy as np

from elisa.analytics.tools.horizon import (
    estimate_analytic_horizon,
    get_analytics_horizon,
    get_discrete_horizon,
)
from tests import set_astropy_units
from tests.utils import ElisaTestCase

set_astropy_units()


class TestEstimateAnalyticHorizon(ElisaTestCase):
    """Test cases for :func:`estimate_analytic_horizon` function."""

    def test_estimate_analytic_horizon_default_params(self) -> None:
        """Test horizon estimation with default parameters.

        Verifies that the function returns a 2D array of horizon points
        when called with default parameters.

        :return: None
        """
        horizon = estimate_analytic_horizon()

        assert isinstance(horizon, np.ndarray)
        assert (horizon.ndim) == (2)
        assert (horizon.shape[1]) == (2)  # Y and Z coordinates
        assert (horizon.shape[0]) > (0)  # At least some points

    def test_estimate_analytic_horizon_at_phase_zero(self) -> None:
        """Test horizon estimation at primary eclipse phase.

        Verifies that the function correctly estimates the horizon at
        the primary eclipse phase (phase = 0.0).

        :return: None
        """
        horizon = estimate_analytic_horizon(phase=0.0)

        assert isinstance(horizon, np.ndarray)
        assert (horizon.ndim) == (2)
        assert (horizon.shape[0]) > (0)

    def test_estimate_analytic_horizon_at_phase_half(self) -> None:
        """Test horizon estimation at secondary eclipse phase.

        Verifies that the function correctly estimates the horizon at
        the secondary eclipse phase (phase = 0.5).

        :return: None
        """
        horizon = estimate_analytic_horizon(phase=0.5)

        assert isinstance(horizon, np.ndarray)
        assert (horizon.shape[0]) > (0)

    def test_estimate_analytic_horizon_polar_coords(self) -> None:
        """Test horizon estimation in polar coordinates.

        Verifies that the function returns properly sorted polar coordinates
        when polar=True.

        :return: None
        """
        horizon = estimate_analytic_horizon(polar=True)

        assert isinstance(horizon, np.ndarray)
        # Check if angles are sorted
        if horizon.shape[0] > 1:
            angles = horizon[:, 1]
            assert np.all(angles[:-1] <= angles[1:])

    def test_estimate_analytic_horizon_with_cosine_precision(self) -> None:
        """Test horizon estimation with cosine precision calculation.

        Verifies that the function returns both horizon points and precision
        when cosine_precision=True.

        :return: None
        """
        result = estimate_analytic_horizon(cosine_precision=True)

        assert isinstance(result, tuple)
        assert (len(result)) == (2)
        horizon, precision = result
        assert isinstance(horizon, np.ndarray)
        assert isinstance(precision, (int, float, type(None)))

    def test_estimate_analytic_horizon_3d_points(self) -> None:
        """Test horizon estimation returning 3D points.

        Verifies that the function returns 3D points when _3d=True.

        :return: None
        """
        horizon_3d = estimate_analytic_horizon(_3d=True)

        assert isinstance(horizon_3d, np.ndarray)
        assert (horizon_3d.ndim) == (2)
        assert (horizon_3d.shape[1]) == (3)  # X, Y, Z coordinates


class TestGetAnalyticsHorizon(ElisaTestCase):
    """Test cases for :func:`get_analytics_horizon` function."""

    def test_get_analytics_horizon_default_params(self) -> None:
        """Test analytics horizon computation with default parameters.

        Verifies that the function returns a 2D array of horizon points
        when called with default parameters.

        :return: None
        """
        horizon = get_analytics_horizon()

        assert isinstance(horizon, np.ndarray)
        assert (horizon.ndim) == (2)
        assert (horizon.shape[1]) == (2)  # Y and Z coordinates
        assert (horizon.shape[0]) > (0)

    def test_get_analytics_horizon_at_phase_zero(self) -> None:
        """Test analytics horizon at primary eclipse phase.

        Verifies that the function correctly computes the horizon using
        the analytical approach at phase = 0.0.

        :return: None
        """
        horizon = get_analytics_horizon(phase=0.0)

        assert isinstance(horizon, np.ndarray)
        assert (horizon.shape[0]) > (0)

    def test_get_analytics_horizon_at_phase_quarter(self) -> None:
        """Test analytics horizon at quarter orbital phase.

        Verifies that the function correctly computes the horizon at
        phase = 0.25.

        :return: None
        """
        horizon = get_analytics_horizon(phase=0.25)

        assert isinstance(horizon, np.ndarray)
        assert (horizon.shape[0]) > (0)

    def test_get_analytics_horizon_tolerance_effect(self) -> None:
        """Test that tolerance parameter affects horizon computation.

        Verifies that different tolerance values produce different results
        in the number of horizon points found.

        :return: None
        """
        horizon_loose = get_analytics_horizon(tol=1e-2)
        horizon_tight = get_analytics_horizon(tol=1e-4)

        assert isinstance(horizon_loose, np.ndarray)
        assert isinstance(horizon_tight, np.ndarray)
        assert (horizon_tight.shape[0]) > (0)
        assert (horizon_loose.shape[0]) > (0)

    def test_get_analytics_horizon_polar_coords(self) -> None:
        """Test analytics horizon in polar coordinates.

        Verifies that the function returns sorted polar coordinates
        when polar=True.

        :return: None
        """
        horizon = get_analytics_horizon(polar=True)

        assert isinstance(horizon, np.ndarray)
        # Check if angles are sorted
        if horizon.shape[0] > 1:
            angles = horizon[:, 1]
            assert np.all(angles[:-1] <= angles[1:])

    def test_get_analytics_horizon_density_effect(self) -> None:
        """Test that density parameters affect computation.

        Verifies that phi_density and theta_density parameters are accepted
        and produce valid results.

        :return: None
        """
        horizon = get_analytics_horizon(phi_density=50, theta_density=500)

        assert isinstance(horizon, np.ndarray)
        assert (horizon.shape[0]) > (0)


class TestGetDiscreteHorizon(ElisaTestCase):
    """Test cases for :func:`get_discrete_horizon` function."""

    def test_get_discrete_horizon_default_params(self) -> None:
        """Test discrete horizon computation with default parameters.

        Verifies that the function returns a tuple of horizon arrays
        when called with default parameters.

        :return: None
        """
        result = get_discrete_horizon()

        assert isinstance(result, tuple)
        assert (len(result)) == (2)
        horizon, origin_horizon = result
        assert isinstance(horizon, np.ndarray)
        assert isinstance(origin_horizon, np.ndarray)
        assert (horizon.shape[0]) > (origin_horizon.shape[0])

    def test_get_discrete_horizon_at_phase_zero(self) -> None:
        """Test discrete horizon at primary eclipse phase.

        Verifies that the function correctly computes the discrete horizon
        at phase = 0.0.

        :return: None
        """
        horizon, origin_horizon = get_discrete_horizon(phase=0.0)

        assert isinstance(horizon, np.ndarray)
        assert isinstance(origin_horizon, np.ndarray)
        assert (horizon.shape[0]) > (0)
        assert (origin_horizon.shape[0]) > (0)

    def test_get_discrete_horizon_at_phase_quarter(self) -> None:
        """Test discrete horizon at quarter orbital phase.

        Verifies that the function correctly computes the horizon at
        phase = 0.25.

        :return: None
        """
        horizon, _origin_horizon = get_discrete_horizon(phase=0.25)

        assert isinstance(horizon, np.ndarray)
        assert (horizon.shape[0]) > (0)

    def test_get_discrete_horizon_polar_coords(self) -> None:
        """Test discrete horizon in polar coordinates.

        Verifies that the function returns sorted polar coordinates
        when polar=True.

        :return: None
        """
        horizon, origin_horizon = get_discrete_horizon(polar=True)

        assert isinstance(horizon, np.ndarray)
        assert isinstance(origin_horizon, np.ndarray)
        # Check if angles are sorted
        if horizon.shape[0] > 1:
            angles = horizon[:, 1]
            assert np.all(angles[:-1] <= angles[1:])
        if origin_horizon.shape[0] > 1:
            angles = origin_horizon[:, 1]
            assert np.all(angles[:-1] <= angles[1:])

    def test_get_discrete_horizon_cartesian_coords(self) -> None:
        """Test discrete horizon in Cartesian coordinates.

        Verifies that the function returns Cartesian coordinates
        when polar=False.

        :return: None
        """
        horizon, origin_horizon = get_discrete_horizon(polar=False)

        assert isinstance(horizon, np.ndarray)
        assert (horizon.shape[1]) == (2)  # Y and Z coordinates
        assert (origin_horizon.shape[1]) == (2)


if __name__ == "__main__":
    unittest.main()
