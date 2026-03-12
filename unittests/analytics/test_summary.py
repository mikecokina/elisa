"""Unit tests for analytics binary fit summary functions.

This module provides comprehensive tests for summary report generation
functions used in light curve and radial velocity fitting.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from elisa import settings, umpy as up
from elisa.analytics.binary_fit.summary import (
    _manage_chain_evaluation,
    evaluate_binary_params,
    evaluate_rv_params,
    fit_lc_summary_with_error_propagation,
    fit_rv_summary_with_error_propagation,
    simple_lc_fit_summary,
    simple_rv_fit_summary,
)


class TestManageChainEvaluation:
    """Test suite for _manage_chain_evaluation function."""

    def test_single_process_evaluation(self) -> None:
        """Test chain evaluation with single process."""
        mock_chain = np.array([[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]])

        def mock_eval_func(dummy_arg: str, chain: np.ndarray) -> np.ndarray:
            """Mock evaluation function."""
            return chain * 2

        with patch.object(settings, "NUMBER_OF_MCMC_PROCESSES", 1):
            result = _manage_chain_evaluation(mock_chain, mock_eval_func, "test_arg")

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, mock_chain * 2)

    def test_multiprocess_evaluation(self) -> None:
        """Test chain evaluation with multiprocessing."""
        mock_chain = np.array([[0.1, 0.2], [0.15, 0.25], [0.2, 0.3], [0.25, 0.35]])

        def mock_eval_func(dummy_arg: str, chain: np.ndarray) -> np.ndarray:
            """Mock evaluation function."""
            return chain * 2

        with patch.object(settings, "NUMBER_OF_MCMC_PROCESSES", 2):
            with patch("elisa.analytics.binary_fit.summary.Pool") as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool

                # Mock async results
                mock_result1 = MagicMock()
                mock_result1.get.return_value = np.array([[0.2, 0.4], [0.3, 0.5]])
                mock_result2 = MagicMock()
                mock_result2.get.return_value = np.array([[0.4, 0.6], [0.5, 0.7]])

                mock_pool.apply_async.side_effect = [mock_result1, mock_result2]

                result = _manage_chain_evaluation(mock_chain, mock_eval_func, "test_arg")

                assert isinstance(result, np.ndarray)
                mock_pool.close.assert_called_once()
                mock_pool.join.assert_called_once()

    def test_eval_function_called_correctly(self) -> None:
        """Test that evaluation function is called with correct arguments."""
        mock_chain = np.array([[0.1, 0.2], [0.15, 0.25]])
        test_args = ("arg1", "arg2")

        eval_func = Mock(return_value=mock_chain)

        with patch.object(settings, "NUMBER_OF_MCMC_PROCESSES", 1):
            _manage_chain_evaluation(mock_chain, eval_func, *test_args)

        eval_func.assert_called_once()
        call_args = eval_func.call_args[0]
        assert "arg1" in call_args
        assert "arg2" in call_args
        np.testing.assert_array_equal(call_args[-1], mock_chain)


class TestSimpleLCFitSummary:
    """Test suite for simple_lc_fit_summary function."""

    @pytest.fixture
    def mock_lc_fit_instance(self) -> Mock:
        """Create a mock LC fit instance."""
        mock_fit = Mock()
        mock_fit.morphology = "detached"
        mock_fit.flat_result = {
            "system@mass_ratio": {"value": 0.5, "unit": "dimensionless"},
            "system@semi_major_axis": {"value": 10.0, "unit": "solRad"},
            "system@inclination": {"value": 90.0, "unit": "deg"},
            "system@eccentricity": {"value": 0.0, "unit": "dimensionless"},
            "system@argument_of_periastron": {"value": 0.0, "unit": "deg"},
            "system@period": {"value": 3.0, "unit": "d"},
            "primary@t_eff": {"value": 5000.0, "unit": "K"},
            "primary@surface_potential": {"value": 5.0, "unit": "dimensionless"},
            "secondary@t_eff": {"value": 6000.0, "unit": "K"},
            "secondary@surface_potential": {"value": 5.0, "unit": "dimensionless"},
            "r_squared": {"value": 0.95, "unit": "dimensionless"},
        }
        return mock_fit

    def test_summary_to_stdout(self, mock_lc_fit_instance: Mock) -> None:
        """Test generating summary to stdout."""
        with patch(
            "elisa.analytics.binary_fit.summary.lc_model.prepare_binary"
        ) as mock_prepare:
            mock_binary = MagicMock()
            mock_binary.mass_ratio = 0.5
            mock_binary.semi_major_axis = 1.0
            mock_binary.inclination = 90.0
            mock_binary.eccentricity = 0.0
            mock_binary.critical_potential.return_value = 2.5
            mock_binary.calculate_bolometric_luminosity.return_value = {
                "primary": 1.0,
                "secondary": 0.5,
            }

            # Mock star instances
            mock_primary = MagicMock()
            mock_primary.mass = 1.0
            mock_primary.equivalent_radius = 0.5
            mock_primary.polar_radius = 0.5
            mock_primary.backward_radius = 0.5
            mock_primary.side_radius = 0.5
            mock_primary.forward_radius = 0.5
            mock_primary.synchronicity = 1.0
            mock_primary.gravity_darkening = 0.32
            mock_primary.albedo = 1.0
            mock_primary.metallicity = 0.0
            mock_primary.has_spots.return_value = False
            mock_primary.has_pulsations.return_value = False

            mock_secondary = MagicMock()
            mock_secondary.mass = 0.5
            mock_secondary.equivalent_radius = 0.35
            mock_secondary.polar_radius = 0.35
            mock_secondary.backward_radius = 0.35
            mock_secondary.side_radius = 0.35
            mock_secondary.forward_radius = 0.35
            mock_secondary.synchronicity = 1.0
            mock_secondary.gravity_darkening = 0.32
            mock_secondary.albedo = 1.0
            mock_secondary.metallicity = 0.0
            mock_secondary.has_spots.return_value = False
            mock_secondary.has_pulsations.return_value = False

            mock_binary.primary = mock_primary
            mock_binary.secondary = mock_secondary
            mock_prepare.return_value = mock_binary

            with patch("builtins.print") as mock_print:
                simple_lc_fit_summary(mock_lc_fit_instance, path=None)

                # Verify print was called
                assert mock_print.called

    def test_summary_to_file(self, mock_lc_fit_instance: Mock, tmp_path: Path) -> None:
        """Test generating summary to file."""
        output_file = tmp_path / "summary.txt"

        with patch(
            "elisa.analytics.binary_fit.summary.lc_model.prepare_binary"
        ) as mock_prepare:
            mock_binary = MagicMock()
            mock_binary.mass_ratio = 0.5
            mock_binary.semi_major_axis = 1.0
            mock_binary.inclination = 90.0
            mock_binary.eccentricity = 0.0
            mock_binary.critical_potential.return_value = 2.5
            mock_binary.calculate_bolometric_luminosity.return_value = {
                "primary": 1.0,
                "secondary": 0.5,
            }

            # Mock star instances
            mock_primary = MagicMock()
            mock_primary.mass = 1.0
            mock_primary.equivalent_radius = 0.5
            mock_primary.polar_radius = 0.5
            mock_primary.backward_radius = 0.5
            mock_primary.side_radius = 0.5
            mock_primary.forward_radius = 0.5
            mock_primary.synchronicity = 1.0
            mock_primary.gravity_darkening = 0.32
            mock_primary.albedo = 1.0
            mock_primary.metallicity = 0.0
            mock_primary.has_spots.return_value = False
            mock_primary.has_pulsations.return_value = False

            mock_secondary = MagicMock()
            mock_secondary.mass = 0.5
            mock_secondary.equivalent_radius = 0.35
            mock_secondary.polar_radius = 0.35
            mock_secondary.backward_radius = 0.35
            mock_secondary.side_radius = 0.35
            mock_secondary.forward_radius = 0.35
            mock_secondary.synchronicity = 1.0
            mock_secondary.gravity_darkening = 0.32
            mock_secondary.albedo = 1.0
            mock_secondary.metallicity = 0.0
            mock_secondary.has_spots.return_value = False
            mock_secondary.has_pulsations.return_value = False

            mock_binary.primary = mock_primary
            mock_binary.secondary = mock_secondary
            mock_prepare.return_value = mock_binary

            simple_lc_fit_summary(mock_lc_fit_instance, path=str(output_file))

            # Verify file was created
            assert output_file.exists()
            content = output_file.read_text()
            assert "BINARY SYSTEM" in content

    def test_summary_dimensionless_radii(self, mock_lc_fit_instance: Mock) -> None:
        """Test summary with dimensionless radii."""
        with patch(
            "elisa.analytics.binary_fit.summary.lc_model.prepare_binary"
        ) as mock_prepare:
            mock_binary = MagicMock()
            mock_binary.mass_ratio = 0.5
            mock_binary.semi_major_axis = 1.0
            mock_binary.inclination = 90.0
            mock_binary.eccentricity = 0.0
            mock_binary.critical_potential.return_value = 2.5
            mock_binary.calculate_bolometric_luminosity.return_value = {
                "primary": 1.0,
                "secondary": 0.5,
            }

            # Mock star instances
            mock_primary = MagicMock()
            mock_primary.mass = 1.0
            mock_primary.equivalent_radius = 0.5
            mock_primary.polar_radius = 0.5
            mock_primary.backward_radius = 0.5
            mock_primary.side_radius = 0.5
            mock_primary.forward_radius = 0.5
            mock_primary.synchronicity = 1.0
            mock_primary.gravity_darkening = 0.32
            mock_primary.albedo = 1.0
            mock_primary.metallicity = 0.0
            mock_primary.has_spots.return_value = False
            mock_primary.has_pulsations.return_value = False

            mock_secondary = MagicMock()
            mock_secondary.mass = 0.5
            mock_secondary.equivalent_radius = 0.35
            mock_secondary.polar_radius = 0.35
            mock_secondary.backward_radius = 0.35
            mock_secondary.side_radius = 0.35
            mock_secondary.forward_radius = 0.35
            mock_secondary.synchronicity = 1.0
            mock_secondary.gravity_darkening = 0.32
            mock_secondary.albedo = 1.0
            mock_secondary.metallicity = 0.0
            mock_secondary.has_spots.return_value = False
            mock_secondary.has_pulsations.return_value = False

            mock_binary.primary = mock_primary
            mock_binary.secondary = mock_secondary
            mock_prepare.return_value = mock_binary

            with patch("builtins.print"):
                # Should not raise exception
                simple_lc_fit_summary(
                    mock_lc_fit_instance, path=None, dimensionless_radii=True
                )


class TestSimpleRVFitSummary:
    """Test suite for simple_rv_fit_summary function."""

    @pytest.fixture
    def mock_rv_fit_instance(self) -> Mock:
        """Create a mock RV fit instance."""
        mock_fit = Mock()
        mock_fit.flat_result = {
            "system@mass_ratio": {"value": 0.5, "unit": "dimensionless"},
            "system@asini": {"value": 15.0, "unit": "solRad"},
            "system@eccentricity": {"value": 0.0, "unit": "dimensionless"},
            "system@argument_of_periastron": {"value": 0.0, "unit": "deg"},
            "system@gamma": {"value": 0.0, "unit": "km/s"},
            "system@period": {"value": 3.0, "unit": "d"},
            "r_squared": {"value": 0.92, "unit": "dimensionless"},
        }
        return mock_fit

    def test_rv_summary_to_stdout(self, mock_rv_fit_instance: Mock) -> None:
        """Test generating RV summary to stdout."""
        with patch("builtins.print"):
            # Should not raise exception
            simple_rv_fit_summary(mock_rv_fit_instance, path=None)

    def test_rv_summary_to_file(
        self, mock_rv_fit_instance: Mock, tmp_path: Path
    ) -> None:
        """Test generating RV summary to file."""
        output_file = tmp_path / "rv_summary.txt"

        simple_rv_fit_summary(mock_rv_fit_instance, path=str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "Mass ratio" in content or "eccentricity" in content.lower()

    def test_rv_summary_with_mass_parameters(self, mock_rv_fit_instance: Mock) -> None:
        """Test RV summary when mass parameters are included."""
        mock_rv_fit_instance.flat_result["primary@mass"] = {
            "value": 1.8,
            "unit": "solMass",
        }
        mock_rv_fit_instance.flat_result["secondary@mass"] = {
            "value": 0.9,
            "unit": "solMass",
        }
        mock_rv_fit_instance.flat_result["system@inclination"] = {
            "value": 85.0,
            "unit": "deg",
        }

        del mock_rv_fit_instance.flat_result["system@mass_ratio"]
        del mock_rv_fit_instance.flat_result["system@asini"]

        with patch("builtins.print"):
            simple_rv_fit_summary(mock_rv_fit_instance, path=None)




# Note: Tests for evaluate_binary_params, fit_lc_summary_with_error_propagation, and
# fit_rv_summary_with_error_propagation integration scenarios are complex and better
# tested as integration tests with real or fully-mocked system parameters.
# The unit tests above cover the main public APIs and data processing pipelines.

