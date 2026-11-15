from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from elisa.analytics.binary_fit import io_tools


class TestWriteLn:
    """Test suite for write_ln function."""

    def test_write_ln_with_float_value(self) -> None:
        """Test write_ln formats float values correctly."""
        output_lines = []
        write_fn = output_lines.append

        io_tools.write_ln(
            write_fn,
            "Temperature",
            5500.123456789,
            100.0,
            -50.0,
            "K",
            "Variable",
            "|",
            precision=2,
        )

        assert len(output_lines) == 1
        output = output_lines[0]
        assert "Temperature" in output
        assert "5500.12" in output
        assert "100" in output
        assert "-50" in output
        assert "K" in output
        assert "Variable" in output
        assert "|" in output

    def test_write_ln_with_string_value(self) -> None:
        """Test write_ln handles string values."""
        output_lines = []
        write_fn = output_lines.append

        io_tools.write_ln(
            write_fn,
            "Parameter",
            "N/A",
            "-",
            "-",
            "-",
            "Derived",
            "|",
        )

        assert len(output_lines) == 1
        output = output_lines[0]
        assert "N/A" in output
        assert "-" in output

    def test_write_ln_formatting_width(self) -> None:
        """Test write_ln maintains proper column widths."""
        output_lines = []
        write_fn = output_lines.append

        io_tools.write_ln(
            write_fn,
            "Test",
            1.0,
            2.0,
            3.0,
            "unit",
            "Status",
            "---",
        )

        output = output_lines[0]
        # Check that Test is left-aligned with 35 char width
        assert output.startswith("Test")

    def test_write_ln_calls_write_function(self) -> None:
        """Test write_ln calls the provided write function."""
        write_fn = Mock()
        io_tools.write_ln(
            write_fn,
            "Param",
            1.0,
            2.0,
            3.0,
            "unit",
            "status",
            "|",
        )
        write_fn.assert_called_once()


class TestWriteParamLn:
    """Test suite for write_param_ln function."""

    @pytest.fixture
    def fit_params_with_confidence(self) -> dict:
        """Fixture for fit parameters with confidence intervals."""
        return {
            "temperature": {
                "value": 5500.0,
                "fixed": False,
                "unit": "K",
                "confidence_interval": {"min": 5450.0, "max": 5550.0},
            },
        }

    @pytest.fixture
    def fit_params_fixed(self) -> dict:
        """Fixture for fixed fit parameters."""
        return {
            "radius": {
                "value": 1.0,
                "fixed": True,
                "unit": "R_sun",
            },
        }

    @pytest.fixture
    def fit_params_constrained(self) -> dict:
        """Fixture for constrained fit parameters."""
        return {
            "mass": {
                "value": 1.5,
                "constraint": "mass_ratio * m_secondary",
                "unit": "M_sun",
            },
        }

    def test_write_param_ln_with_confidence_interval(self, fit_params_with_confidence: dict) -> None:
        """Test write_param_ln with confidence interval."""
        output_lines = []
        write_fn = output_lines.append

        with patch("elisa.analytics.binary_fit.io_tools.write_ln"):
            io_tools.write_param_ln(
                fit_params_with_confidence,
                "temperature",
                "Temperature",
                write_fn,
                "|",
            )

    def test_write_param_ln_fixed_parameter(self, fit_params_fixed: dict) -> None:
        """Test write_param_ln correctly identifies fixed parameters."""
        output_lines = []
        write_fn = output_lines.append

        with patch("elisa.analytics.binary_fit.io_tools.write_ln") as mock_write:
            io_tools.write_param_ln(
                fit_params_fixed,
                "radius",
                "Radius",
                write_fn,
                "|",
            )
            # Check that write_ln was called with "Fixed" status
            call_args = mock_write.call_args[0]
            assert "Fixed" in call_args

    def test_write_param_ln_variable_parameter(self, fit_params_with_confidence: dict) -> None:
        """Test write_param_ln identifies variable parameters."""
        output_lines = []
        write_fn = output_lines.append

        with patch("elisa.analytics.binary_fit.io_tools.write_ln") as mock_write:
            io_tools.write_param_ln(
                fit_params_with_confidence,
                "temperature",
                "Temperature",
                write_fn,
                "|",
            )
            call_args = mock_write.call_args[0]
            assert "Variable" in call_args

    def test_write_param_ln_constrained_parameter(self, fit_params_constrained: dict) -> None:
        """Test write_param_ln handles constrained parameters."""
        output_lines = []
        write_fn = output_lines.append

        with patch("elisa.analytics.binary_fit.io_tools.write_ln") as mock_write:
            io_tools.write_param_ln(
                fit_params_constrained,
                "mass",
                "Mass",
                write_fn,
                "|",
            )
            call_args = mock_write.call_args[0]
            assert "mass_ratio * m_secondary" in call_args

    def test_write_param_ln_without_unit(self) -> None:
        """Test write_param_ln handles parameters without unit."""
        fit_params = {
            "r_squared": {
                "value": 0.95,
            },
        }
        output_lines = []
        write_fn = output_lines.append

        with patch("elisa.analytics.binary_fit.io_tools.write_ln") as mock_write:
            io_tools.write_param_ln(
                fit_params,
                "r_squared",
                "R-squared",
                write_fn,
                "|",
            )
            call_args = mock_write.call_args[0]
            assert "-" in call_args  # Unit should be "-"

    def test_write_param_ln_r_squared_is_derived(self) -> None:
        """Test write_param_ln marks r_squared as Derived."""
        fit_params = {
            "r_squared": {"value": 0.95},
        }

        with patch("elisa.analytics.binary_fit.io_tools.write_ln") as mock_write:
            io_tools.write_param_ln(
                fit_params,
                "r_squared",
                "R-squared",
                Mock(),
                "|",
            )
            call_args = mock_write.call_args[0]
            assert "Derived" in call_args


class TestWritePropagatedLn:
    """Test suite for write_propagated_ln function."""

    def test_write_propagated_ln_valid_values(self) -> None:
        """Test write_propagated_ln with valid values."""
        values = np.array([5500.0, 50.0, 50.0])
        fit_params = {
            "temperature": {
                "value": 5500.0,
                "fixed": False,
                "unit": "K",
            },
        }

        with patch("elisa.analytics.binary_fit.io_tools.write_ln") as write_line:
            result = io_tools.write_propagated_ln(
                values,
                fit_params,
                "temperature",
                "Temperature",
                Mock(),
                "|",
                "K",
            )

        assert result is None
        write_line.assert_called_once()
        assert write_line.call_args.args[1:6] == ("Temperature", 5500.0, 50.0, 50.0, "K")

    def test_write_propagated_ln_with_nan_returns_none(self) -> None:
        """Test write_propagated_ln returns None for NaN values."""
        values = np.array([5500.0, np.nan, 50.0])
        fit_params = {}

        result = io_tools.write_propagated_ln(
            values,
            fit_params,
            "temp",
            "Temperature",
            Mock(),
            "|",
            "K",
        )

        assert result is None

    def test_write_propagated_ln_derived_parameter(self) -> None:
        """Test write_propagated_ln marks derived parameters."""
        values = np.array([1.5, 0.1, 0.1])
        fit_params = {}  # Parameter not in fit_params

        with patch("elisa.analytics.binary_fit.io_tools.write_ln") as mock_write:
            io_tools.write_propagated_ln(
                values,
                fit_params,
                "derived_param",
                "Derived Param",
                Mock(),
                "|",
                "-",
            )
            call_args = mock_write.call_args[0]
            assert "Derived" in call_args

    def test_write_propagated_ln_fixed_parameter(self) -> None:
        """Test write_propagated_ln with fixed parameter."""
        values = np.array([1.0, 0.0, 0.0])
        fit_params = {
            "radius": {
                "value": 1.0,
                "fixed": True,
            },
        }

        with patch("elisa.analytics.binary_fit.io_tools.write_ln") as mock_write:
            io_tools.write_propagated_ln(
                values,
                fit_params,
                "radius",
                "Radius",
                Mock(),
                "|",
                "R_sun",
            )
            call_args = mock_write.call_args[0]
            assert "Fixed" in call_args

    def test_write_propagated_ln_all_nan_returns_none(self) -> None:
        """Test write_propagated_ln returns None when all values are NaN."""
        values = np.array([np.nan, np.nan, np.nan])
        fit_params = {}

        result = io_tools.write_propagated_ln(
            values,
            fit_params,
            "param",
            "Parameter",
            Mock(),
            "|",
            "-",
        )

        assert result is None

    def test_write_propagated_ln_very_small_errors(self) -> None:
        """Test write_propagated_ln handles very small error values."""
        values = np.array([5500.0, 1e-16, 1e-16])
        fit_params = {"temperature": {"fixed": False}}

        with patch("elisa.analytics.binary_fit.io_tools.write_ln") as mock_write:
            io_tools.write_propagated_ln(
                values,
                fit_params,
                "temperature",
                "Temperature",
                Mock(),
                "|",
                "K",
            )
            assert mock_write.called

    def test_write_propagated_ln_constrained_parameter(self) -> None:
        """Test write_propagated_ln with constrained parameter."""
        values = np.array([1.5, 0.1, 0.1])
        fit_params = {
            "mass": {
                "constraint": "q * m2",
            },
        }

        with patch("elisa.analytics.binary_fit.io_tools.write_ln") as mock_write:
            io_tools.write_propagated_ln(
                values,
                fit_params,
                "mass",
                "Mass",
                Mock(),
                "|",
                "M_sun",
            )
            call_args = mock_write.call_args[0]
            assert "q * m2" in call_args
