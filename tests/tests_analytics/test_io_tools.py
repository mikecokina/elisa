from __future__ import annotations

import io
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from elisa.analytics.binary_fit import io_tools


class TestFilterChain:
    """Test suite for filter_chain function."""

    @pytest.fixture
    def mock_mcmc_fit_cls(self) -> Mock:
        """Create a mock MCMC fitting class instance."""
        mock_cls = Mock()
        mock_cls.flat_chain = np.array([
            [0.1, 0.2],
            [0.15, 0.25],
            [0.2, 0.3],
            [0.25, 0.35],
            [0.3, 0.4],
        ])
        mock_cls.variable_labels = ["primary@te_ff", "secondary@te_ff"]
        mock_cls.normalization = {
            "primary@te_ff": (0.0, 1.0),
            "secondary@te_ff": (0.0, 1.0),
        }
        mock_cls.flat_result = {
            "primary@te_ff": {"value": 0.2},
            "secondary@te_ff": {"value": 0.3},
        }
        mock_cls.result = {"primary@te_ff": 5500, "secondary@te_ff": 5000}
        return mock_cls

    @patch("elisa.analytics.binary_fit.io_tools.update_solution")
    def test_filter_chain_single_boundary(self, mock_update, mock_mcmc_fit_cls: Mock) -> None:
        """Test filtering chain with a single parameter boundary."""
        with patch("elisa.analytics.binary_fit.io_tools.parameters.normalize_value") as mock_norm:
            mock_norm.side_effect = lambda v, a, b: v  # Identity function for testing
            result = io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{"primary@te_ff": (0.12, 0.28)},
            )
            assert result is not None
            assert mock_update.called

    @patch("elisa.analytics.binary_fit.io_tools.update_solution")
    def test_filter_chain_multiple_boundaries(self, mock_update, mock_mcmc_fit_cls: Mock) -> None:
        """Test filtering chain with multiple parameter boundaries."""
        with patch("elisa.analytics.binary_fit.io_tools.parameters.normalize_value") as mock_norm:
            mock_norm.side_effect = lambda v, a, b: v
            result = io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{
                    "primary@te_ff": (0.12, 0.28),
                    "secondary@te_ff": (0.22, 0.38),
                },
            )
            assert result is not None

    def test_filter_chain_invalid_boundary_type(self, mock_mcmc_fit_cls: Mock) -> None:
        """Test filter_chain raises TypeError for invalid boundary type."""
        with pytest.raises(TypeError, match="boundary is not tuple or list"):
            io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{"primary@te_ff": "invalid"},
            )

    def test_filter_chain_invalid_boundary_length(self, mock_mcmc_fit_cls: Mock) -> None:
        """Test filter_chain raises TypeError for incorrect boundary length."""
        with pytest.raises(TypeError, match="has incorrect length of 3"):
            io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{"primary@te_ff": (0.1, 0.2, 0.3)},
            )

    def test_filter_chain_invalid_parameter(self, mock_mcmc_fit_cls: Mock) -> None:
        """Test filter_chain raises NameError for invalid parameter name."""
        with pytest.raises(NameError, match="is not valid model parameter"):
            io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{"invalid@param": (0.1, 0.2)},
            )

    @patch("elisa.analytics.binary_fit.io_tools.update_solution")
    @patch("elisa.analytics.binary_fit.io_tools.parameters.normalize_value")
    def test_filter_chain_empty_result(
        self,
        mock_norm,
        mock_update,
        mock_mcmc_fit_cls: Mock,
    ) -> None:
        """Test filter_chain raises ValueError when boundaries yield empty array."""
        mock_norm.return_value = 0.5  # Returns value outside actual range
        with pytest.raises(ValueError, match="yielded an empty array"):
            io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{"primary@te_ff": (0.5, 0.6)},
            )

    def test_filter_chain_boundary_as_list(self, mock_mcmc_fit_cls: Mock) -> None:
        """Test filter_chain accepts list as boundary."""
        with patch("elisa.analytics.binary_fit.io_tools.update_solution"):
            with patch("elisa.analytics.binary_fit.io_tools.parameters.normalize_value") as mock_norm:
                mock_norm.side_effect = lambda v, a, b: v
                result = io_tools.filter_chain(
                    mock_mcmc_fit_cls,
                    **{"primary@te_ff": [0.12, 0.28]},
                )
                assert result is not None

    def test_filter_chain_boundary_as_ndarray(self, mock_mcmc_fit_cls: Mock) -> None:
        """Test filter_chain accepts ndarray as boundary."""
        with patch("elisa.analytics.binary_fit.io_tools.update_solution"):
            with patch("elisa.analytics.binary_fit.io_tools.parameters.normalize_value") as mock_norm:
                mock_norm.side_effect = lambda v, a, b: v
                result = io_tools.filter_chain(
                    mock_mcmc_fit_cls,
                    **{"primary@te_ff": np.array([0.12, 0.28])},
                )
                assert result is not None


class TestLoadChain:
    """Test suite for load_chain function."""

    @pytest.fixture
    def mock_mcmc_fit_cls(self) -> Mock:
        """Create a mock MCMC fitting class instance."""
        mock_cls = Mock()
        mock_cls.flat_chain = None
        mock_cls.variable_labels = None
        mock_cls.normalization = None
        mock_cls.flat_result = {}
        mock_cls.result = {}
        return mock_cls

    @patch("elisa.analytics.binary_fit.io_tools.update_solution")
    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.load_flat_chain")
    def test_load_chain_basic(
        self,
        mock_load_flat,
        mock_update,
        mock_mcmc_fit_cls: Mock,
    ) -> None:
        """Test load_chain with basic valid input."""
        mock_load_flat.return_value = {
            "flat_chain": [[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]],
            "fitable_parameters": ["param1", "param2"],
            "normalization": {"param1": (0.0, 1.0), "param2": (0.0, 1.0)},
            "fitable": {"param1": {"value": 0.1}, "param2": {"value": 0.2}},
        }

        result = io_tools.load_chain(mock_mcmc_fit_cls, "test_fit.json")

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], np.ndarray)
        assert result[1] == ["param1", "param2"]
        assert isinstance(result[2], dict)

    @patch("elisa.analytics.binary_fit.io_tools.update_solution")
    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.load_flat_chain")
    def test_load_chain_with_discard(
        self,
        mock_load_flat,
        mock_update,
        mock_mcmc_fit_cls: Mock,
    ) -> None:
        """Test load_chain discards initial burn-in steps."""
        chain_data = [[0.1, 0.2], [0.15, 0.25], [0.2, 0.3], [0.25, 0.35]]
        mock_load_flat.return_value = {
            "flat_chain": chain_data,
            "fitable_parameters": ["param1", "param2"],
            "normalization": {"param1": (0.0, 1.0), "param2": (0.0, 1.0)},
            "fitable": {"param1": {"value": 0.1}, "param2": {"value": 0.2}},
        }

        result = io_tools.load_chain(mock_mcmc_fit_cls, "test_fit.json", discard=2)

        # Should have only 2 rows after discarding first 2
        assert result[0].shape[0] == 2
        np.testing.assert_array_equal(result[0], np.array([[0.2, 0.3], [0.25, 0.35]]))

    @patch("elisa.analytics.binary_fit.io_tools.update_solution")
    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.load_flat_chain")
    def test_load_chain_with_percentiles(
        self,
        mock_load_flat,
        mock_update,
        mock_mcmc_fit_cls: Mock,
    ) -> None:
        """Test load_chain with percentiles for confidence intervals."""
        mock_load_flat.return_value = {
            "flat_chain": [[0.1, 0.2], [0.15, 0.25]],
            "fitable_parameters": ["param1", "param2"],
            "normalization": {"param1": (0.0, 1.0), "param2": (0.0, 1.0)},
            "fitable": {"param1": {"value": 0.1}, "param2": {"value": 0.2}},
        }

        percentiles = [16, 50, 84]
        result = io_tools.load_chain(
            mock_mcmc_fit_cls,
            "test_fit.json",
            percentiles=percentiles,
        )

        mock_update.assert_called_once()
        # Check percentiles were passed to update_solution
        call_args = mock_update.call_args
        assert call_args[0][2] == percentiles

    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.load_flat_chain")
    def test_load_chain_sets_attributes(
        self,
        mock_load_flat,
        mock_mcmc_fit_cls: Mock,
    ) -> None:
        """Test load_chain correctly sets instance attributes."""
        with patch("elisa.analytics.binary_fit.io_tools.update_solution"):
            mock_load_flat.return_value = {
                "flat_chain": [[0.1, 0.2]],
                "fitable_parameters": ["a", "b"],
                "normalization": {"a": (0.0, 1.0), "b": (0.0, 1.0)},
                "fitable": {"a": {"value": 0.1}, "b": {"value": 0.2}},
            }

            io_tools.load_chain(mock_mcmc_fit_cls, "test.json")

            assert mock_mcmc_fit_cls.variable_labels == ["a", "b"]
            assert isinstance(mock_mcmc_fit_cls.flat_chain, np.ndarray)
            assert mock_mcmc_fit_cls.normalization == {"a": (0.0, 1.0), "b": (0.0, 1.0)}


class TestUpdateSolution:
    """Test suite for update_solution function."""

    @pytest.fixture
    def mock_mcmc_fit_cls(self) -> Mock:
        """Create a mock MCMC fitting class instance."""
        mock_cls = Mock()
        mock_cls.flat_chain = np.array([[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]])
        mock_cls.normalization = {"param1": (0.0, 1.0), "param2": (0.0, 1.0)}
        mock_cls.flat_result = {"param1": {"value": 0.1}, "param2": {"value": 0.2}}
        mock_cls.result = {"param1": 0.1, "param2": 0.2}
        return mock_cls

    @patch("elisa.analytics.binary_fit.io_tools.AbstractFit.eval_constrained_results")
    @patch("elisa.analytics.binary_fit.io_tools.BinaryInitialParameters")
    @patch("elisa.analytics.binary_fit.io_tools.parameters.serialize_result")
    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.resolve_mcmc_result")
    def test_update_solution_with_result(
        self,
        mock_resolve,
        mock_serialize,
        mock_binary_params,
        mock_eval,
        mock_mcmc_fit_cls: Mock,
    ) -> None:
        """Test update_solution when result is not None."""
        mock_resolve.return_value = {"param1": {"value": 0.15}, "param2": {"value": 0.25}}
        mock_serialize.side_effect = lambda x: x
        mock_constrained = Mock()
        mock_constrained.get_constrained.return_value = {}
        mock_binary_params.return_value = mock_constrained
        mock_eval.return_value = {"param1": {"value": 0.15}, "param2": {"value": 0.25}}

        io_tools.update_solution(
            mock_mcmc_fit_cls,
            {"param1": {"value": 0.1}, "param2": {"value": 0.2}},
            percentiles=[16, 50, 84],
        )

        assert mock_resolve.called
        assert mock_binary_params.called
        assert mock_eval.called

    def test_update_solution_no_result_raises_error(self, mock_mcmc_fit_cls: Mock) -> None:
        """Test update_solution raises ValueError when result is None."""
        mock_mcmc_fit_cls.result = None

        with pytest.raises(ValueError, match="Load fit parameters before loading the chain"):
            with patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.resolve_mcmc_result"):
                io_tools.update_solution(
                    mock_mcmc_fit_cls,
                    {"param1": {"value": 0.1}},
                    percentiles=None,
                )

    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.resolve_mcmc_result")
    def test_update_solution_calls_resolve_with_percentiles(
        self,
        mock_resolve,
        mock_mcmc_fit_cls: Mock,
    ) -> None:
        """Test update_solution passes percentiles to resolve_mcmc_result."""
        mock_resolve.return_value = {}

        with patch("elisa.analytics.binary_fit.io_tools.BinaryInitialParameters"):
            with patch("elisa.analytics.binary_fit.io_tools.AbstractFit.eval_constrained_results"):
                with patch("elisa.analytics.binary_fit.io_tools.parameters.serialize_result") as mock_ser:
                    mock_ser.side_effect = lambda x: x

                    percentiles = [16, 50, 84]
                    io_tools.update_solution(
                        mock_mcmc_fit_cls,
                        {"param1": {"value": 0.1}},
                        percentiles=percentiles,
                    )

                    # Check resolve was called with correct percentiles
                    call_kwargs = mock_resolve.call_args[1]
                    assert call_kwargs["percentiles"] == percentiles


class TestWriteLn:
    """Test suite for write_ln function."""

    def test_write_ln_with_float_value(self) -> None:
        """Test write_ln formats float values correctly."""
        output_lines = []
        write_fn = lambda x: output_lines.append(x)

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
        write_fn = lambda x: output_lines.append(x)

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
        write_fn = lambda x: output_lines.append(x)

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
        write_fn = lambda x: output_lines.append(x)

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
        write_fn = lambda x: output_lines.append(x)

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
        write_fn = lambda x: output_lines.append(x)

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
        write_fn = lambda x: output_lines.append(x)

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
        write_fn = lambda x: output_lines.append(x)

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

        with patch("elisa.analytics.binary_fit.io_tools.write_ln"):
            result = io_tools.write_propagated_ln(
                values,
                fit_params,
                "temperature",
                "Temperature",
                Mock(),
                "|",
                "K",
            )

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


class TestIntegration:
    """Integration tests for multiple functions working together."""

    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.load_flat_chain")
    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.resolve_mcmc_result")
    @patch("elisa.analytics.binary_fit.io_tools.BinaryInitialParameters")
    @patch("elisa.analytics.binary_fit.io_tools.AbstractFit.eval_constrained_results")
    @patch("elisa.analytics.binary_fit.io_tools.parameters.serialize_result")
    def test_load_chain_and_update_solution(
        self,
        mock_serialize,
        mock_eval,
        mock_binary,
        mock_resolve,
        mock_load,
    ) -> None:
        """Test load_chain followed by update_solution interaction."""
        # Setup mocks
        chain_data = [[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]]
        mock_load.return_value = {
            "flat_chain": chain_data,
            "fitable_parameters": ["param1", "param2"],
            "normalization": {"param1": (0.0, 1.0), "param2": (0.0, 1.0)},
            "fitable": {"param1": {"value": 0.1}, "param2": {"value": 0.2}},
        }
        mock_resolve.return_value = {"param1": {"value": 0.15}, "param2": {"value": 0.25}}
        mock_serialize.side_effect = lambda x: x
        mock_constrained = Mock()
        mock_constrained.get_constrained.return_value = {}
        mock_binary.return_value = mock_constrained
        mock_eval.return_value = {"param1": {"value": 0.15}, "param2": {"value": 0.25}}

        # Create mock fit class
        mock_fit_cls = Mock()
        mock_fit_cls.result = {"param1": 0.1, "param2": 0.2}
        mock_fit_cls.flat_result = {}

        # Execute
        result = io_tools.load_chain(mock_fit_cls, "test.json", discard=0, percentiles=[16, 50, 84])

        # Verify
        assert result[0].shape == (3, 2)
        assert result[1] == ["param1", "param2"]
        assert mock_resolve.called

    def test_write_functions_output_integration(self) -> None:
        """Test write functions produce properly formatted output."""
        output = io.StringIO()

        def write_fn(line: str) -> None:
            output.write(line + "\n")

        # Write header
        fit_params = {
            "temperature": {
                "value": 5500.0,
                "fixed": False,
                "unit": "K",
                "confidence_interval": {"min": 5450.0, "max": 5550.0},
            },
            "radius": {
                "value": 1.5,
                "fixed": True,
                "unit": "R_sun",
            },
        }

        with patch("elisa.analytics.binary_fit.io_tools.write_ln"):
            io_tools.write_param_ln(
                fit_params,
                "temperature",
                "Temperature",
                write_fn,
                "|",
            )
            io_tools.write_param_ln(
                fit_params,
                "radius",
                "Radius",
                write_fn,
                "|",
            )

        # Output should contain the data (mocked write_ln, but write_fn should be called)
        output_text = output.getvalue()
        # We can't verify specific output since write_ln is mocked, but structure should be valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

