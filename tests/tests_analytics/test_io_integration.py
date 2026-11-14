from __future__ import annotations

import io
from unittest.mock import Mock, patch

import numpy as np
import pytest

from elisa.analytics.binary_fit import io_tools


class TestIntegration:
    """Integration tests for multiple functions working together."""

    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.load_flat_chain")
    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.resolve_mcmc_result")
    @patch("elisa.analytics.binary_fit.io_tools._result_initial_parameters")
    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.resolve_constrained_result")
    @patch("elisa.analytics.binary_fit.io_tools.parameters.serialize_result")
    def test_load_chain_and_update_solution(
        self,
        mock_serialize: Mock,
        mock_eval: Mock,
        mock_initial_parameters: Mock,
        mock_resolve: Mock,
        mock_load: Mock,
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
        constrained = {"param2": Mock(constraint="param1 * 2")}
        fixed = {"param3": Mock(value=0.5)}
        mock_constrained.get_constrained.return_value = constrained
        mock_constrained.get_fixed.return_value = fixed
        mock_initial_parameters.return_value = mock_constrained
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
        np.testing.assert_array_equal(result[0], chain_data)
        assert result[2] == mock_load.return_value["normalization"]
        mock_initial_parameters.assert_called_once_with(mock_resolve.return_value)
        fitable = mock_resolve.call_args.args[1]
        assert {key: meta.value for key, meta in fitable.items()} == {"param1": 0.1, "param2": 0.2}
        assert mock_serialize.call_count == 2
        mock_eval.assert_called_once_with(
            mock_fit_cls.flat_chain,
            mock_resolve.call_args.args[1],
            mock_fit_cls.normalization,
            constrained,
            fixed,
            [16, 50, 84],
        )
        assert mock_resolve.call_args.kwargs["percentiles"] == [16, 50, 84]
        assert mock_fit_cls.flat_result == mock_eval.return_value
        assert mock_fit_cls.result == mock_eval.return_value

    def test_write_functions_output_integration(self) -> None:
        """Verify parameter summaries delegate both fitted and fixed rows to the writer."""
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

        with patch("elisa.analytics.binary_fit.io_tools.write_ln") as write_line:
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

        # Formatting is mocked; verify the two distinct parameter rows are forwarded.
        assert write_line.call_count == 2
        temperature_row, radius_row = write_line.call_args_list
        assert temperature_row.args == (write_fn, "Temperature", 5500.0, 50.0, 50.0, "K", "Variable", "|")
        assert radius_row.args == (write_fn, "Radius", 1.5, "-", "-", "R_sun", "Fixed", "|")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
