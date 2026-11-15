from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from elisa.analytics.binary_fit import io_tools
from elisa.base.error import ElisaMonteCarloError


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

    @patch("elisa.analytics.binary_fit.io_tools._update_solution")
    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.load_flat_chain")
    def test_load_chain_basic(
        self,
        mock_load_flat: Mock,
        mock_update: Mock,
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
        mock_update.assert_called_once()

    @patch("elisa.analytics.binary_fit.io_tools._update_solution")
    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.load_flat_chain")
    def test_load_chain_with_discard(
        self,
        mock_load_flat: Mock,
        mock_update: Mock,
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
        mock_update.assert_called_once()

    @patch("elisa.analytics.binary_fit.io_tools._update_solution")
    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.load_flat_chain")
    def test_load_chain_with_percentiles(
        self,
        mock_load_flat: Mock,
        mock_update: Mock,
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
        io_tools.load_chain(
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
        mock_load_flat: Mock,
        mock_mcmc_fit_cls: Mock,
    ) -> None:
        """Test load_chain correctly sets instance attributes."""
        with patch("elisa.analytics.binary_fit.io_tools._update_solution"):
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

    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.resolve_constrained_result")
    @patch("elisa.analytics.binary_fit.io_tools._result_initial_parameters")
    @patch("elisa.analytics.binary_fit.io_tools.parameters.serialize_result")
    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.resolve_mcmc_result")
    def test_update_solution_with_result(
        self,
        mock_resolve: Mock,
        mock_serialize: Mock,
        mock_initial_parameters: Mock,
        mock_eval: Mock,
        mock_mcmc_fit_cls: Mock,
    ) -> None:
        """Test update_solution when result is not None."""
        mock_resolve.return_value = {"param1": {"value": 0.15}, "param2": {"value": 0.25}}
        mock_serialize.side_effect = lambda x: x
        mock_constrained = Mock()
        constrained = {"param2": Mock(constraint="param1 * 2")}
        fixed = {"param3": Mock(value=0.5)}
        mock_constrained.get_constrained.return_value = constrained
        mock_constrained.get_fixed.return_value = fixed
        mock_initial_parameters.return_value = mock_constrained
        mock_eval.return_value = {"param1": {"value": 0.15}, "param2": {"value": 0.25}}

        io_tools._update_solution(
            mock_mcmc_fit_cls,
            {"param1": {"value": 0.1}, "param2": {"value": 0.2}},
            percentiles=[16, 50, 84],
        )

        assert mock_resolve.called
        mock_initial_parameters.assert_called_once_with(mock_resolve.return_value)
        fitable = mock_resolve.call_args.args[1]
        assert {key: meta.value for key, meta in fitable.items()} == {"param1": 0.1, "param2": 0.2}
        assert mock_serialize.call_count == 2
        mock_eval.assert_called_once_with(
            mock_mcmc_fit_cls.flat_chain,
            mock_resolve.call_args.args[1],
            mock_mcmc_fit_cls.normalization,
            constrained,
            fixed,
            [16, 50, 84],
        )
        assert mock_mcmc_fit_cls.flat_result == mock_eval.return_value
        assert mock_mcmc_fit_cls.result == mock_eval.return_value

    def test_update_solution_no_result_raises_error(self, mock_mcmc_fit_cls: Mock) -> None:
        """Test update_solution raises ValueError when result is None."""
        mock_mcmc_fit_cls.result = None

        with (
            pytest.raises(ElisaMonteCarloError, match="Load fit parameters before loading the chain"),
            patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.resolve_mcmc_result"),
        ):
            io_tools._update_solution(
                mock_mcmc_fit_cls,
                {"param1": {"value": 0.1}},
                percentiles=None,
            )

    @patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.resolve_mcmc_result")
    def test_update_solution_calls_resolve_with_percentiles(
        self,
        mock_resolve: Mock,
        mock_mcmc_fit_cls: Mock,
    ) -> None:
        """Test update_solution passes percentiles to resolve_mcmc_result."""
        mock_resolve.return_value = {}

        with (
            patch("elisa.analytics.binary_fit.io_tools._result_initial_parameters") as mock_initial_parameters,
            patch("elisa.analytics.binary_fit.io_tools.MCMCMixin.resolve_constrained_result") as mock_eval,
            patch("elisa.analytics.binary_fit.io_tools.parameters.serialize_result") as mock_ser,
        ):
            mock_ser.side_effect = lambda x: x

            percentiles = [16, 50, 84]
            io_tools._update_solution(
                mock_mcmc_fit_cls,
                {"param1": {"value": 0.1}},
                percentiles=percentiles,
            )

            # Check resolve was called with correct percentiles
            call_kwargs = mock_resolve.call_args[1]
            assert call_kwargs["percentiles"] == percentiles
            mock_initial_parameters.assert_called_once_with(mock_mcmc_fit_cls.flat_result)
            assert mock_eval.call_args.args[-1] == percentiles
