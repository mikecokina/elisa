from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from elisa.analytics.binary_fit import io_tools
from elisa.base.error import ElisaMonteCarloError


class TestFilterChain:
    """Test suite for filter_chain function."""

    @pytest.fixture
    def mock_mcmc_fit_cls(self) -> Mock:
        """Create a mock MCMC fitting class instance."""
        mock_cls = Mock()
        mock_cls.flat_chain = np.array(
            [
                [0.1, 0.2],
                [0.15, 0.25],
                [0.2, 0.3],
                [0.25, 0.35],
                [0.3, 0.4],
            ],
        )
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

    @patch("elisa.analytics.binary_fit.io_tools._update_solution")
    def test_filter_chain_single_boundary(self, mock_update: Mock, mock_mcmc_fit_cls: Mock) -> None:
        """Test filtering chain with a single parameter boundary."""
        with patch("elisa.analytics.binary_fit.io_tools.parameters.normalize_value") as mock_norm:
            mock_norm.side_effect = lambda v, _a, _b: v  # Identity function for testing
            result = io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{"primary@te_ff": (0.12, 0.28)},
            )
            assert result is not None
            assert mock_update.called

    @patch("elisa.analytics.binary_fit.io_tools._update_solution")
    def test_filter_chain_multiple_boundaries(self, mock_update: Mock, mock_mcmc_fit_cls: Mock) -> None:
        """Test filtering chain with multiple parameter boundaries."""
        with patch("elisa.analytics.binary_fit.io_tools.parameters.normalize_value") as mock_norm:
            mock_norm.side_effect = lambda v, _a, _b: v
            result = io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{
                    "primary@te_ff": (0.12, 0.28),
                    "secondary@te_ff": (0.22, 0.38),
                },
            )
            assert result is not None
        mock_update.assert_called_once()

    def test_filter_chain_invalid_boundary_type(self, mock_mcmc_fit_cls: Mock) -> None:
        """Test filter_chain raises TypeError for invalid boundary type."""
        with pytest.raises(ElisaMonteCarloError, match="boundary is not tuple or list"):
            io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{"primary@te_ff": "invalid"},
            )

    def test_filter_chain_invalid_boundary_length(self, mock_mcmc_fit_cls: Mock) -> None:
        """Test filter_chain raises TypeError for incorrect boundary length."""
        with pytest.raises(ElisaMonteCarloError, match="has incorrect length of 3"):
            io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{"primary@te_ff": (0.1, 0.2, 0.3)},
            )

    def test_filter_chain_invalid_parameter(self, mock_mcmc_fit_cls: Mock) -> None:
        """Test filter_chain raises NameError for invalid parameter name."""
        with pytest.raises(ElisaMonteCarloError, match="is not valid model parameter"):
            io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{"invalid@param": (0.1, 0.2)},
            )

    @patch("elisa.analytics.binary_fit.io_tools._update_solution")
    @patch("elisa.analytics.binary_fit.io_tools.parameters.normalize_value")
    def test_filter_chain_empty_result(
        self,
        mock_norm: Mock,
        mock_update: Mock,
        mock_mcmc_fit_cls: Mock,
    ) -> None:
        """Test filter_chain raises ValueError when boundaries yield empty array."""
        mock_norm.return_value = 0.5  # Returns value outside actual range
        with pytest.raises(ElisaMonteCarloError, match="yielded an empty array"):
            io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{"primary@te_ff": (0.5, 0.6)},
            )
        mock_update.assert_not_called()

    def test_filter_chain_boundary_as_list(self, mock_mcmc_fit_cls: Mock) -> None:
        """Test filter_chain accepts list as boundary."""
        with (
            patch("elisa.analytics.binary_fit.io_tools._update_solution"),
            patch("elisa.analytics.binary_fit.io_tools.parameters.normalize_value") as mock_norm,
        ):
            mock_norm.side_effect = lambda v, _a, _b: v
            result = io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{"primary@te_ff": [0.12, 0.28]},
            )
            assert result is not None

    def test_filter_chain_boundary_as_ndarray(self, mock_mcmc_fit_cls: Mock) -> None:
        """Test filter_chain accepts ndarray as boundary."""
        with (
            patch("elisa.analytics.binary_fit.io_tools._update_solution"),
            patch("elisa.analytics.binary_fit.io_tools.parameters.normalize_value") as mock_norm,
        ):
            mock_norm.side_effect = lambda v, _a, _b: v
            result = io_tools.filter_chain(
                mock_mcmc_fit_cls,
                **{"primary@te_ff": np.array([0.12, 0.28])},
            )
            assert result is not None
