from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elisa.analytics.binary_fit import io_tools
from elisa.analytics.binary_fit.least_squares import (
    CentralRadialVelocity as LstSqrCentralRV,
)
from elisa.analytics.binary_fit.mcmc import CentralRadialVelocity as MCMCCentralRV
from elisa.analytics.binary_fit.summary import (
    fit_lc_summary_with_error_propagation,
    fit_rv_summary_with_error_propagation,
    simple_lc_fit_summary,
    simple_rv_fit_summary,
)
from elisa.analytics.params import parameters
from elisa.analytics.params.result_handler import FitResultHandler
from elisa.binary_system.utils import resolve_json_kind
from elisa.logger import getLogger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.analytics.params.parameters import BinaryInitialParameters

logger = getLogger("analytics.binary_fit.rv_fit")


class RVFit(FitResultHandler):
    """Base class with common methods used during RV fitting.

    This class provides shared functionality for radial velocity fitting operations,
    including methods for calculating coefficient of determination and managing
    the fit method instance.
    """

    def __init__(self) -> None:
        """Initialize RVFit instance.

        :returns: None.
        :rtype: None
        """
        super().__init__()
        self.fit_method_instance: RVFitLeastSquares | RVFitMCMC | None = None

    def coefficient_of_determination(
        self,
        model_parameters: dict[str, Any],
        data: dict[str, Any],
        discretization: float,
        interp_treshold: int,
    ) -> float:
        """Calculate coefficient of determination (R²) for given parameters.

        Returns R² for given model parameters and observed data.

        :param model_parameters: Set of model parameters in JSON format.
        :type model_parameters: dict[str, Any]
        :param data: Observed radial velocities for each component.
        :type data: dict[str, Any]
        :param discretization: Discretization factor for the primary component.
        :type discretization: float
        :param interp_treshold: Number of observation points above which the synthetic curves
            will be calculated using equally spaced points that will be subsequently
            interpolated to the desired times of observation.
        :type interp_treshold: int
        :returns: Coefficient of determination (1.0 means perfect fit).
        :rtype: float
        """
        b_parameters = parameters.BinaryInitialParameters(**model_parameters)
        b_parameters.validate_rv_parameters()
        args = model_parameters, data, discretization, interp_treshold
        return self.fit_method_instance.coefficient_of_determination(*args)


class RVFitMCMC(RVFit):
    """MCMC-based radial velocity fitting.

    Implements radial velocity fitting using the Markov Chain Monte Carlo (MCMC) method.
    """

    def __init__(self) -> None:
        """Initialize RVFitMCMC instance.

        :returns: None.
        :rtype: None
        """
        super().__init__()
        self.fit_method_instance: MCMCCentralRV = MCMCCentralRV()

        self.flat_chain: NDArray | None = None
        self.flat_chain_path: str | None = None
        self.normalization: dict[str, Any] | None = None
        self.variable_labels: list[str] | None = None

    def fit(
        self,
        x0: BinaryInitialParameters,
        data: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Perform MCMC sampling for radial velocity fitting.

        Executes MCMC sampling on the RVFitMCMC instance with the provided initial
        parameters and observational data.

        :param x0: Initial model parameters with information about status (fixed, variable,
            constrained), bounds (prior distribution), and initial values.
        :type x0: BinaryInitialParameters
        :param data: Observational data (radial velocities for both components).
        :type data: dict[str, Any]
        :param kwargs: Keyword arguments passed to the fitting method.
            See ``AnalyticsTask.fit`` kwargs for MCMC or
            ``mcmc.CentralRadialVelocity.fit`` for details.
        :type kwargs: Any
        :returns: Optimized model parameters in JSON format.
        :rtype: dict[str, Any]
        """
        x0.validate_rv_parameters()
        self.result = self.fit_method_instance.fit(data=data, x0=x0, **kwargs)
        self.flat_result = self.fit_method_instance.flat_result

        # noinspection PyArgumentList
        self.flat_chain = self.fit_method_instance.last_sampler.get_chain(flat=True)
        self.flat_chain_path = self.fit_method_instance.flat_chain_path
        self.normalization = self.fit_method_instance.normalization
        self.variable_labels = list(self.fit_method_instance.fitable.keys())

        logger.info("Fitting and processing of results finished successfully.")
        self.fit_summary()

        return self.result

    def load_chain(
        self,
        filename: str,
        discard: int = 0,
        percentiles: list[float] | None = None,
    ) -> tuple[NDArray, list[str], dict[str, tuple[float, float]]]:
        """Load MCMC chain from JSON file with auxiliary data.

        Loads MCMC chain along with auxiliary data from JSON file created after each MCMC run.

        :param filename: Chain identifier or filename (ending with .json) containing the chain.
        :type filename: str
        :param discard: Number of steps to discard from the chain as part of the
            thermalization phase (default: 0).
        :type discard: int
        :param percentiles: Percentile intervals used to generate confidence intervals,
            provided as [lower, center, upper].
        :type percentiles: list[float] | None
        :returns: Tuple containing flattened MCMC chain, labels of variables in
            flat_chain columns, and dictionary of boundaries for reconstructing
            real values from normalized flat_chain array.
        :rtype: tuple[NDArray, list[str], dict[str, tuple[float, float]]]
        """
        return io_tools.load_chain(self, filename, discard, percentiles)

    def fit_summary(
        self,
        filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Produce detailed RV fitting summary with optional error propagation.

        Generates detailed summary about the current RV fitting task with complete
        error propagation for RV parameters if ``propagate_errors`` is True.

        :param filename: Path where to store summary.
        :type filename: str | None
        :param kwargs: Keyword arguments controlling summary generation.
            Supported options are:

            - ``propagate_errors`` (bool): If True, errors of fitted parameters will be
              propagated to the rest of EB parameters (takes a while to calculate).
              Default: False.
            - ``percentiles`` (list[float]): Percentiles used to evaluate confidence
              intervals from posterior distribution of EB parameters in MCMC chain.
              Used only when ``propagate_errors`` is True. Default: [16, 50, 84].
            - ``dimensionless_radii`` (bool): Whether to use dimensionless radii in output.
              Default: True.

        :type kwargs: Any
        :returns: None.
        :rtype: None
        """
        propagate_errors = kwargs.get("propagate_errors", False)
        percentiles = kwargs.get("percentiles", [16, 50, 84])
        dimensionless_radii = kwargs.get("dimensionless_radii", True)

        kind_of = resolve_json_kind(data=self.result, _sin=True)
        if not propagate_errors:
            if kind_of == "community":
                simple_rv_fit_summary(self, filename)
            else:
                simple_lc_fit_summary(self, filename, dimensionless_radii=True)
            return

        if kind_of == "community":
            fit_rv_summary_with_error_propagation(self, filename, percentiles)
        else:
            fit_lc_summary_with_error_propagation(
                self,
                filename,
                percentiles,
                dimensionless_radii=dimensionless_radii,
            )

    def filter_chain(self, **boundaries: tuple[float, float]) -> NDArray:
        """Filter MCMC chain down to given parameter intervals.

        Useful for filtering bimodal distributions in MCMC chain.

        :param boundaries: Dictionary of boundaries mapping parameter names to
            (lower, upper) tuples. Example: ``{'primary@te_ff': (5000, 6000)}``.
        :type boundaries: tuple[float, float]
        :returns: Filtered flat chain.
        :rtype: NDArray
        """
        return io_tools.filter_chain(self, **boundaries)


class RVFitLeastSquares(RVFit):
    """Least-Squares based radial velocity fitting.

    Implements radial velocity fitting using the Least-Squares optimization method.
    """

    def __init__(self) -> None:
        """Initialize RVFitLeastSquares instance.

        :returns: None.
        :rtype: None
        """
        super().__init__()
        self.fit_method_instance: LstSqrCentralRV = LstSqrCentralRV()

    def fit(
        self,
        x0: BinaryInitialParameters,
        data: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Perform Least-Squares optimization for radial velocity fitting.

        Executes Least-Squares optimization on the RVFitLeastSquares instance with
        the provided initial parameters and observational data.

        :param x0: Initial model parameters with information about status (fixed, variable,
            constrained), bounds (prior distribution), and initial values.
        :type x0: BinaryInitialParameters
        :param data: Observational data (radial velocities for both components).
        :type data: dict[str, Any]
        :param kwargs: Keyword arguments passed to the fitting method.
            See ``AnalyticsTask.fit`` kwargs for Least-Squares or
            ``least_squares.CentralRadialVelocity.fit`` for details.
        :type kwargs: Any
        :returns: Optimized model parameters in JSON format.
        :rtype: dict[str, Any]
        """
        x0.validate_rv_parameters()
        self.result = self.fit_method_instance.fit(data=data, x0=x0, **kwargs)
        self.flat_result = self.fit_method_instance.flat_result

        logger.info("Fitting and processing of results finished successfully.")
        self.fit_summary()

        return self.result

    def fit_summary(self, path: str | None = None) -> None:
        """Produce detailed RV fitting summary.

        Generates detailed summary of the current RV fitting task.

        :param path: Path where to store summary.
        :type path: str | None
        :returns: None.
        :rtype: None
        """
        simple_rv_fit_summary(self, path)
