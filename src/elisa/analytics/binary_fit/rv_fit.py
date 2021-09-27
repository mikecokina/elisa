from typing import Union
from . mcmc import CentralRadialVelocity as MCMCCentralRV
from . least_squares import CentralRadialVelocity as LstSqrCentralRV
from . import io_tools
from . summary import (
    fit_rv_summary_with_error_propagation,
    fit_lc_summary_with_error_propagation,
    simple_rv_fit_summary,
    simple_lc_fit_summary
)
from .. params import parameters
from .. params.result_handler import FitResultHandler
from .. params.parameters import BinaryInitialParameters
from ... binary_system.utils import resolve_json_kind
from ... logger import getLogger


logger = getLogger('analytics.binary_fit.rv_fit')

DASH_N = 126


class RVFit(FitResultHandler):
    """
    Class with common methods used during an RV fit.
    """
    def __init__(self):
        super().__init__()
        self.fit_method_instance: Union[RVFitLeastSquares, RVFitMCMC, None] = None

    def coefficient_of_determination(self, model_parameters, data, discretization, interp_treshold):
        """
        Function returns R^2 for given model parameters and observed data.

        :param model_parameters: Dict; set of model parameters in json format
        :param data: Dict[str, RVData]; observed RVs for each component
        :param discretization: float; discretization factor for the primary component
        :param interp_treshold: int; a number of observation points above which the synthetic curves will be calculated
                                     using `interp_treshold` equally spaced points that will be subsequently
                                     interpolated to the desired times of observation
        :return: float; coefficient of determination (1.0 means a perfect fit to the observations)
        """
        b_parameters = parameters.BinaryInitialParameters(**model_parameters)
        b_parameters.validate_rv_parameters()
        args = model_parameters, data, discretization, interp_treshold
        return self.fit_method_instance.coefficient_of_determination(*args)


class RVFitMCMC(RVFit):
    """
    Class for RV fitting using the MCMC method.
    """
    def __init__(self):
        super().__init__()
        self.fit_method_instance = MCMCCentralRV()

        self.flat_chain = None
        self.flat_chain_path = None
        self.normalization = None
        self.variable_labels = None

    def fit(self, x0: BinaryInitialParameters, data, **kwargs):
        """
        Perform MCMC sampling on RVFitMCMC instance.

        :param x0: BinaryInitialParameters; initial info about the model parameters such as status
                                            (fixed, variable, constrained), bounds (prior distribution) and
                                            initial value
        :param data: Dict[RVData]; observational data (radial velocities for both components)
        :param kwargs: Dict; arguments passed to the fitting method (see AnalyticsTask.fit kwargs for MCMC or
                             mcmc.CentralRadialVelocity.fit for further info)
        :return: Dict; optimized model parameters in JSON format
        """
        x0.validate_rv_parameters()
        self.result = self.fit_method_instance.fit(data=data, x0=x0, **kwargs)
        self.flat_result = self.fit_method_instance.flat_result

        self.flat_chain = self.fit_method_instance.last_sampler.get_chain(flat=True)
        self.flat_chain_path = self.fit_method_instance.flat_chain_path
        self.normalization = self.fit_method_instance.normalization
        self.variable_labels = list(self.fit_method_instance.fitable.keys())

        logger.info('Fitting and processing of results finished successfully.')
        self.fit_summary()

        return self.result

    def load_chain(self, filename, discard=0, percentiles=None):
        """
        Function loads MCMC chain along with auxiliary data from json file created after each MCMC run.

        :param percentiles: List; percentile intervals used to generate confidence intervals, provided in form:
                                  [percentile for lower bound of confidence interval, percentile of the centre,
                                  percentile for the upper bound of confidence interval]
        :param discard: int; Discard the first discard steps in the chain as a part of the thermalization phase
                             (default: 0).
        :param filename: str; str; chain identificator or filename (ending with .json) containing the chain
        :return: Tuple[numpy.ndarray, List, Dict]; flattened mcmc chain, labels of variables in `flat_chain` columns,
                                                  {var_name: (min_boundary, max_boundary), ...} dictionary of
                                                  boundaries defined by user for each variable needed
                                                  to reconstruct real values from normalized `flat_chain` array
        """
        return io_tools.load_chain(self, filename, discard, percentiles)

    def fit_summary(self, filename=None, **kwargs):
        """
        Function produces detailed summary about the current RV fitting task with the complete error propagation for RV
        parameters if `propagate_errors` is True

        :param filename: str; path where to store summary
        :param kwargs: Dict;
        :**kwargs options**:
            * :propagate_errors: bool; errors of fitted parameters will be propagated to the rest of EB
                                       parameters (takes a while to calculate)
            * :percentiles: List; percentiles used to evaluate confidence intervals from posterior
                                  distribution of EB parameters in MCMC chain . Used only when if
                                  `propagate_errors` is True.
        """
        propagate_errors = kwargs.get('propagate_errors', False)
        percentiles = kwargs.get('percentiles', [16, 50, 84])
        dimensionless_radii = kwargs.get('dimensionless_radii', True)

        kind_of = resolve_json_kind(data=self.result, _sin=True)
        if not propagate_errors:
            if kind_of in ["community"]:
                simple_rv_fit_summary(self, filename)
            else:
                simple_lc_fit_summary(self, filename, dimensionless_radii=True)
            return

        if kind_of in ["community"]:
            fit_rv_summary_with_error_propagation(self, filename, percentiles)
        else:
            fit_lc_summary_with_error_propagation(self, filename, percentiles, dimensionless_radii=dimensionless_radii)

    def filter_chain(self, **boundaries):
        """
        Filtering MCMC chain down to given parameter intervals. This function is useful in case of bimodal distribution
        of the MCMC chain.

        :param boundaries: Dict; dictionary of boundaries e.g. {'primary@te_ff': (5000, 6000), other parameters ...}
        :return: numpy.array; filtered flat chain
        """
        return io_tools.filter_chain(self, **boundaries)


class RVFitLeastSquares(RVFit):
    """
    Class for LC fitting using the Least-Squares method.
    """
    def __init__(self):
        super().__init__()
        self.fit_method_instance = LstSqrCentralRV()

    def fit(self, x0: BinaryInitialParameters, data, **kwargs):
        """
        Perform Least-Squares optimization on RVFitLeastSquares instance.

        :param x0: BinaryInitialParameters; initial info about the model parameters such as status
                                            (fixed, variable, constrained), bounds (prior distribution) and
                                            initial value
        :param data: Dict[RVData]; observational data (radial velocities for both components)
        :param kwargs: Dict; arguments passed to the fitting method (see AnalyticsTask.fit kwargs for Least-Squares or
                             least_squares.CentralRadialVelocity.fit for further info)
        :return: Dict; optimized model parameters in JSON format
        """
        x0.validate_rv_parameters()
        self.result = self.fit_method_instance.fit(data=data, x0=x0, **kwargs)
        self.flat_result = self.fit_method_instance.flat_result

        logger.info('Fitting and processing of results finished successfully.')
        self.fit_summary()

        return self.result

    def fit_summary(self, path=None, **kwargs):
        """
        Function produces detailed summary of the current RV fitting task.

        :param path: str; path, place to store summary
        :param kwargs:
        :return:
        """
        simple_rv_fit_summary(self, path)

