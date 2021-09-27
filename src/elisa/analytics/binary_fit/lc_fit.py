from abc import abstractmethod
from typing import Union

from ... logger import getLogger

from .. params.parameters import BinaryInitialParameters
from .. params import parameters
from .. params.result_handler import FitResultHandler
from . summary import fit_lc_summary_with_error_propagation, simple_lc_fit_summary
from . import least_squares
from . import mcmc
from . import io_tools


logger = getLogger('analytics.binary_fit.lc_fit')

DASH_N = 126


class LCFit(FitResultHandler):
    """
    Class with common methods used during an LC fit.
    """
    def __init__(self, morphology):
        super().__init__()
        self.morphology = morphology
        self.fit_method_instance: Union[LCFitLeastSquares, LCFitMCMC, None] = None

    def coefficient_of_determination(self, model_parameters, data, discretization, interp_treshold):
        """
        Function returns R^2 for given model parameters and observed data.

        :param model_parameters: Dict; set of model parameters in json format
        :param data: Dict[str; LCData]; observational data in each passband
        :param discretization: float; discretization factor for the primary component
        :param interp_treshold: int; a number of observation points above which the synthetic curves will be calculated
                                     using `interp_treshold` equally spaced points that will be subsequently
                                     interpolated to the desired times of observation
        :return: float; coefficient of determination (1.0 means a perfect fit to the observations)
        """
        b_parameters = parameters.BinaryInitialParameters(**model_parameters)
        b_parameters.validate_lc_parameters(morphology=self.morphology)
        args = model_parameters, data, discretization, interp_treshold
        return self.fit_method_instance.coefficient_of_determination(*args)

    @abstractmethod
    def resolve_fit_cls(self, morphology: str):
        pass


class LCFitMCMC(LCFit):
    """
    Class for LC fitting using the MCMC method.
    """
    def __init__(self, morphology):
        super().__init__(morphology)
        self.fit_method_instance = self.resolve_fit_cls(morphology)()

        self.flat_chain = None
        self.flat_chain_path = None
        self.normalization = None
        self.variable_labels = None

    def filter_chain(self, **boundaries):
        """
        Filtering MCMC chain down to given parameter intervals. This function is useful in case of bimodal distribution
        of the MCMC chain.

        :param boundaries: Dict; dictionary of boundaries e.g. {'primary@te_ff': (5000, 6000), other parameters ...}
        :return: numpy.array; filtered flat chain
        """
        return io_tools.filter_chain(self, **boundaries)

    def fit(self, x0: BinaryInitialParameters, data, **kwargs):
        """
        Perform MCMC sampling on LCFitMCMC instance.

        :param x0: BinaryInitialParameters; initial info about the model parameters such as status
                                            (fixed, variable, constrained), bounds (prior distribution) and
                                            initial value
        :param data: Dict[LCData]; observational data (light curves in multiple filters)
        :param kwargs: Dict; arguments passed to the fitting method (see AnalyticsTask.fit kwargs for MCMC or
                             mcmc.LightCurveFit.fit for further info)
        :return: Dict; optimized model parameters in JSON format
        """
        x0.validate_lc_parameters(morphology=self.morphology)
        self.result = self.fit_method_instance.fit(data=data, x0=x0, **kwargs)
        self.flat_result = self.fit_method_instance.flat_result

        self.flat_chain = self.fit_method_instance.last_sampler.get_chain(flat=True)
        self.flat_chain_path = self.fit_method_instance.flat_chain_path
        self.normalization = self.fit_method_instance.normalization
        self.variable_labels = list(self.fit_method_instance.fitable.keys())

        logger.info('Fitting and processing of results finished successfully.')
        self.fit_summary()

    def fit_summary(self, filename=None, **kwargs):
        """
        Function produces detailed summary of the current LC fitting task with the possibility to propagate
        uncertainties of the fitted binary model parameters if `propagate_errors` is True.

        :param filename: str; path, place to store summary
        :param kwargs: Dict;
        :**kwargs options**:
            * :propagate_errors: bool; errors of fitted parameters will be propagated to the rest of EB
                                       parameters (takes a while to calculate)
            * :percentiles: List; percentiles used to evaluate confidence intervals from posterior
                                  distribution of EB parameters in MCMC chain . Used only when if
                                  `propagate_errors` is True.
            * :dimensionless_radii: bool; if True (default), radii are provided in SMA, otherwise solRad are used

        """
        propagate_errors, percentiles = kwargs.get('propagate_errors', False), kwargs.get('percentiles', [16, 50, 84])
        dimensionless_radii = kwargs.get('dimensionless_radii', True)
        if not propagate_errors:
            simple_lc_fit_summary(self, filename, dimensionless_radii=dimensionless_radii)
            return

        fit_lc_summary_with_error_propagation(self, filename, percentiles, dimensionless_radii=dimensionless_radii)

        return self.result

    def load_chain(self, filename, discard=0, percentiles=None):
        """
        Function loads MCMC chain along with auxiliary data from json file created after each MCMC run.

        :param percentiles: List; percentile intervals used to generate confidence intervals, provided in form:
                                  [percentile for lower bound of confidence interval, percentile of the centre,
                                  percentile for the upper bound of confidence interval]
        :param discard: int; Discard the first discard steps in the chain as a part of the thermalization phase
                             (default: 0).
        :param filename: str; chain identificator or filename (ending with .json) containing the chain
        :return: Tuple[numpy.ndarray, List, Dict]; flattened mcmc chain, labels of variables in `flat_chain` columns,
                                                   {var_name: (min_boundary, max_boundary), ...} dictionary of
                                                   boundaries defined by user for each variable needed
                                                   to reconstruct real values from normalized `flat_chain` array
        """
        return io_tools.load_chain(self, filename, discard, percentiles)

    def resolve_fit_cls(self, morphology: str):
        """
        Function returns MCMC fitting class suitable for the model based on its morphology.

        :param morphology: str; `detached` or `overcontact`
        :return: Union[mcmc.DetachedLightCurveFit, mcmc.OvercontactLightCurveFit]
        """
        _cls = {"detached": mcmc.DetachedLightCurveFit, "over-contact": mcmc.OvercontactLightCurveFit}
        return _cls[morphology]


class LCFitLeastSquares(LCFit):
    """
    Class for LC fitting using the Least-Squares method.
    """
    def __init__(self, morphology):
        super().__init__(morphology)
        self.fit_method_instance = self.resolve_fit_cls(morphology)()

    def fit(self, x0: BinaryInitialParameters, data, **kwargs):
        """
        Perform Least-Squares optimization on LCFitLeastSquares instance.

        :param x0: BinaryInitialParameters; initial info about the model parameters such as status
                                            (fixed, variable, constrained), bounds (prior distribution) and
                                            initial value
        :param data: Dict[LCData]; observational data (light curves in multiple filters)
        :param kwargs: Dict; arguments passed to the fitting method (see AnalyticsTask.fit kwargs for Least-Squares or
                             least_squares.LightCurveFit.fit for further info)
        :return: Dict; optimized model parameters in JSON format
        """
        x0.validate_lc_parameters(morphology=self.morphology)
        self.result = self.fit_method_instance.fit(data=data, x0=x0, **kwargs)
        self.flat_result = self.fit_method_instance.flat_result
        logger.info('Fitting and processing of results finished successfully.')
        self.fit_summary()
        return self.result

    def fit_summary(self, path=None, **kwargs):
        """
        Function produces detailed summary of the current LC fitting task.

        :param path: str; path, place to store summary
        :param kwargs:
        :return:
        """
        simple_lc_fit_summary(self, path)

    def resolve_fit_cls(self, morphology: str):
        """
        Function returns Least-Squares fitting class suitable for the model based on its morphology.

        :param morphology: str; `detached` or `overcontact`
        :return: Union[least_squares.DetachedLightCurveFit, least_squares.OvercontactLightCurveFit]
        """
        _cls = {"detached": least_squares.DetachedLightCurveFit, "over-contact": least_squares.OvercontactLightCurveFit}
        return _cls[morphology]
