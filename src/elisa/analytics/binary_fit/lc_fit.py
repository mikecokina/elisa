import json

from abc import abstractmethod
from typing import Union

from ... logger import getLogger

from .. params.parameters import BinaryInitialParameters
from .. params import parameters
from . summary import fit_lc_summary_with_error_propagation, simple_lc_fit_summary
from . shared import eval_constraint_in_dict
from . import least_squares
from . import mcmc
from . import io_tools


logger = getLogger('analytics.binary_fit.lc_fit')

DASH_N = 126


class LCFit(object):

    def __init__(self, morphology):
        self.morphology = morphology

        self.result = None
        self.flat_result = None
        self.fit_method_instance: Union[LCFitLeastSquares, LCFitMCMC, None] = None

    def get_result(self):
        return self.result

    def set_result(self, result):
        result = eval_constraint_in_dict(result)
        self.result = result
        self.flat_result = parameters.deserialize_result(self.result)

    def load_result(self, path):
        """
        Function loads fitted parameters of given model.

        :param path: str;
        """
        with open(path, 'r') as f:
            loaded_result = json.load(f)
        loaded_result = eval_constraint_in_dict(loaded_result)
        self.result = loaded_result
        self.flat_result = parameters.deserialize_result(self.result)

    def save_result(self, path):
        """
        Save result as json.

        :param path: str; path to file
        """
        if self.result is None:
            raise IOError("No result to store.")

        with open(path, 'w') as f:
            json.dump(self.result, f, separators=(',', ': '), indent=4)

    def coefficient_of_determination(self, model_parameters, data, discretization, interp_treshold):
        """
        Function returns R^2 for given model parameters and observed data.

        :param model_parameters: Dict; serialized form
        :param data: DataSet; observational data
        :param discretization: float;
        :param interp_treshold: int;
        :return: float;
        """
        b_parameters = parameters.BinaryInitialParameters(**model_parameters)
        b_parameters.validate_lc_parameters(morphology=self.morphology)
        args = model_parameters, data, discretization, interp_treshold
        return self.fit_method_instance.coefficient_of_determination(*args)

    @abstractmethod
    def resolve_fit_cls(self, morphology: str):
        pass


class LCFitMCMC(LCFit):
    def __init__(self, morphology):
        super().__init__(morphology)
        self.fit_method_instance = self.resolve_fit_cls(morphology)()

        self.flat_chain = None
        self.flat_chain_path = None
        self.normalization = None
        self.variable_labels = None

    def resolve_fit_cls(self, morphology: str):
        _cls = {"detached": mcmc.DetachedLightCurveFit, "over-contact": mcmc.OvercontactLightCurveFit}
        return _cls[morphology]

    def fit(self, x0: BinaryInitialParameters, data, **kwargs):
        x0.validate_lc_parameters(morphology=self.morphology)
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

        :param percentiles: List;
        :param self: Union[] instance of fitting cls based on method (mcmc, lsqr) and type(lc, rv)
        :param discard: int; Discard the first discard steps in the chain as burn-in. (default: 0)
        :param filename: str; full name of the json file
        :return: Tuple[numpy.ndarray, List, Dict]; flattened mcmc chain, labels of variables in `flat_chain` columns,
                                                  {var_name: (min_boundary, max_boundary), ...} dictionary of
                                                  boundaries defined by user for each variable needed
                                                  to reconstruct real values from normalized `flat_chain` array
        """
        return io_tools.load_chain(self, filename, discard, percentiles)

    def fit_summary(self, filename=None, **kwargs):
        """
        Function produces detailed summary about the current LC fitting task with complete error propagation for the LC
        parameters if `propagate_errors` is True.

        :param path: str; path, where to store summary
        :param kwargs: Dict;
        :**kwargs options**:
            * ** propagate_errors ** * - bool -- errors of fitted parameters will be propagated to the rest of EB
                                                 parameters (takes a while)
            * ** percentiles ** * - List -- percentiles used to evaluate confidence intervals from forward distribution
                                            of EB parameters. Useless if `propagate_errors` is False.
            * ** dimensionless_radii ** * - if True (default), radii are provided in SMA, otherwise solRad are used

        """
        propagate_errors, percentiles = kwargs.get('propagate_errors', False), kwargs.get('percentiles', [16, 50, 84])
        dimensionless_radii = kwargs.get('dimensionless_radii', True)
        if not propagate_errors:
            simple_lc_fit_summary(self, filename, dimensionless_radii=dimensionless_radii)
            return

        fit_lc_summary_with_error_propagation(self, filename, percentiles, dimensionless_radii=dimensionless_radii)

    def filter_chain(self, **boundaries):
        """
        Filtering mcmc chain to given set of intervals.

        :param boundaries: Dict; dictionary of boundaries e.g. {'primary@te_ff': (5000, 6000), other parameters ...}
        :return: numpy.array; filtered flat chain
        """
        return io_tools.filter_chain(self, **boundaries)


class LCFitLeastSquares(LCFit):
    def __init__(self, morphology):
        super().__init__(morphology)
        self.fit_method_instance = self.resolve_fit_cls(morphology)()

    def resolve_fit_cls(self, morphology: str):
        _cls = {"detached": least_squares.DetachedLightCurveFit, "over-contact": least_squares.OvercontactLightCurveFit}
        return _cls[morphology]

    def fit(self, x0: BinaryInitialParameters, data, **kwargs):
        x0.validate_lc_parameters(morphology=self.morphology)
        self.result = self.fit_method_instance.fit(data=data, x0=x0, **kwargs)
        self.flat_result = self.fit_method_instance.flat_result
        logger.info('Fitting and processing of results finished successfully.')
        self.fit_summary()
        return self.result

    def fit_summary(self, path=None, **kwargs):
        simple_lc_fit_summary(self, path)

