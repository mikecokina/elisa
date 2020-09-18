import json

from typing import Union
from . mcmc import CentralRadialVelocity as MCMCCentralRV
from . least_squares import CentralRadialVelocity as LstSqrCentralRV
from . import io_tools
from . summary import (
    fit_rv_summary_with_error_propagation,
    fit_lc_summary_with_error_propagation,
    simple_rv_fit_summary
)
from .. params import parameters
from .. params.parameters import BinaryInitialParameters
from ... binary_system.utils import resolve_json_kind
from ... logger import getLogger

logger = getLogger('analytics.binary_fit.rv_fit')

DASH_N = 126


class RVFit(object):
    def __init__(self):
        self.result = None
        self.flat_result = None
        self.fit_method_instance: Union[RVFitLeastSquares, RVFitMCMC, None] = None

    def get_result(self):
        return self.result

    def set_result(self, result):
        self.result = result
        self.flat_result = parameters.deserialize_result(self.result)

    def load_result(self, path):
        """
        Function loads fitted parameters of given model.

        :param path: str;
        """
        with open(path, 'r') as f:
            loaded_result = json.load(f)
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
            json.dump(self.result, f, separators=(',\n', ': '))


class RVFitMCMC(RVFit):
    def __init__(self):
        super().__init__()
        self.fit_method_instance = MCMCCentralRV()

        self.flat_chain = None
        self.flat_chain_path = None
        self.normalization = None
        self.variable_labels = None

    def fit(self, x0: BinaryInitialParameters, data, **kwargs):
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

    def fit_summary(self, path=None, **kwargs):
        """
        Function produces detailed summary about the current RV fitting task with the complete error propagation for RV
        parameters if `propagate_errors` is True

        :param path: str; path, where to store summary
        :param kwargs: Dict;
        :**kwargs options**:
            * ** propagate_errors ** * - bool -- errors of fitted parameters will be propagated to the rest of EB
                                                 parameters (takes a while)
            * ** percentiles ** * - List -- percentiles used to evaluate confidence intervals from forward distribution
                                            of EB parameters. Useless if `propagate_errors` is False.
        """
        propagate_errors = kwargs.get('propagate_errors', False)
        percentiles = kwargs.get('percentiles', [16, 50, 84])
        if not propagate_errors:
            simple_rv_fit_summary(self, path)
            return

        kind_of = resolve_json_kind(data=self.result, _sin=True)

        if kind_of in ["community"]:
            fit_rv_summary_with_error_propagation(self, path, percentiles)
        else:
            fit_lc_summary_with_error_propagation(self, path, percentiles)


class RVFitLeastSquares(RVFit):
    def __init__(self):
        super().__init__()
        self.fit_method_instance = LstSqrCentralRV()

    def fit(self, x0: BinaryInitialParameters, data, **kwargs):
        x0.validate_rv_parameters()
        self.result = self.fit_method_instance.fit(data=data, x0=x0, **kwargs)
        self.flat_result = self.fit_method_instance.flat_result

        logger.info('Fitting and processing of results finished successfully.')
        self.fit_summary()

        return self.result

    def fit_summary(self, path=None, **kwargs):
        simple_rv_fit_summary(self, path)

