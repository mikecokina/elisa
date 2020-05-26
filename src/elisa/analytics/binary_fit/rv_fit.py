import json

from typing import Union, Dict
from elisa.analytics.params import parameters
from elisa.logger import getLogger
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.analytics.binary_fit.mcmc import CentralRadialVelocity as MCMCCentralRV
from elisa.analytics.binary_fit.least_squares import CentralRadialVelocity as LstSqrCentralRV
from elisa.analytics.binary_fit import io_tools

logger = getLogger('analytics.binary_fit.rv_fit')

DESH_N = 126


class RVFit(object):
    def __init__(self):
        self.result = None
        self.flat_result = None
        self.fit_method_instance: Union[RVFitLeastSquares, RVFitMCMC, None] = None

    def fit_summary(self, path=None):
        """
        Producing a summary of the fit in more human readable form.

        :param path: Union[str, None]; if not None, summary is stored in file, otherwise it is printed into console
        :return:
        """
        f = None
        if path is not None:
            f = open(path, 'w')
            write_fn = f.write
            line_sep = '\n'
        else:
            write_fn = print
            line_sep = ''

        try:
            write_fn(f"\n{'-' * DESH_N}{line_sep}")
            io_tools.write_ln(write_fn, 'Parameter', 'value', '-1 sigma', '+1 sigma', 'unit', 'status', line_sep)
            write_fn(f"{'-'*DESH_N}{line_sep}")
            result_dict: Dict = self.fit_method_instance.flat_result

            if 'system@mass_ratio' in result_dict:
                io_tools.write_param_ln(result_dict, 'system@mass_ratio', 'Mass ratio (q=M_2/M_1):',
                                        write_fn, line_sep, 3)
                io_tools.write_param_ln(result_dict, 'system@asini', 'a*sin(i):', write_fn, line_sep, 2)
            else:
                io_tools.write_param_ln(result_dict, 'primary@mass', 'Primary mass:', write_fn, line_sep, 3)
                io_tools.write_param_ln(result_dict, 'secondary@mass', 'Secondary mass:', write_fn, line_sep, 3)
                io_tools.write_param_ln(result_dict, 'system@inclination', 'Inclination(i):', write_fn, line_sep, 3)

            io_tools.write_param_ln(result_dict, 'system@eccentricity', 'Eccentricity (e):', write_fn, line_sep)
            io_tools.write_param_ln(result_dict, 'system@argument_of_periastron',
                                    'Argument of periastron (omega):', write_fn, line_sep)
            io_tools.write_param_ln(result_dict, 'system@gamma', 'Centre of mass velocity (gamma):', write_fn, line_sep)
            io_tools.write_param_ln(result_dict, 'system@period', 'Orbital period (P):', write_fn, line_sep)
            if 'system@primary_minimum_time' in result_dict.keys():
                io_tools.write_param_ln(result_dict, 'system@primary_minimum_time',
                                        'Time of primary minimum (T0):', write_fn, line_sep)

            write_fn(f"{'-' * DESH_N}{line_sep}")

            if result_dict['r_squared']['value'] is not None:
                io_tools.write_param_ln(result_dict, 'r_squared', 'Fit R^2: ', write_fn, line_sep, 6)

            write_fn(f"{'-' * DESH_N}{line_sep}")
        finally:
            if f is not None:
                f.close()

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
