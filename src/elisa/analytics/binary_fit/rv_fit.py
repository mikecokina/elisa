import json
from ...logger import getLogger
from copy import copy

from ... import utils
from elisa.analytics.binary.least_squares import central_rv as lstsqr_central_rv
from elisa.analytics.binary.mcmc import central_rv as mcmc_central_rv
from elisa.analytics.binary_fit.plot import RVPlot
from elisa.analytics import utils as autils
from elisa.analytics.binary_fit import shared

logger = getLogger('analytics.binary_fit.rv_fit')


class RVFit(object):
    MANDATORY_KWARGS = ['radial_velocities', ]
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    MANDATORY_FIT_PARAMS = ['eccentricity', 'argument_of_periastron', 'gamma']
    OPTIONAL_FIT_PARAMS = ['period', 'primary_minimum_time', 'p__mass', 's__mass', 'inclination', 'asini', 'mass_ratio']
    ALL_FIT_PARAMS = MANDATORY_FIT_PARAMS + OPTIONAL_FIT_PARAMS

    FIT_PARAMS_COMBINATIONS = {
        'standard': ['p__mass', 's__mass', 'inclination', 'eccentricity', 'argument_of_periastron', 'gamma', 'period',
                     'primary_minimum_time'],
        'community': ['mass_ratio', 'asini', 'eccentricity', 'argument_of_periastron', 'gamma', 'period',
                      'primary_minimum_time']
    }

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs, RVFit.ALL_KWARGS, RVFit)
        utils.check_missing_kwargs(self.__class__.MANDATORY_KWARGS, kwargs, instance_of=self.__class__)

        self.radial_velocities = None
        self.fit_params = None
        self.ranges = None
        self.period = None

        # MCMC specific variables
        self.flat_chain = None
        self.flat_chain_path = None
        self.normalization = None
        self.variable_labels = None

        self.plot = RVPlot(self)

        # values of properties
        logger.debug(f"setting properties of RVFit")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    def fit(self, x0, method='least_squares', **kwargs):
        """
        Function encapsulates various fitting functions for fitting radial velocities.
        
        :param x0: Dict; starting values of the fit
        :param method: string;
        :param kwargs: Dict;
        :return: dict: fit_params
        """
        x0 = autils.transform_initial_values(x0)

        param_names = {key: value['value'] for key, value in x0.items()}
        utils.invalid_kwarg_checker(param_names, RVFit.ALL_FIT_PARAMS, RVFit)
        utils.check_missing_kwargs(RVFit.MANDATORY_FIT_PARAMS, param_names, instance_of=RVFit)

        x_data, y_data, yerr = dict(), dict(), dict()
        for component, data in self.radial_velocities.items():
            x_data[component] = data.x_data
            y_data[component] = data.y_data
            yerr[component] = data.yerr

        if method == 'least_squares':
            self.fit_params = lstsqr_central_rv.fit(xs=x_data, ys=y_data, x0=x0, yerr=yerr, **kwargs)

        elif str(method).lower() in ['mcmc']:
            self.fit_params = mcmc_central_rv.fit(xs=x_data, ys=y_data, x0=x0, yerr=yerr, **kwargs)
            self.flat_chain = mcmc_central_rv.last_sampler.get_chain(flat=True)
            self.flat_chain_path = mcmc_central_rv.last_fname
            self.normalization = mcmc_central_rv.last_normalization
            self.variable_labels = mcmc_central_rv.labels

        logger.info('Fitting and processing of results finished successfully.')

        return self.fit_params

    def load_chain(self, filename):
        """
        Function loads MCMC chain along with auxiliary data from json file created after each MCMC run.

        :param filename: str; full name of the json file
        :return: Tuple[numpy.ndarray, list, Dict]; flattened mcmc chain, labels of variables in `flat_chain` columns,
                                                  {var_name: (min_boundary, max_boundary), ...} dictionary of
                                                  boundaries defined by user for each variable needed
                                                  to reconstruct real values from normalized `flat_chain` array
        """
        return shared.load_mcmc_chain(self, filename)

    def store_parameters(self, parameters=None, filename=None):
        """
        Function converts model parameters to json compatibile format and stores model parameters.

        :param parameters: Dict; {'name': {'value': numpy.ndarray, 'unit': Union[astropy.unit, str], ...}, ...}
        :param filename: str;
        :return:
        """
        parameters = copy(self.fit_params) if parameters is None else parameters
        parameters = self.fit_params if parameters is None else parameters

        parameters = autils.unify_unit_string_representation(parameters)

        json_params = autils.convert_dict_to_json_format(parameters)
        with open(filename, 'w') as f:
            json.dump(json_params, f, separators=(',\n', ': '))

    def load_parameters(self, filename=None):
        with open(filename, 'r') as f:
            prms = json.load(f)

        prms = autils.convert_json_to_dict_format(prms)
        self.period = prms['period']['value']
        self.fit_params = prms

        return prms
