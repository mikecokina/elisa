import json
from ...logger import getLogger
from copy import copy
from astropy import units as u

from ... import utils
from elisa.analytics.binary_fit.plot import (
    LCPlot, params
)
from elisa.analytics import utils as autils
from elisa.analytics.binary.least_squares import (
    binary_detached as lst_detached_fit,
    binary_overcontact as lst_over_contact_fit
)
from elisa.analytics.binary.mcmc import (
    binary_detached as mcmc_detached_fit,
    binary_overcontact as mcmc_over_contact_fit
)
from elisa import units
from elisa.analytics.binary_fit import shared

logger = getLogger('analytics.binary_fit.rv_fit')


class LCFit(object):
    MANDATORY_KWARGS = ['light_curves', ]
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    MANDATORY_FIT_PARAMS = ['eccentricity', 'argument_of_periastron', 'period', 'inclination']
    OPTIONAL_FIT_PARAMS = ['period', 'primary_minimum_time', 'p__mass', 's__mass', 'semi_major_axis',
                           'asini', 'mass_ratio', 'p__t_eff', 's__t_eff', 'p__surface_potential',
                           's__surface_potential', 'p__gravity_darkening', 's__gravity_darkening', 'p__albedo',
                           's__albedo', 'additional_light', 'phase_shift']
    ALL_FIT_PARAMS = MANDATORY_FIT_PARAMS + OPTIONAL_FIT_PARAMS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs, LCFit.ALL_KWARGS, LCFit)
        utils.check_missing_kwargs(self.__class__.MANDATORY_KWARGS, kwargs, instance_of=self.__class__)

        self.light_curves = None
        self.fit_params = None
        self.ranges = None
        self.period = None

        # MCMC specific variables
        self.flat_chain = None
        self.normalization = None
        self.variable_labels = None

        self.plot = LCPlot(self)

        # values of properties
        logger.debug(f"setting properties of LCFit")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    def fit(self, X0, morphology='detached', method='least_squares', discretization=3, **kwargs):
        X0 = autils.transform_initial_values(X0)

        param_names = {key: value['value'] for key, value in X0.items()}
        utils.invalid_kwarg_checker(param_names, LCFit.ALL_FIT_PARAMS, LCFit)
        utils.check_missing_kwargs(LCFit.MANDATORY_FIT_PARAMS, param_names, instance_of=LCFit)

        if X0['period']['fixed'] is not True:
            logger.warning('Orbital period is expected to be known apriori the fit and thus it is considered fixed')
        period_dict = X0.pop('period')
        # period_dict = X0['period']
        self.period = period_dict['value']

        x_data, y_data, yerr = dict(), dict(), dict()
        for fltr, data in self.light_curves.items():
            x_data[fltr] = data.x_data
            y_data[fltr] = data.y_data
            yerr[fltr] = data.yerr

        if method == 'least_squares':
            fit_fn = lst_detached_fit if morphology == 'detached' else lst_over_contact_fit
            self.fit_params = fit_fn.fit(xs=x_data, ys=y_data, period=period_dict['value'], x0=X0, yerr=yerr,
                                         discretization=discretization, **kwargs)

        elif str(method).lower() in ['mcmc']:
            fit_fn = mcmc_detached_fit if morphology == 'detached' else mcmc_over_contact_fit
            self.fit_params = fit_fn.fit(xs=x_data, ys=y_data, period=period_dict['value'], x0=X0, yerr=yerr,
                                         discretization=discretization, **kwargs)
            self.flat_chain = fit_fn.last_sampler.get_chain(flat=True)
            self.normalization = fit_fn.last_normalization
            self.variable_labels = fit_fn.labels

        else:
            raise ValueError('Method name not recognised. Available methods `least_squares`, `mcmc` or `MCMC`.')

        self.fit_params.update({'period': {'value': self.period, 'unit': params.PARAMS_UNITS_MAP['period']}})
        return self.fit_params

    def load_chain(self, filename):
        """
        Function loads MCMC chain along with auxiliary data from json file created after each MCMC run

        :param filename: str; full name of the json file
        :return: Tuple[numpy.ndarray, list, Dict]; flattened mcmc chain, labels of variables in `flat_chain` columns,
        {var_name: (min_boundary, max_boundary), ...} dictionary of boundaries defined by user for each variable needed
        to reconstruct real values from normalized `flat_chain` array
        """
        return shared.load_mcmc_chain(self, filename)

    def store_parameters(self, parameters=None, filename=None):
        """
        Function converts model parameters to json compatibile format and stores model parameters.

        :param parameters: dict; {'name': {'value': numpy.ndarray, 'unit': Union[astropy.unit, str], ...}, ...}
        :param filename: str;
        :return:
        """
        parameters = copy(self.fit_params) if parameters is None else parameters
        parameters.update({'period': {'value': self.period, 'unit': units.PERIOD_UNIT}})

        for key, val in parameters.items():
            if 'unit' in val.keys():
                val['unit'] = u.Unit(val['unit']) if isinstance(val['unit'], str) else val['unit']
                val['unit'] = val['unit'].to_string()

        json_params = autils.convert_dict_to_json_format(parameters)
        with open(filename, 'w') as fl:
            json.dump(json_params, fl, separators=(',\n', ': '))

    def load_parameters(self, filename=None):
        """
        Function loads fitted parameters of given model.

        :param filename: str;
        :return: dict; {'name': {'value': numpy.ndarray, 'unit': Union[astropy.unit, str], ...}, ...}
        """
        with open(filename, 'r') as fl:
            prms = json.load(fl)

        prms = autils.convert_json_to_dict_format(prms)
        self.period = prms['period']['value']
        self.fit_params = prms

        return prms
