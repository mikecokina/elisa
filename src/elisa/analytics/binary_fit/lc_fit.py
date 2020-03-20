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

logger = getLogger('analytics.binary_fit.lc_fit')


class LCFit(object):
    MANDATORY_KWARGS = ['light_curves', ]
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    MANDATORY_FIT_PARAMS = ['eccentricity', 'argument_of_periastron', 'period', 'inclination']
    OPTIONAL_FIT_PARAMS = ['period', 'primary_minimum_time', 'p__mass', 's__mass', 'semi_major_axis',
                           'asini', 'mass_ratio', 'p__t_eff', 's__t_eff', 'p__surface_potential',
                           's__surface_potential', 'p__gravity_darkening', 's__gravity_darkening', 'p__albedo',
                           's__albedo', 'additional_light', 'phase_shift', 'p__spots', 's__spots', 'p__pulsations',
                           's__pulsations']
    ALL_FIT_PARAMS = MANDATORY_FIT_PARAMS + OPTIONAL_FIT_PARAMS

    FIT_PARAMS_COMBINATIONS = {
        'standard': ['p__mass', 's__mass', 'inclination', 'eccentricity', 'argument_of_periastron', 'period',
                     'primary_minimum_time', 'p__t_eff', 's__t_eff', 'p__surface_potential',
                     's__surface_potential', 'p__gravity_darkening', 's__gravity_darkening', 'p__albedo',
                     's__albedo', 'additional_light', 'phase_shift', 'p__spots', 's__spots', 'p__pulsations',
                           's__pulsations'],
        'community': ['mass_ratio', 'semi_major_axis', 'inclination', 'eccentricity', 'argument_of_periastron', 'period',
                      'primary_minimum_time', 'p__t_eff', 's__t_eff', 'p__surface_potential',
                      's__surface_potential', 'p__gravity_darkening', 's__gravity_darkening', 'p__albedo',
                      's__albedo', 'additional_light', 'phase_shift', 'p__spots', 's__spots', 'p__pulsations',
                           's__pulsations']
    }

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

    def fit(self, x0, morphology='detached', method='least_squares', discretization=3, interp_treshold=None, **kwargs):
        """
        Function solves an inverse task of inferring parameters of the eclipsing binary from the observed light curve.
        Least squares method is adopted from scipy.optimize.least_squares.
        MCMC uses emcee package to perform sampling.

        :param x0: Dict; initial state (metadata included) {param_name: {`value`: value, `unit`: astropy.unit, ...},
                   ...}
        :param morphology: str; `detached` (default) or `over-contact`
        :param method: str; 'least_squares` (default) or `mcmc`
        :param discretization: Union[int, float]; discretization factor of the primary component, default value: 3
        :param kwargs: dict; method-dependent
        :param interp_treshold: bool; Above this total number of datapoints, light curve will be interpolated using
            model containing `interp_treshold` equidistant points per epoch
        :**kwargs options for least_squares**: passes arguments of scipy.optimize.least_squares method except
                                               `fun`, `x0` and `bounds`
        :**kwargs options for mcmc**:
            * ** nwalkers (int) ** * - The number of walkers in the ensemble. Minimum is 2 * number of free parameters.
            * ** nsteps (int) ** * - The number of steps to run. Default is 1000.
            * ** initial_state (numpy.ndarray) ** * - The initial state or position vector made of free parameters with
                    shape (nwalkers, number of free parameters). The order of the parameters is specified by their order
                    in `x0`. Be aware that initial states should be supplied in normalized form (0, 1). For example, 0
                    means value of the parameter at `min` and 1.0 at `max` value in `x0`. By default, they are generated
                    randomly uniform distribution.
            * ** burn_in ** * - The number of steps to run to achieve equilibrium where useful sampling can start.
                    Default value is nsteps / 10.
            * ** percentiles ** * - List of percentiles used to create error results and confidence interval from MCMC
                    chain. Default value is [16, 50, 84] (1-sigma confidence interval)
        :return: dict; resulting parameters {param_name: {`value`: value, `unit`: astropy.unit, ...}, ...}
        """
        # treating a lack of `value` key in constrained parameters
        x0 = autils.prep_constrained_params(x0)

        shared.check_initial_param_validity(x0, LCFit.ALL_FIT_PARAMS, LCFit.MANDATORY_FIT_PARAMS)

        # transforming initial parameters to base units
        x0 = autils.transform_initial_values(x0)

        if x0['period']['fixed'] is not True:
            logger.warning('Orbital period is expected to be known apriori the fit and thus it is considered fixed')
        period_dict = x0.pop('period')
        self.period = period_dict['value']

        x_data, y_data, yerr = dict(), dict(), dict()
        for fltr, data in self.light_curves.items():
            x_data[fltr] = data.x_data
            y_data[fltr] = data.y_data
            yerr[fltr] = data.yerr

        if method == 'least_squares':
            fit_fn = lst_detached_fit if morphology == 'detached' else lst_over_contact_fit
            self.fit_params = fit_fn.fit(xs=x_data, ys=y_data, period=period_dict['value'], x0=x0, yerr=yerr,
                                         discretization=discretization, interp_treshold=interp_treshold, **kwargs)

        elif str(method).lower() in ['mcmc']:
            fit_fn = mcmc_detached_fit if morphology == 'detached' else mcmc_over_contact_fit
            self.fit_params = fit_fn.fit(xs=x_data, ys=y_data, period=period_dict['value'], x0=x0, yerr=yerr,
                                         discretization=discretization, interp_treshold=interp_treshold, **kwargs)
            self.flat_chain = fit_fn.last_sampler.get_chain(flat=True)
            self.normalization = fit_fn.last_normalization
            self.variable_labels = fit_fn.labels

        else:
            raise ValueError('Method name not recognised. Available methods `least_squares`, `mcmc` or `MCMC`.')

        self.fit_params.update({'period': {'value': self.period, 'unit': params.PARAMS_UNITS_MAP['period']}})
        return self.fit_params

    def load_chain(self, filename, discard=0):
        """
        Function loads MCMC chain along with auxiliary data from json file created after each MCMC run.

        :param discard: Discard the first `discard` steps in the chain as burn-in. (default: 0)
        :param filename: str; full name of the json file
        :return: Tuple[numpy.ndarray, list, Dict]; flattened mcmc chain, labels of variables in `flat_chain` columns,
                                                   {var_name: (min_boundary, max_boundary), ...} dictionary of 
                                                   boundaries defined by user for each variable needed
                                                   to reconstruct real values from normalized `flat_chain` array
        """
        return shared.load_mcmc_chain(self, filename, discard=discard)

    def store_parameters(self, parameters=None, filename=None):
        """
        Function converts model parameters to json compatibile format and stores model parameters.

        :param parameters: Dict; {'name': {'value': numpy.ndarray, 'unit': Union[astropy.unit, str], ...}, ...}
        :param filename: str;
        """
        parameters = copy(self.fit_params) if parameters is None else parameters
        parameters.update({'period': {'value': self.period, 'unit': units.PERIOD_UNIT}})

        parameters = autils.unify_unit_string_representation(parameters)

        json_params = autils.convert_dict_to_json_format(parameters)
        with open(filename, 'w') as f:
            json.dump(json_params, f, separators=(',\n', ': '))

    def load_parameters(self, filename=None):
        """
        Function loads fitted parameters of given model.

        :param filename: str;
        :return: Dict; {'name': {'value': numpy.ndarray, 'unit': Union[astropy.unit, str], ...}, ...}
        """
        with open(filename, 'r') as f:
            prms = json.load(f)

        prms = autils.convert_json_to_dict_format(prms)
        self.period = prms['period']['value']
        self.fit_params = prms

        return prms
