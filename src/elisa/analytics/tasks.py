import json
from abc import ABCMeta
from typing import Union, Dict

from . import transform
from . params import bonds, parameters
from . binary_fit import rv_fit, lc_fit
from . binary_fit.plot import (
    RVPlotMCMC, RVPlotLsqr,
    LCPlotLsqr, LCPlotMCMC
)
from .. import utils
from .. logger import getLogger

logger = getLogger('analytics.tasks')


class AnalyticsTask(metaclass=ABCMeta):
    """
    Abstract class defining fitting task. This structure aims to provide a framework for solving inverse problem inside
    one object that embeds observed data and fitting methods and provides unified output from fitting methods along with
    capability to visualize the fit.

    :param name: str; arbitrary name of instance
    :param data: Dict;
    """

    ID = 1
    ALLOWED_METHODS = ('least_squares', 'mcmc')
    MANDATORY_KWARGS = ["data", ]
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS
    CONSTRAINT_OPERATORS = bonds.ALLOWED_CONSTRAINT_METHODS + bonds.ALLOWED_CONSTRAINT_CHARS
    FIT_CLS = None
    PLOT_CLS = None
    TRANSFORM_PROPERTIES_CLS = None

    def __init__(self, method, name=None, **kwargs):
        self.data = dict()
        self.method = method

        if utils.is_empty(name):
            self.name = str(AnalyticsTask.ID)
            logger.debug(f"name of class instance {self.__class__.__name__} set to {self.name}")
            self.__class__.ID += 1
        else:
            self.name = str(name)

        utils.invalid_kwarg_checker(kwargs, self.ALL_KWARGS, self.__class__)
        utils.check_missing_kwargs(self.MANDATORY_KWARGS, kwargs, instance_of=AnalyticsTask)
        kwargs = self.transform_input(**kwargs)
        self.init_properties(**kwargs)

        logger.debug(f'initializing fitting module in class instance '
                     f'{self.__class__.__name__} / {self.name}.')
        self.fit_cls = self.__class__.FIT_CLS()
        self.plot = self.__class__.PLOT_CLS(instance=self.fit_cls, data=self.data)

    @staticmethod
    def validate_method(method):
        if method not in ['least_squares', 'mcmc']:
            raise ValueError(f'Invalid fitting method. Use one of: {", ".join(AnalyticsTask.ALLOWED_METHODS)}')

    def load_result(self, filename):
        self.fit_cls.load_result(filename)

    def save_result(self, filename):
        self.fit_cls.save_result(filename)

    def set_result(self, result):
        self.fit_cls.set_result(result)
        return self

    def get_result(self):
        return self.fit_cls.get_result()

    def result_summary(self, filename=None, **kwargs):
        self.fit_cls.fit_summary(filename, **kwargs)

    fit_summary = result_summary

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

        if self.method not in ['mcmc']:
            raise IOError('Load chain method can be used only with mcmc task.')
        self.fit_cls.load_chain(filename, discard, percentiles)
        return self

    def fit(self, x0: Union[Dict, parameters.BinaryInitialParameters], **kwargs):
        """
        Function solves an inverse task of inferring parameters of the eclipsing binary from the observed light curve.
        Least squares method is adopted from scipy.optimize.least_squares.
        MCMC uses emcee package to perform sampling.

        :param x0: Dict; initial state (metadata included) {param_name: {`value`: value,
                                                           `unit`: astropy.unit, ...}, ...}
        :param kwargs: Dict; method-dependent
        :**additional light curve kwargs**:
            * **morphology** * - str - `detached` or `over-contact`
            * **interp_treshold** * - bool - Above this total number of datapoints, light curve will be interpolated
                                             using model containing `interp_treshold` equidistant points per epoch
            * **discretization** * - Union[int, float] - discretization factor of the primary component, default: 5

        :**kwargs options for least_squares**: passes arguments of scipy.optimize.least_squares method except
                                               `fun`, `x0` and `bounds`

        :**kwargs options for mcmc**:
            * **nwalkers (int)** * - The number of walkers in the ensemble. Minimum is 2 * number of free parameters.
            * **nsteps (int)** * - The number of steps to run. Default is 1000.
            * **initial_state (numpy.ndarray)** * - The initial state or position vector made of free parameters with
                    shape (nwalkers, number of free parameters). The order of the parameters is specified by their order
                    in `x0`. Be aware that initial states should be supplied in normalized form (0, 1). For example, 0
                    means value of the parameter at `min` and 1.0 at `max` value in `x0`. By default, they are generated
                    randomly uniform distribution.
            * **burn_in** * - The number of steps to run to achieve equilibrium where useful sampling can start.
                    Default value is nsteps / 10.
            * **percentiles** * - List of percentiles used to create error results and confidence interval from MCMC
                                  chain. Default value is [16, 50, 84] (1-sigma confidence interval)
            * **save** * - bool - save chain
            * **fit_id** * - str - identificator of stored chain
        :return: Dict; resulting parameters {param_name: {`value`: value, `unit`: astropy.unit, ...}, ...}
        """
        if isinstance(x0, dict):
            x0 = parameters.BinaryInitialParameters(**x0)
        return self.fit_cls.fit(x0, data=self.data, **kwargs)

    @classmethod
    def transform_input(cls, **kwargs):
        """
        Transform and validate input kwargs.

        :param kwargs: Dict;
        :return: Dict;
        """
        return cls.TRANSFORM_PROPERTIES_CLS.transform_input(**kwargs)

    def init_properties(self, **kwargs):
        """
        Setup system properties from input.

        :param kwargs: Dict; all supplied input properties
        """
        logger.debug(f"initialising properties of analytics task {self.name}")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])


class LCBinaryAnalyticsTask(AnalyticsTask):
    FIT_CLS = None
    PLOT_CLS = None
    FIT_PARAMS_COMBINATIONS = json.dumps({
        "standard": {"system": ["inclination", "eccentricity", "argument_of_periastron", "period",
                                "primary_minimum_time", "additional_light", "phase_shift"],
                     "primary": ["mass", "t_eff", "surface_potential", "gravity_darkening", "albedo",
                                 "synchronicity", "metallicity", "spots", "pulsations"],
                     "secondary": ["mass", "t_eff", "surface_potential", "gravity_darkening", "albedo",
                                   "synchronicity", "metallicity", "spots", "pulsations"]
                     },
        "community": {"system": ["inclination", "eccentricity", "argument_of_periastron", "period", "semi_major_axis",
                                 "primary_minimum_time", "additional_light", "phase_shift" "mass_ratio"],
                      "primary": ["t_eff", "surface_potential", "gravity_darkening", "albedo",
                                  "synchronicity", "metallicity", "spots", "pulsations"],
                      "secondary": ["t_eff", "surface_potential", "gravity_darkening", "albedo",
                                    "synchronicity", "metallicity", "spots", "pulsations"]
                      },
        "spots": ["longitude", "latituded", "angular_radius", "temperature_factor"],
        "pulsations": ["l", "m", "amplitude", "frequency", "start_phase", "mode_axis_theta", "mode_axis_phi"]
    }, indent=4)
    TRANSFORM_PROPERTIES_CLS = transform.LCBinaryAnalyticsProperties

    def __init__(self, method, expected_morphology='detached', name=None, **kwargs):
        self.validate_method(method)
        if method in 'mcmc':
            self.__class__.FIT_CLS = lambda: lc_fit.LCFitMCMC(morphology=expected_morphology)
            self.__class__.PLOT_CLS = LCPlotMCMC
        elif method in 'least_squares':
            self.__class__.FIT_CLS = lambda: lc_fit.LCFitLeastSquares(morphology=expected_morphology)
            self.__class__.PLOT_CLS = LCPlotLsqr
        super().__init__(method, name, **kwargs)


class RVBinaryAnalyticsTask(AnalyticsTask):
    FIT_CLS = None
    PLOT_CLS = None
    FIT_PARAMS_COMBINATIONS = json.dumps({

        'community': {
            'system': ['mass_ratio', 'asini', 'eccentricity', 'argument_of_periastron',
                       'gamma', 'period', 'primary_minimum_time']
        },

        'standard': {
            'primary': ['mass'],
            'secondary': ['mass'],
            'system': ['inclination', 'eccentricity', 'argument_of_periastron',
                       'gamma', 'period', 'primary_minimum_time'],
        }
    }, indent=4)
    TRANSFORM_PROPERTIES_CLS = transform.RVBinaryAnalyticsTask

    def __init__(self, method, name=None, **kwargs):
        self.validate_method(method)
        if method in 'mcmc':
            self.__class__.FIT_CLS = rv_fit.RVFitMCMC
            self.__class__.PLOT_CLS = RVPlotMCMC
        elif method in 'least_squares':
            self.__class__.FIT_CLS = rv_fit.RVFitLeastSquares
            self.__class__.PLOT_CLS = RVPlotLsqr
        super().__init__(method, name, **kwargs)
