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
from elisa.analytics.binary_fit.shared import eval_constraint_in_dict
from elisa import settings

logger = getLogger('analytics.tasks')


class AnalyticsTask(metaclass=ABCMeta):
    """
    Abstract class defining fitting task. This structure aims to provide a framework for solving inverse problem for
    one object that embeds observed data and fitting methods and provides unified output from fitting methods along with
    capability to visualize the resulting fit.

    :param name: str; arbitrary name of instance
    :param data: Dict; data to bwe analyzed with the Analytics task instance
    """
    ID = 1
    LS_NAMES = ('least_squares', 'least_squares', 'ls', 'LS')
    MCMC_NAMES = ('mcmc', 'MCMC')
    ALLOWED_METHODS = LS_NAMES + MCMC_NAMES
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

    @classmethod
    def validate_method(cls, method):
        """
        Checking if the user supplied the correct name for the optimization method.

        :param method: str; name of the optimization method provided by the user
        :return: Union[None, ValueError];
        """
        if method not in cls.ALLOWED_METHODS:
            raise ValueError(f'Invalid fitting method. Use one of: {", ".join(cls.ALLOWED_METHODS)}')

    def load_result(self, filename):
        """
        Function loads a JSON file containing model parameters and stores it as an attribute of AnalyticsTask fitting
        instance. This is useful if you want to examine already calculated results using functionality provided by the
        AnalyticsTask instances (e.g: LCBinaryAnalyticsTask, RVBinaryAnalyticsTask, etc.). I also returns model
        parameters in standard dict (JSON) format.

        :param filename: str;
        :return: Dict; model parameters in a standardized format
        """
        self.fit_cls.load_result(filename)
        return self.fit_cls.get_result()

    def save_result(self, filename):
        """
        Save result as JSON file.

        :param filename: str; path to file
        """
        self.fit_cls.save_result(filename)

    def set_result(self, result):
        """
        Set model parameters in dictionary (JSON format) as an attribute of AnalyticsTask fitting instance. This is
        useful if you want to examine already calculated results using functionality provided by the AnalyticsTask
        instances (e.g: LCBinaryAnalyticsTask, RVBinaryAnalyticsTask, etc.).

        :param result: Dict; model parameters in JSON format
        """
        self.fit_cls.set_result(result)

    def get_result(self):
        """
        Returns model parameters in standard dict (JSON) format.

        :return: Dict; model parameters in a standardized format
        """
        return self.fit_cls.get_result()

    def result_summary(self, filename=None, **kwargs):
        """
        Function produces detailed summary of the current fitting task with the possibility to propagate
        uncertainties of the fitted binary model parameters if MCMC method was used and `propagate_errors` is True.

        :param filename: path where to store summary
        :param kwargs: Dict;
        :**kwargs options for MCMC method**:
            * :propagate_errors: bool; errors of fitted parameters will be propagated to the rest of EB
                                       parameters (takes a while to calculate)
            * :percentiles: List; percentiles used to evaluate confidence intervals from posterior
                                  distribution of EB parameters in MCMC chain . Used only when if
                                  `propagate_errors` is True.
            * :dimensionless_radii: bool; if True (default), radii are provided in SMA, otherwise solRad are used,
                                          available only for light curve fitting

        :return:
        """
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

        if self.method not in self.MCMC_NAMES:
            raise ValueError('Load chain method can be used only with mcmc task.')
        self.fit_cls.load_chain(filename, discard, percentiles)
        return self

    def filter_chain(self, **boundaries):
        """
        Filtering MCMC chain down to given parameter intervals. This function is useful in case of bimodal distribution
        of the MCMC chain.

        :param boundaries: Dict; dictionary of boundaries e.g. {'primary@te_ff': (5000, 6000), other parameters ...}
        :return: numpy.array; filtered flat chain
        """
        if self.method not in self.MCMC_NAMES:
            raise ValueError('Filter chain method can be used only with mcmc task.')
        self.fit_cls.filter_chain(**boundaries)

    def fit(self, x0: Union[Dict, parameters.BinaryInitialParameters], **kwargs):
        """
        Function solves an inverse task of inferring parameters of the eclipsing binary from the observed light curve.
        Least squares method is adopted from scipy.optimize.least_squares.
        MCMC uses emcee package to perform sampling.

        :param x0: Dict; initial state of the sampler, model parameters in standard JSON format
        :param kwargs: Dict; method-dependent
        :**additional light curve kwargs**:
            * **morphology** * - str - `detached` or `over-contact`
            * **interp_treshold** * - int - Above this total number of datapoints, light curve will be interpolated
                                            using model containing `interp_treshold` equidistant points per epoch
            * **discretization** * - Union[int, float] - discretization factor of the primary component, default: 5
            * **samples** * - Union[str, List]; 'uniform' (equidistant sampling in phase), 'adaptive'
                                                (equidistant sampling on curve) or list with phases in (0, 1) interval

        :**kwargs options for least_squares**: passes arguments of scipy.optimize.least_squares method except
                                               `fun`, `x0` and `bounds`

        :**kwargs options for MCMC method**:
            * **nwalkers (int)** * - The number of walkers in the ensemble. Minimum is 2 * number of free parameters.
            * **nsteps (int)** * - The number of steps to run. Default is 1000.
            * **initial_state (numpy.ndarray)** * - The initial state or position vector made of free parameters with
                    shape (nwalkers, number of free parameters). The order of the parameters is specified by their order
                    in `x0`. Be aware that initial states should be supplied in normalized form (0, 1). For example, 0
                    means value of the parameter at `min` and 1.0 at `max` value in `x0`. By default, they are generated
                    randomly uniform distribution.
            * **burn_in** * - The expected number of steps to run to achieve equilibrium where useful sampling can
                              start.Default value is nsteps / 10.
            * **progress** * - bool - display the progress bar of the sampling
            * **percentiles** * - List of percentiles used to create error results and confidence interval from MCMC
                                  chain. Default value is [16, 50, 84] (1-sigma confidence interval)
            * **save** * - bool - save chain
            * **fit_id** * - str - identificator or location of stored chain

        :return: Dict; resulting parameters {param_name: {`value`: value, `unit`: astropy.unit, ...}, ...}
        """
        if isinstance(x0, dict):
            x0 = parameters.BinaryInitialParameters(**x0)
        return self.fit_cls.fit(x0, data=self.data, **kwargs)

    def coefficient_of_determination(self, model_parameters=None, discretization=5, interpolation_treshold=None):
        """
        Function returns R^2 for given model parameters and observed data.

        :param model_parameters: Dict; if None, get_result() is called
        :param discretization: float;
        :param interpolation_treshold: int; if None settings.MAX_CURVE_DATA_POINTS is used
        :return: float;
        """
        model_parameters = self.get_result() if model_parameters is None else model_parameters
        interpolation_treshold = settings.MAX_CURVE_DATA_POINTS \
            if interpolation_treshold is None else interpolation_treshold

        r2 = self.fit_cls.coefficient_of_determination(
            model_parameters,
            self.data,
            discretization,
            interpolation_treshold
        )
        model_parameters['r_squared'] = r2
        self.set_result(model_parameters)

        return r2

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
    """
    Fitting task class aimed to fit light curves of eclipsing binary stars.
    """
    FIT_CLS = None
    PLOT_CLS = None
    FIT_PARAMS_COMBINATIONS = json.dumps({
        "standard": {"system": ["inclination", "eccentricity", "argument_of_periastron", "period",
                                "primary_minimum_time", "additional_light", "phase_shift"],
                     "primary": ["mass", "t_eff", "surface_potential", "gravity_darkening", "albedo",
                                 "synchronicity", "metallicity", "spots", "pulsations"],
                     "secondary": ["mass", "t_eff", "surface_potential", "gravity_darkening", "albedo",
                                   "synchronicity", "metallicity", "spots", "pulsations"],
                     "nuisance": ['ln_f']
                     },
        "community": {"system": ["inclination", "eccentricity", "argument_of_periastron", "period", "semi_major_axis",
                                 "primary_minimum_time", "additional_light", "phase_shift" "mass_ratio"],
                      "primary": ["t_eff", "surface_potential", "gravity_darkening", "albedo",
                                  "synchronicity", "metallicity", "spots", "pulsations"],
                      "secondary": ["t_eff", "surface_potential", "gravity_darkening", "albedo",
                                    "synchronicity", "metallicity", "spots", "pulsations"],
                      "nuisance": ['ln_f']
                      },
        "spots": ["longitude", "latituded", "angular_radius", "temperature_factor"],
        "pulsations": ["l", "m", "amplitude", "frequency", "start_phase", "mode_axis_theta", "mode_axis_phi"]
    }, indent=4)
    TRANSFORM_PROPERTIES_CLS = transform.LCBinaryAnalyticsProperties

    def __init__(self, method, expected_morphology='detached', name=None, **kwargs):
        self.validate_method(method)
        if method in self.MCMC_NAMES:
            self.__class__.FIT_CLS = lambda: lc_fit.LCFitMCMC(morphology=expected_morphology)
            self.__class__.PLOT_CLS = LCPlotMCMC
        elif method in self.LS_NAMES:
            self.__class__.FIT_CLS = lambda: lc_fit.LCFitLeastSquares(morphology=expected_morphology)
            self.__class__.PLOT_CLS = LCPlotLsqr
        super().__init__(method, name, **kwargs)


class RVBinaryAnalyticsTask(AnalyticsTask):
    """
    Fitting task class aimed to fit radial velocity (RV) curves of eclipsing binary stars. For now, the method
    support only kinematic method for calculation of radial velocities (regarding stars as point masses).
    """
    FIT_CLS = None
    PLOT_CLS = None
    FIT_PARAMS_COMBINATIONS = json.dumps({

        'community': {
            'system': ['mass_ratio', 'asini', 'eccentricity', 'argument_of_periastron',
                       'gamma', 'period', 'primary_minimum_time'],
            'nuisance': ['ln_f']
        },

        'standard': {
            'primary': ['mass'],
            'secondary': ['mass'],
            'system': ['inclination', 'eccentricity', 'argument_of_periastron',
                       'gamma', 'period', 'primary_minimum_time'],
            'nuisance': ['ln_f']
        }
    }, indent=4)
    TRANSFORM_PROPERTIES_CLS = transform.RVBinaryAnalyticsTask

    def __init__(self, method, name=None, **kwargs):
        self.validate_method(method)
        if method in self.MCMC_NAMES:
            self.__class__.FIT_CLS = rv_fit.RVFitMCMC
            self.__class__.PLOT_CLS = RVPlotMCMC
        elif method in self.LS_NAMES:
            self.__class__.FIT_CLS = rv_fit.RVFitLeastSquares
            self.__class__.PLOT_CLS = RVPlotLsqr
        super().__init__(method, name, **kwargs)
