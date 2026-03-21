"""Framework for solving inverse problems with observed data and fitting methods."""
from __future__ import annotations

import json
from abc import ABCMeta
from typing import TYPE_CHECKING, Any

from elisa import settings, utils
from elisa.analytics import transform
from elisa.analytics.binary_fit import lc_fit, rv_fit
from elisa.analytics.binary_fit.plot import (
    LCPlotLsqr,
    LCPlotMCMC,
    RVPlotLsqr,
    RVPlotMCMC,
)
from elisa.analytics.params import bonds, parameters
from elisa.logger import getLogger

if TYPE_CHECKING:
    from elisa.types import Float

logger = getLogger("analytics.tasks")


class AnalyticsTask(metaclass=ABCMeta):  # noqa: B024
    """Abstract base class defining fitting task framework.

    This structure provides a framework for solving inverse problems by
    embedding observed data and fitting methods, providing unified output
    from fitting methods along with visualization capabilities.

    :param method: Name of the optimization/fitting method
    :type method: str
    :param name: Arbitrary name of instance
    :type name: str | None
    :param kwargs: Additional keyword arguments for configuration
    :type kwargs: dict

    **kwargs options:**

    - **data** (:class:`dict`) - Data to be analyzed with the Analytics task instance
    - **atmosphere_model** (:class:`str`) - Atmosphere model configuration
    - **limb_darkening_coefficients** (:class:`dict`) - Limb darkening coefficients

    **examples:**

    Basic usage with light curves fitting::

        task = LCBinaryAnalyticsTask(
            method='mcmc',
            data={'Generic.Bessell.V': lc_data},
            atmosphere_model='blackbody',
            limb_darkening_coefficients={'bolometric': [0.5, 0.3]}
        )
        result = task.fit(initial_params)
    """

    ID: int = 1
    LS_NAMES: tuple[str, ...] = ("least_squares", "least_squares", "ls", "LS")
    MCMC_NAMES: tuple[str, ...] = ("mcmc", "MCMC")
    ALLOWED_METHODS: tuple[str, ...] = LS_NAMES + MCMC_NAMES
    MANDATORY_KWARGS: tuple[str, ...] = ("data",)
    OPTIONAL_KWARGS: tuple[str, ...] = ("atmosphere_model", "limb_darkening_coefficients")
    ALL_KWARGS: tuple[str, ...] = MANDATORY_KWARGS + OPTIONAL_KWARGS
    CONSTRAINT_OPERATORS: tuple = (
        bonds.ALLOWED_CONSTRAINT_METHODS + bonds.ALLOWED_CONSTRAINT_CHARS
    )
    FIT_CLS: Any = None
    PLOT_CLS: Any = None
    TRANSFORM_PROPERTIES_CLS: Any = None

    def __init__(
        self,
        method: str,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize analytics task with method and configuration.

        :param method: Name of the fitting method (least_squares or mcmc)
        :type method: str
        :param name: Instance name (defaults to auto-generated ID if not provided)
        :type name: str | None
        :param kwargs: Additional configuration options
        :type kwargs: dict
        """
        self.data: dict = {}
        self.method: str = method
        self.validate_method(method)

        if utils.is_empty(name):
            self.name: str = str(AnalyticsTask.ID)
            logger.debug(
                "name of class instance %s set to %s",
                self.__class__.__name__,
                self.name,
            )
            self.__class__.ID += 1
        else:
            self.name = str(name)

        utils.invalid_kwarg_checker(kwargs, self.ALL_KWARGS, self.__class__)
        utils.check_missing_kwargs(
            self.MANDATORY_KWARGS, kwargs, instance_of=AnalyticsTask,
        )
        kwargs = self.transform_input(**kwargs)
        self.init_properties(**kwargs)

        logger.debug(
            "initializing fitting module in class instance %s / %s",
            self.__class__.__name__,
            self.name,
        )
        # noinspection PyCallingNonCallable
        self.fit_cls: Any = self.__class__.FIT_CLS()
        # noinspection PyCallingNonCallable
        self.plot: Any = self.__class__.PLOT_CLS(instance=self.fit_cls, data=self.data)

    @classmethod
    def validate_method(cls, method: str) -> None:
        """Validate if user supplied correct optimization method name.

        Checks if the provided method name is one of the allowed fitting methods
        (the least squares or MCMC variants).

        :param method: Name of the optimization method provided by the user
        :type method: str
        :raises ValueError: If method name is not in ALLOWED_METHODS
        """
        if method not in cls.ALLOWED_METHODS:
            error_msg: str = (
                f"Invalid fitting method. Use one of: {', '.join(cls.ALLOWED_METHODS)}"
            )
            raise ValueError(error_msg)

    def load_result(self, filename: str, *, autofill_sma: bool = False) -> dict:
        """Load model parameters from JSON file.

        Function loads a JSON file containing model parameters and stores it as an
        attribute of AnalyticsTask fitting instance. This is useful if you want to
        examine already calculated results using functionality provided by the
        AnalyticsTask instances (e.g: LCBinaryAnalyticsTask, RVBinaryAnalyticsTask, etc.).
        It also returns model parameters in standard dict (JSON) format.

        :param filename: Path to JSON file containing model parameters
        :type filename: str
        :param autofill_sma: If True, the semi-major axis will be autofilled to fitting
                             parameters if absent (default: False)
        :type autofill_sma: bool
        :return: Model parameters in standardized format
        :rtype: dict
        """
        self.fit_cls.load_result(filename, autofill_sma=autofill_sma)
        return self.fit_cls.get_result()

    def save_result(self, filename: str) -> None:
        """Save fitting result as JSON file.

        :param filename: Path to output file where result will be saved
        :type filename: str
        """
        self.fit_cls.save_result(filename)

    def set_result(self, result: dict, *, autofill_sma: bool = False) -> None:
        """Set model parameters from dictionary in JSON format.

        Set model parameters in dictionary (JSON format) as an attribute of AnalyticsTask
        fitting instance. This is useful if you want to examine already calculated results
        using functionality provided by the AnalyticsTask instances
        (e.g: LCBinaryAnalyticsTask, RVBinaryAnalyticsTask, etc.).

        :param result: Model parameters in JSON format
        :type result: dict
        :param autofill_sma: If True, the semi-major axis will be autofilled to fitting
                             parameters if absent (default: False)
        :type autofill_sma: bool
        """
        self.fit_cls.set_result(result, autofill_sma=autofill_sma)

    def get_result(self) -> dict:
        """Return model parameters in standard dict (JSON) format.

        :return: Model parameters in standardized format
        :rtype: dict
        """
        return self.fit_cls.get_result()

    def result_summary(self, filename: str | None = None, **kwargs: Any) -> None:
        """Produce detailed summary of current fitting task.

        Function produces detailed summary of the current fitting task with the
        possibility to propagate uncertainties of the fitted binary model parameters
        if MCMC method was used and `propagate_errors` is True.

        :param filename: Path where to store summary (if None, prints to console)
        :type filename: str | None
        :param kwargs: Method-dependent options
        :type kwargs: dict

        **kwargs options for MCMC method:**

        - **propagate_errors** (:class:`bool`) - Errors of fitted parameters will be
          propagated to the rest of EB parameters (takes a while to calculate)
        - **percentiles** (:class:`list`) - Percentiles used to evaluate confidence
          intervals from posterior distribution of EB parameters in MCMC chain.
          Used only when `propagate_errors` is True. Default: [16, 50, 84]
        - **dimensionless_radii** (:class:`bool`) - If True (default), radii are
          provided in SMA, otherwise in solar radii. Available only for light
          curve fitting.
        """
        self.fit_cls.fit_summary(filename, **kwargs)

    fit_summary = result_summary

    def load_chain(
        self,
        filename: str,
        *,
        discard: int = 0,
        percentiles: list | None = None,
    ) -> AnalyticsTask:
        """Load MCMC chain along with auxiliary data from JSON file.

        Function loads MCMC chain along with auxiliary data from json file created
        after each MCMC run. The chain can be optionally filtered by discarding
        burn-in steps and computing percentiles for confidence intervals.

        :param filename: Full name/path of the JSON file containing MCMC chain
        :type filename: str
        :param discard: Number of first steps to discard as burn-in (default: 0)
        :type discard: int
        :param percentiles: List of percentiles used to create error results and
                           confidence interval from MCMC chain
        :type percentiles: list | None
        :return: Self for method chaining
        :rtype: AnalyticsTask
        :raises ValueError: If method is not MCMC
        """
        if self.method not in self.MCMC_NAMES:
            error_msg: str = "load_chain method can be used only with MCMC task."
            raise ValueError(error_msg)
        self.fit_cls.load_chain(filename, discard, percentiles)
        return self

    def filter_chain(self, **boundaries: tuple[Float, Float]) -> None:
        """Filter MCMC chain down to given parameter intervals.

        Filtering MCMC chain down to given parameter intervals. This function is
        useful in case of bimodal distribution of the MCMC chain. Allows restricting
        the posterior distribution to specific parameter ranges.

        :param boundaries: Dictionary of param  eter boundaries in format
                          {param_name: (min_value, max_value), ...}
                          Example: {'primary@t_eff': (5000, 6000), ...}
        :type boundaries: dict
        :raises ValueError: If method is not MCMC
        """
        if self.method not in self.MCMC_NAMES:
            error_msg: str = "filter_chain method can be used only with MCMC task."
            raise ValueError(error_msg)
        self.fit_cls.filter_chain(**boundaries)

    def fit(
        self,
        x0: dict | parameters.BinaryInitialParameters,
        **kwargs: Any,
    ) -> dict:
        """Solve inverse problem of inferring binary parameters from observed data.

        Function solves an inverse task of inferring parameters of the eclipsing
        binary from the observed light curve or radial velocities. Least squares
        method is adopted from scipy.optimize.least_squares. MCMC uses emcee
        package to perform sampling.

        :param x0: Initial state or model parameters in standard JSON format.
                   Can be a dictionary or BinaryInitialParameters instance
        :type x0: dict | BinaryInitialParameters
        :param kwargs: Method-dependent options
        :type kwargs: dict
        :return: Resulting parameters in format
                 {param_name: {`value`: value, `unit`: astropy.unit, ...}, ...}
        :rtype: dict

        **additional light curve kwargs:**

        - **morphology** (:class:`str`) - `detached` or `over-contact`
        - **interp_threshold** (:class:`int`) - Above this total number of datapoints,
          light curve will be interpolated using model containing `interp_threshold`
          equidistant points per epoch
        - **discretization** (:class:`int` | :class:`float`) - Discretization factor
          of the primary component (default: 5)
        - **samples** (:class:`str` | :class:`list`) - 'uniform' (equidistant sampling
          in phase), 'adaptive' (equidistant sampling on curve) or list with phases
          in (0, 1) interval

        **kwargs options for least_squares:**

        Passes arguments of scipy.optimize.least_squares method except `fun`, `x0`,
        and `bounds`

        **kwargs options for MCMC method:**

        - **nwalkers** (:class:`int`) - The number of walkers in the ensemble.
          Minimum is 2 * number of free parameters.
        - **nsteps** (:class:`int`) - The number of steps to run (default: 1000)
        - **initial_state** (:class:`ndarray`) - The initial state or position vector
          made of free parameters with shape (nwalkers, number of free parameters).
          The order is specified by parameter order in `x0`. Initial states should
          be supplied in normalized form (0, 1). For example, 0 means value at `min`
          and 1.0 at `max` value in `x0`. By default, randomly generated.
        - **burn_in** (:class:`int`) - Expected number of steps to achieve equilibrium
          where useful sampling can start (default: nsteps / 10)
        - **progress** (:class:`bool`) - Display the progress bar of the sampling
        - **percentiles** (:class:`list`) - Percentiles used to create error results
          and confidence interval from MCMC chain (default: [16, 50, 84])
        - **save** (:class:`bool`) - Save chain to file
        - **fit_id** (:class:`str`) - Identifier or location of stored chain
        - **samples** (:class:`str` | :class:`list`) - 'uniform' (equidistant sampling
          in phase), 'adaptive' (equidistant sampling on curve) or list with phases
          in (0, 1) interval
        """
        if isinstance(x0, dict):
            x0 = parameters.BinaryInitialParameters(**x0)
        return self.fit_cls.fit(x0, data=self.data, **kwargs)

    def coefficient_of_determination(
        self,
        model_parameters: dict | None = None,
        *,
        discretization: int = 5,
        interpolation_treshold: int | None = None,
    ) -> Float:
        """Return R² (coefficient of determination) for model and observed data.

        Function returns R² for given model parameters and observed data.
        If successful, the calculated R² value is stored in the result dictionary.

        :param model_parameters: Model parameters (if None, get_result() is called)
        :type model_parameters: dict | None
        :param discretization: Discretization factor for model calculation
        :type discretization: int
        :param interpolation_treshold: Maximum curve datapoints threshold.
                                        If None, settings.MAX_CURVE_DATAPOINTS is used
        :type interpolation_treshold: int | None
        :return: R² value (coefficient of determination)
        :rtype: float
        """
        model_parameters = (
            self.get_result() if model_parameters is None else model_parameters
        )
        interpolation_treshold = (
            settings.MAX_CURVE_DATAPOINTS
            if interpolation_treshold is None
            else interpolation_treshold
        )

        r2: Float = self.fit_cls.coefficient_of_determination(
            model_parameters,
            self.data,
            discretization,
            interpolation_treshold,
        )
        model_parameters["r_squared"] = r2
        self.set_result(model_parameters)

        return r2

    @classmethod
    def transform_input(cls, **kwargs: Any) -> dict:
        """Transform and validate input kwargs.

        :param kwargs: Input keyword arguments
        :type kwargs: dict
        :return: Transformed keyword arguments
        :rtype: dict
        """
        return cls.TRANSFORM_PROPERTIES_CLS.transform_input(**kwargs)

    def init_properties(self, **kwargs: Any) -> None:
        """Initialize system properties from input arguments.

        Setup system properties from input by setting all kwargs as instance
        attributes. This allows dynamic property assignment based on provided
        configuration.

        :param kwargs: All supplied input properties
        :type kwargs: dict
        """
        logger.debug(
            "initialising properties of analytics task %s",
            self.name,
        )
        for kwarg, value in kwargs.items():
            setattr(self, kwarg, value)


class LCBinaryAnalyticsTask(AnalyticsTask):
    """Fitting task class for light curves of eclipsing binary stars.

    This class specializes AnalyticsTask for fitting light curves of eclipsing
    binary systems. It supports various morphologies (detached, semi-detached,
    over-contact) and provides comprehensive fitting parameter combinations for
    standard and community analysis approaches.

    **class attributes:**

    - **FIT_PARAMS_COMBINATIONS** - JSON format string containing available
      parameter sets for fitting
    - **TRANSFORM_PROPERTIES_CLS** - Class for transforming and validating input
    """

    FIT_CLS: Any = None
    PLOT_CLS: Any = None
    FIT_PARAMS_COMBINATIONS: str = json.dumps(
        {
            "standard": {
                "system": [
                    "inclination",
                    "eccentricity",
                    "argument_of_periastron",
                    "period",
                    "primary_minimum_time",
                    "additional_light",
                    "phase_shift",
                ],
                "primary": [
                    "mass",
                    "t_eff",
                    "surface_potential",
                    "gravity_darkening",
                    "albedo",
                    "synchronicity",
                    "metallicity",
                    "spots",
                    "pulsations",
                ],
                "secondary": [
                    "mass",
                    "t_eff",
                    "surface_potential",
                    "gravity_darkening",
                    "albedo",
                    "synchronicity",
                    "metallicity",
                    "spots",
                    "pulsations",
                ],
                "nuisance": ["ln_f"],
            },
            "community": {
                "system": [
                    "inclination",
                    "eccentricity",
                    "argument_of_periastron",
                    "period",
                    "semi_major_axis",
                    "primary_minimum_time",
                    "additional_light",
                    "phase_shift",
                    "mass_ratio",
                ],
                "primary": [
                    "t_eff",
                    "surface_potential",
                    "gravity_darkening",
                    "albedo",
                    "synchronicity",
                    "metallicity",
                    "spots",
                    "pulsations",
                ],
                "secondary": [
                    "t_eff",
                    "surface_potential",
                    "gravity_darkening",
                    "albedo",
                    "synchronicity",
                    "metallicity",
                    "spots",
                    "pulsations",
                ],
                "nuisance": ["ln_f"],
            },
            "spots": [
                "longitude",
                "latituded",
                "angular_radius",
                "temperature_factor",
            ],
            "pulsations": [
                "l",
                "m",
                "amplitude",
                "frequency",
                "start_phase",
                "mode_axis_theta",
                "mode_axis_phi",
            ],
        },
        indent=4,
    )
    TRANSFORM_PROPERTIES_CLS: Any = transform.LCBinaryAnalyticsProperties

    def __init__(
        self,
        method: str,
        *,
        name: str | None = None,
        expected_morphology: str = "detached",
        atmosphere_models: str | None = None,
        limb_darkening_coefficients: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize light curve fitting task.

        Sets up LC fitting task with specified fitting method and configuration.
        Initializes appropriate fitting class (MCMC or Least Squares) and plotting
        class based on the provided method.

        :param method: Fitting method name (least_squares or mcmc)
        :type method: str
        :param name: Instance name (auto-generated if not provided)
        :type name: str | None
        :param expected_morphology: System morphology (default: 'detached').
                                    Options: 'detached', 'semi-detached', 'over-contact'
        :type expected_morphology: str
        :param atmosphere_models: Atmosphere model configuration
        :type atmosphere_models: str | None
        :param limb_darkening_coefficients: Limb darkening coefficient configuration
        :type limb_darkening_coefficients: dict | None
        :param kwargs: Additional configuration options (data, atmosphere_model, etc.)
        :type kwargs: dict
        """
        self.validate_method(method)
        if method in self.MCMC_NAMES:
            def _create_lc_fit_mcmc() -> lc_fit.LCFitMCMC:
                return lc_fit.LCFitMCMC(
                    morphology=expected_morphology,
                    atmosphere_model=atmosphere_models,
                    limb_darkening_coefficients=limb_darkening_coefficients,
                )
            self.__class__.FIT_CLS = _create_lc_fit_mcmc
            self.__class__.PLOT_CLS = LCPlotMCMC
        elif method in self.LS_NAMES:
            def _create_lc_fit_least_squares() -> lc_fit.LCFitLeastSquares:
                return lc_fit.LCFitLeastSquares(
                    morphology=expected_morphology,
                    atmosphere_model=atmosphere_models,
                    limb_darkening_coefficients=limb_darkening_coefficients,
                )
            self.__class__.FIT_CLS = _create_lc_fit_least_squares
            self.__class__.PLOT_CLS = LCPlotLsqr
        super().__init__(method, name=name, **kwargs)

    def load_result(self, filename: str, *, autofill_sma: bool = True) -> dict:
        """Load result with default autofill_sma=True for light curves.

        Load model parameters from JSON file with automatic semi-major axis
        autofilling enabled by default (since SMA can be calculated from
        LC observables for eccentric systems).

        :param filename: Path to JSON file containing model parameters
        :type filename: str
        :param autofill_sma: Auto-fill semi-major axis if absent (default: True)
        :type autofill_sma: bool
        :return: Model parameters in standardized format
        :rtype: dict
        """
        return super().load_result(filename, autofill_sma=autofill_sma)

    def set_result(self, result: dict, *, autofill_sma: bool = True) -> None:
        """Set result with default autofill_sma=True for light curves.

        Set model parameters with automatic semi-major axis auto-filling
        enabled by default.

        :param result: Model parameters in JSON format
        :type result: dict
        :param autofill_sma: Auto-fill semi-major axis if absent (default: True)
        :type autofill_sma: bool
        """
        return super().set_result(result, autofill_sma=autofill_sma)


class RVBinaryAnalyticsTask(AnalyticsTask):
    """Fitting task class for radial velocity curves of eclipsing binary stars.

    This class specializes AnalyticsTask for fitting radial velocity (RV) data
    from eclipsing binary systems. Currently, supports kinematic method for
    calculation of radial velocities (treating stars as point masses).

    The RV task provides parameter combinations suitable for both community
    (mass ratio) and standard (individual masses) approaches to binary parameter
    estimation.

    **class attributes:**

    - **FIT_PARAMS_COMBINATIONS** - JSON format string containing available
      parameter sets for RV fitting
    - **TRANSFORM_PROPERTIES_CLS** - Class for transforming and validating input
    """

    FIT_CLS: Any = None
    PLOT_CLS: Any = None
    FIT_PARAMS_COMBINATIONS: str = json.dumps(
        {
            "community": {
                "system": [
                    "mass_ratio",
                    "asini",
                    "eccentricity",
                    "argument_of_periastron",
                    "gamma",
                    "period",
                    "primary_minimum_time",
                ],
                "nuisance": ["ln_f"],
            },
            "standard": {
                "primary": ["mass"],
                "secondary": ["mass"],
                "system": [
                    "inclination",
                    "eccentricity",
                    "argument_of_periastron",
                    "gamma",
                    "period",
                    "primary_minimum_time",
                ],
                "nuisance": ["ln_f"],
            },
        },
        indent=4,
    )
    TRANSFORM_PROPERTIES_CLS: Any = transform.RVBinaryAnalyticsTask

    # noinspection PyUnusedLocal
    def __init__(
        self,
        method: str,
        *,
        name: str | None = None,
        atmosphere_models: str | None = None,  # noqa: ARG002
        limb_darkening_factor: dict | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        """Initialize radial velocity fitting task.

        Sets up RV fitting task with specified fitting method and configuration.
        Initializes appropriate fitting class (MCMC or Least Squares) and plotting
        class based on the provided method. Supports kinematic method for
        radial velocity calculation.

        :param method: Fitting method name (least_squares or mcmc)
        :type method: str
        :param name: Instance name (auto-generated if not provided)
        :type name: str | None
        :param atmosphere_models: Atmosphere model configuration
        :type atmosphere_models: str | None
        :param limb_darkening_factor: Limb darkening factor configuration
        :type limb_darkening_factor: dict | None
        :param kwargs: Additional configuration options (data, atmosphere_model, etc.)
        :type kwargs: dict

        **note:**

        Parameters `atmosphere_models` and `limb_darkening_factor` are accepted for
        API consistency with LCBinaryAnalyticsTask but are not used in RV calculations
        since RV method uses kinematic approach (point masses).
        """
        self.validate_method(method)
        if method in self.MCMC_NAMES:
            def _create_rv_fit_mcmc() -> rv_fit.RVFitMCMC:
                return rv_fit.RVFitMCMC()

            self.__class__.FIT_CLS = _create_rv_fit_mcmc
            self.__class__.PLOT_CLS = RVPlotMCMC
        elif method in self.LS_NAMES:
            def _create_rv_fit_least_squares() -> rv_fit.RVFitLeastSquares:
                return rv_fit.RVFitLeastSquares()

            self.__class__.FIT_CLS = _create_rv_fit_least_squares
            self.__class__.PLOT_CLS = RVPlotLsqr
        super().__init__(method, name=name, **kwargs)
