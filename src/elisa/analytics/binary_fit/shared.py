from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import interpolate

from elisa import settings, units
from elisa.analytics.models import lc as lc_model
from elisa.analytics.models import rv as rv_model
from elisa.analytics.params import parameters
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.analytics.tools.utils import (
    lightcurves_mean_error,
    radialcurves_mean_error,
    time_layer_resolver,
)
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from elisa.logger import getPersistentLogger
from elisa.observer.observer import Observer
from elisa.observer.utils import normalize_light_curve
from elisa.utils import is_empty

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from numpy.typing import NDArray

    from elisa.types import Float

logger = getPersistentLogger("analytics.binary_fit.shared")


class AbstractFit(metaclass=ABCMeta):
    """General framework for solution of the inverse problem in ELISa.

    Base class for binary system fitting. Handles common functionality for both
    light curve and radial velocity fitting, including data setup, normalization,
    and constraint evaluation.

    Slotted attributes:

        * ``atmosphere_model``: Dict | None; atmosphere models for each component
        * ``constrained``: Dict; list of constrained model parameters in flat JSON format
        * ``discretization``: float; discretization factor for primary component
        * ``fitable``: Dict; list of optimized model parameters in flat JSON format
        * ``fixed``: Dict; list of constant model parameters in flat JSON format
        * ``fit_xs``: NDArray | None; photometric phases for synthetic observations (LC only)
        * ``flat_result``: Dict; flattened result dictionary with fit results
        * ``initial_vector``: List; array of starting values for variable parameters
        * ``interp_treshold``: int; Above this total number of datapoints, curve will be interpolated
          using model containing ``interp_treshold`` equidistant points per epoch
        * ``limb_darkening_coefficients``: Dict | None; custom limb-darkening coefficients
        * ``normalization``: Dict[str, tuple]; normalization boundaries of variable parameters
          {parameter@name: (min, max), ...}
        * ``num_of_points``: Dict[str, int]; number of points of observations in each passband
        * ``observer``: Observer; Observer instance
        * ``x_data``: Dict[str, NDArray]; phases or times of observations in each filter
        * ``x_data_reduced``: NDArray; phases or times of observations grouped from all filters
        * ``x_data_reducer``: Dict[str, NDArray]; x_data[passband] = x_data_reduced[x_data_reducer[passband]]
        * ``y_data``: Dict[str, NDArray]; fluxes in each filter
        * ``y_err``: Dict[str, NDArray]; observational errors of y_data
    """

    MEAN_ERROR_FN: Callable[..., Float]

    # Attributes from set_up method (defined here to avoid "Instance attribute defined outside __init__" warnings)
    constrained: dict[str, Any]
    discretization: float
    fitable: dict[str, Any]
    fixed: dict[str, Any]
    fit_xs: NDArray[Float] | None
    flat_result: dict[str, Any]
    initial_vector: list[Any]
    interp_treshold: int
    normalization: dict[str, tuple[Any, ...]]
    num_of_points: dict[str, int]
    observer: Any
    x_data: dict[str, Any]
    x_data_reduced: NDArray
    x_data_reducer: dict[str, NDArray]
    y_data: dict[str, Any]
    y_err: dict[str, Any]
    atmosphere_model: dict[str, Any] | None
    limb_darkening_coefficients: dict[str, Any] | None

    __slots__ = [
        "atmosphere_model",
        "constrained",
        "discretization",
        "fit_xs",
        "fitable",
        "fixed",
        "flat_result",
        "initial_vector",
        "interp_treshold",
        "limb_darkening_coefficients",
        "normalization",
        "num_of_points",
        "observer",
        "x_data",
        "x_data_reduced",
        "x_data_reducer",
        "y_data",
        "y_err",
    ]

    def set_up(
        self,
        x0: BinaryInitialParameters,
        data: dict[str, Any],
        passband: Iterable | None = None,
        **kwargs: Any,
    ) -> None:
        """Set up class attributes listed in __slots__.

        Initializes all necessary attributes for fitting including observer,
        data, normalization, and constraint parameters.

        :param x0: Initial state of model parameters.
        :type x0: BinaryInitialParameters
        :param data: Observational data in photometric filters or radial velocity.
        :type data: dict[str, Any]
        :param passband: List of used passbands, use None in case of RV fit.
        :type passband: Iterable | None
        :param kwargs: Class dependent content (see inheritor classes).
        :type kwargs: Any
        """
        self._check_data_param_consistency(x0, data)
        self.fixed = x0.get_fixed(jsonify=False)
        self.constrained = x0.get_constrained(jsonify=False)
        self.fitable = x0.get_fitable(jsonify=False)
        self.normalization = x0.get_normalization_map()

        observer = Observer(passband="bolometric" if passband is None else passband, system=None)
        self.observer = observer
        self.observer.system_cls = kwargs.get("observer_system_cls")

        self.x_data = {key: val.x_data for key, val in data.items()}
        self.y_data = {key: val.y_data for key, val in data.items()}
        self.num_of_points = {key: np.shape(val.y_data)[0] for key, val in data.items()}

        err = {
            key: (abs(self.__class__.MEAN_ERROR_FN(val)) if data[key].y_err is None else data[key].y_err)
            for key, val in self.y_data.items()
        }
        self.y_err = err

        x_data_reduced, x_data_reducer = parameters.xs_reducer(
            {key: val.x_data for key, val in data.items()},
        )
        self.x_data_reduced = x_data_reduced
        self.x_data_reducer = x_data_reducer

        self.initial_vector = [val.value for val in self.fitable.values()]
        self.flat_result = {}

        if not isinstance(getattr(self, "atmosphere_model", None), (dict, type(None))):
            error_msg = (
                "Please provide argument `atmosphere_model` in format "
                "{component (primary or secondary): atmosphere_model_name (ck04, bb, ...), }"
            )
            raise TypeError(error_msg)

        if not isinstance(getattr(self, "limb_darkening_coefficients", None), (dict, type(None))):
            error_msg = (
                "Please provide argument `limb_darkening_coefficients` in format "
                "{component (primary or secondary): {passband: ld_coeffs (float, list), }, }"
            )
            raise TypeError(error_msg)

    @staticmethod
    def _check_data_param_consistency(x0: BinaryInitialParameters, data: dict[str, Any]) -> None:
        """Check compatibility between observational data and initial parameters.

        Validates that input observations have time stamps either in JD or photometric phase,
        and that they are compatible with the supplied initial parameters (i.e., if using JD
        times, primary_minimum_time must be defined).

        :param x0: Initial state of model parameters.
        :type x0: BinaryInitialParameters
        :param data: Observational data in photometric filters or RV.
        :type data: dict[str, Any]
        :raises ValueError: When data and parameters are incompatible.
        """
        x_units = {key: val.x_unit for key, val in data.items()}
        x_units_reduced = list(set(x_units.values()))
        if len(x_units_reduced) > 1:
            error_msg = (
                f"Please make sure that all synthetic observations are supplied "
                f"in BJD or photometric phases. Current state: {x_units}"
            )
            raise ValueError(error_msg)

        try:
            # noinspection PyUnresolvedReferences
            _ = x0.primary_minimum_time
        except AttributeError as exc:
            if x_units_reduced[0] == units.d:
                error_msg = (
                    "Your initial parameters do not contain `primary_minimum_time`, yet you provided "
                    "your synthetic observations in JD. Either convert your observations to photometric "
                    "phases using `DataSet.convert_to_phases(period, t0)` or include "
                    "`primary_minimum_time` to your initial fit parameters."
                )
                raise ValueError(error_msg) from exc
        else:
            if x_units_reduced[0] == units.dimensionless_unscaled:
                error_msg = (
                    "Your initial parameters contain `primary_minimum_time`, yet you provided "
                    "your synthetic observations in photometric phases. Either convert your observations "
                    "to JD using `DataSet.convert_to_time(period, t0)` or remove `primary_minimum_"
                    "time` from your initial fit parameters."
                )
                raise ValueError(error_msg)

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Abstract fit method to be implemented by subclasses.

        :param args: Positional arguments.
        :type args: Any
        :param kwargs: Keyword arguments.
        :type kwargs: Any
        :return: Fit results.
        :rtype: Any
        """
        ...

    @staticmethod
    def eval_constrained_results(
        result_dict: dict[str, dict[str, Any]],
        constraints: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Add constrained parameters to result dictionary.

        Evaluates values of constrained parameters based on their dependent variables
        and adds them to the result dictionary.

        :param result_dict: Result dictionary with format {'name': {'value': value, 'unit': unit, ...}}.
        :type result_dict: dict[str, dict[str, Any]]
        :param constraints: Dictionary containing constrained parameters.
        :type constraints: dict[str, Any]
        :return: Dictionary with evaluated constraint values added.
        :rtype: dict[str, dict[str, Any]]
        """
        if is_empty(constraints):
            return result_dict

        res_val_dict = {key: val["value"] for key, val in result_dict.items()}
        constrained_values = parameters.constraints_evaluator(res_val_dict, constraints)
        result_dict.update(
            {
                key: {
                    "value": val,
                    "constraint": constraints[key].constraint,
                    "unit": constraints[key].to_dict()["unit"],
                }
                for key, val in constrained_values.items()
            },
        )
        return result_dict


class AbstractRVFit(AbstractFit):
    """Abstract implementation of the RV fit."""

    MEAN_ERROR_FN = radialcurves_mean_error

    # Attributes from parent class (set by set_up method)
    y_err: dict[str, Any]
    y_data: dict[str, NDArray]
    x_data_reducer: dict[str, NDArray]
    x_data_reduced: NDArray
    observer: Any
    fitable: dict[str, Any]
    constrained: dict[str, Any]
    fixed: dict[str, Any]
    initial_vector: list[Any]
    normalization: dict[str, tuple[Any, ...]]
    flat_result: dict[str, Any]

    __slots__: list = []  # inheriting attributes from parent

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Abstract fit method to be implemented by subclasses.

        :param args: Positional arguments.
        :type args: Any
        :param kwargs: Keyword arguments.
        :type kwargs: Any
        :return: Fit results.
        :rtype: Any
        """
        ...

    def set_up(self, x0: BinaryInitialParameters, data: dict[str, Any], **kwargs: Any) -> None:
        """Set up class attributes inherited in __slots__.

        :param x0: Initial state of model parameters.
        :type x0: BinaryInitialParameters
        :param data: RV data for primary and secondary component.
        :type data: dict[str, Any]
        :param kwargs: Dictionary with optional arguments:

            * ``observer_system_cls`` - Union[BinarySystem, RadialVelocitySystem]; system used to
              evaluate synthetic observations.
        :type kwargs: Any
        """
        super().set_up(x0, data, passband=None, **kwargs)

    # noinspection PyUnusedLocal
    def coefficient_of_determination(
        self,
        model_parameters: dict[str, Any],
        data: dict[str, Any],
        discretization: Float,  # noqa: ARG002
        interp_treshold: int,  # noqa: ARG002
    )-> Float:
        """Return R^2 for given model parameters and observed data.

        :param model_parameters: Set of model parameters in JSON format.
        :type model_parameters: dict[str, Any]
        :param data: Observed RVs for each component.
        :type data: dict[str, Any]
        :param discretization: Not (yet) used.
        :type discretization: float
        :param interp_treshold: Not (yet) used.
        :type interp_treshold: int
        :return: Coefficient of determination (1.0 means a perfect fit).
        :rtype: float
        """
        self.set_up(
            parameters.BinaryInitialParameters(**model_parameters),
            data,
            observer_system_cls=RadialVelocitySystem,
        )
        r_squared_args: tuple[Any, ...] = (
            self.x_data_reduced,
            self.y_data,
            self.x_data_reducer,
            self.observer.system_cls,
        )
        flat_result = parameters.deserialize_result(model_parameters)
        r_dict: dict[str, Any] = {key: value["value"] for key, value in flat_result.items()}

        logger.info("Evaluating light curve for calculation of R^2.")
        r_squared_result = rv_r_squared(rv_model.central_rv_synthetic, *r_squared_args, **r_dict)
        logger.info("Calculation of R^2 finished.")
        return r_squared_result


class AbstractLCFit(AbstractFit):
    """Abstract implementation the LC fitting."""

    MEAN_ERROR_FN = lightcurves_mean_error

    # Attributes from parent class (set by set_up method)
    fit_xs: NDArray[Float] | None
    y_err: dict[str, Any]
    y_data: dict[str, NDArray]
    x_data_reducer: dict[str, NDArray]
    x_data_reduced: NDArray
    interp_treshold: int
    discretization: float
    observer: Observer
    atmosphere_model: None
    limb_darkening_coefficients: None
    fitable: dict[str, Any]
    constrained: dict[str, Any]
    fixed: dict[str, Any]
    initial_vector: list[Float] | NDArray[Float]
    normalization: dict[str, tuple[Any, ...]]
    flat_result: dict[str, Any]

    __slots__: list = []  # inheriting attributes from parent

    def set_up(
        self,
        x0: BinaryInitialParameters,
        data: dict[str, Any],
        passband: Iterable | None = None,
        **kwargs: Any,
    ) -> None:
        """Set up class attributes inherited in __slots__.

        :param x0: Initial state of model parameters.
        :type x0: BinaryInitialParameters
        :param data: Observations in each filter.
        :type data: dict[str, Any]
        :param passband: List of used passbands.
        :type passband: Iterable | None
        :param kwargs: Optional arguments:

            * ``observer_system_cls`` - System used to evaluate synthetic observations.
            * ``discretization`` - Discretization factor for the primary component.
            * ``samples`` - Union[str, List]; 'uniform', 'adaptive' or list with phases in (0, 1).
        :type kwargs: Any
        """
        super().set_up(x0, data, passband, observer_system_cls=kwargs.get("observer_system_cls"))
        self.discretization = kwargs.pop("discretization")
        self.interp_treshold = kwargs.pop("interp_treshold")
        fit_xs = self.generate_sample_phases(kwargs.pop("samples"))
        self.fit_xs = fit_xs
        self.normalize_data(kind="average")

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Abstract fit method to be implemented by subclasses.

        :param args: Positional arguments.
        :type args: Any
        :param kwargs: Keyword arguments.
        :type kwargs: Any
        :return: Fit results.
        :rtype: Any
        """
        ...

    def normalize_data(
        self,
        kind: str = "global_maximum",
        top_fraction_to_average: float = 0.1,
    ) -> None:
        """Normalize input observational data using different methods.

        The result is assigned back to the respective data attributes ``y_data`` and ``y_err``.

        :param kind: Normalization method. Options:

            * ``average`` - Each curve is normalized to its average.
            * ``global_average`` - Curves are normalized to their global average.
            * ``maximum`` - Each curve is normalized to its own maximum.
            * ``global_maximum`` - Curves are normalized to their global maximum (default).
        :type kind: str
        :param top_fraction_to_average: Top portion of the dataset (in y-axis direction) used
            in the normalization process, from (0, 1) interval.
        :type top_fraction_to_average: float
        """
        y_data, y_err = normalize_light_curve(self.y_data, self.y_err, kind, top_fraction_to_average)
        self.y_data = y_data
        self.y_err = y_err

    def generate_sample_phases(self, samples: str | list) -> NDArray[Float] | None:
        """Generate photometric phases for synthetic observations.

        Sampling photometric phases are generated according to a specified rule or array
        of orbital phases.

        :param samples: Sampling method or phase array. Options:

            * ``uniform`` - Phase equidistant sampling (use for initial fitting stages).
            * ``adaptive`` - Equidistant sampling along the synthetic curve (useful for narrow eclipses).
            * List/array - Manually provided array of photometric phases in (0, 1).
        :type samples: str | list
        :return: Photometric phases or None if insufficient datapoints.
        :rtype: list | None
        :raises ValueError: When ``samples`` parameter is invalid.
        """
        kwargs: dict[str, Any] = parameters.prepare_properties_set(
            self.initial_vector,
            self.fitable.keys(),
            self.constrained,
            self.fixed,
        )
        phases, kwargs = time_layer_resolver(self.x_data_reduced, pop=False, **kwargs)
        if np.shape(phases)[0] < self.interp_treshold:
            return None

        if samples == "uniform":
            diff = 1.0 / self.interp_treshold
            return np.linspace(0.0 - diff, 1.0 + diff, num=self.interp_treshold + 2)
        if samples == "adaptive":
            logger.info("Generating equidistant samples along the light curve using adaptive sampling method")
            return self.adaptive_sampling()
        if isinstance(samples, (list, np.ndarray)):
            return np.sort(samples)
        error_msg = (
            "Parameter `samples` has to be either string with values `uniform` or `adaptive` or "
            "array of phases in (0, 1) interval"
        )
        raise ValueError(error_msg)

    def adaptive_sampling(self) -> NDArray[Float]:
        """Generate sampling equidistantly along the curve.

        Generates photometric phases equidistantly along the curve defined by the initial vector.

        :return: Photometric phases.
        :rtype: NDArray[Float]
        :raises RuntimeError: When initial parameters are invalid.
        """
        n = 3 * settings.MAX_CURVE_DATAPOINTS
        diff = 1.0 / n
        x = np.linspace(0.0 - diff, 1.0 + diff, num=n)

        kwargs: dict[str, Any] = parameters.prepare_properties_set(
            self.initial_vector,
            self.fitable.keys(),
            self.constrained,
            self.fixed,
        )

        kwargs = parameters.extend_json_with_atm_params(
            kwargs,
            atmosphere_model=self.atmosphere_model,
            limb_darkening_coefficients=self.limb_darkening_coefficients,
        )

        observer = Observer(passband="bolometric", system=None)
        observer.system_cls = self.observer.system_cls
        try:
            synthetic = lc_model.synthetic_binary(x, self.discretization, observer, **kwargs)
            synthetic, _ = normalize_light_curve(synthetic, kind="average")
        except Exception as e:
            error_msg = "Your initial parameters are invalid and phase sampling could not be generated."
            raise RuntimeError(error_msg) from e

        curve = np.column_stack((x, synthetic["bolometric"]))
        lengths = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
        crv_lengths = np.cumsum(np.concatenate(([0], lengths)))
        segments = np.linspace(0, crv_lengths[-1], num=self.interp_treshold)

        return np.interp(segments, crv_lengths, x)

    def coefficient_of_determination(
        self,
        model_parameters: dict[str, Any],
        data: dict[str, Any],
        discretization: Float,
        interp_treshold: int,
    )-> Float:
        """Return R^2 for given model parameters and observed data.

        :param model_parameters: Set of model parameters in JSON format.
        :type model_parameters: dict[str, Any]
        :param data: Observations in each filter.
        :type data: dict[str, Any]
        :param discretization: Discretization factor for the primary component.
        :type discretization: float
        :param interp_treshold: Threshold above which light curve will be interpolated.
        :type interp_treshold: int
        :return: Coefficient of determination (1.0 means a perfect fit).
        :rtype: float
        """
        self.set_up(
            x0=parameters.BinaryInitialParameters(**model_parameters),
            data=data,
            passband=data.keys(),
            discretization=discretization,
            interp_treshold=interp_treshold,
            samples="uniform",
            observer_system_cls=BinarySystem,
        )

        r_squared_args: tuple[Any, ...] = (
            self.x_data_reduced,
            self.y_data,
            self.observer.passband,
            discretization,
            self.x_data_reducer,
            1.0 / self.interp_treshold,
            self.interp_treshold,
            self.observer.system_cls,
        )
        flat_result = parameters.deserialize_result(model_parameters)

        r_dict: dict[str, Any] = {key: value["value"] for key, value in flat_result.items()}
        r_dict = parameters.extend_json_with_atm_params(
            r_dict,
            atmosphere_model=self.atmosphere_model,
            limb_darkening_coefficients=self.limb_darkening_coefficients,
        )

        logger.info("Evaluating light curve for calculation of R^2.")
        r_squared_result = lc_r_squared(lc_model.synthetic_binary, *r_squared_args, **r_dict)
        logger.info("Calculation of R^2 finished.")
        return r_squared_result


def lc_r_squared(synthetic: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    """Compute R^2 coefficient of determination between synthetic and observed light curves.

    :param synthetic: Callable method for creating synthetic LC observations
        (e.g., ``elisa.analytics.models.lc.synthetic_binary``).
    :type synthetic: Callable[..., Any]
    :param args: Tuple containing:

        * ``x_data_reduced`` - Phases in AbstractFit.x_data_reduced slot.
        * ``y_data`` - Dict[str, NDArray]; fluxes from observation normalized to max value.
        * ``passband`` - Union[str, List[str]]; list of used photometric filters.
        * ``discretization`` - Discretization factor for the primary component.
        * ``x_data_reducer`` - Dict[str, NDArray]; mask stored in AbstractFit.x_data_reducer slot.
        * ``diff`` - Float; auxiliary parameter for interpolation.
        * ``interp_treshold`` - Int; threshold above which synthetic curves are interpolated.
        * ``cls`` - BinarySystem; class used for observations.
    :type args: Any
    :param kwargs: Model parameters to compute binary system.
    :type kwargs: Any
    :return: Coefficient of determination (1.0 means perfect fit).
    :rtype: float
    """
    x_data_reduced, y_data, passband, discretization, x_data_reducer, diff, interp_treshold, cls = args

    x_data_reduced, kwargs = time_layer_resolver(x_data_reduced, pop=False, **kwargs)
    # noinspection PyUnresolvedReferences
    fit_xs = (
        np.linspace(
            np.min(x_data_reduced) - diff,
            np.max(x_data_reduced) + diff,
            num=interp_treshold + 2,
        )
        if np.shape(x_data_reduced)[0] > interp_treshold
        else x_data_reduced
    )

    observer = Observer(passband=passband, system=None)
    observer._system_cls = cls  # noqa: SLF001
    synthetic_result = synthetic(fit_xs, discretization, observer, **kwargs)

    if np.shape(fit_xs) != np.shape(x_data_reduced):
        synthetic_result = {
            fltr: interpolate.interp1d(fit_xs, curve, kind="cubic")(x_data_reduced)
            for fltr, curve in synthetic_result.items()
        }
    synthetic_result = {band: synthetic_result[band][x_data_reducer[band]] for band in synthetic_result}
    synthetic_result, _ = normalize_light_curve(synthetic_result, kind="average")

    return r_squared(synthetic_result, y_data)


def rv_r_squared(synthetic: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    """Compute R^2 coefficient of determination between synthetic and observed RV curves.

    :param synthetic: Callable method for creating synthetic RV observations
        (e.g., ``elisa.analytics.models.rv.central_rv_synthetic``).
    :type synthetic: Callable[..., Any]
    :param args: Tuple containing:

        * ``x_data_reduced`` - Phases in AbstractFit.x_data_reduced slot.
        * ``y_data`` - Dict[str, NDArray]; radial velocities for both components.
        * ``x_data_reducer`` - Dict[str, NDArray]; mask stored in AbstractFit.x_data_reducer slot.
        * ``cls`` - Union[BinarySystem, RadialVelocitySystem]; class used for observations.
    :type args: Any
    :param kwargs: Model parameters to compute radial velocities curve.
    :type kwargs: Any
    :return: Coefficient of determination (1.0 means perfect fit).
    :rtype: float
    """
    x_data_reduced, y_data, x_data_reducer, cls = args

    observer = Observer(passband="bolometric", system=None)
    observer._system_cls = cls  # noqa: SLF001
    synthetic_result = synthetic(x_data_reduced, observer, **kwargs)
    synthetic_result = {comp: synthetic_result[comp][x_data_reducer[comp]] for comp in synthetic_result}

    return r_squared(synthetic_result, y_data)


def r_squared(synthetic: dict[str, Any], observed: dict[str, Any])-> Float:
    """Return coefficient of determination between model and observations.

    :param synthetic: Dictionary mapping component names to synthetic values.
    :type synthetic: dict[str, Any]
    :param observed: Dictionary mapping component names to observed values.
    :type observed: dict[str, Any]
    :return: Coefficient of determination (1.0 means perfect fit).
    :rtype: float
    """
    variability = np.sum(
        [np.sum(np.power(observed[item] - np.mean(observed[item]), 2)) for item in observed],
    )
    residual = np.sum([np.sum(np.power(synthetic[item] - observed[item], 2)) for item in observed])

    # noinspection PyUnresolvedReferences
    return 1 - (residual / variability)


def extend_observations_to_desired_interval(
    start_phase: Float,
    stop_phase: Float,
    x_data: dict[str, Any],
    y_data: dict[str, Any],
    y_err: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Extend left and right boundaries of phase-folded observations to desired interval.

    Extends the phase coverage of observational data by concatenating adjacent phase cycles
    to allow for better interpolation and fitting at phase boundaries.

    :param start_phase: Start phase of desired interval.
    :type start_phase: float
    :param stop_phase: Stop phase of desired interval.
    :type stop_phase: float
    :param x_data: Dictionary of phase data for each band.
    :type x_data: dict[str, Any]
    :param y_data: Dictionary of flux data for each band.
    :type y_data: dict[str, Any]
    :param y_err: Dictionary of flux errors for each band.
    :type y_err: dict[str, Any]
    :return: Tuple of (x_data, y_data, y_err) with extended observations.
    :rtype: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    """
    for item, curve in x_data.items():
        phases_extended = np.concatenate((curve - 1.0, curve, curve + 1.0))
        phases_extended_filter = np.logical_and(start_phase < phases_extended, phases_extended < stop_phase)
        x_data[item] = phases_extended[phases_extended_filter]

        y_data[item] = np.tile(y_data[item], 3)[phases_extended_filter]
        if y_err[item] is not None:
            y_err[item] = np.tile(y_err[item], 3)[phases_extended_filter]

    return x_data, y_data, y_err


def check_for_boundary_surface_potentials(
    result_dict: dict[str, dict[str, Any]],
    morphology: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Check and correct surface potentials near critical values.

    Checks if surface potential values are within errors below critical potentials
    (which would break BinarySystem initialization). If they are close but still within
    errors, they are snapped to critical potential.

    :param result_dict: Flat dictionary of fit results with confidence intervals.
    :type result_dict: dict[str, dict[str, Any]]
    :param morphology: Expected morphology ('over-contact' for synchronized potentials).
    :type morphology: str | None
    :return: Corrected flat dictionary of fit results.
    :rtype: dict[str, dict[str, Any]]
    """
    if "primary@surface_potential" not in result_dict or "secondary@surface_potential" not in result_dict:
        return result_dict

    if result_dict["primary@surface_potential"]["value"] == result_dict["secondary@surface_potential"]["value"]:
        return result_dict

    for component in settings.BINARY_COUNTERPARTS:
        pot = result_dict[f"{component}@surface_potential"]
        if "fixed" not in pot or "value" not in pot:
            continue

        sigma = pot["value"] - pot["confidence_interval"]["min"] if "confidence_interval" in pot else 0.001

        synchronicity = (
            result_dict[f"{component}@synchronicity"]["value"] if f"{component}@synchronicity" in result_dict else 1.0
        )

        mass_ratio = (
            result_dict["system@mass_ratio"]["value"]
            if "system@mass_ratio" in result_dict
            else result_dict["secondary@mass"]["value"] / result_dict["primary@mass"]["value"]
        )

        periastron_distance = 1 - result_dict["system@eccentricity"]["value"]

        l1 = BinarySystem.critical_potential_static(
            component=component,
            components_distance=periastron_distance,
            mass_ratio=mass_ratio,
            synchronicity=synchronicity,
        )

        # if resulting potential is too close critical potentials (within errors), it will snap potential
        # to critical one to avoid problems
        if 5 * sigma >= l1 - pot["value"] >= 0.0:
            pot["value"] = l1 + 1e-5 * sigma

        # test for over-contact overflow trough L2 point
        l2 = BinarySystem.libration_potentials_static(periastron_distance, mass_ratio)[2]
        if 5 * sigma >= l2 - pot["value"] >= 0.0:
            pot["value"] = l2 - 1e-5 * sigma

    if morphology == "over-contact":
        result_dict["secondary@surface_potential"]["value"] = result_dict["primary@surface_potential"]["value"]

    return result_dict


def eval_constraint_in_dict(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate constraints and update parameter dictionary.

    Evaluates constraints defined in the user-given model parameters and updates
    the dictionary with constraint values based on dependent variables.

    :param input_dict: Standard JSON format of model parameters.
    :type input_dict: dict[str, Any]
    :return: Same as ``input_dict`` but with values added/updated for constrained parameters.
    :rtype: dict[str, Any]
    """
    input_dict1 = parameters.deserialize_result(input_dict)
    result_dict: dict[str, dict[str, Any]] = {key: val for key, val in input_dict1.items() if "fixed" in val}

    reduced_dict: dict[str, Any] = {key: val["value"] for key, val in result_dict.items()}

    b_parameters = BinaryInitialParameters(**input_dict)
    constraints = b_parameters.get_constrained(jsonify=False)

    constrained_values = parameters.constraints_evaluator(reduced_dict, constraints)
    result_dict.update(
        {
            key: {
                "value": val,
                "constraint": constraints[key].constraint,
                "unit": constraints[key].unit,
            }
            for key, val in constrained_values.items()
        },
    )

    return parameters.serialize_result(result_dict)
