from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import atm, settings, units
from elisa.analytics.binary_fit.lc_prediction import (
    predict_light_curves,
    validate_observations,
)
from elisa.analytics.dataset.base import DataSet
from elisa.analytics.models import cost_fns
from elisa.analytics.models import lc as lc_model
from elisa.analytics.models import rv as rv_model
from elisa.analytics.params import parameters
from elisa.analytics.params.conf import JOINT_POSTERIOR_PERCENTILES_ESTIMATE
from elisa.analytics.params.parameters import BinaryInitialParameters

# noinspection PyProtectedMember
from elisa.analytics.tools.utils import (
    _radialcurves_mean_error,
    lightcurves_mean_error,
    time_layer_resolver,
)
from elisa.base.error import (
    ElisaPredictionAlignmentError,
    raise_elisa_analytics_data_error,
    raise_elisa_analytics_task_error,
    raise_elisa_numeric_error,
    raise_elisa_prediction_alignment_error,
)
from elisa.base.transform import StarProperties
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from elisa.conf.enums import SamplingMethod
from elisa.logger import getPersistentLogger
from elisa.observer.observer import Observer
from elisa.observer.utils import normalize_light_curve
from elisa.utils import is_empty

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from numpy.typing import NDArray

    from elisa.types import Float, UnitType

logger = getPersistentLogger(__name__)


def _fit_passbands(passband: Iterable[str] | None, data: Mapping[str, DataSet]) -> str | list[str]:
    """Resolve observer passbands and check their dataset correspondence.

    :param passband: Requested LC passbands, or None for RV fitting.
    :param data: Observations keyed by passband or RV component.
    :returns: Observer passbands, using bolometric for RV fitting.
    :raises ElisaAnalyticsTaskError: If requested LC passbands differ from the dataset keys.
    """
    if passband is None:
        return "bolometric"
    bands = list(passband)
    if set(bands) != set(data):
        msg = "Fitting passbands must match observation dataset identifiers"
        raise_elisa_analytics_task_error(msg)
    return bands


def _observation_coordinate_unit(data: Mapping[str, DataSet]) -> UnitType:
    """Validate dataset identifiers and resolve their common coordinate unit.

    :param data: Nonempty observations keyed by dataset identifier.
    :returns: Common coordinate unit for the supplied observations.
    :raises ElisaAnalyticsTaskError: If dataset identifiers are invalid.
    :raises ElisaAnalyticsDataError: If observations are absent or coordinate units differ.
    """
    if not data:
        msg = "Fitting requires at least one observational dataset"
        raise_elisa_analytics_data_error(msg)
    if any(not isinstance(key, str) or not key for key in data):
        msg = "Observation dataset identifiers must be nonempty strings"
        raise_elisa_analytics_task_error(msg)
    x_units = {key: dataset.x_unit for key, dataset in data.items()}
    unique_units = set(x_units.values())
    if len(unique_units) > 1:
        msg = (
            "Please make sure that all synthetic observations are supplied "
            f"in BJD or photometric phases. Current state: {x_units}"
        )
        raise_elisa_analytics_data_error(msg)
    return unique_units.pop()


def _validate_lc_normalization(
    y_data: Mapping[str, NDArray],
    kind: str,
    top_fraction_to_average: Float,
) -> None:
    """Check normalization settings and the observed mean flux scales.

    :param y_data: Observed flux vectors keyed by passband.
    :param kind: Requested normalization method.
    :param top_fraction_to_average: Finite fraction in ``(0, 1]``, used only for
        maximum, global_maximum and minimum normalization; ignored for mean modes.
    :raises ElisaAnalyticsTaskError: If the normalization fraction is invalid.
    :raises ElisaAnalyticsDataError: If a mean normalization scale is unusable.
    """
    uses_fraction = kind in ("maximum", "global_maximum", "minimum")
    if uses_fraction and (not np.isfinite(top_fraction_to_average) or not 0 < top_fraction_to_average <= 1):
        msg = "LC normalization fraction must be finite and satisfy 0 < fraction <= 1"
        raise_elisa_analytics_task_error(msg)
    if kind != "average":
        return
    for key, values in y_data.items():
        scale = np.mean(values)
        if not np.isfinite(scale) or scale <= 0:
            msg = f"LC normalization requires a finite positive mean flux for {key}"
            raise_elisa_analytics_data_error(msg)


def _validated_normalized_errors(
    y_data: Mapping[str, NDArray],
    y_err: Mapping[str, NDArray | None] | None,
) -> dict[str, NDArray]:
    """Check normalized observations and return their usable uncertainties.

    :param y_data: Normalized observed flux vectors.
    :param y_err: Normalized errors returned by the public normalization utility.
    :returns: Matching finite, strictly positive uncertainty vectors.
    :raises ElisaAnalyticsDataError: If normalized fluxes or errors are nonfinite, or errors are nonpositive.
    :raises ValueError: If prepared uncertainties are missing.
    """
    if y_err is None:
        msg = "LC fitting requires prepared uncertainties"
        raise ValueError(msg)
    validated: dict[str, NDArray] = {}
    for key, values in y_data.items():
        errors = y_err[key]
        if errors is None:
            msg = f"LC normalization produced nonfinite data or uncertainties for {key}"
            raise ValueError(msg)
        finite_flux = np.all(np.isfinite(values))
        finite_errors = np.all(np.isfinite(errors))
        if not finite_flux or not finite_errors:
            msg = f"LC normalization produced nonfinite data or uncertainties for {key}"
            raise_elisa_analytics_data_error(msg)
        if np.any(errors <= 0):
            msg = f"LC normalization requires strictly positive uncertainties for {key}"
            raise_elisa_analytics_data_error(msg)
        validated[key] = errors
    return validated


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

    _prepared_observations: cost_fns.PreparedObservations | None

    __slots__ = [
        "_candidate_failure_diagnostics",
        "_prepared_observations",
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
        passband: Iterable[str] | None = None,
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
        :type passband: Iterable[str] | None
        :param kwargs: Class dependent content (see inheritor classes).
        :type kwargs: Any
        """
        # InitialParameter validated input metadata on construction. Candidate
        # values are validated by model properties before physical calculations.
        x0.constraint_validator()
        self.atmosphere_model = getattr(self, "atmosphere_model", None)
        self.limb_darkening_coefficients = getattr(self, "limb_darkening_coefficients", None)
        self._check_data_param_consistency(x0, data)
        self.fixed = x0.get_fixed(jsonify=False)
        self.constrained = x0.get_constrained(jsonify=False)
        self.fitable = x0.get_fitable(jsonify=False)
        self.normalization = x0.get_normalization_map()

        observer = Observer(passband=_fit_passbands(passband, data), system=None)
        self.observer = observer
        self.observer.system_cls = kwargs.get("observer_system_cls")

        self.x_data = {key: val.x_data for key, val in data.items()}
        self.y_data = {key: val.y_data for key, val in data.items()}

        self.y_err = self._prepare_observation_errors(data)
        self._prepared_observations = None
        self.num_of_points = {key: len(values) for key, values in self.y_data.items()}

        x_data_reduced, x_data_reducer = parameters.xs_reducer(
            {key: val.x_data for key, val in data.items()},
        )
        self.x_data_reduced = x_data_reduced
        self.x_data_reducer = x_data_reducer

        self.initial_vector = [val.value for val in self.fitable.values()]
        self.flat_result = {}

        self._validate_atmosphere_parameters()

    def _prepare_observation_errors(self, data: Mapping[str, DataSet]) -> dict[str, NDArray[np.float64]]:
        """Validate observations and prepare supplied or estimated errors once.

        :param data: Observations keyed by passband or RV component.
        :returns: Uncertainty vectors, warning for each estimated dataset.
        :raises ElisaAnalyticsDataError: If dataset validation fails or an estimated scale is unusable.
        """
        errors: dict[str, NDArray[np.float64]] = {}
        for key, dataset in data.items():
            DataSet.check_data_validity(x_data=dataset.x_data, y_data=dataset.y_data, y_err=dataset.y_err)
            if dataset.y_err is not None:
                errors[key] = dataset.y_err
                continue
            scale = abs(self.__class__.MEAN_ERROR_FN(dataset.y_data))
            if not np.isfinite(scale) or scale <= 0:
                msg = f"Cannot estimate positive y_err for {key}; supply explicit positive y_err"
                raise_elisa_analytics_data_error(msg)
            errors[key] = np.full(dataset.y_data.shape, scale, dtype=float)
            logger.warning(
                "Dataset %s uses estimated y_err=%s: heuristic weights, not calibrated measurement "
                "uncertainties; MCMC posterior credible intervals depend on this assumed scale.",
                key,
                scale,
            )
        return errors

    def _validate_atmosphere_parameters(self) -> None:
        """Validate optional atmosphere and limb-darkening configuration dictionaries.

        :raises ElisaAnalyticsTaskError: If configuration dictionaries or component names are invalid.
        """
        if not isinstance(getattr(self, "atmosphere_model", None), (dict, type(None))):
            error_msg = (
                "Please provide argument `atmosphere_model` in format "
                "{component (primary or secondary): atmosphere_model_name (ck04, bb, ...), }"
            )
            raise_elisa_analytics_task_error(error_msg)

        if not isinstance(getattr(self, "limb_darkening_coefficients", None), (dict, type(None))):
            error_msg = (
                "Please provide argument `limb_darkening_coefficients` in format "
                "{component (primary or secondary): {passband: ld_coeffs (float, list), }, }"
            )
            raise_elisa_analytics_task_error(error_msg)

        for component, atlas in (self.atmosphere_model or {}).items():
            if component not in settings.BINARY_COUNTERPARTS:
                msg = f"Unknown atmosphere component: {component}"
                raise_elisa_analytics_task_error(msg)
            atm.validated_atlas(atlas)
        for component, coefficients in (self.limb_darkening_coefficients or {}).items():
            if component not in settings.BINARY_COUNTERPARTS:
                msg_0 = f"Unknown limb-darkening component: {component}"
                raise_elisa_analytics_task_error(msg_0)
            StarProperties.limb_darkening_coefficients(coefficients)

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
        :raises ElisaAnalyticsTaskError: If data and parameter configuration are incompatible.
        :raises ElisaAnalyticsDataError: If observations are absent or their coordinate units differ.
        """
        coordinate_unit = _observation_coordinate_unit(data)
        has_epoch = hasattr(x0, "primary_minimum_time")
        if coordinate_unit == units.d and not has_epoch:
            msg = (
                "Your initial parameters do not contain `primary_minimum_time`, yet you provided "
                "your synthetic observations in JD. Either convert your observations to photometric "
                "phases using `DataSet.convert_to_phases(period, t0)` or include "
                "`primary_minimum_time` to your initial fit parameters."
            )
            raise_elisa_analytics_task_error(msg)
        if coordinate_unit == units.dimensionless_unscaled and has_epoch:
            msg = (
                "Your initial parameters contain `primary_minimum_time`, yet you provided "
                "your synthetic observations in photometric phases. Either convert your observations "
                "to JD using `DataSet.convert_to_time(period, t0)` or remove `primary_minimum_"
                "time` from your initial fit parameters."
            )
            raise_elisa_analytics_task_error(msg)

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

    MEAN_ERROR_FN = _radialcurves_mean_error

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
        x0.validate_rv_parameters()
        if kwargs.get("observer_system_cls") is RadialVelocitySystem and settings.RV_METHOD != "kinematic":
            msg = "Community RV fitting supports only the kinematic RV method."
            raise_elisa_analytics_task_error(msg)
        if not set(data).issubset(settings.BINARY_COUNTERPARTS):
            msg = "RV datasets must be keyed by primary or secondary"
            raise_elisa_analytics_task_error(msg)
        super().set_up(x0, data, passband=None, **kwargs)
        validate_observations(self.x_data_reduced, self.y_data, self.x_data_reducer, self.x_data, self.y_err)

    # noinspection PyUnusedLocal
    def coefficient_of_determination(
        self,
        model_parameters: dict[str, Any],
        data: dict[str, Any],
        discretization: Float,  # noqa: ARG002
        interp_treshold: int,  # noqa: ARG002
    ) -> Float:
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
    atmosphere_model: dict[str, Any] | None
    limb_darkening_coefficients: dict[str, Any] | None
    fitable: dict[str, Any]
    constrained: dict[str, Any]
    fixed: dict[str, Any]
    initial_vector: list[Float] | NDArray[Float]
    normalization: dict[str, tuple[Any, ...]]
    flat_result: dict[str, Any]

    _validated_reducers: dict[str, NDArray] | None

    __slots__ = ["_validated_reducers"]

    def set_up(
        self,
        x0: BinaryInitialParameters,
        data: dict[str, Any],
        passband: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Set up class attributes inherited in __slots__.

        :param x0: Initial state of model parameters.
        :type x0: BinaryInitialParameters
        :param data: Observations in each filter.
        :type data: dict[str, Any]
        :param passband: List of used passbands.
        :type passband: Iterable[str] | None
        :param kwargs: Optional arguments:

            * ``observer_system_cls`` - System used to evaluate synthetic observations.
            * ``discretization`` - Discretization factor for the primary component.
            * ``samples`` - SamplingMethod, a legacy string value, or a manual phase list/array.
        :type kwargs: Any
        """
        x0.validate_lc_parameters(getattr(self, "MORPHOLOGY", "detached"))
        super().set_up(x0, data, passband, observer_system_cls=kwargs.get("observer_system_cls"))
        self.discretization = kwargs.pop("discretization")
        self.interp_treshold = kwargs.pop("interp_treshold")
        self.normalize_data(kind="average")
        self.fit_xs = self.generate_sample_phases(kwargs.pop("samples"))

    def validate_observations(self) -> None:
        """Validate and prepare static observation mappings for this setup.

        Called once by setup through normalize_data(). Observation coordinates,
        shapes and reducers must remain unchanged during fitting. After manually
        editing these internal arrays/dictionaries, call this method before any
        evaluation, or use set_up() to replace the dataset. No global cache or
        content fingerprint is retained. Flux/error value edits with unchanged
        shapes do not change the mappings; normalize_data() revalidates them.

        :raises ElisaPredictionAlignmentError: If observation alignment is invalid.
        """
        self._validated_reducers = None
        self._validated_reducers = validate_observations(
            self.x_data_reduced,
            self.y_data,
            self.x_data_reducer,
            self.x_data,
            self.y_err,
        )
        self.num_of_points = {band: len(curve) for band, curve in self.y_data.items()}

    def prediction_r_squared(self, model_parameters: dict[str, Any]) -> Float:
        """Compute R² using this fitter's prepared mappings, support and Observer.

        :param model_parameters: Flat physical parameters with constraints resolved.
        :return: Coefficient of determination for observation-point predictions.
        :raises ElisaPredictionAlignmentError: If dynamic synthesis alignment is invalid.
        """
        synthetic = self.predict_light_curves(model_parameters)
        return r_squared(synthetic, self.y_data)

    def predict_light_curves(self, model_parameters: dict[str, Any], **kwargs: Any) -> dict:
        """Predict observation-point fluxes using this setup's validated mappings.

        Static observation dimensions, keys, indices and coordinate correspondence
        were validated by setup. This method reuses those mappings and the existing
        Observer while retaining parameter-dependent phase, interpolation-support
        and synthesis-output checks on every call.

        :param model_parameters: Flat physical parameters with constraints resolved.
        :param kwargs: Optional prediction arguments, such as a synthesis callable.
        :return: Per-band mean-normalized flux arrays in observation order.
        :raises ElisaPredictionAlignmentError: If dynamic prediction alignment is invalid.
        Synthesis exceptions propagate unchanged.
        """
        return predict_light_curves(
            self.x_data_reduced,
            self.y_data,
            self.x_data_reducer,
            self.discretization,
            self.observer,
            self.interp_treshold,
            model_parameters,
            fit_xs=self.fit_xs,
            x_data=self.x_data,
            y_err=self.y_err,
            _validated_reducers=self._validated_reducers,
            atmosphere_model=self.atmosphere_model,
            limb_darkening_coefficients=self.limb_darkening_coefficients,
            **kwargs,
        )

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

        The result is assigned back to the respective data attributes ``y_data``
        and ``y_err``. Static observation alignment is validated before rescaling,
        refreshing the prepared mappings for setup or a later normalization call.

        :param kind: Normalization method. Options:

            * ``average`` - Each curve is normalized to its average.
            * ``global_average`` - Curves are normalized to their global average.
            * ``maximum`` - Each curve is normalized to its own maximum.
            * ``global_maximum`` - Curves are normalized to their global maximum (default).
        :type kind: str
        :param top_fraction_to_average: Finite fraction in ``(0, 1]`` used by maximum,
            global_maximum and minimum normalization; ignored for average and global_average.
        :type top_fraction_to_average: float
        """
        _validate_lc_normalization(self.y_data, kind, top_fraction_to_average)
        self.validate_observations()
        y_data, y_err = normalize_light_curve(self.y_data, self.y_err, kind, top_fraction_to_average)
        normalized_errors = _validated_normalized_errors(y_data, y_err)
        self.y_data = y_data
        self.y_err = normalized_errors
        if self._prepared_observations is not None:
            self._prepared_observations = cost_fns.prepare_observations(self.y_data, self.y_err)

    def generate_sample_phases(self, samples: SamplingMethod | str | list | NDArray) -> NDArray[Float] | None:
        """Generate photometric phases for synthetic observations.

        Sampling photometric phases are generated according to a specified rule or array
        of orbital phases.

        :param samples: Sampling method or phase array. Options:

            * ``SamplingMethod.UNIFORM`` - Equidistant phase support.
            * ``SamplingMethod.ADAPTIVE`` - Equidistant support along the initial curve.
            * List/array - Manually provided photometric phases.

        Legacy string values are converted to SamplingMethod at this boundary.
        Below the sampling threshold, direct synthesis remains in use.
        :type samples: SamplingMethod | str | list | NDArray
        :return: Photometric phases or None if insufficient datapoints.
        :rtype: NDArray | None
        :raises ElisaAnalyticsTaskError: When ``samples`` parameter is invalid.
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

        if isinstance(samples, (list, np.ndarray)):
            return np.sort(samples)
        try:
            method = SamplingMethod(samples)
        except (ValueError, TypeError) as exc:
            error_msg = (
                "Parameter `samples` must be SamplingMethod.UNIFORM, SamplingMethod.ADAPTIVE, "
                "their string values, or a manual phase list/array"
            )
            raise_elisa_analytics_task_error(error_msg, exc)
        if method is SamplingMethod.UNIFORM:
            diff = 1.0 / self.interp_treshold
            return np.linspace(0.0 - diff, 1.0 + diff, num=self.interp_treshold + 2)
        logger.info("Generating equidistant samples along the light curve using adaptive sampling method")
        return self.adaptive_sampling()

    def adaptive_sampling(self) -> NDArray[Float]:
        """Generate sampling equidistantly along the curve.

        Generates photometric phases equidistantly along the curve defined by the initial vector.

        :return: Photometric phases.
        :rtype: NDArray[Float]
        :raises RuntimeError: When initial parameters are invalid.
        :raises ElisaPredictionAlignmentError: If synthesis reports a structural alignment error.
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
        except ElisaPredictionAlignmentError:
            raise
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
    ) -> Float:
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
            samples=SamplingMethod.UNIFORM,
            observer_system_cls=BinarySystem,
        )

        flat_result = parameters.deserialize_result(model_parameters)
        r_dict = {key: value["value"] for key, value in flat_result.items()}
        logger.info("Evaluating light curve for calculation of R^2.")
        r_squared_result = self.prediction_r_squared(r_dict)
        logger.info("Calculation of R^2 finished.")
        return r_squared_result


def lc_r_squared(
    synthetic: Callable[..., Any],
    *args: Any,
    fit_xs: Any = None,
    x_data: Any = None,
    **kwargs: Any,
) -> float:
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
    :param fit_xs: Prepared phase support from fitting. If omitted, use the
        same uniform/direct support rule as fitting setup.
    :param x_data: Optional original per-band coordinates for reducer validation.
    :param kwargs: Model parameters to compute binary system.
    :type kwargs: Any
    :return: Coefficient of determination (1.0 means perfect fit).
    :rtype: float
    """
    x_data_reduced, y_data, passband, discretization, x_data_reducer, _diff, interp_treshold, cls = args

    observer = Observer(passband=passband, system=None)
    observer.system_cls = cls
    synthetic_result = predict_light_curves(
        x_data_reduced,
        y_data,
        x_data_reducer,
        discretization,
        observer,
        interp_treshold,
        kwargs,
        fit_xs=fit_xs,
        x_data=x_data,
        synthesis=synthetic,
    )

    return r_squared(synthetic_result, y_data)


def map_rv_predictions(
    synthetic: dict[str, NDArray],
    x_data_reduced: NDArray,
    x_data_reducer: dict[str, NDArray],
) -> dict[str, NDArray]:
    """Select observed RV components and map their curves into observation order.

    :param synthetic: Component curves on the full synthesis coordinate grid.
        Extra components are ignored.
    :param x_data_reduced: Coordinate grid supplied to RV synthesis.
    :param x_data_reducer: Prepared component indices into that grid.
    :returns: Curves in each component's original observation order.
    :raises ElisaPredictionAlignmentError: If a required component is missing, or an
        observed component's raw curve has the wrong shape or nonfinite values.
    """
    if not x_data_reducer.keys() <= synthetic.keys():
        msg = "RV synthesis and observation components differ"
        raise_elisa_prediction_alignment_error(msg)
    expected_shape = np.shape(x_data_reduced)
    predictions = {}
    for component, reducer in x_data_reducer.items():
        curve = synthetic[component]
        if np.shape(curve) != expected_shape:
            msg = (
                f"RV synthetic shape for {component} must match coordinate grid {expected_shape}, got {np.shape(curve)}"
            )
            raise_elisa_prediction_alignment_error(msg)
        if not np.all(np.isfinite(curve)):
            msg = f"RV synthetic values are nonfinite for {component}"
            raise_elisa_numeric_error(msg)
        predictions[component] = curve[reducer]
    return predictions


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
    :raises ElisaPredictionAlignmentError: If raw RV components or shapes do not match the synthesis grid.
    """
    x_data_reduced, y_data, x_data_reducer, cls = args

    observer = Observer(passband="bolometric", system=None)
    observer._system_cls = cls  # noqa: SLF001
    synthetic_result = synthetic(x_data_reduced, observer, **kwargs)
    synthetic_result = map_rv_predictions(synthetic_result, x_data_reduced, x_data_reducer)

    return r_squared(synthetic_result, y_data)


def r_squared(synthetic: dict[str, Any], observed: dict[str, Any]) -> Float:
    """Return coefficient of determination between model and observations.

    :param synthetic: Dictionary mapping component names to synthetic values.
    :type synthetic: dict[str, Any]
    :param observed: Dictionary mapping component names to observed values.
    :type observed: dict[str, Any]
    :return: Coefficient of determination (1.0 means perfect fit).
    :rtype: float
    """
    variability = sum(np.sum((observed[item] - np.mean(observed[item])) ** 2) for item in observed)
    residual = sum(np.sum((synthetic[item] - observed[item]) ** 2) for item in observed)

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

    # A marginal constrained median need not equal the constraint at marginal free medians.
    for key in list(constraints):
        previous = input_dict1[key]
        if previous.get("estimate") == JOINT_POSTERIOR_PERCENTILES_ESTIMATE:
            result_dict[key] = previous
            del constraints[key]

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

    result = parameters.serialize_result(result_dict)
    if "r_squared" in input_dict:
        result["r_squared"] = input_dict["r_squared"]
    return result
