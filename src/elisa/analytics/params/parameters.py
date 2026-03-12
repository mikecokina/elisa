"""Analytics fitting parameters definition and manipulation utilities."""

from __future__ import annotations

import abc
import json
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

try:
    # noinspection PyProtectedMember
    from collections.abc import Iterable
except ImportError:
    from collections.abc import Iterable

from jsonschema import ValidationError

from elisa import settings, utils
from elisa import units as u
from elisa.analytics.params import bonds, conf
from elisa.analytics.params.bonds import (
    ALLOWED_CONSTRAINT_CHARS,
    ALLOWED_CONSTRAINT_METHODS,
    TRANSFORM_TO_METHODS,
)
from elisa.analytics.params.transform import (
    BinaryInitialProperties,
    NuisanceInitialProperties,
    PulsationModeInitialProperties,
    SpotInitialProperties,
    StarInitialProperties,
)
from elisa.base.error import InitialParamsError
from elisa.binary_system.utils import calculate_sma_estimate
from elisa.logger import getLogger
from elisa.utils import is_empty

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import AstropyUnit, Float, Number, TransformProperties

logger = getLogger("analytics.params.parameters")


def deflate_phenomena(flatten: dict) -> dict:
    """Convert phenomena parameters from flat format to nested JSON format.

    Converts model parameters concerning spots or pulsations from flat format
    (e.g., 'primary@spot@spot1@longitude') into the standard (nested) JSON format.

    :param flatten: Flat format phenomena parameters dictionary.
    :type flatten: dict
    :returns: Nested format phenomena parameters dictionary.
    :rtype: dict
    """
    result = {}
    for phenom_uid, phenom_meta in flatten.items():
        _, _, label, param = str(phenom_uid).split(conf.PARAM_PARSER)
        if label not in result:
            result[label] = {"label": label}
        result[label][param] = phenom_meta
    return result


def deserialize_result(result_dict: dict) -> dict:  # noqa: C901, PLR0912
    """Convert dictionary of results in user format JSON to flat format.

    :param result_dict: Dict; standard input format::
        >>> # noinspection PyShadowingNames
        >>> result_dict = {
        >>>     'system': {
        >>>         'inclination': {},
        >>>         # ...
        >>>     },
        >>>     'primary': {
        >>>         't_eff': {},
        >>>         'spots': [
        >>>             {
        >>>                 'label': 'spot1'
        >>>                 'longitude': {},
        >>>                 ...
        >>>             },
        >>>             ...
        >>>         ]
        >>>     }
        >>>     ...
        >>> }

    :return: Dict; model parameters in flat format::

        >>> {
        >>>     'system@inclination': {},
        >>>     'primary@t_eff': {},
        >>>     'primary@spots@spot1@longitude': {},
        >>>     ...
        >>> }

    """
    data = {}

    if "r_quared" in result_dict:
        data["r_squared"] = result_dict["r_squared"]

    for system_slot in BinaryInitialParameters.__slots__:
        if system_slot in result_dict["system"]:
            system_prop = result_dict["system"][system_slot]
            data.update({f"system{conf.PARAM_PARSER}{system_slot}": system_prop})

        if system_slot == conf.NUISANCE_PARSER and system_slot in result_dict:
            for nuisance_slot in NuisanceInitialPrameters.__slots__:
                if nuisance_slot in result_dict[conf.NUISANCE_PARSER] and conf.NUISANCE_PARSER in result_dict:
                    nuisance_prop = result_dict[conf.NUISANCE_PARSER][nuisance_slot]
                    data.update({f"{conf.NUISANCE_PARSER}{conf.PARAM_PARSER}{nuisance_slot}": nuisance_prop})

        if system_slot in ["primary", "secondary"] and system_slot in result_dict:
            component_prop = result_dict[system_slot]

            for component_slot in StarInitialParameters.__slots__:
                if component_slot not in ["spots", "pulsations"] and component_slot in result_dict[system_slot]:
                    data.update({f"{system_slot}@{component_slot}": component_prop[component_slot]})

                elif component_slot in ["spots", "pulsations"] and component_slot in result_dict[system_slot]:
                    for phenom in component_prop[component_slot]:
                        for phenom_key, phenom_value in phenom.items():
                            if phenom_key != "label":
                                key_path = f"{system_slot}@{component_slot[:-1]}@{phenom['label']}@{phenom_key}"
                                data.update({key_path: phenom_value})
    return data


def serialize_result(result_dict: dict) -> dict:  # noqa: C901, PLR0912
    """Convert dictionary of fit parameters in flat format back to user format.

    :param result_dict: Dict; fit parameters in flat format::

        >>> {
        >>>     'system@inclination': {},
        >>>     'primary@t_eff': {},
        >>>     'primary@spots@spot1@longitude': {},
        >>>     ...
        >>> }

    :return: Dict; standard (nested) fit parameter JSON::

        >>> retval = {
        >>>     'system': {
        >>>         'inclination': {},
        >>>         # ...
        >>>     },
        >>>     'primary': {
        >>>         't_eff': {},
        >>>         'spots': [
        >>>             {
        >>>                 'label': 'spot1'
        >>>                 'longitude': {},
        >>>                 ...
        >>>             },
        >>>             ...
        >>>         ]
        >>>     }
        >>>     # ...
        >>> }

    """
    ret_dict = {}

    for param, value in result_dict.items():
        if param == "r_squared":
            ret_dict["r_squared"] = result_dict["r_squared"]
            continue

        identificators = param.split(conf.PARAM_PARSER)

        if identificators[0] not in ret_dict:
            ret_dict[identificators[0]] = {}

        if str(identificators[1]) in conf.COMPOSITE_FLAT_PARAMS:
            identificators[1] = f"{identificators[1]}s"

        if identificators[1] not in ret_dict[identificators[0]]:
            ret_dict[identificators[0]][identificators[1]] = {}

        if str(identificators[0]).startswith("system"):
            ret_dict["system"].update(
                {
                    identificators[1]: value,
                },
            )
            continue

        if str(identificators[1]) not in ["spots", "pulsations"]:
            ret_dict[identificators[0]].update(
                {
                    identificators[1]: value,
                },
            )
        else:
            if str(identificators[2]) not in ret_dict[identificators[0]][identificators[1]]:
                ret_dict[identificators[0]][identificators[1]][identificators[2]] = {}

            ret_dict[identificators[0]][identificators[1]][identificators[2]].update(
                {
                    "label": identificators[2],
                    identificators[3]: value,
                },
            )

    # renormalize spots and pulastions if presented
    for component in settings.BINARY_COUNTERPARTS:
        if component in ret_dict:
            for phenomena in ["spots", "pulsations"]:
                if phenomena in ret_dict[component]:
                    ret_dict[component][phenomena] = list(ret_dict[component][phenomena].values())

    return ret_dict


def extend_json_with_atm_params(
    params: dict,
    atmosphere_model: dict | None = None,
    limb_darkening_coefficients: dict | None = None,
) -> dict:
    """Extend initialization JSON with custom atmosphere-related parameters.

    Adds atmosphere model and/or limb-darkening coefficient parameters to the
    fitting parameter dictionary for specified stellar components.

        :param params: Dict; flattened JSON used to initialize the system e.g.:

        >>> {
        >>>     'system@mass_ratio': 1.0,
        >>>         ...
        >>>     'primary@teff': 80000,
        >>>         ...
        >>>     'secondary@teff': 5000,
        >>>         ...
        >>> }

    :param atmosphere_model: dict; desired atmosphere `atmosphere` models::

        >>> {
        >>>     'primary': 'bb',
        >>>     'secondary': 'ck04'
        >>> }

    :param limb_darkening_coefficients: dict; custom limb-darkening coefficients::

        >>> {
        >>>     'primary': {
        >>>         'bolometric': [0.5, 0.5],  # for logarithmic and square_root law
        >>>         'TESS': [0.68, 0.32]
        >>>     },
        >>>     'secondary': ...
        >>> }

    :return: dict; updated parameter JSON, e.g.:

        >>> {
        >>>     'system@mass_ratio': 1.0,
        >>>         ...
        >>>     'primary@teff': 80000,
        >>>     'primary@atmosphere': 'bb',
        >>>     'primary@limb_darkening_coefficients': {
        >>>         'bolometric': [0.5, 0.5],
        >>>         'TESS': [0.68, 0.32]
        >>>      }
        >>>         ...
        >>>     'secondary@teff': 5000,
        >>>     'secondary@atmosphere': 'ck04',
        >>>         ...
        >>> }

    """
    var_names = ["atmosphere", "limb_darkening_coefficients"]
    for ii, atmosphere_param in enumerate([atmosphere_model, limb_darkening_coefficients]):
        if atmosphere_param is None:
            continue

        component_list = ",".join(list(params.keys()))
        for component, atm_param in atmosphere_param.items():
            if not atm_param:
                continue
            if component in component_list:
                params[f"{component}{conf.PARAM_PARSER}{var_names[ii]}"] = atm_param
            else:
                error_msg = (
                    f"Component {component} does not figure in your fit parameters JSON. Make sure that "
                    f"your `{var_names[ii]}` contain `primary` and `secondary` components in case of "
                    f"binary system and `star` in case of a single star."
                )
                raise ValueError(error_msg)

    return params


def check_initial_param_validity(
    x0: dict[str, InitialParameter],
    all_fit_params: list[str],
    mandatory_fit_params: list[str],
) -> None:
    """Validate that initial system parameters contain all necessary values and no invalid entries.

    Checks that all required parameters are present, parameter values are within valid ranges,
    and no unexpected parameters appear in the model. Also validates spot and pulsation parameters.

    :param x0: Dictionary of initial parameters with InitialParameter values.
    :type x0: dict[str, InitialParameter]
    :param all_fit_params: List of all valid system parameter names (spot and pulsation parameters excluded).
    :type all_fit_params: list[str]
    :param mandatory_fit_params: List of mandatory system parameter names (spot and pulsation parameters excluded).
    :type mandatory_fit_params: list[str]
    :raises InitialParamsError: If mandatory parameters are missing or invalid parameters are found.
    """
    param_names = {
        key: val.value for key, val in x0.items() if not re.search(r"|".join(conf.COMPOSITE_FLAT_PARAMS), key)
    }
    utils.invalid_param_checker(param_names, all_fit_params, "x0")
    utils.check_missing_params(list(mandatory_fit_params), list(param_names.keys()), "x0")

    # checking validity of spot parameters and pulsations parameters
    params_map = {"spot": conf.SPOTS_PARAMETERS, "pulsation": conf.PULSATIONS_PARAMETERS}
    for phenom_type, phenom_params in params_map.items():
        var_ = {key: value for key, value in x0.items() if re.search(phenom_type, key)}
        var_ = deflate_phenomena(var_)

        for label, meta in var_.items():
            meta_copy = meta.copy()
            meta_copy.pop("label", None)

            utils.invalid_param_checker(meta_copy.keys(), phenom_params, label)
            utils.check_missing_params(phenom_params, meta_copy, label)


def xs_reducer(xs: dict) -> tuple[NDArray, dict]:
    """Reduce passband/component-specific phases to unique sorted array with inverse mapping.

    Converts phases defined separately for each passband (light curves) or component
    (radial velocities) into a single sorted array of unique phases, while maintaining
    an inverse mapping for reconstruction.

    :param xs: Phases for each passband or component.
    :type xs: dict[str, NDArray]
    :returns: Tuple of (unique sorted phases, inverse mapping dict).
    :rtype: tuple[NDArray, dict[str, list[int]]]

    Example:
        >>> # noinspection PyShadowingNames
        >>> xs = {'V': np.array([0, 0.25, 0.5]), 'I': np.array([0, 0.5])}
        >>> # noinspection PyShadowingNames
        >>> xs_reduced, inverse = xs_reducer(xs)
        >>> # xs_reduced contains unique phases sorted
        >>> # inverse['V'] contains indices to reconstruct V phases from xs_reduced

    """
    # Convert all phases to flattened array
    x = np.hstack(list(xs.values())).flatten()
    y = np.arange(len(x)).tolist()
    reverse_dict = {}

    # Build reverse mapping for each passband/component
    for xs_key, phases in xs.items():
        reverse_dict[xs_key] = y[: len(phases)]
        del y[: len(phases)]

    # Find unique phases and their inverse indices
    xs_reduced, inverse = np.unique(x, return_inverse=True)
    reverse = {band: inverse[indices] for band, indices in reverse_dict.items()}
    return xs_reduced, reverse


def renormalize_value(val: Float | NDArray, _min: Float, _max: Float) -> Float:
    """Convert normalized value to actual parameter value within specified interval.

    Inverse function to :func:`normalize_value`. Converts a value from the
    normalized interval [0, 1] back to its actual value within the range
    defined by `_min` and `_max`.

    :param val: Normalized value (e.g., 0.5).
    :type val: Float | NDArray
    :param _min: Bottom normalization boundary (e.g., 5000 K).
    :type _min: float
    :param _max: Top normalization boundary (e.g., 7000 K).
    :type _max: float
    :returns: Actual parameter value (e.g., 6000 K).
    :rtype: float

    Example::

        >>> renormalize_value(0.5, 5000.0, 7000.0)
        6000.0

    """
    return (val * (_max - _min)) + _min


def normalize_value(val: float, _min: float, _max: float) -> float:
    """Normalize actual parameter value to interval [0, 1] based on min/max bounds.

    Converts an actual parameter value to a normalized value within the [0, 1]
    interval based on the specified minimum and maximum boundaries.

    :param val: Actual parameter value (e.g., 6000 K).
    :type val: float
    :param _min: Bottom normalization boundary (e.g., 5000 K).
    :type _min: float
    :param _max: Top normalization boundary (e.g., 7000 K).
    :type _max: float
    :returns: Normalized value (e.g., 0.5).
    :rtype: float

    Example:
        >>> normalize_value(6000.0, 5000.0, 7000.0)
        0.5

    """
    return (val - _min) / (_max - _min)


def vector_renormalizer(
    vector: Iterable[Float] | NDArray[Float], properties: Iterable[str], normalization: dict,
) -> list:
    """Convert normalized parameter vector to actual values using normalization boundaries.

    Denormalizes an array of normalized parameters [0, 1] to their actual values
    according to the specified normalization boundaries for each parameter.

    :param vector: Array of normalized parameter values.
    :type vector: Iterable[Float] | NDArray[Float]
    :param properties: Parameter names corresponding to vector elements.
    :type properties: Iterable[str]
    :param normalization: Normalization map with min/max boundaries.
    :type normalization: dict[str, tuple[float, float]]
    :returns: Actual parameter values.
    :rtype: list[float]

    Example:
        >>> # noinspection PyShadowingNames
        >>> normalization = {'param1': (0, 10), 'param2': (1.0, 2.0)}
        >>> vector_renormalizer([0.5, 0.5], ('param1', 'param2'), normalization)
        [5.0, 1.5]

    """
    return [renormalize_value(value, *normalization[prop]) for value, prop in zip(vector, properties, strict=False)]


def vector_normalizer(vector: Iterable[Float] | NDArray[Float], properties: Iterable[str], normalization: dict) -> list:
    """Normalize array of parameter values from actual values to [0, 1] interval.

    Normalizes an array of actual parameter values to the [0, 1] interval based
    on normalization boundaries for each parameter.

    :param vector: Array of actual parameter values.
    :type vector: Iterable[Float] | NDArray[Float]
    :param properties: Parameter names corresponding to vector elements.
    :type properties: Iterable[str]
    :param normalization: Normalization map with min/max boundaries.
    :type normalization: dict[str, tuple[float, float]]
    :returns: Normalized parameter values.
    :rtype: list[float]

    Example:
        >>> # noinspection PyShadowingNames
        >>> normalization = {'param1': (0, 10), 'param2': (1.0, 2.0)}
        >>> vector_normalizer([5.0, 1.5], ('param1', 'param2'), normalization)
        [0.5, 0.5]

    """
    return [normalize_value(value, *normalization[prop]) for value, prop in zip(vector, properties, strict=False)]


def prepare_properties_set(xn: NDArray, properties: list[str] | Iterable[str], constrained: dict, fixed: dict) -> dict:
    """Prepare final keyword arguments for synthetic model evaluation.

    Combines variable parameters with their values, applies constraints,
    and adds fixed parameters to create a complete parameter set.

    :param xn: Array of variable parameter values.
    :type xn: NDArray
    :param properties: Parameter names corresponding to `xn` values.
    :type properties: list[str]
    :param constrained: Constrained initial parameters' dictionary.
    :type constrained: dict
    :param fixed: Fixed initial parameters' dictionary.
    :type fixed: dict
    :returns: Flat model parameters dictionary {'param@name': value, ...}.
    :rtype: dict
    """
    kwargs = dict(zip(properties, xn, strict=True))
    kwargs.update(constraints_evaluator(kwargs, constrained))
    fixed_values = {key: val.value if isinstance(val, InitialParameter) else val for key, val in fixed.items()}
    kwargs.update(fixed_values)
    return kwargs


def prepare_nuisance_properties_set(xn: NDArray | Iterable, properties: list[str] | Iterable[str], fixed: dict) -> dict:
    """Extract nuisance parameter values used during MCMC sampling.

    Extracts and combines nuisance parameters (such as `ln_f` jitter parameter)
    from both the variable vector and fixed parameters.

    :param xn: Array of variable parameter values.
    :type xn: NDArray
    :param properties: Variable parameter names corresponding to `xn`.
    :type properties: list[str]
    :param fixed: Fixed initial parameters' dictionary.
    :type fixed: dict
    :returns: Dictionary of nuisance parameter values.
    :rtype: dict
    """
    kwargs = {key: item for item, key in zip(xn, properties, strict=True) if conf.NUISANCE_PARSER in key}
    fixed_nuisance = {
        key: val.value if isinstance(val, InitialParameter) else val
        for key, val in fixed.items()
        if conf.NUISANCE_PARSER in key
    }
    kwargs.update(fixed_nuisance)
    return kwargs


def check_for_invalid_constraint(constrained: dict, allowed_params: list | dict | Any) -> None:
    """Validate that constraint expressions only reference allowed parameters.

    Ensures that constraint expressions, after substitution of valid parameters
    with their values, do not contain any invalid model parameters.

    :param constrained: Constrained parameters dictionary with expressions.
    :type constrained: dict[str, str]
    :param allowed_params: Names of valid model parameters that can be used.
    :type allowed_params: list | dict | Any
    :raises InitialParamsError: If constraint contains non-variable parameters.
    """
    for c_param, constraint in constrained.items():
        if conf.PARAM_PARSER in constraint:
            error_msg = (
                f"Your constraint for parameter {c_param} currently looks: {constraint} and "
                f"contains non-variable parameter. Only following parameters can be used to "
                f"define a constrained parameter: {allowed_params}"
            )
            raise InitialParamsError(error_msg)


def constraints_evaluator(substitution: dict, constrained: dict) -> dict:
    """Substitute variables in constraints with their values and evaluate to numbers.

    Replaces parameter references in constraint expressions with their actual values,
    then evaluates the mathematical expressions to produce final constraint values.

    :param substitution: Dictionary of parameter values for substitution.
    :type substitution: dict[str, float | InitialParameter]
    :param constrained: Dictionary of constraint expressions.
    :type constrained: dict[str, str | InitialParameter]
    :returns: Dictionary of evaluated constraint values.
    :rtype: dict[str, float]
    :raises InitialParamsError: If constraint syntax is invalid or evaluation fails.
    """
    if is_empty(constrained):
        return {}

    if isinstance(list(substitution.values())[-1], InitialParameter):
        substitution = {key: val.value for key, val in substitution.items()}

    if isinstance(list(constrained.values())[-1], InitialParameter):
        constrained = {key: val.constraint for key, val in constrained.items()}

    numpy_methods = [f"bonds.{method}" for method in TRANSFORM_TO_METHODS]
    allowed_methods = bonds.ALLOWED_CONSTRAINT_METHODS
    constrained = constrained.copy()

    subst = {
        key: utils.str_repalce(val, substitution.keys(), substitution.values()) for key, val in constrained.items()
    }
    numpy_callable = {key: utils.str_repalce(val, allowed_methods, numpy_methods) for key, val in subst.items()}

    # Check for invalid parameters in constraint
    check_for_invalid_constraint(numpy_callable, substitution.keys())

    try:
        evaluated = {key: eval(val) for key, val in numpy_callable.items()}  # noqa: S307
    except (ValueError, NameError, TypeError) as e:
        error_msg = f"Invalid syntax or value in constraint: {e!s}."
        raise InitialParamsError(error_msg) from e
    return evaluated


def extend_result_with_sma(fit_parameters: dict) -> dict:
    """Extend result dictionary with semi-major axis parameter.

    For light curve-based fits without a directly fitted semi-major axis,
    this function calculates and adds a fixed SMA value to the result
    parameters based on physical consistency.

    :param fit_parameters: Input fitting parameters' dictionary.
    :type fit_parameters: dict
    :returns: Result dictionary with semi-major axis added if applicable.
    :rtype: dict
    """
    if "mass_ratio" not in fit_parameters["system"]:
        logger.debug(
            "Binary system parameters supplied in standard format where SMA does not figure. Nothing to add.",
        )
        return fit_parameters

    if "semi_major_axis" in fit_parameters["system"]:
        logger.debug("Binary system parameters already contains `semi_major_axis` parameter.")
        return fit_parameters

    sma_estimate = []
    mid_g = 270  # m.s^-2, reasonable estimate of surface gravity
    mass_ratio = fit_parameters["system"]["mass_ratio"]["value"]
    period = fit_parameters["system"]["period"]["value"]
    for component in settings.BINARY_COUNTERPARTS:
        synchronicity = fit_parameters[component].get("synchronicity", {"value": 1.0})["value"]
        potential = fit_parameters[component]["surface_potential"]["value"]
        sma_estimate.append(
            calculate_sma_estimate(mass_ratio, synchronicity, potential, period, component, mid_g),
        )

    fit_parameters["system"]["semi_major_axis"] = {
        "value": np.mean(sma_estimate),
        "fixed": True,
        "unit": u.solRad,
    }
    return fit_parameters


class ParameterMeta:
    """Auxiliary handler for fit parameters that ensures each parameter attribute is present in results.

    Stores metadata about individual fit parameters including value, bounds, units, and constraints.
    This class provides a structured representation of parameter information used throughout
    the fitting process.

    :param kwargs: Keyword arguments containing parameter metadata.
    :type kwargs: dict
    :**kwargs contents**:
        * **param** (str) - Parameter name
        * **value** (float) - Parameter value
        * **unit** (astropy.unit.Unit) - Unit of the parameter
        * **fixed** (bool) - Whether parameter is fixed
        * **constraint** (str) - Constraint expression if applicable
        * **min** (float) - Minimum allowed value
        * **max** (float) - Maximum allowed value
        * **sigma** (float) - Standard deviation of prior distribution

    """

    def __init__(self, **kwargs) -> None:
        self.unit = kwargs.get("unit")
        self.param = kwargs.get("param")
        self.property = self.param
        self.fixed = kwargs.get("fixed")
        self.constraint = kwargs.get("constraint")
        self.value = kwargs.get("value")
        self.min = kwargs.get("min")
        self.max = kwargs.get("max")
        self.sigma = kwargs.get("sigma")

    def to_dict(self) -> dict:
        """Convert ParameterMeta to dictionary representation.

        :returns: Dictionary containing all parameter metadata.
        :rtype: dict
        """
        return {
            "value": self.value,
            "param": self.param,
            "min": self.min,
            "max": self.max,
            "unit": str(self.unit) if self.unit is not None else None,
            "fixed": self.fixed,
            "constraint": self.constraint,
            "sigma": self.sigma,
        }


class InitialParameter:
    """Store and manage attributes of a fit parameter.

    Encapsulates parameter metadata including value, bounds, units, and constraints.
    Handles unit conversions and parameter validation according to fitting requirements.

    :param transform_cls: Class type implementing TransformProperties protocol to transform fit parameters.
    :type transform_cls: type[TransformProperties]
    :param kwargs: Fit parameter attributes dictionary.
    :type kwargs: dict

    :**kwargs contents**:
        * **param** (str) - Parameter name
        * **value** (float) - Value assigned to the fit parameter
        * **unit** (astropy.unit.Unit) - Unit assigned to the value
        * **fixed** (bool) - If True, parameter is fixed during fit
        * **constraint** (str | None) - Mathematical expression constraining parameter to other variables
        * **min** (float | None) - Minimum allowed value
        * **max** (float | None) - Maximum allowed value (must be > min)
        * **sigma** (float | None) - Standard deviation of prior distribution for MCMC

    """

    DEFAULT: ClassVar = {
        "param": None,
        "value": None,
        "fixed": None,
        "constraint": None,
        "min": None,
        "max": None,
        "unit": None,
        "sigma": None,
    }

    def __init__(self, transform_cls: TransformProperties, **kwargs) -> None:
        self.unit: AstropyUnit | None = kwargs.get("unit")
        self.param: str | None = kwargs.get("param")
        self.property: str | None = self.param
        self.fixed: bool = kwargs.get("fixed", False)
        self.constraint: str | None = kwargs.get("constraint")
        self.value: Number | Any | None = None
        self.min: Number | Any | None = None
        self.max: Number | Any | None = None
        self.sigma: Number | Any | None = None

        # units transformaton
        self.unit = u.Unit(self.unit) if self.unit is not None else self.unit
        if self.unit is not None and self.unit is not u.dimensionless_unscaled and self.constraint is None:
            kwargs.update(
                {
                    "value": kwargs.get("value") * self.unit,
                    "min": kwargs.get("min") * self.unit,
                    "max": kwargs.get("max") * self.unit,
                },
            )
            if kwargs.get("sigma") is not None:
                kwargs.update({"sigma": kwargs.get("sigma") * self.unit})

        if self.constraint is None:
            self.value = transform_cls.transform_input(**{self.param: kwargs.get("value")})[self.param]
            self.min = transform_cls.transform_input(**{self.param: kwargs.get("min")})[self.param]
            self.max = transform_cls.transform_input(**{self.param: kwargs.get("max")})[self.param]
            if kwargs.get("sigma") is not None:
                self.sigma = transform_cls.transform_input(**{self.param: kwargs.get("sigma")})[self.param]

        if self.fixed:
            self.min, self.max, self.sigma = None, None, None

        self.unit = conf.DEFAULT_FLOAT_UNITS[self.property]

    def copy(self) -> InitialParameter:
        """Create independent deep copy of the InitialParameter.

        :returns: Deep copy of this InitialParameter instance.
        :rtype: InitialParameter
        """
        return deepcopy(self)

    def to_dict(self) -> dict:
        """Transform InitialParameter into dictionary format.

        :returns: Dictionary containing all InitialParameter metadata.
        :rtype: dict
        """
        return dict(
            value=self.value,
            param=self.param,
            min=self.min,
            max=self.max,
            unit=str(self.unit) if self.unit is not None else None,
            sigma=self.sigma,
            **{"fixed": self.fixed} if self.fixed is not None else {},
            **{"constraint": self.constraint} if self.constraint is not None else {},
        )

    def __repr__(self) -> str:
        return json.dumps(
            {
                "param": self.param,
                "value": self.value,
                "fixed": self.fixed,
                "constraint": self.constraint,
                "min": self.min,
                "max": self.max,
                "sigma": self.sigma,
                "unit": str(self.unit) if self.unit is not None else None,
            },
            indent=4,
        )

    __str__ = __repr__


class InitialParameters(metaclass=abc.ABCMeta):
    """Handle sets of initial fit parameters for system modeling.

    Abstract base class for managing collections of initial fit parameters,
    including validation, transformation, and constraint handling for various
    binary system or single-star fitting scenarios.
    """

    TRANSFORM_PROPERTIES_CLS: ClassVar[TransformProperties]
    DEFAULT_NORMALIZATION: ClassVar[dict[str, tuple]]

    @property
    def slots_(self) -> list:
        """Get list of attribute slot names for this object.

        :returns: List of slot names or empty list if no slots defined.
        :rtype: list[str]
        """
        if hasattr(self, "__slots__"):
            return self.__slots__
        return []

    def validity_check(self) -> None:
        """Examine whether inputted definitions of initial fit parameters are valid.

        Verifies that InitialParameter.min <= InitialParameter.value <= InitialParameter.max
        and that InitialParameter is either:

        * **fixed**: InitialParameter.fixed = True, InitialParameter.constraint = None
        * **variable**: InitialParameter.fixed = False, InitialParameter.constraint = None
        * **constrained**: InitialParameter.fixed = None, constraint = valid expression

        :raises InitialParamsError: If bounds are invalid or both fixed and constraint are set.
        """
        for slot in self.slots_:
            if not hasattr(self, str(slot)):
                continue
            prop = getattr(self, str(slot))

            if not isinstance(prop, InitialParameter):
                continue

            if prop.constraint is None and not prop.fixed and not (prop.min <= prop.value <= prop.max):
                error_msg = (
                    f"Initial parameters in parameter `{prop.param}` are not valid. "
                    f"Invalid bounds: {prop.min} <= {prop.value} <= {prop.max}"
                )
                raise InitialParamsError(error_msg)
            if prop.fixed is not None and prop.constraint is not None:
                error_msg = f"It is not allowed for `{prop.param}` to contain both `fixed` and `constraint` parameter."
                raise InitialParamsError(error_msg)

    def init_parameter(self, parameter: str, items: dict) -> InitialParameter:
        """Initialize InitialParameter instance from dictionary definition.

        Creates a properly configured InitialParameter object with unit conversion,
        bounds setting, and validation according to the transformation class specifications.

        :param parameter: Name of the fit parameter.
        :type parameter: str
        :param items: Dictionary definition of the fit parameter containing value, unit, fixed, min, max, sigma, etc.
        :type items: dict
        :returns: Initialized InitialParameter instance.
        :rtype: InitialParameter
        """
        _kwarg = InitialParameter.DEFAULT.copy()
        _kwarg.update(dict(param=parameter, **items))
        _kwarg.update(
            {
                "min": items.get("min", self.DEFAULT_NORMALIZATION[parameter][0]),
                "max": items.get("max", self.DEFAULT_NORMALIZATION[parameter][1]),
            },
        )
        return InitialParameter(transform_cls=self.__class__.TRANSFORM_PROPERTIES_CLS, **_kwarg)


class SpotInitialParameters(InitialParameters):
    """Manage spot-related fit parameters.

    Handles initialization, validation, and storage of stellar spot model parameters
    including position, size, and temperature properties.
    """

    __slots__ = ("angular_radius", "label", "latitude", "longitude", "temperature_factor")

    TRANSFORM_PROPERTIES_CLS = SpotInitialProperties
    DEFAULT_NORMALIZATION = conf.DEFAULT_NORMALIZATION_SPOT

    def __init__(self, **kwargs) -> None:
        """Initialize spot parameters from provided dictionary.

        :param kwargs: Spot parameter definitions.
        :type kwargs: dict
        """
        self.label = f"spot{conf.PARAM_PARSER}{kwargs.pop('label')}"
        for parameter, items in kwargs.items():
            value = self.init_parameter(parameter, items)
            setattr(self, parameter, value)
        self.validity_check()


class PulsationInitialParameters(InitialParameters):
    """Manage pulsation mode-related fit parameters.

    Handles initialization, validation, and storage of asteroseismic mode parameters
    including frequency, amplitude, and spherical harmonic degree/order.
    """

    __slots__ = ("amplitude", "frequency", "l", "label", "m", "mode_axis_phi", "mode_axis_theta", "start_phase")

    TRANSFORM_PROPERTIES_CLS = PulsationModeInitialProperties
    DEFAULT_NORMALIZATION = conf.DEFAULT_NORMALIZATION_PULSATION

    def __init__(self, **kwargs) -> None:
        """Initialize pulsation mode parameters from provided dictionary.

        :param kwargs: Pulsation mode parameter definitions.
        :type kwargs: dict
        """
        self.label = f"pulsation{conf.PARAM_PARSER}{kwargs.pop('label')}"
        self.mode_axis_phi = 0
        self.mode_axis_theta = 0

        for parameter, items in kwargs.items():
            value = self.init_parameter(parameter, items)
            setattr(self, parameter, value)
        self.validity_check()


class StarInitialParameters(InitialParameters):
    """Manage stellar component-related fit parameters.

    Handles initialization, validation, and storage of single stellar component parameters
    including effective temperature, potential, gravity darkening, and surface phenomena
    (spots and pulsations).
    """

    __slots__ = [
        "albedo",
        "gravity_darkening",
        "mass",
        "metallicity",
        "pulsations",
        "spots",
        "surface_potential",
        "synchronicity",
        "t_eff",
    ]

    TRANSFORM_PROPERTIES_CLS = StarInitialProperties
    DEFAULT_NORMALIZATION = conf.DEFAULT_NORMALIZATION_STAR

    def __init__(self, **kwargs) -> None:
        """Initialize stellar component parameters from provided dictionary.

        :param kwargs: Stellar parameter definitions including optional spots and pulsations.
        :type kwargs: dict
        """
        self.label = None

        spots = kwargs.pop("spots", [])
        pulsations = kwargs.pop("pulsations", [])

        for parameter, items in kwargs.items():
            value = self.init_parameter(parameter, items)
            setattr(self, parameter, value)

        if not is_empty(spots):
            spots = [SpotInitialParameters(**spot) for spot in spots]
            self.spots = spots

        if not is_empty(pulsations):
            pulsations = [PulsationInitialParameters(**pulsation) for pulsation in pulsations]
            self.pulsations = pulsations

        self.validity_check()


class NuisanceInitialPrameters(InitialParameters):
    """Manage auxiliary nuisance fit parameters for MCMC sampling.

    Handles initialization and storage of auxiliary parameters used during MCMC
    fitting to produce realistic confidence interval estimates.
    """

    __slots__ = ["ln_f"]

    TRANSFORM_PROPERTIES_CLS = NuisanceInitialProperties
    DEFAULT_NORMALIZATION = conf.DEFAULT_NORMALIZATION_NUISANCE

    def __init__(self, **kwargs) -> None:
        """Initialize nuisance parameters from provided dictionary.

        :param kwargs: Nuisance parameter definitions.
        :type kwargs: dict
        """
        for parameter, items in kwargs.items():
            value = self.init_parameter(parameter, items)
            setattr(self, parameter, value)
        self.validity_check()


# noinspection PyTypeHints
class BinaryInitialParameters(InitialParameters):
    """Manage binary system-related fit parameters.

    Comprehensive handler for all fitting parameters of an eclipsing binary system,
    including system orbital parameters, component stellar parameters, and optional
    surface phenomena (spots and pulsations).
    """

    __slots__ = [
        "primary",
        "secondary",
        "eccentricity",
        "argument_of_periastron",
        "inclination",
        "gamma",
        "period",
        "mass_ratio",
        "asini",
        "semi_major_axis",
        "additional_light",
        "phase_shift",
        "primary_minimum_time",
        conf.NUISANCE_PARSER,
    ]

    TRANSFORM_PROPERTIES_CLS = BinaryInitialProperties
    DEFAULT_NORMALIZATION = conf.DEFAULT_NORMALIZATION_SYSTEM

    def __init__(self, **kwargs) -> None:
        """Initialize binary system parameters from provided dictionary.

        :param kwargs: Binary system parameter definitions including system, primary, and secondary components.
        :type kwargs: dict
        """
        self._primary: dict | None = kwargs.pop("primary", None)
        self._secondary: dict | None = kwargs.pop("secondary", None)
        self._nuisance = kwargs.pop(conf.NUISANCE_PARSER, None)
        system: dict | None = kwargs.pop("system")

        # nuisance params
        if self._nuisance is None:
            self._nuisance = {"ln_f": {"value": -20, "fixed": True}}
        self.nuisance = NuisanceInitialPrameters(**self._nuisance)

        # system params
        for parameter, items in system.items():
            value = self.init_parameter(parameter, items)
            setattr(self, parameter, value)

        # components params
        self.primary: StarInitialParameters | None = None
        self.secondary: StarInitialParameters | None = None

        for component in settings.BINARY_COUNTERPARTS:
            props = getattr(self, f"_{component}")
            if is_empty(props):
                continue
            star_params = StarInitialParameters(**props)
            setattr(self, component, star_params)
            star_params.label = component

        self.validity_check()
        self.unique_labels_validation()
        self._data = self.serialize_flat_set()
        self.validate_data()

    @property
    def data(self) -> dict:
        """Return complete set of binary system model parameters in standard format.

        :returns: Binary system parameters compiled from BinaryInitialParameters instance.
        :rtype: dict[str, InitialParameter]
        """
        return self._data

    def __getitem__(self, item: str | list) -> Any:
        """Get parameter or parameters from the model.

        :param item: Parameter name or list of parameter names to retrieve.
        :type item: str | list
        :returns: Single parameter or list of parameters.
        :rtype: InitialParameter | list[InitialParameter]
        """
        iterable = True
        if not isinstance(item, Iterable):
            iterable = True
            item = [item]
        data = [self.data[_item] for _item in item]
        return data if iterable else data[-1]

    def to_flat_json(self) -> list[dict[Any, Any]]:
        """Return a complete set of binary system model parameters in flat JSON format.

        Compiles all parameters from the BinaryInitialParameters instance into a
        flat JSON-compatible list format, with each parameter converted to its dictionary representation.

        :returns: List of parameter dictionaries.
        :rtype: list[dict[Any, Any]]
        """
        return [{key: val.to_dict()} for key, val in self.data.items()]

    def unique_labels_validation(self) -> None:
        """Validate that all spot and pulsation labels are unique within components.

        Ensures no two spots or pulsations in the same stellar component have
        the same label identifier.

        :raises InitialParamsError: If duplicate labels are found.
        """

        def _test(_what: str) -> None:
            for component in settings.BINARY_COUNTERPARTS:
                if hasattr(self, component):
                    inst = getattr(self, component)
                    if hasattr(inst, _what):
                        phenom = getattr(inst, _what)
                        _all = [_phenom.label for _phenom in phenom]
                        if len(_all) != len(set(_all)):
                            msg = f"It is not allowed to have multiple {_what} with same label."
                            raise InitialParamsError(msg)

        _test("spots")
        _test("pulsations")

    def serialize_flat_set(self) -> dict:  # noqa: C901, PLR0912
        """Return flat set of InitialParameters in JSON-compatible format.

        Converts InitialParameters to nested structure matching fit results JSON format,
        with @ delimiters used for parameter path identification.

        :returns: Nested InitialParameter dictionary.
        :rtype: dict[str, InitialParameter | dict]
        """
        data = {}

        for system_slot in self.__slots__:
            if not hasattr(self, str(system_slot)):
                continue
            system_prop = getattr(self, str(system_slot))
            if isinstance(system_prop, InitialParameter):
                data.update({f"system{conf.PARAM_PARSER}{system_slot}": system_prop})
            elif isinstance(system_prop, NuisanceInitialPrameters):
                for nuisance_slot in system_prop.__slots__:
                    key = f"{conf.NUISANCE_PARSER}{conf.PARAM_PARSER}{nuisance_slot}"
                    data.update({key: getattr(system_prop, nuisance_slot)})
            elif isinstance(system_prop, StarInitialParameters):
                for component_slot in system_prop.__slots__:
                    if not hasattr(system_prop, str(component_slot)):
                        continue
                    component_prop = getattr(system_prop, str(component_slot))
                    if isinstance(component_prop, InitialParameter):
                        data.update({f"{system_prop.label}{conf.PARAM_PARSER}{component_slot}": component_prop})

                    elif isinstance(component_prop, list):
                        for phenomena_prop in component_prop:
                            if isinstance(phenomena_prop, (PulsationInitialParameters, SpotInitialParameters)):
                                for phenomena_slot in phenomena_prop.__slots__:
                                    if not hasattr(phenomena_prop, str(phenomena_slot)):
                                        continue
                                    prop = getattr(phenomena_prop, str(phenomena_slot))

                                    if isinstance(prop, InitialParameter):
                                        data.update(
                                            {
                                                f"{system_prop.label}"
                                                f"{conf.PARAM_PARSER}{phenomena_prop.label}"
                                                f"{conf.PARAM_PARSER}{phenomena_slot}": prop,
                                            },
                                        )

        return data

    def constraint_validator(self) -> None:
        """Validate constraints for valid syntax and allowed mathematical operations.

        Ensures constraints only use allowed functions and characters. Allowed methods:
        sqrt, arcsin, arccos, arctan, log, sin, cos, tan, exp, degrees, radians.

        Allowed characters: '(', ')', '+', '-', '*', '/', '.', 'e'

        :raises ValidationError: If constraint contains forbidden syntax or characters.
        """
        constrained = self.get_constrained(jsonify=False)
        substitution = self.get_substitution_dict()
        try:
            subst = {
                param: utils.str_repalce(
                    utils.str_repalce(constraint.constraint, substitution.keys(), substitution.values()),
                    ALLOWED_CONSTRAINT_METHODS,
                    [""] * len(ALLOWED_CONSTRAINT_METHODS),
                ).replace(" ", "")
                for param, constraint in constrained.items()
            }
        except KeyError as exc:
            msg = (
                "It seems your constraint contain variable that cannot be resolved. "
                "Make sure that linked constraint variable is not fixed or check for typos in variable name in "
                "constraint expression."
            )
            raise ValidationError(msg) from exc

        for key, val in subst.items():
            if not np.all(np.isin(list(val), ALLOWED_CONSTRAINT_CHARS)):
                msg = f"Constraint {key} contain forbidden characters. Allowed: {ALLOWED_CONSTRAINT_CHARS}"
                raise ValidationError(msg)

    def validate_data(self) -> None:
        """Validate that fit parameters contain at least one variable parameter.

        Checks that the model has parameters that can be varied during fitting and
        validates all constrained parameters for mathematical correctness.

        :raises ValidationError: If no variable parameters exist or constraints are invalid.
        """
        # validate that at least one parameter is not fixed
        if len(self.get_fitable(jsonify=False)) == 0:
            msg = "There are no variable parameters to fit."
            raise ValidationError(msg)

        # constraint validation
        self.constraint_validator()

    def adjust_overcontact_potential(self, morphology: str) -> None:
        """Adjust secondary potential constraint for over-contact morphology.

        For over-contact systems during fitting, this function constrains
        the secondary component's surface potential to equal the primary's.

        :param morphology: System morphology string ('over-contact', 'detached', etc.).
        :type morphology: str
        """
        if self.is_overcontact(morphology):
            update_surface_potential = self.primary.surface_potential.copy()
            update_surface_potential.fixed = None
            update_surface_potential.constraint = "primary@surface_potential"
            self.secondary.surface_potential = update_surface_potential
            self.data["secondary@surface_potential"] = update_surface_potential

    def validate_lc_parameters(self, morphology: str) -> None:
        """Validate parameters for light curve fitting.

        Ensures all mandatory light curve parameters are present and valid.
        Checks consistency constraints for over-contact morphology systems.

        :param morphology: System morphology ('over-contact' or 'detached').
        :type morphology: str
        :raises ValidationError: If required parameters are missing or incompatible.
        """
        mandatory_fit_params = [
            "system@eccentricity",
            "system@argument_of_periastron",
            "system@period",
            "system@inclination",
            "system@period",
        ] + [
            f"{component}@{param}"
            for param in ["t_eff", "surface_potential"]
            for component in settings.BINARY_COUNTERPARTS
        ]

        optional_fit_params = [
            "system@semi_major_axis",
            "system@primary_minimum_time",
            "system@phase_shift",
            "system@asini",
            "system@mass_ratio",
            "system@additional_light",
            "nuisance@ln_f",
        ] + [
            f"{component}@{param}"
            for param in ["mass", "synchronicity", "metallicity", "spots", "pulsations", "gravity_darkening", "albedo"]
            for component in settings.BINARY_COUNTERPARTS
        ]

        all_fit_params = mandatory_fit_params + optional_fit_params
        utils.check_missing_kwargs(mandatory_fit_params, self.data, instance_of=self.__class__)
        check_initial_param_validity(self.data, all_fit_params, mandatory_fit_params)

        is_oc = self.is_overcontact(morphology)
        are_same = self.data["primary@surface_potential"].value == self.data["secondary@surface_potential"].value

        is_fixed_omega_1 = self.data["primary@surface_potential"].fixed or False
        is_fixed_omega_2 = self.data["secondary@surface_potential"].fixed or False

        any_fixed = is_fixed_omega_1 | is_fixed_omega_2
        all_fixed = is_fixed_omega_1 & is_fixed_omega_2

        if is_oc and all_fixed and are_same:
            return
        if is_oc and all_fixed and not are_same:
            msg = "Different potential in over-contact morphology with all fixed (potential) value are not allowed."
            raise ValidationError(msg)
        if is_oc and any_fixed:
            msg = "Just one fixed potential in over-contact morphology is not allowed."
            raise ValidationError(msg)

        # adjust constraint for secondary potential
        self.adjust_overcontact_potential(morphology)

    def validate_rv_parameters(self) -> None:
        """Validate parameters for radial velocity curve fitting.

        Ensures all mandatory RV parameters are present and valid.
        Checks consistency of time and period parameters.

        :raises ValidationError: If required parameters are missing or inconsistent.
        """
        mandatory_fit_params = ["system@eccentricity", "system@argument_of_periastron", "system@gamma"]
        optional_fit_params = [
            "system@period",
            "system@primary_minimum_time",
            "primary@mass",
            "secondary@mass",
            "system@inclination",
            "system@asini",
            "system@mass_ratio",
            "nuisance@ln_f",
        ]
        all_fit_params = mandatory_fit_params + optional_fit_params
        utils.check_missing_kwargs(mandatory_fit_params, self.data, instance_of=self.__class__)
        check_initial_param_validity(self.data, all_fit_params, mandatory_fit_params)

        # validate consistency of parameters (system has to be definable)
        has_primary_minimum_time, has_period = "system@primary_minimum_time" in self.data, "system@period" in self.data
        if has_primary_minimum_time:
            if not (has_primary_minimum_time and has_period):
                msg = "Input requires both, period and primary minimum time."
                raise ValidationError(msg)
        elif not has_period:
            msg = "Input requires at least period."
            raise ValidationError(msg)

    def _get_kind_of(self, kind_of: str, *, jsonify: bool = False) -> dict:
        """Extract parameters of a specific kind from the model.

        :param kind_of: Property name to filter by ('fixed', 'constraint', etc.).
        :type kind_of: str
        :param jsonify: If True, convert to JSON-compatible format. Defaults to False.
        :type jsonify: bool
        :returns: Dictionary of parameters matching the filter.
        :rtype: dict
        """
        if jsonify:
            return {key: val.to_dict() for key, val in self.data.items() if getattr(val, kind_of)}
        return {key: val for key, val in self.data.items() if getattr(val, kind_of)}

    def get_fixed(self, *, jsonify: bool = False) -> dict:
        """Get all fixed parameters from the model.

        :param jsonify: If True, convert to JSON-compatible format. Defaults to False.
        :type jsonify: bool
        :returns: Dictionary of fixed parameters.
        :rtype: dict[str, InitialParameter | dict]
        """
        return self._get_kind_of(kind_of="fixed", jsonify=jsonify)

    def get_constrained(self, *, jsonify: bool = False) -> dict:
        """Get all constrained parameters from the model.

        Parameters with mathematical constraints (e.g., secondary@surface_potential = 2.0 * primary@surface_potential).

        :param jsonify: If True, convert to JSON-compatible format. Defaults to False.
        :type jsonify: bool
        :returns: Dictionary of constrained parameters.
        :rtype: dict[str, InitialParameter | dict]
        """
        return self._get_kind_of(kind_of="constraint", jsonify=jsonify)

    def get_fitable(self, *, jsonify: bool = False) -> dict:
        """Get all fitable (variable) parameters from the model.

        These are parameters that should be varied during fitting.

        :param jsonify: If True, convert to JSON-compatible format. Defaults to False.
        :type jsonify: bool
        :returns: Dictionary of fitable parameters.
        :rtype: dict[str, InitialParameter | dict]
        """
        if jsonify:
            return {key: val.to_dict() for key, val in self.data.items() if not val.constraint and not val.fixed}
        return {key: val for key, val in self.data.items() if not val.constraint and not val.fixed}

    def get_substitution_dict(self) -> dict:
        """Return dictionary of model parameters with their values for substitution.

        Excludes constrained parameters (which will be evaluated from constraints).

        :returns: Dictionary of parameter names and values for substitution.
        :rtype: dict[str, float | Any]
        """
        return {key: val.value for key, val in self.data.items() if not val.constraint}

    def get_normalization_map(self) -> dict:
        """Return normalization boundaries (min, max) for all parameters.

        :returns: Mapping of parameter names to (min, max) boundary tuples.
        :rtype: dict[str, tuple[float, float]]
        """
        return {key: (val.min, val.max) for key, val in self.data.items()}

    @staticmethod
    def is_overcontact(morphology: str) -> bool:
        """Check if system morphology is over-contact.

        :param morphology: System morphology string ('over-contact', 'detached', etc.).
        :type morphology: str
        :returns: True if over-contact morphology, False otherwise.
        :rtype: bool
        """
        return morphology in ["over-contact", "overcontact"]
