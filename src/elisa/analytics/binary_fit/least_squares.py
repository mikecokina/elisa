from __future__ import annotations

import functools
import logging
from abc import ABCMeta
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares

from elisa import const, settings
from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.analytics.binary_fit.sampling import SamplingMethod
from elisa.analytics.binary_fit.shared import (
    AbstractLCFit,
    AbstractRVFit,
    map_rv_predictions,
    r_squared,
    rv_r_squared,
)
from elisa.analytics.models import cost_fns
from elisa.analytics.models import lc as lc_model
from elisa.analytics.models import rv as rv_model
from elisa.analytics.params import parameters
from elisa.base.types import FLOAT
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from elisa.logger import getPersistentLogger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.analytics.dataset.base import LCData, RVData
    from elisa.types import Float

logger = getPersistentLogger(__name__)

def _warn(e: Exception) -> None:
    """Log exception as a warning if warnings are not suppressed."""
    if not settings.SUPPRESS_WARNINGS:
        error_msg = f"lsqrt hit invalid parameters, exception: {e!s}"
        logger.warning(error_msg)


def logger_decorator(*, suppress_logger: bool = False) -> Any:
    """Decorate function to add optional parameter logging capability.

    :param suppress_logger: If True, skip logging of parameter values.
    :type suppress_logger: bool
    :return: Decorated function wrapper.
    :rtype: Any
    """

    def do(func: Any) -> Any:
        """Decorate function with logging capability.

        :param func: Function to decorate.
        :type func: Any
        :return: Wrapped function.
        :rtype: Any
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrap function with optional parameter logging.

            :param args: Positional arguments to pass to function.
            :type args: Any
            :param kwargs: Keyword arguments to pass to function.
            :type kwargs: Any
            :return: Result of the original function.
            :rtype: Any
            """
            if not suppress_logger:
                logger.debug("current xn value: %s", kwargs)
            return func(*args, **kwargs)

        return wrapper

    return do


# Create logger decorator instance for reuse
_log_decorator = logger_decorator(suppress_logger=False)


class LightCurveFit(AbstractLCFit, metaclass=ABCMeta):
    """General class for solving inverse problem in case of LC data.

    This class provides the foundation for light curve fitting using least-squares
    optimization. It handles synthetic light curve generation, cost function calculation,
    and parameter optimization for both detached and over-contact binary systems.
    """

    MORPHOLOGY: str | None

    # Attributes from parent class (set by set_up method)
    fit_xs: None
    y_err: dict[str, Any]
    y_data: dict[str, NDArray]
    x_data_reducer: dict[str, NDArray]
    x_data_reduced: NDArray
    interp_treshold: int
    discretization: float
    observer: Any
    atmosphere_model: None
    limb_darkening_coefficients: None
    fitable: dict[str, Any]
    constrained: dict[str, Any]
    fixed: dict[str, Any]
    initial_vector: list[Any]
    normalization: dict[str, tuple[Any, ...]]
    flat_result: dict[str, Any]


    def model_to_fit(self, xn: NDArray) -> NDArray:
        """Return standardized observation residuals for SciPy least squares.

        Concatenate (observed - predicted) / uncertainty in observation-dictionary
        insertion order. Linear loss then minimizes 0.5 * chi²; robust losses act
        on individual standardized observations, and Jacobians have one row per
        datum with derivatives in normalized parameter coordinates.

        :param xn: Normalized free-parameter vector in [0, 1].
        :return: One-dimensional residual vector of fixed observation length.
            Invalid physical synthesis returns a finite vector whose squared norm
            is MAX_USABLE_FLOAT, preserving that constant as the chi² penalty.
        :raises PredictionAlignmentError: If dynamic synthesis alignment fails.
        """
        xn_list = parameters.vector_renormalizer(xn, self.fitable.keys(), self.normalization)
        xn = np.asarray(xn_list, dtype=FLOAT)
        kwargs: dict[str, Any] = parameters.prepare_properties_set(
            xn,
            self.fitable.keys(),
            self.constrained,
            self.fixed,
        )
        try:
            synthetic = self.predict_light_curves(kwargs, synthesis=_log_decorator(lc_model.synthetic_binary))
        except SYNTHESIS_FAILURES as e:
            # Only the shared policy's types/subclasses become proposal penalties.
            # Other exceptions escape this callback and abort least-squares fitting.
            _warn(e)
            count = sum(self.num_of_points.values())
            return np.full(count, np.sqrt(const.MAX_USABLE_FLOAT / count))

        residuals = cost_fns.standardized_residuals(self.y_data, self.y_err, synthetic, validate=False)

        if logger.isEnabledFor(logging.INFO):
            logger.info("current R2: %s", r_squared(synthetic, self.y_data))

        return residuals

    def fit(
        self,
        data: dict[str, LCData],
        x0: parameters.BinaryInitialParameters,
        *,
        discretization: Float = 5.0,
        interp_treshold: int | None = None,
        samples: SamplingMethod | str | list | NDArray = SamplingMethod.UNIFORM,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fit light curve using non-linear least squares optimization.

        Uses scipy.optimize.least_squares for fitting. Can handle both physical
        parameters (component masses) and astro community parameters (asini, q).

        :param data: Observational light curve data in photometric filters.
        :type data: dict[str, LCData]
        :param x0: Initial state of binary system parameters.
        :type x0: parameters.BinaryInitialParameters
        :param discretization: Discretization factor for primary component surface.
            Default is 5.0.
        :type discretization: Float
        :param interp_treshold: Number of datapoints above which light curve will be
            interpolated using equidistant points. If None, uses MAX_CURVE_DATAPOINTS
            from settings.
        :type interp_treshold: int | None
        :param samples: SamplingMethod.UNIFORM (equidistant in phase),
            SamplingMethod.ADAPTIVE (equidistant on curve), or a manual phase
            list/array. Legacy string values remain accepted. Defaults to UNIFORM.
        :type samples: SamplingMethod | str | list | NDArray
        :param kwargs: Optional arguments for scipy.optimize.least_squares function.
            Common options: jac, method, ftol, xtol, gtol, x_scale, loss, f_scale,
            diff_step, tr_solver, tr_options, jac_sparsity, max_nfev, verbose.
            Jacobians and jac_sparsity have one row per observation and one
            column per normalized free parameter. Robust loss and f_scale apply
            to individual standardized residuals. Solver diagnostics default to
            verbose=0; pass verbose=2 explicitly to print SciPy iteration tables.
        :type kwargs: Any
        :return: Optimized model parameters in standard JSON format.
        :rtype: dict[str, Any]
        """
        self.set_up(
            x0,
            data,
            passband=data.keys(),
            discretization=discretization,
            morphology=self.MORPHOLOGY,
            interp_treshold=(settings.MAX_CURVE_DATAPOINTS if interp_treshold is None else interp_treshold),
            observer_system_cls=BinarySystem,
            samples=samples,
        )


        initial_vector: NDArray = np.asarray(
            parameters.vector_normalizer(
                self.initial_vector,
                self.fitable.keys(),
                self.normalization,
            ),
            dtype=FLOAT,
        )

        logger.info("fitting started...")
        result = least_squares(
            self.model_to_fit,
            initial_vector,
            jac=kwargs.get("jac", "2-point"),
            bounds=(0, 1),
            method=kwargs.get("method", "trf"),
            ftol=kwargs.get("ftol", 1e-7),
            xtol=kwargs.get("xtol", 1e-8),
            gtol=kwargs.get("gtol", 1e-8),
            x_scale=kwargs.get("x_scale", 1.0),
            loss=kwargs.get("loss", "linear"),
            f_scale=kwargs.get("f_scale", 1.0),
            diff_step=kwargs.get("diff_step"),
            tr_solver=kwargs.get("tr_solver"),
            tr_options=kwargs.get("tr_options", {}),
            jac_sparsity=kwargs.get("jac_sparsity"),
            max_nfev=kwargs.get("max_nfev"),
            verbose=kwargs.get("verbose", 0),
            args=kwargs.get("args", ()),
            kwargs=kwargs.get("kwargs", {}),
        )
        logger.info("fitting finished")

        result_normalized: NDArray = np.asarray(
            parameters.vector_renormalizer(
                result.x,
                self.fitable.keys(),
                self.normalization,
            ),
            dtype=FLOAT,
        )

        # Build result dictionary from fitted parameters
        result_dict: dict[str, dict[str, Any]] = {
            lbl: {
                "value": result_normalized[i],
                "fixed": False,
                "unit": self.fitable[lbl].to_dict()["unit"],
                "min": self.fitable[lbl].min,
                "max": self.fitable[lbl].max,
            }
            for i, lbl in enumerate(self.fitable.keys())
        }

        # Add fixed parameters
        result_dict.update(
            {
                lbl: {"value": val.value, "fixed": True, "unit": val.to_dict()["unit"]}
                for lbl, val in self.fixed.items()
            },
        )
        result_dict = self.eval_constrained_results(result_dict, self.constrained)

        # Use the fitted prediction path, support and existing Observer for R².
        r_dict = {key: value["value"] for key, value in result_dict.items()}
        r_squared_result = self.prediction_r_squared(r_dict)
        result_dict["r_squared"] = {"value": r_squared_result, "unit": None}

        self.flat_result = result_dict
        return parameters.serialize_result(result_dict)


class OvercontactLightCurveFit(LightCurveFit):
    """Optimization class for solving inverse problem for overcontact systems.

    This class extends LightCurveFit to provide specialized functionality for
    fitting overcontact binary systems where the component stars share a common
    envelope.
    """

    MORPHOLOGY: str = "over-contact"


class DetachedLightCurveFit(LightCurveFit):
    """Optimization class for solving inverse problem for detached systems.

    This class extends LightCurveFit to provide specialized functionality for
    fitting detached binary systems where the component stars are not in contact.
    """

    MORPHOLOGY: str = "detached"


class CentralRadialVelocity(AbstractRVFit):
    """Class for fitting radial velocities using kinematic method.

    This class provides functionality to fit radial velocity observations using
    least-squares optimization. It handles synthetic RV generation and cost
    function calculation for binary star systems.
    """

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


    def prepare_synthetic(self, xn: NDArray) -> dict[str, NDArray]:
        """Return synthetic radial velocity observations for given parameters.

        Generates synthetic radial velocity observations for the given set of
        normalized variable parameters.

        :param xn: Vector of variable model parameters in normalized form [0, 1].
        :type xn: NDArray
        :return: Synthetic radial velocity observations for each component.
        :rtype: dict[str, NDArray]
        """
        xn_list = parameters.vector_renormalizer(xn, self.fitable.keys(), self.normalization)
        xn_normalized = np.asarray(xn_list, dtype=FLOAT)
        kwargs: dict[str, Any] = parameters.prepare_properties_set(
            xn_normalized,
            self.fitable.keys(),
            self.constrained,
            self.fixed,
        )
        fn = rv_model.central_rv_synthetic
        synthetic: dict[str, NDArray] = _log_decorator(fn)(
            self.x_data_reduced,
            self.observer,
            **kwargs,
        )
        return synthetic

    def central_rv_model_to_fit(self, xn: NDArray) -> NDArray:
        """Return per-observation standardized RV residuals for SciPy.

        :param xn: Normalized free-parameter vector.
        :return: Concatenated (observed - predicted) / uncertainty, in observation
            dictionary insertion order. Its squared norm is the actual chi².
        :raises PredictionAlignmentError: If raw RV components or shapes do not match the synthesis grid.
        """
        try:
            synthetic: dict[str, NDArray] = self.prepare_synthetic(xn)
        except SYNTHESIS_FAILURES as e:
            # Only the shared policy's types/subclasses become proposal penalties.
            # Other exceptions escape this callback and abort least-squares fitting.
            _warn(e)
            count = sum(self.num_of_points.values())
            return np.full(count, np.sqrt(const.MAX_USABLE_FLOAT / count))
        synthetic = map_rv_predictions(synthetic, self.x_data_reduced, self.x_data_reducer)
        return cost_fns.standardized_residuals(self.y_data, self.y_err, synthetic, validate=False)

    def fit(
        self,
        data: dict[str, RVData],
        x0: parameters.BinaryInitialParameters,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fit radial velocity curves using non-linear least squares optimization.

        Performs fitting of radial velocities using scipy.optimize.least_squares.
        Can handle both physical parameters (component masses) and astro community
        parameters (asini, q).

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        :param data: Radial velocity observations for primary and secondary components.
        :type data: dict[str, RVData]
        :param x0: Initial state of binary system model parameters.
        :type x0: parameters.BinaryInitialParameters
        :param kwargs: Optional arguments for scipy.optimize.least_squares function.
            Common options: jac, method, ftol, xtol, gtol, x_scale, loss, f_scale,
            diff_step, tr_solver, tr_options, jac_sparsity, max_nfev, verbose.
            Jacobians and jac_sparsity have one row per observation and one
            column per normalized free parameter. Robust loss and f_scale apply
            to individual standardized residuals. Solver diagnostics default to
            verbose=0; pass verbose=2 explicitly to print SciPy iteration tables.
        :type kwargs: Any
        :return: Optimized model parameters in standard JSON format.
        :rtype: dict[str, Any]
        """
        self.set_up(x0, data, observer_system_cls=RadialVelocitySystem)
        logger.info("fitting radial velocity light curve...")

        func = self.central_rv_model_to_fit
        initial_vector_list = parameters.vector_normalizer(
            self.initial_vector,
            self.fitable.keys(),
            self.normalization,
        )
        initial_vector: NDArray = np.asarray(initial_vector_list, dtype=FLOAT)
        result = least_squares(
            func,
            initial_vector,
            jac=kwargs.get("jac", "2-point"),
            bounds=(0, 1),
            method=kwargs.get("method", "trf"),
            ftol=kwargs.get("ftol", 1e-8),
            xtol=kwargs.get("xtol", 1e-8),
            gtol=kwargs.get("gtol", 1e-8),
            x_scale=kwargs.get("x_scale", 1.0),
            loss=kwargs.get("loss", "linear"),
            f_scale=kwargs.get("f_scale", 1.0),
            diff_step=kwargs.get("diff_step"),
            tr_solver=kwargs.get("tr_solver"),
            tr_options=kwargs.get("tr_options", {}),
            jac_sparsity=kwargs.get("jac_sparsity"),
            max_nfev=kwargs.get("max_nfev"),
            verbose=kwargs.get("verbose", 0),
            args=kwargs.get("args", ()),
            kwargs=kwargs.get("kwargs", {}),
        )
        logger.info("fitting finished...")
        result_normalized: NDArray = np.asarray(
            parameters.vector_renormalizer(
                result.x,
                self.fitable.keys(),
                self.normalization,
            ),
            dtype=FLOAT,
        )

        # Build result dictionary from fitted parameters
        result_dict: dict[str, dict[str, Any]] = {
            lbl: {
                "value": result_normalized[i],
                "fixed": False,
                "unit": self.fitable[lbl].to_dict()["unit"],
                "min": self.fitable[lbl].min,
                "max": self.fitable[lbl].max,
            }
            for i, lbl in enumerate(self.fitable.keys())
        }

        # Add fixed parameters
        result_dict.update(
            {
                lbl: {"value": val.value, "fixed": True, "unit": val.to_dict()["unit"]}
                for lbl, val in self.fixed.items()
            },
        )
        result_dict = self.eval_constrained_results(result_dict, self.constrained)

        # Calculate R-squared
        r_squared_args: tuple[Any, ...] = (
            self.x_data_reduced,
            self.y_data,
            self.x_data_reducer,
            self.observer.system_cls,
        )
        r_dict: dict[str, Any] = {key: value["value"] for key, value in result_dict.items()}

        r_squared_result: Float = rv_r_squared(
            rv_model.central_rv_synthetic,
            *r_squared_args,
            **r_dict,
        )
        result_dict["r_squared"] = {"value": r_squared_result, "unit": None}

        self.flat_result = result_dict
        return parameters.serialize_result(result_dict)
