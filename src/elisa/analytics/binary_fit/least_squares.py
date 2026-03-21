from __future__ import annotations

import functools
import logging
from abc import ABCMeta
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import interpolate
from scipy.optimize import least_squares

from elisa import const, settings
from elisa.analytics.binary_fit.shared import (
    AbstractLCFit,
    AbstractRVFit,
    lc_r_squared,
    r_squared,
    rv_r_squared,
)
from elisa.analytics.models import cost_fns
from elisa.analytics.models import lc as lc_model
from elisa.analytics.models import rv as rv_model
from elisa.analytics.params import parameters
from elisa.analytics.tools.utils import time_layer_resolver
from elisa.base.types import FLOAT
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from elisa.logger import getPersistentLogger
from elisa.observer.utils import normalize_light_curve

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.analytics.dataset.base import LCData, RVData
    from elisa.types import Float

logger = getPersistentLogger("analytics.binary_fit.least_squares")


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


    def model_to_fit(self, xn: NDArray) -> NDArray | float:
        """Cost function minimized during solution of inverse problem using The Least Squares.

        This method calculates weighted sum of squared residuals between observed and
        synthetic light curves for the given normalized parameter vector.

        :param xn: Vector containing normalized values of model parameters optimized
            during fit. Values are in range [0, 1] and normalized using stored
            normalization parameters.
        :type xn: NDArray
        :return: Error weighted sum of squared residuals. Large values (MAX_USABLE_FLOAT)
            are returned if model parameters lead to invalid binary system.
        :rtype: NDArray | float
        """
        diff = 1.0 / self.interp_treshold

        xn_list = parameters.vector_renormalizer(xn, self.fitable.keys(), self.normalization)
        xn = np.asarray(xn_list, dtype=FLOAT)
        kwargs: dict[str, Any] = parameters.prepare_properties_set(
            xn,
            self.fitable.keys(),
            self.constrained,
            self.fixed,
        )
        phases, kwargs = time_layer_resolver(self.x_data_reduced, pop=False, **kwargs)

        if self.fit_xs is None:
            fit_xs = (
                np.linspace(
                    np.min(phases) - diff,
                    np.max(phases) + diff,
                    num=self.interp_treshold + 2,
                )
                if np.shape(phases)[0] > self.interp_treshold
                else phases
            )
        else:
            fit_xs = self.fit_xs

        args: tuple[Any, ...] = (fit_xs, self.discretization, self.observer)

        kwargs = parameters.extend_json_with_atm_params(
            kwargs,
            atmosphere_model=self.atmosphere_model,
            limb_darkening_coefficients=self.limb_darkening_coefficients,
        )

        fn = lc_model.synthetic_binary

        try:
            synthetic: dict[str, Any] = _log_decorator(fn)(*args, **kwargs)
        except (ValueError, RuntimeError, TypeError) as e:
            error_msg = f"your initial parameters lead during fitting to invalid binary system, exception: {e!s}"
            logger.exception(error_msg)
            return const.MAX_USABLE_FLOAT

        if np.shape(fit_xs) != np.shape(phases):
            synthetic = {
                band: interpolate.interp1d(fit_xs, curve, kind="cubic")(
                    phases[self.x_data_reducer[band]],
                )
                for band, curve in synthetic.items()
            }
        else:
            synthetic = {band: val[self.x_data_reducer[band]] for band, val in synthetic.items()}

        synthetic, _ = normalize_light_curve(synthetic, kind="average")

        residuals = cost_fns.wssr(self.y_data, self.y_err, synthetic)

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
        samples: str = "uniform",
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
        :param samples: Sampling method: 'uniform' (equidistant in phase), 'adaptive'
            (equidistant on curve), or list of phases in (0, 1) interval.
            Default is 'uniform'.
        :type samples: str
        :param kwargs: Optional arguments for scipy.optimize.least_squares function.
            Common options: jac, method, ftol, xtol, gtol, x_scale, loss, f_scale,
            diff_step, tr_solver, tr_options, jac_sparsity, max_nfev, verbose.
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
            verbose=kwargs.get("verbose", 2),
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

        # Calculate R-squared
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

        r_dict: dict[str, Any] = {key: value["value"] for key, value in result_dict.items()}
        r_dict = parameters.extend_json_with_atm_params(
            r_dict,
            atmosphere_model=self.atmosphere_model,
            limb_darkening_coefficients=self.limb_darkening_coefficients,
        )

        r_squared_result: Float = lc_r_squared(lc_model.synthetic_binary, *r_squared_args, **r_dict)
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

    def central_rv_model_to_fit(self, xn: NDArray) -> float:
        """Cost function minimized during radial velocity fitting.

        Calculates weighted sum of squared residuals between observed and
        synthetic radial velocities for the given normalized parameter vector.

        :param xn: Vector of variable model parameters in normalized form.
        :type xn: NDArray
        :return: Error weighted sum of squared residuals.
        :rtype: float
        """
        synthetic: dict[str, NDArray] = self.prepare_synthetic(xn)
        synthetic = {comp: synthetic[comp][self.x_data_reducer[comp]] for comp in synthetic}
        return cost_fns.wssr(self.y_data, self.y_err, synthetic)

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
