"""Cost function implementations for likelihood and least squares fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from elisa.base.error import PredictionAlignmentError
from elisa.const import PI

_TWO_PI = 2.0 * PI

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float


def s_squared(
    y_errs: dict[str, NDArray[Any]],
    ln_f: Float,
) -> dict[str, NDArray[Any]]:
    """Calculate error component of likelihood function.

    Computes the error component used in the likelihood function, accounting
    for observational errors and an error underestimation factor. The total
    variance for each dataset is the sum of squared observational errors and
    the scaled underestimation factor.

    :param y_errs: Dictionary mapping observable names to their error arrays.
    :type y_errs: dict[str, NDArray[Any]]
    :param ln_f: Error underestimation/marginalization parameter.
    :type ln_f: Float
    :returns: Dictionary mapping observable names to computed variance arrays
        (sigma squared).
    :rtype: dict[str, NDArray[Any]]
    """
    exp_factor = 1.0 + np.exp(2.0 * ln_f)  # scalar - compute once, not once per passband
    return {key: errors**2 * exp_factor for key, errors in y_errs.items()}


def likelihood_fn(
    y_data: dict[str, NDArray[Any]],
    y_errs: dict[str, NDArray[Any]],
    synthetic: dict[str, NDArray[Any]],
    ln_f: Float,
) -> Float:
    """Calculate likelihood function value for observational data.

    Computes the logarithm of the likelihood function for observational data
    given a synthetic model. Assumes normal distribution of observables around
    the synthetic model values, with variances that include observational
    errors and an error underestimation factor.

    :param y_data: Dictionary mapping observable names to observed data arrays.
    :type y_data: dict[str, NDArray[Any]]
    :param y_errs: Dictionary mapping observable names to their error arrays.
    :type y_errs: dict[str, NDArray[Any]]
    :param synthetic: Dictionary mapping observable names to synthetic model
        arrays.
    :type synthetic: dict[str, NDArray[Any]]
    :param ln_f: Error underestimation/marginalization parameter (currently
        supported as single parameter for error penalization).
    :type ln_f: Float
    :returns: Log-likelihood value.
    :rtype: Float
    """
    sigma2 = s_squared(y_errs, ln_f)

    return -0.5 * sum(
        np.sum((y_data[key] - synthetic[key]) ** 2 / sigma2[key] + np.log(_TWO_PI * sigma2[key]))
        for key in synthetic
    )


def wssr(
    y_data: dict[str, NDArray[Any]],
    y_err: dict[str, NDArray[Any]],
    synthetic: dict[str, NDArray[Any]],
) -> Float:
    """Calculate weighted sum of squared residuals.

    Computes the error-weighted sum of squared residuals (WSSR) between
    synthetic model predictions and observational data. Each residual is
    weighted by the inverse of the squared observational error.

    :param y_data: Dictionary mapping observable names to observed data arrays.
    :type y_data: dict[str, NDArray[Any]]
    :param y_err: Dictionary mapping observable names to their error arrays.
    :type y_err: dict[str, NDArray[Any]]
    :param synthetic: Dictionary mapping observable names to synthetic model
        arrays.
    :type synthetic: dict[str, NDArray[Any]]
    :returns: Weighted sum of squared residuals value.
    :rtype: Float
    """
    return sum(np.sum(((synthetic[item] - y_data[item]) / y_err[item]) ** 2) for item in synthetic)


def standardized_residuals(
    y_data: dict[str, NDArray[Any]],
    y_err: dict[str, NDArray[Any]],
    synthetic: dict[str, NDArray[Any]],
    *,
    validate: bool = True,
) -> NDArray[Any]:
    """Return a flat standardized residual vector in observation insertion order.

    :param y_data: Per-dataset observed vectors; their insertion order fixes the
        residual and Jacobian row order throughout an optimization.
    :param y_err: Per-datum errors or the existing per-dataset scalar fallback.
    :param synthetic: Per-dataset predictions already mapped to observation order.
    :param validate: Check dataset and array shapes. Prepared LC fitting passes
        False after setup validation and dynamic prediction alignment checks.
    :return: Concatenated (observed - synthetic) / uncertainty. Its squared norm
        is chi² and SciPy's default linear-loss cost is half that value.
    :raises PredictionAlignmentError: If prediction keys or shapes differ from observations.
    :raises ValueError: If observation or uncertainty shapes are invalid.
        Scalar uncertainties remain supported.
    """
    if validate:
        _validate_residual_shapes(y_data, y_err, synthetic)
    return np.concatenate([(observed - synthetic[key]) / y_err[key] for key, observed in y_data.items()])


def _validate_residual_shapes(y_data: dict, y_err: dict, synthetic: dict) -> None:
    """Reject broadcasting mismatches in RV or standalone residual construction.

    :param y_data: Observed vectors by dataset.
    :param y_err: Matching uncertainty vectors or scalar uncertainty fallbacks.
    :param synthetic: Predictions mapped into each observation vector's order.
    :raises PredictionAlignmentError: If prediction keys or shapes differ.
    :raises ValueError: If observation or uncertainty keys/shapes are invalid.
    """
    if set(y_data) != set(synthetic):
        message = "Residual observation and prediction datasets differ"
        raise PredictionAlignmentError(message)
    if set(y_data) != set(y_err):
        message = "Residual observation, prediction and uncertainty datasets differ"
        raise ValueError(message)
    for key, observed in y_data.items():
        if np.ndim(observed) != 1:
            message = f"Residual prediction shape differs from observations for {key}"
            raise ValueError(message)
        if np.shape(synthetic[key]) != np.shape(observed):
            message = f"Residual prediction shape differs from observations for {key}"
            raise PredictionAlignmentError(message)
        if np.shape(y_err[key]) not in ((), np.shape(observed)):
            message = f"Residual uncertainty shape differs from observations for {key}"
            raise ValueError(message)
