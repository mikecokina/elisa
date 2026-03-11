"""Cost function implementations for likelihood and least squares fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from elisa.const import PI

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
    return {key: np.power(errors, 2) + np.power(errors, 2) * np.exp(2 * ln_f) for key, errors in y_errs.items()}


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

    return -0.5 * (
        np.sum(
            [
                np.sum(
                    (np.power((y_data[key] - synthetic[key]), 2) / sigma2[key]) + np.log(2.0 * PI * sigma2[key]),
                )
                for key in synthetic
            ],
        )
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
    return np.sum(
        [np.sum(np.power((synthetic[item] - y_data[item]) / y_err[item], 2)) for item in synthetic],
    )
