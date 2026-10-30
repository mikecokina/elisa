"""Cost function implementations for likelihood and least squares fitting."""

from __future__ import annotations

from math import exp, log1p
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from elisa.base.error import raise_elisa_prediction_alignment_error
from elisa.const import PI

_TWO_PI = 2.0 * PI
_LOG_TWO = np.log(2.0)
_HALF_LOG_TWO_PI = 0.5 * np.log(_TWO_PI)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float


class PreparedObservations(NamedTuple):
    """Per-setup observation arrays and Gaussian constants; no global cache."""

    labels: tuple[str, ...]
    observed: tuple[NDArray[np.float64], ...]
    log_errors: tuple[NDArray[np.float64], ...]
    log_normalization: Float
    size: int


def _log_noise_scale(ln_f: Float) -> Float:
    """Return log(sqrt(1 + exp(2*ln_f))) without exponential overflow.

    :param ln_f: Current finite scalar noise parameter.
    :return: Logarithm of the standard-deviation multiplier.
    :raises ValueError: If the current noise parameter is malformed.
    """
    if np.ndim(ln_f) != 0 or not np.isfinite(ln_f):
        msg = "ln_f must be finite and scalar"
        raise ValueError(msg)
    small = exp(-abs(ln_f))
    return max(0.0, ln_f) + 0.5 * log1p(small * small)


def prepare_observations(
    y_data: dict[str, NDArray], y_errs: dict[str, NDArray],
) -> PreparedObservations:
    """Validate immutable data and prepare arrays/statistical constants once.

    :param y_data: Nonempty finite observed vectors in residual order.
    :param y_errs: Matching strictly positive errors in observation units.
    :return: Reusable arrays and constants. New setup replaces this record.
    :raises ValueError: If keys, dimensions, values or uncertainties are invalid.
    """
    if not y_data or y_data.keys() != y_errs.keys():
        msg = "Likelihood observation, model and uncertainty dataset keys must match"
        raise ValueError(msg)
    observed_arrays, log_errors = [], []
    for key, data in y_data.items():
        observed = np.asarray(data, dtype=float)
        errors = np.asarray(y_errs[key], dtype=float)
        if observed.ndim != 1 or observed.size == 0 or errors.shape != observed.shape:
            msg = f"Likelihood arrays for {key} must be matching nonempty one-dimensional vectors"
            raise ValueError(msg)
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(errors)) or np.any(errors <= 0):
            msg = f"Likelihood arrays for {key} must be finite with strictly positive uncertainties"
            raise ValueError(msg)
        observed_arrays.append(observed)
        log_errors.append(np.log(errors))
    size = sum(array.size for array in observed_arrays)
    normalization = sum(np.sum(logs + _HALF_LOG_TWO_PI) for logs in log_errors)
    return PreparedObservations(tuple(y_data), tuple(observed_arrays), tuple(log_errors), normalization, size)


def validate_predictions(prepared: PreparedObservations, synthetic: dict[str, NDArray]) -> None:
    """Validate current synthetic output at a standalone public cost boundary.

    Fitters check dynamic output in the forward mapping and omit this duplicate.

    :param prepared: Validated observations and expected array dimensions.
    :param synthetic: Current synthetic output.
    :raises ElisaPredictionAlignmentError: If prediction keys or shapes differ.
    :raises ValueError: If predictions contain nonfinite values.
    """
    if len(synthetic) != len(prepared.labels) or any(key not in synthetic for key in prepared.labels):
        msg = "Likelihood observation, model and uncertainty dataset keys must match"
        raise_elisa_prediction_alignment_error(msg)
    for key, observed in zip(prepared.labels, prepared.observed, strict=True):
        model = np.asarray(synthetic[key])
        if model.shape != observed.shape:
            msg = f"Likelihood prediction shape differs from observations for {key}"
            raise_elisa_prediction_alignment_error(msg)
        if not np.all(np.isfinite(model)):
            msg = f"Likelihood arrays for {key} must be matching finite one-dimensional vectors"
            raise ValueError(msg)


def prepared_likelihood(prepared: PreparedObservations, synthetic: dict[str, NDArray], ln_f: Float) -> Float:
    """Evaluate the existing Gaussian using prepared data and checked predictions.

    :param prepared: Per-setup observation statistics.
    :param synthetic: Dynamically checked, observation-aligned predictions.
    :param ln_f: Current scalar noise parameter.
    :return: Gaussian log likelihood; no observation validation occurs here.
    """
    log_scale = _log_noise_scale(ln_f)
    try:
        with np.errstate(over="raise", invalid="raise", divide="ignore", under="ignore"):
            total = -prepared.log_normalization - prepared.size * log_scale
            for key, observed, logs in zip(prepared.labels, prepared.observed, prepared.log_errors, strict=True):
                log_residual = np.log(np.abs(observed - synthetic[key]))
                total -= np.sum(np.exp(2.0 * (log_residual - logs - log_scale) - _LOG_TWO))
    except FloatingPointError:
        total = _extended_likelihood(prepared, synthetic, log_scale)
    if not np.isfinite(total):
        # Direct callers receive the numerical error; MCMC rejects this candidate.
        msg = "Nonfinite likelihood: Gaussian log density is not representable as a finite float"
        raise FloatingPointError(msg)
    return total


def _extended_likelihood(prepared: PreparedObservations, synthetic: dict[str, NDArray], log_scale: Float) -> Float:
    """Resolve rare intermediate overflows before testing float64 representability.

    Half the quadratic is evaluated directly, so a representable density is not
    rejected merely because its squared standardized residual exceeds float64.
    Extended subtraction also handles opposite-sign observations near float64 max.

    :param prepared: Validated observation statistics.
    :param synthetic: Aligned model predictions.
    :param log_scale: Stable logarithm of the noise multiplier.
    :return: Finite float64 density.
    :raises FloatingPointError: If the resulting density cannot be represented.
    """
    with np.errstate(over="raise", invalid="raise", divide="ignore", under="ignore"):
        total = np.longdouble(0)
        for key, observed, logs in zip(prepared.labels, prepared.observed, prepared.log_errors, strict=True):
            residual = observed.astype(np.longdouble) - np.asarray(synthetic[key], dtype=np.longdouble)
            log_sigma = logs.astype(np.longdouble) + log_scale
            half_square = np.exp(2 * (np.log(np.abs(residual)) - log_sigma) - _LOG_TWO)
            total -= np.sum(half_square + log_sigma + _HALF_LOG_TWO_PI)
        return float(total)


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
    prepared = prepare_observations(y_data, y_errs)
    validate_predictions(prepared, synthetic)
    return prepared_likelihood(prepared, synthetic, ln_f)


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
    :raises ElisaPredictionAlignmentError: If prediction keys or shapes differ from observations.
        Observation and uncertainty shapes must also align; scalar uncertainties remain supported.
    """
    if validate:
        _validate_residual_shapes(y_data, y_err, synthetic)
    return np.concatenate([(observed - synthetic[key]) / y_err[key] for key, observed in y_data.items()])


def _validate_residual_shapes(y_data: dict, y_err: dict, synthetic: dict) -> None:
    """Reject broadcasting mismatches in RV or standalone residual construction.

    :param y_data: Observed vectors by dataset.
    :param y_err: Matching uncertainty vectors or scalar uncertainty fallbacks.
    :param synthetic: Predictions mapped into each observation vector's order.
    :raises ElisaPredictionAlignmentError: If observation, prediction or uncertainty keys/shapes differ.
    """
    if set(y_data) != set(synthetic):
        message = "Residual observation and prediction datasets differ"
        raise_elisa_prediction_alignment_error(message)
    if set(y_data) != set(y_err):
        message = "Residual observation, prediction and uncertainty datasets differ"
        raise_elisa_prediction_alignment_error(message)
    for key, observed in y_data.items():
        if np.ndim(observed) != 1:
            message = f"Residual prediction shape differs from observations for {key}"
            raise_elisa_prediction_alignment_error(message)
        if np.shape(synthetic[key]) != np.shape(observed):
            message = f"Residual prediction shape differs from observations for {key}"
            raise_elisa_prediction_alignment_error(message)
        if np.shape(y_err[key]) not in ((), np.shape(observed)):
            message = f"Residual uncertainty shape differs from observations for {key}"
            raise_elisa_prediction_alignment_error(message)
