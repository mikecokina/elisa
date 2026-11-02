"""Utility functions for analytics tools module."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from elisa.base.error import raise_elisa_analytics_data_error
from elisa.base.types import FLOAT
from elisa.binary_system import t_layer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from elisa.types import Float


def lightcurves_mean_error(lc: NDArray) -> Float:
    """Return synthetic error for light curve observations.

    If observation errors are not provided, the default 5 percent relative
    error is used to generate synthetic errors for the light curve.

    :param lc: Light curve observations
    :type lc: NDArray
    :return: Synthetic error value (5% of mean)
    :rtype: float
    """
    return FLOAT(np.mean(lc) * 0.05)


def _radialcurves_mean_error(rv: NDArray | Sequence[Float]) -> Float:
    """Estimate heuristic RV errors from translation-invariant spread.

    Uses 5% of Gaussian-scaled MAD (1.482602218505602 times MAD), falling
    back to 5% of sample standard deviation when ties make MAD zero.
    Centering and rescaling preserve small representable variations without
    squaring tiny or huge velocities. No absolute velocity floor is used.
    Orbital variation affects these weights; they do not estimate instrument
    precision. MCMC credible intervals depend on the assumed error scale.

    :param rv: Finite one-dimensional velocities for a single component.
    :returns: Positive heuristic uncertainty in the input velocity unit.
    :raises ElisaAnalyticsDataError: If a finite positive scale cannot be estimated.
    """
    scaled, amplitude = _scaled_rv_deviations(rv)
    mad = np.median(np.abs(scaled))
    spread = 1.482602218505602 * mad if mad > 0 else np.std(scaled, ddof=1)
    uncertainty = (0.05 * spread) * amplitude
    if not np.isfinite(uncertainty) or uncertainty <= 0:
        msg = "RV variation cannot produce a finite positive uncertainty; supply explicit positive y_err"
        raise_elisa_analytics_data_error(msg)
    return FLOAT(uncertainty)


def _scaled_rv_deviations(rv: NDArray | Sequence[Float]) -> tuple[NDArray[np.float64], Float]:
    """Prepare centered, scaled velocities for stable spread estimation.

    :param rv: Velocity observations for one component.
    :returns: Median-centered deviations divided by their positive amplitude,
        together with that amplitude in the original velocity unit.
    :raises ElisaAnalyticsDataError: If observations cannot resolve a finite positive spread.
    """
    try:
        velocities = np.asarray(rv, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        msg = "Estimating RV y_err requires numeric observations; supply explicit positive y_err"
        raise_elisa_analytics_data_error(msg, exc)
    minimum_samples = 2
    if velocities.ndim != 1 or velocities.size < minimum_samples or not np.all(np.isfinite(velocities)):
        msg = "Estimating RV y_err requires at least two finite observations; supply explicit positive y_err"
        raise_elisa_analytics_data_error(msg)
    with np.errstate(over="ignore", invalid="ignore"):
        centered = velocities - velocities[0]
        deviations = centered - np.median(centered)
        amplitude = np.max(np.abs(deviations))
    if not np.isfinite(amplitude) or amplitude <= 0:
        msg = "RV observations have no resolvable positive velocity scale; supply explicit positive y_err"
        raise_elisa_analytics_data_error(msg)
    return deviations / amplitude, amplitude


def is_time_dependent(labels: list[str]) -> bool:
    """Check if fit parameters are time-dependent.

    If 'system@primary_minimum_time' is located in the fit parameters,
    the fit parameters are considered time dependent and observations
    are therefore expected to be supplied in Julian Date (JD) format
    rather than orbital phases.

    :param labels: List of parameter labels
    :type labels: list[str]
    :return: True if both period and primary_minimum_time are present
    :rtype: bool
    """
    return (
        "system@period" in labels
        and "system@primary_minimum_time" in labels
    )


def time_layer_resolver(
    x_data: NDArray,
    *,
    pop: bool = False,
    **kwargs: Any,
) -> tuple[NDArray, dict]:
    """Resolve time layer and convert between JD and phase data.

    If kwargs contain `period` and `primary_minimum_time`, then x_data
    is expected to be Julian Date (JD) time not orbital phases. In that
    case, x_data (observational time) is converted to orbital phases
    using the system period and primary minimum time.

    :param x_data: Observational times or phases
    :type x_data: NDArray
    :param pop: If True, remove the 'system@primary_minimum_time' parameter
                from the fit parameters after conversion (default: False)
    :type pop: bool
    :param kwargs: Dictionary containing fit parameters including period
                   and primary_minimum_time if time-dependent
    :type kwargs: dict
    :return: Tuple containing converted x_data and (possibly modified) kwargs
    :rtype: tuple[NDArray, dict]
    """
    if is_time_dependent(list(kwargs.keys())):
        t0: Float = kwargs["system@primary_minimum_time"]
        if pop:
            kwargs.pop("system@primary_minimum_time")
        period: Float = kwargs["system@period"]
        x_data_new: NDArray = t_layer.jd_to_phase(t0, period, x_data, centre=0.5)
    else:
        x_data_new = x_data % 1.0
    return x_data_new, kwargs


