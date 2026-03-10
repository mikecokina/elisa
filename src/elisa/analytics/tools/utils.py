"""Utility functions for analytics tools module."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from elisa.base.types import FLOAT
from elisa.binary_system import t_layer

if TYPE_CHECKING:
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


def radialcurves_mean_error(rv: NDArray) -> Float:
    """Return synthetic error for radial velocity observations.

    If observation errors are not provided, the default 5 percent relative
    error is used to generate synthetic errors for the radial velocities.

    :param rv: Radial velocity observations
    :type rv: NDArray
    :return: Synthetic error value (5% of mean)
    :rtype: float
    """
    return FLOAT(np.mean(rv) * 0.05)


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


