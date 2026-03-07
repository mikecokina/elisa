from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa.base import error

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from elisa.types import Float


def validate_period(period: Float) -> None:
    """Validate that the orbital period is positive.

    :param period: Orbital period of the binary system.
    :type period: Float
    :return: ``None``.
    :rtype: None
    :raises error.ValidationError: If ``period`` is not greater than zero.
    """
    if period <= 0:
        message = "Period has to be > 0."
        raise error.ValidationError(message)


def validate_primary_minimum_time(t0: Float) -> None:
    """Validate that the reference primary minimum time is positive.

    :param t0: Reference primary minimum time.
    :type t0: Float
    :return: ``None``.
    :rtype: None
    :raises error.ValidationError: If ``t0`` is not greater than zero.
    """
    if t0 <= 0:
        message = "Primary minimum time has to be > 0."
        raise error.ValidationError(message)


def adjust_phases(
        phases: Float | ArrayLike,
        centre: Float = 0.5,
) -> Float | NDArray[np.float64]:
    """Shift phases to center them on the given value.

    The returned phases are wrapped into an interval spanning approximately
    ``centre - 0.5`` to ``centre + 0.5``.

    :param phases: Input phase value or phase array.
    :type phases: Float | ArrayLike
    :param centre: Center around which phases will be calculated.
    :type centre: Float
    :return: Shifted phase value or shifted phase array.
    :rtype: Float | NDArray[numpy.float64]
    """
    if np.isscalar(phases):
        shift = centre - 0.5
        return (phases - shift) % 1.0 + shift

    phase_array = np.asarray(phases, dtype=np.float64)
    shift = centre - 0.5
    return np.asarray((phase_array - shift) % 1.0 + shift, dtype=np.float64)


def jd_to_phase(
        t0: Float,
        period: Float,
        jd: Float | ArrayLike,
        centre: Float = 0.5,
) -> Float | NDArray[np.float64]:
    """Convert Julian Date time to orbital phase.

    The phase is wrapped around the interval centered on ``centre``.

    :param t0: Reference primary minimum time.
    :type t0: Float
    :param period: Period of the binary system.
    :type period: Float
    :param jd: Measurement Julian Date values.
    :type jd: Float | ArrayLike
    :param centre: Center around which phases will be calculated.
    :type centre: Float
    :return: Converted phase value or phase array.
    :rtype: Float | NDArray[numpy.float64]
    :raises error.ValidationError: If ``period`` or ``t0`` is invalid.
    """
    validate_period(period)
    validate_primary_minimum_time(t0)

    shift = centre - 0.5

    if np.isscalar(jd):
        return (((jd - t0) / period) - shift) % 1.0 + shift

    jd_array = np.asarray(jd, dtype=np.float64)
    return np.asarray(
        (((jd_array - t0) / period) - shift) % 1.0 + shift,
        dtype=np.float64,
    )


def phase_to_jd(
        t0: Float,
        period: Float,
        phases: Float | ArrayLike,
) -> Float | NDArray[np.float64]:
    """Convert orbital phase to Julian Date time.

    :param t0: Reference primary minimum time.
    :type t0: Float
    :param period: Period of the binary system.
    :type period: Float
    :param phases: Phase value or phase array.
    :type phases: Float | ArrayLike
    :return: Converted Julian Date value or array.
    :rtype: Float | NDArray[numpy.float64]
    :raises error.ValidationError: If ``period`` or ``t0`` is invalid.
    """
    validate_period(period)
    validate_primary_minimum_time(t0)

    if np.isscalar(phases):
        return (period * phases) + t0

    phase_array = np.asarray(phases, dtype=np.float64)
    return np.asarray((period * phase_array) + t0, dtype=np.float64)
