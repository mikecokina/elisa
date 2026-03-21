"""Temperature estimation utilities based on B-V color index."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float


def ballesteros_formula(color_bv: Float | NDArray) -> Float | NDArray:
    """Estimate stellar temperature from B-V color index using Ballesteros formula.

    The Ballesteros formula provides a temperature estimation based on the
    B-V (Johnson B minus Johnson V) color index. This formula is empirical
    and derived from observational data of stellar color and effective temperature.

    The B-V index is defined as: B - V = -2.5 * log(F_b / F_v)

    :param color_bv: B-V color index value(s)
    :type color_bv: float | NDArray
    :return: Estimated stellar effective temperature in Kelvin
    :rtype: float | NDArray
    """
    return 4600.0 * (
        (1.0 / ((0.92 * color_bv) + 1.7)) + (1.0 / ((0.92 * color_bv) + 0.62))
    )


def pogsons_formula(
    f1: Float | NDArray,
    f2: Float | NDArray,
) -> Float | NDArray:
    """Calculate magnitude difference using Pogson's formula.

    Pogson's formula relates the magnitude difference to the flux ratio
    between two objects. A magnitude difference of 1 corresponds to a
    flux ratio of exactly 2.512 (the fifth root of 100).

    :param f1: First flux value(s)
    :type f1: float | NDArray
    :param f2: Second flux value(s)
    :type f2: float | NDArray
    :return: Magnitude difference (m2 - m1 = -2.5 * log10(f1 / f2))
    :rtype: float | NDArray
    """
    return -2.5 * np.log10(f1 / f2)


def _overcontact_temperature_estimation(b_v: Float | NDArray) -> Float | NDArray:
    """Estimate temperature for overcontact binary systems.

    Uses empirical coefficients optimized for overcontact (contact binary)
    star systems. These coefficients differ from the solar-like formula
    due to the different physical properties of contact binaries.

    :param b_v: B-V color index value(s)
    :type b_v: float | NDArray
    :return: Estimated stellar effective temperature in Kelvin
    :rtype: float | NDArray
    """
    a: Float = 1270.92384801
    b: Float = 1.73588834
    c: Float = 1290.17377233
    return a / np.log10(b + b_v) + c


def _solar_like_temperature_estimation(b_v: Float | NDArray) -> Float | NDArray:
    """Estimate temperature for detached/solar-like binary systems.

    Uses empirical coefficients optimized for solar-like and detached binary
    star systems. These coefficients are based on calibrations using stars
    with well-determined temperatures from other methods.

    :param b_v: B-V color index value(s)
    :type b_v: float | NDArray
    :return: Estimated stellar effective temperature in Kelvin
    :rtype: float | NDArray
    """
    a: Float = 1768.60111726
    b: Float = 1.78240258
    c: Float = -264.71844987
    return a / np.log10(b + b_v) + c


def elisa_bv_temperature(
    b_v: Float | NDArray,
    *,
    morphology: str = "detached",
) -> Float | NDArray:
    """Estimate stellar temperature from B-V index based on binary morphology.

    This function selects the appropriate temperature estimation formula
    based on the binary system morphology. Contact binaries (overcontact
    systems) have different temperature-color relationships compared to
    detached systems due to tidal forces and mass transfer effects.

    :param b_v: B-V color index value(s)
    :type b_v: float | NDArray
    :param morphology: Binary system morphology ('detached' or 'overcontact').
                       Default is 'detached'. Unknown values fall back to
                       solar-like formula.
    :type morphology: str
    :return: Estimated stellar effective temperature in Kelvin
    :rtype: float | NDArray
    """
    if morphology == "overcontact":
        return _overcontact_temperature_estimation(b_v)
    return _solar_like_temperature_estimation(b_v)
