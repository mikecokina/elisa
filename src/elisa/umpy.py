"""Unified mathematical and array operations wrapper.

This module provides a unified interface to NumPy and SciPy functions,
with backward compatibility patches for different versions of these libraries.
It abstracts away version-specific differences in function names and APIs.

The module serves as a compatibility layer between different versions of:
- NumPy: Handles changes like trapz -> trapezoid
- SciPy: Handles changes like simps -> simpson, sph_harm API changes

Functions and variables are aliased to their NumPy/SciPy counterparts for easy access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import integrate, optimize, special

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float

# numpy
degrees = np.degrees
radians = np.radians

cos = np.cos
sin = np.sin
tan = np.tan
arctan = np.arctan
arcsin = np.arcsin
arccos = np.arccos
arctan2 = np.arctan2
pi = np.pi
zeros = np.zeros
log10 = np.log10
divide = np.divide
log = np.log
exp = np.exp

isnan = np.isnan
NaN = np.nan
power = np.power
matmul = np.matmul
multiply = np.multiply
inner = np.inner
arange = np.arange
# noinspection PyShadowingBuiltins
abs = np.abs  # noqa: A001
ceil = np.ceil
where = np.where
invert = np.invert
concatenate = np.concatenate
sqrt = np.sqrt
sign = np.sign
# noinspection PyShadowingBuiltins
sum = np.sum  # noqa: A001

logical_xor = np.logical_xor
logical_and = np.logical_and
logical_or = np.logical_or
mod = np.mod
equal = np.equal
less = np.less
ones = np.ones
# noinspection PyShadowingBuiltins
round = np.round  # noqa: A001

# scipy
lpmv = special.lpmv
optimize = optimize  # noqa: PLW0127

# patches
# Keep old SciPy sph_harm(m, n, theta, phi) API for the rest of the codebase.
# Old sph_harm expects: theta=azimuth, phi=polar.
# New sph_harm_y expects: theta=polar, phi=azimuth, and argument order (n, m, theta, phi).
try:
    # noinspection PyUnresolvedReferences
    sph_harm = special.sph_harm  # SciPy <= 1.16
except AttributeError:
    _sph_harm_y = special.sph_harm_y  # SciPy >= 1.17


    def sph_harm(
            m: int,
            n: int,
            theta: Float | NDArray,
            phi: Float | NDArray,
            out: NDArray | None = None,
    ) -> NDArray | complex:
        """Calculate spherical harmonics using legacy API.

        Provides backward compatibility for the old SciPy sph_harm API.
        Maps the old convention (m, n, azimuth, polar) to the new convention
        (n, m, polar, azimuth) used by scipy.special.sph_harm_y.

        :param m: int; order of the spherical harmonic
        :param n: int; degree of the spherical harmonic (n >= |m|)
        :param theta: float or numpy.ndarray; azimuthal angle in radians (maps to phi in new API)
        :param phi: float or numpy.ndarray; polar angle in radians (maps to theta in new API)
        :param out: numpy.ndarray | None; optional output array for the result

        :return: numpy.ndarray or complex; spherical harmonic value(s) at (theta, phi)
        """
        # Map old convention -> new convention:
        # old: (m, n, azimuth, polar)
        # new: (n, m, polar, azimuth)
        y = _sph_harm_y(n, m, phi, theta)
        if out is not None:
            out[...] = y
            return out
        return y

try:
    # noinspection PyUnresolvedReferences
    trapz = np.trapz  # NumPy < 2.4  # noqa: NPY201
except AttributeError:
    trapz = np.trapezoid  # NumPy >= 2.4

try:
    # noinspection PyUnresolvedReferences
    simps = integrate.simps  # older SciPy
except AttributeError:
    # newer SciPy: simpson is the replacement
    # noinspection PyUnusedLocal
    def simps(
            y: NDArray,
            x: NDArray | None = None,
            dx: Float = 1.0,
            axis: int = -1,
            even: str | None = None,  # noqa: ARG001
    ) -> float | np.ndarray:
        """Integrate using Simpson's rule.

        Provides backward compatibility wrapper for SciPy's simpson function.
        Integrates y(x) using samples along the given axis and Simpson's rule.

        :param y: numpy.ndarray; array to integrate (y values)
        :param x: numpy.ndarray | None; sample locations (x values), optional.
                  If None, spacing is assumed to be uniform with spacing dx
        :param dx: float; spacing between sample points when x is None (default: 1.0)
        :param axis: int; axis along which to integrate (default: -1)
        :param even: str | None; deprecated parameter, ignored for compatibility with older API

        :return: float or numpy.ndarray; integral of y(x) along the specified axis
        """
        # ignore `even` (removed in simpson); most code never used it explicitly
        return integrate.simpson(y, x=x, dx=dx, axis=axis)
