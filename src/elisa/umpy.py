import numpy as np

from scipy import special
from scipy import optimize
from scipy import integrate

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
abs = np.abs
ceil = np.ceil
where = np.where
invert = np.invert
concatenate = np.concatenate
sqrt = np.sqrt
sign = np.sign
sum = np.sum

logical_xor = np.logical_xor
logical_and = np.logical_and
logical_or = np.logical_or
mod = np.mod
equal = np.equal
less = np.less
ones = np.ones
# noinspection PyShadowingBuiltins
round = np.round

# scipy
lpmv = special.lpmv
optimize = optimize

# patches
# Keep old SciPy sph_harm(m, n, theta, phi) API for the rest of the codebase.
# Old sph_harm expects: theta=azimuth, phi=polar.
# New sph_harm_y expects: theta=polar, phi=azimuth, and argument order (n, m, theta, phi).
try:
    # noinspection PyUnresolvedReferences
    sph_harm = special.sph_harm  # SciPy <= 1.16
except AttributeError:
    _sph_harm_y = special.sph_harm_y  # SciPy >= 1.17


    def sph_harm(m, n, theta, phi, out=None):
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
    trapz = np.trapz  # NumPy < 2.4
except AttributeError:
    trapz = np.trapezoid  # NumPy >= 2.4

try:
    # noinspection PyUnresolvedReferences
    simps = integrate.simps  # older SciPy
except AttributeError:
    # newer SciPy: simpson is the replacement
    def simps(y, x=None, dx=1.0, axis=-1, even=None):
        # ignore `even` (removed in simpson); most code never used it explicitly
        return integrate.simpson(y, x=x, dx=dx, axis=axis)
