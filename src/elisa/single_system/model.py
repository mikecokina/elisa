from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import const

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float


def surface_potential_from_polar_log_g(
        polar_log_g: NDArray[Float] | Float,
        mass: NDArray[Float] | Float,
) -> NDArray[Float] | Float:
    """Calculate surface potential from polar surface gravity.

    The polar surface gravity (log10) and mass are combined to produce a
    surface potential value for the polar point. The function accepts
    scalar or array-like inputs and returns a NumPy array of float values.

    :param polar_log_g: Polar gravity acceleration in log10 units.
    :type polar_log_g: numpy.typing.NDArray[elisa.types.Float] | elisa.types.Float
    :param mass: Stellar mass.
    :type mass: numpy.typing.NDArray[elisa.types.Float] | elisa.types.Float
    :returns: Surface potential value(s) (negative-valued).
    :rtype: numpy.typing.NDArray[elisa.types.Float] | elisa.types.Float
    """
    polar_gravity_acceleration = np.power(10.0, polar_log_g)
    return -np.power(const.G * mass * polar_gravity_acceleration, 0.5)


def potential_fn(
        radius: NDArray[Float] | Float,
        precalc_vals: NDArray[Float] | Float,
        target_potential: Float,
) -> NDArray[Float] | Float:
    """Implicit potential function used by the root solver.

    This adapts the generic :func:`potential` to the solver's calling
    convention by taking precomputed coefficients and the desired target
    potential and returning the residual: potential(radius) - target.

    :param radius: Radial coordinate(s) in spherical units.
    :type radius: numpy.typing.NDArray[elisa.types.Float] | elisa.types.Float
    :param precalc_vals: Precomputed coefficients (``a``, ``b``) produced by :func:`pre_calculate_for_potential_value`.
    :type precalc_vals: numpy.typing.NDArray[elisa.types.Float] | elisa.types.Float
    :param target_potential: Desired potential value to match.
    :type target_potential: elisa.types.Float
    :returns: Residual(s) of the potential equation.
    :rtype: numpy.typing.NDArray[elisa.types.Float] | elisa.types.Float
    """
    return potential(radius, precalc_vals[0], precalc_vals[1]) - target_potential


def potential(
    radius: NDArray[Float],
    a: NDArray[Float],
    b: NDArray[Float],
) -> NDArray[Float]:
    r"""Compute potential ``\Psi`` at a given radius.

    The potential is given by the expression::

        \Psi(r) = -a / r - b r^2

    :param radius: Radial position(s).
    :type radius: numpy.typing.NDArray[elisa.types.Float]
    :param a: Coefficient ``a`` (G*M).
    :type a: numpy.typing.NDArray[elisa.types.Float]
    :param b: Coefficient ``b`` (rotation term).
    :type b: numpy.typing.NDArray[elisa.types.Float]
    :returns: Potential value(s) at supplied radius(es).
    :rtype: numpy.typing.NDArray[elisa.types.Float]
    """
    return -a / radius - b * np.power(radius, 2.0)


def pre_calculate_for_potential_value(
    *args: Float,
    return_as_tuple: bool = False,
) -> NDArray[Float] | tuple[Float, Float] | tuple[NDArray[Float], NDArray[Float]]:
    r"""Precompute coefficients ``a`` and ``b`` for potential evaluation.

    ``a`` equals ``G * mass`` and ``b`` equals ``0.5 * (omega * sin(theta))**2``.
    The function accepts scalar or array-like ``theta``. When ``theta`` is
    scalar the scalars ``(a, b)`` are returned. For array ``theta`` the
    function returns either a stacked ``NDArray`` of shape (N, 2) or a tuple
    ``(a_array, b_array)`` when ``return_as_tuple`` is ``True``.

    :param args: Tuple (mass, angular_velocity, theta).
    :type args: tuple[elisa.types.Float, elisa.types.Float, elisa.types.Float]
    :param return_as_tuple: If True return a tuple of arrays instead of a
        2-column array (keyword-only).
    :type return_as_tuple: bool
    :returns: Coefficients for potential evaluation. Either a 2-column array
        or two arrays/floats.
    :rtype: numpy.typing.NDArray[elisa.types.Float] |
        tuple[elisa.types.Float, elisa.types.Float] |
        tuple[numpy.typing.NDArray[elisa.types.Float], numpy.typing.NDArray[elisa.types.Float]]
    """
    mass, angular_velocity, theta = args

    a = const.G * mass
    b = 0.5 * np.power(angular_velocity * np.sin(theta), 2)

    if np.isscalar(theta):
        return a, b

    aa = a * np.ones(np.shape(theta))
    return (aa, b) if return_as_tuple else np.column_stack((aa, b))


def radial_potential_derivative(radius: NDArray[Float], a: NDArray[Float], b: NDArray[Float]) -> NDArray[Float]:
    r"""Radial derivative of the potential in spherical coordinates.

    The radial derivative of :math:`\Psi` is::

        d\Psi/dr = a / r^2 - 2 b r

    :param radius: Radial coordinate(s).
    :type radius: numpy.typing.NDArray[elisa.types.Float]
    :param a: Coefficient a.
    :type a: numpy.typing.NDArray[elisa.types.Float]
    :param b: Coefficient b.
    :type b: numpy.typing.NDArray[elisa.types.Float]
    :returns: Radial derivative values.
    :rtype: numpy.typing.NDArray[elisa.types.Float]
    """
    return a / np.power(radius, 2) - 2 * b * radius
