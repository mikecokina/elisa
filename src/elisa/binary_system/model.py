from __future__ import annotations

from typing import Literal, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

from elisa import umpy as up
from elisa.base.types import FLOAT
from elisa.types import Float

NumericLike: TypeAlias = Float | NDArray[Float]


def static_potential_primary_fn(radius: NumericLike, *args: Float) -> NumericLike:
    """Evaluate the implicit primary surface potential in spherical coordinates.

    This helper is exposed for multiprocessing so that callers do not need to
    pickle classes, loggers, or other non-trivial objects.

    The expected argument layout is::

        (mass_ratio, surface_potential, b, c, d, e)

    where the coefficients satisfy::

        Psi1 = 1 / r + q / sqrt(b + r^2 - c r) - d r + e r^2

    :param radius: Radial coordinate in spherical coordinates.
    :type radius: Float | NDArray[Float]
    :param args: Precomputed coefficients and target surface potential in the
        order ``(mass_ratio, surface_potential, b, c, d, e)``.
    :type args: Float
    :return: Difference between the evaluated potential and the requested
        surface potential.
    :rtype: Float | NDArray[Float]
    """
    mass_ratio, surface_potential, b, c, d, e = args
    radius2 = up.power(radius, 2)
    potential = 1 / radius + mass_ratio / up.sqrt(b + radius2 - c * radius) - d * radius + e * radius2
    return potential - surface_potential


def static_potential_secondary_fn(radius: NumericLike, *args: Float) -> NumericLike:
    """Evaluate the implicit secondary surface potential in spherical coordinates.

    This helper is exposed for multiprocessing so that callers do not need to
    pickle classes, loggers, or other non-trivial objects.

    The expected argument layout is::

        (mass_ratio, surface_potential, b, c, d, e, f)

    where the coefficients satisfy::

        Psi2 = q / r + 1 / sqrt(b + r^2 - c r) - d r + e r^2 + f

    :param radius: Radial coordinate in spherical coordinates.
    :type radius: Float | NDArray[Float]
    :param args: Precomputed coefficients and target surface potential in the
        order ``(mass_ratio, surface_potential, b, c, d, e, f)``.
    :type args: Float
    :return: Difference between the evaluated potential and the requested
        surface potential.
    :rtype: Float | NDArray[Float]
    """
    mass_ratio, surface_potential, b, c, d, e, f = args
    radius2 = up.power(radius, 2)
    potential = mass_ratio / radius + 1.0 / up.sqrt(b + radius2 - c * radius) - d * radius + e * radius2 + f
    return potential - surface_potential


def static_potential_primary_cylindrical_fn(
    radius: NumericLike,
    *args: Float,
) -> NumericLike:
    """Evaluate the implicit primary surface potential in cylindrical coordinates.

    This form is useful for W UMa systems.

    The expected argument layout is::

        (mass_ratio, surface_potential, a, b, c, d, e, f)

    where the coefficients satisfy::

        Psi1 = 1 / sqrt(a + r^2) + q / sqrt(b + r^2) - c + d (e + f r^2)

    (a, b, c, d, e, f) are precalculated values to decrease runtime:

        (a, b, c, d, e, f) such that: Psi1 = 1/sqrt(a+r^2) + q/sqrt(b + r^2) - c + d*(e+f*r^2)

    :param radius: Radial coordinate in cylindrical coordinates.
    :type radius: Float | NDArray[Float]
    :param args: Precomputed coefficients and target surface potential in the
        order ``(mass_ratio, surface_potential, a, b, c, d, e, f)``.
    :type args: Float
    :return: Difference between the evaluated potential and the requested
        surface potential.
    :rtype: Float | NDArray[Float]
    """
    mass_ratio, surface_potential, a, b, c, d, e, f = args
    radius2 = up.power(radius, 2)
    return 1 / up.sqrt(a + radius2) + mass_ratio / up.sqrt(b + radius2) - c + d * (e + f * radius2) - surface_potential


def static_potential_secondary_cylindrical_fn(
    radius: NumericLike,
    *args: Float,
) -> NumericLike:
    """Evaluate the implicit secondary surface potential in cylindrical coordinates.

    This form is useful for W UMa systems.

    The expected argument layout is::

        (mass_ratio, surface_potential, a, b, c, d, e, f)

    (a, b, c, d, e, f) are precalculated values to decrease runtime:

        (a, b, c, d, e, f) such that: Psi1 = 1/sqrt(a+r^2) + q/sqrt(b + r^2) - c + d*(e+f*r^2)

    :param radius: Radial coordinate in cylindrical coordinates.
    :type radius: Float | NDArray[Float]
    :param args: Precomputed coefficients and target surface potential in the
        order ``(mass_ratio, surface_potential, a, b, c, d, e, f)``.
    :type args: Float
    :return: Difference between the evaluated potential and the requested
        surface potential.
    :rtype: Float | NDArray[Float]
    """
    mass_ratio, surface_potential, a, b, c, d, e, f = args
    radius2 = up.power(radius, 2)
    return (
        mass_ratio / up.sqrt(a + radius2) + 1.0 / up.sqrt(b + radius2) + c * (d + e * radius2) + f - surface_potential
    )


def potential_value_primary(radius: NumericLike, *args: Float) -> NumericLike:
    """Calculate the modified Kopal potential from the primary perspective.

    The expected argument layout is::

        (mass_ratio, b, c, d, e)

    such that::

        Psi1 = 1 / r + q / sqrt(b + r^2 - c r) - d r + e r^2

    :param radius: Spherical radial coordinate.
    :type radius: Float | NDArray[Float]
    :param args: Precomputed coefficients in the order
        ``(mass_ratio, b, c, d, e)``.
    :type args: Float
    :return: Potential value.
    :rtype: Float | NDArray[Float]
    """
    mass_ratio, b, c, d, e = args
    radius2 = up.power(radius, 2)
    return 1 / radius + mass_ratio / up.sqrt(b + radius2 - c * radius) - d * radius + e * radius2


def potential_value_secondary(radius: NumericLike, *args: Float) -> NumericLike:
    """Calculate the modified Kopal potential from the secondary perspective.

    The expected argument layout is::

        (mass_ratio, b, c, d, e, f)

    such that::

        Psi2 = q / r + 1 / sqrt(b + r^2 - c r) - d r + e r^2 + f

    :param radius: Spherical radial coordinate.
    :type radius: Float | NDArray[Float]
    :param args: Precomputed coefficients in the order
        ``(mass_ratio, b, c, d, e, f)``.
    :type args: Float
    :return: Potential value.
    :rtype: Float | NDArray[Float]
    """
    mass_ratio, b, c, d, e, f = args
    radius2 = up.power(radius, 2)
    return mass_ratio / radius + 1.0 / up.sqrt(b + radius2 - c * radius) - d * radius + e * radius2 + f


def potential_primary_fn(
    radius: NumericLike,
    coefficients: tuple[Float, Float, Float, Float, Float],
    surface_potential: Float,
) -> NumericLike:
    """Evaluate the implicit primary potential equation.

    :param radius: Spherical radial coordinate.
    :type radius: Float | NDArray[Float]
    :param coefficients: Precomputed primary potential coefficients in the
        order ``(mass_ratio, b, c, d, e)``.
    :type coefficients: tuple[Float, Float, Float, Float, Float]
    :param surface_potential: Target surface potential.
    :type surface_potential: Float
    :return: Difference between the evaluated potential and the requested
        surface potential.
    :rtype: Float | NDArray[Float]
    """
    return potential_value_primary(radius, *coefficients) - surface_potential


def potential_secondary_fn(
    radius: NumericLike,
    coefficients: tuple[Float, Float, Float, Float, Float, Float],
    surface_potential: Float,
) -> NumericLike:
    """Evaluate the implicit secondary potential equation.

    :param radius: Spherical radial coordinate.
    :type radius: Float | NDArray[Float]
    :param coefficients: Precomputed secondary potential coefficients in the
        order ``(mass_ratio, b, c, d, e, f)``.
    :type coefficients: tuple[Float, Float, Float, Float, Float, Float]
    :param surface_potential: Target surface potential.
    :type surface_potential: Float
    :return: Difference between the evaluated potential and the requested
        surface potential.
    :rtype: Float | NDArray[Float]
    """
    return potential_value_secondary(radius, *coefficients) - surface_potential


def potential_value_primary_cylindrical(
    radius: NumericLike,
    *args: Float,
) -> NumericLike:
    """Calculate the primary potential in cylindrical coordinates.

    The coordinate system is ``(r_n, phi_n, z_n)``, where ``z_n = x`` and the
    axis points along the Cartesian ``x`` direction.

    This function is intended for generation of necks of W UMa systems, so
    component distance ``= 1`` and synchronicity ``= 1`` are assumed by the
    surrounding derivation.

    The expected argument layout is::

        (mass_ratio, a, b, c, d, e)

    such that::

        Psi1 = 1 / sqrt(a + r^2) + q / sqrt(b + r^2) - c + d (a + e r^2)

    :param radius: Cylindrical radial coordinate.
    :type radius: Float | NDArray[Float]
    :param args: Precomputed coefficients in the order
        ``(mass_ratio, a, b, c, d, e)``.
    :type args: Float
    :return: Potential value.
    :rtype: Float | NDArray[Float]
    """
    mass_ratio, a, b, c, d, e = args
    radius2 = up.power(radius, 2)
    return 1 / up.sqrt(a + radius2) + mass_ratio / up.sqrt(b + radius2) - c + d * (a + e * radius2)


def potential_value_secondary_cylindrical(
    radius: NumericLike,
    *args: Float,
) -> NumericLike:
    """Calculate the secondary potential in cylindrical coordinates.

    The coordinate system is ``(r_n, phi_n, z_n)``, where ``z_n = x`` and the
    axis points along the Cartesian ``x`` direction.

    The expected argument layout is::

        (mass_ratio, a, b, c, d, e, f)

    such that::

        Psi2 = q / sqrt(a + r^2) + 1 / sqrt(b + r^2) - c + d (a + e r^2) + f

    :param radius: Cylindrical radial coordinate.
    :type radius: Float | NDArray[Float]
    :param args: Precomputed coefficients in the order
        ``(mass_ratio, a, b, c, d, e, f)``.
    :type args: Float
    :return: Potential value.
    :rtype: Float | NDArray[Float]
    """
    mass_ratio, a, b, c, d, e, f = args
    radius2 = up.power(radius, 2)
    return mass_ratio / up.sqrt(a + radius2) + 1.0 / up.sqrt(b + radius2) - c + d * (e * radius2 + a) + f


def potential_primary_cylindrical_fn(
    radius: NumericLike,
    coefficients: tuple[Float, Float, Float, Float, Float, Float],
    surface_potential: Float,
) -> NumericLike:
    """Evaluate the implicit primary cylindrical potential equation.

    :param radius: Cylindrical radial coordinate.
    :type radius: Float | NDArray[Float]
    :param coefficients: Precomputed primary cylindrical coefficients in the
        order ``(mass_ratio, a, b, c, d, e)``.
    :type coefficients: tuple[Float, Float, Float, Float, Float, Float]
    :param surface_potential: Target surface potential.
    :type surface_potential: Float
    :return: Difference between the evaluated potential and the requested
        surface potential.
    :rtype: Float | NDArray[Float]
    """
    return potential_value_primary_cylindrical(radius, *coefficients) - surface_potential


def potential_secondary_cylindrical_fn(
    radius: NumericLike,
    coefficients: tuple[Float, Float, Float, Float, Float, Float, Float],
    surface_potential: Float,
) -> NumericLike:
    """Evaluate the implicit secondary cylindrical potential equation.

    :param radius: Cylindrical radial coordinate.
    :type radius: Float | NDArray[Float]
    :param coefficients: Precomputed secondary cylindrical coefficients in the
        order ``(mass_ratio, a, b, c, d, e, f)``.
    :type coefficients: tuple[Float, Float, Float, Float, Float, Float, Float]
    :param surface_potential: Target surface potential.
    :type surface_potential: Float
    :return: Difference between the evaluated potential and the requested
        surface potential.
    :rtype: Float | NDArray[Float]
    """
    return potential_value_secondary_cylindrical(radius, *coefficients) - surface_potential


def radial_primary_potential_derivative(
    radius: NumericLike,
    *args: Float,
) -> NumericLike:
    """Calculate the radial derivative of the primary potential.

    The expected argument layout is::

        (mass_ratio, b, c, d, e)

    such that::

        dPsi1 / dr = -1 / r^2 + 0.5 q (c - 2 r) / (b - c r + r^2)^(3/2)
                     - d + 2 e r

    :param radius: Radius of the evaluation point in spherical coordinates.
    :type radius: Float | NDArray[Float]
    :param args: Precomputed coefficients in the order
        ``(mass_ratio, b, c, d, e)``.
    :type args: Float
    :return: Radial derivative value.
    :rtype: Float | NDArray[Float]
    """
    mass_ratio, b, c, d, e = args
    radius2 = up.power(radius, 2)

    return (
        -1 / radius2
        + 0.5 * mass_ratio * (c - 2 * radius) / up.power(b - c * radius + radius2, 1.5)
        - d
        + 2 * e * radius
    )


def radial_secondary_potential_derivative(
    radius: NumericLike,
    *args: Float,
) -> NumericLike:
    """Calculate the radial derivative of the secondary potential.

    The expected argument layout is::

        (mass_ratio, b, c, d, e, f)

    such that::

        dPsi2 / dr = -q / r^2 + (0.5 c - r) / (b - c r + r^2)^(3/2)
                     - d + 2 e r

    :param radius: Radius of the evaluation point in spherical coordinates.
    :type radius: Float | NDArray[Float]
    :param args: Precomputed coefficients in the order
        ``(mass_ratio, b, c, d, e, f)``.
    :type args: Float
    :return: Radial derivative value.
    :rtype: Float | NDArray[Float]
    """
    mass_ratio, b, c, d, e, _f = args
    radius2 = up.power(radius, 2)

    return -mass_ratio / radius2 + (0.5 * c - radius) / up.power(b - c * radius + radius2, 1.5) - d + 2 * e * radius


@overload
def pre_calculate_for_potential_value_primary(
    synchronicity: Float,
    mass_ratio: Float,
    distance: Float,
    phi: Float,
    theta: Float,
    *,
    return_as_tuple: Literal[False] = False,
) -> tuple[Float, Float, Float, Float]: ...


@overload
def pre_calculate_for_potential_value_primary(
    synchronicity: Float,
    mass_ratio: Float,
    distance: Float,
    phi: NDArray[Float],
    theta: NDArray[Float],
    *,
    return_as_tuple: Literal[False] = False,
) -> NDArray[Float]: ...


@overload
def pre_calculate_for_potential_value_primary(
    synchronicity: Float,
    mass_ratio: Float,
    distance: Float,
    phi: NDArray[Float],
    theta: NDArray[Float],
    *,
    return_as_tuple: Literal[True],
) -> tuple[
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
]: ...


def pre_calculate_for_potential_value_primary(
    synchronicity: Float,
    mass_ratio: Float,
    distance: Float,
    phi: Float | NDArray[Float],
    theta: Float | NDArray[Float],
    *,
    return_as_tuple: bool = False,
) -> (
    tuple[Float, Float, Float, Float]
    | NDArray[Float]
    | tuple[
        NDArray[Float],
        NDArray[Float],
        NDArray[Float],
        NDArray[Float],
    ]
):
    """Precompute auxiliary values for the primary spherical potential.

    The coefficients are used to avoid repeated work during iterative solving.

    The returned coefficients satisfy::

        Psi1 = 1 / r + q / sqrt(b + r^2 - c r) - d r + e r^2

    :param synchronicity: Component synchronicity.
    :type synchronicity: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param distance: Distance between components.
    :type distance: Float
    :param phi: Azimuth angle in radians.
    :type phi: Float | NDArray[Float]
    :param theta: Latitude angle in radians.
    :type theta: Float | NDArray[Float]
    :param return_as_tuple: If ``True``, return coefficient vectors as a tuple
        instead of a stacked matrix.
    :type return_as_tuple: bool
    :return: Either scalar coefficients ``(b, c, d, e)``, a coefficient matrix,
        or a tuple of coefficient vectors.
    :rtype: tuple[Float, Float, Float, Float]
            | NDArray[Float]
            | tuple[NDArray[Float], NDArray[Float], NDArray[Float], NDArray[numpy.float64]]
    """
    sin_theta = up.sin(theta)
    cs = up.cos(phi) * sin_theta

    b = up.power(distance, 2)
    c = 2 * distance * cs
    d = (mass_ratio * cs) / b
    e = 0.5 * up.power(synchronicity, 2) * (1 + mass_ratio) * up.power(sin_theta, 2)

    if np.isscalar(phi):
        return b, c, d, e

    phi_array = np.asarray(phi, dtype=FLOAT)
    c_array = np.asarray(c, dtype=FLOAT)
    d_array = np.asarray(d, dtype=FLOAT)
    e_array = np.asarray(e, dtype=FLOAT)
    b_array = b * np.ones(np.shape(phi_array), dtype=FLOAT)

    if return_as_tuple:
        return b_array, c_array, d_array, e_array

    return np.column_stack((b_array, c_array, d_array, e_array))


@overload
def pre_calculate_for_potential_value_secondary(
    synchronicity: Float,
    mass_ratio: Float,
    distance: Float,
    phi: Float,
    theta: Float,
    *,
    return_as_tuple: Literal[False] = False,
) -> tuple[Float, Float, Float, Float, Float]: ...


@overload
def pre_calculate_for_potential_value_secondary(
    synchronicity: Float,
    mass_ratio: Float,
    distance: Float,
    phi: NDArray[Float],
    theta: NDArray[Float],
    *,
    return_as_tuple: Literal[False] = False,
) -> NDArray[Float]: ...


@overload
def pre_calculate_for_potential_value_secondary(
    synchronicity: Float,
    mass_ratio: Float,
    distance: Float,
    phi: NDArray[Float],
    theta: NDArray[Float],
    *,
    return_as_tuple: Literal[True],
) -> tuple[
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
]: ...


def pre_calculate_for_potential_value_secondary(
    synchronicity: Float,
    mass_ratio: Float,
    distance: Float,
    phi: Float | NDArray[Float],
    theta: Float | NDArray[Float],
    *,
    return_as_tuple: bool = False,
) -> (
    tuple[Float, Float, Float, Float, Float]
    | NDArray[Float]
    | tuple[
        NDArray[Float],
        NDArray[Float],
        NDArray[Float],
        NDArray[Float],
        NDArray[Float],
    ]
):
    """Precompute auxiliary values for the secondary spherical potential.

    The coefficients are used to avoid repeated work during iterative solving.

    The returned coefficients satisfy::

        Psi2 = q / r + 1 / sqrt(b + r^2 - c r) - d r + e r^2 + f

    :param synchronicity: Component synchronicity.
    :type synchronicity: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param distance: Distance between components.
    :type distance: Float
    :param phi: Azimuth angle in radians.
    :type phi: Float | NDArray[Float]
    :param theta: Latitude angle in radians.
    :type theta: Float | NDArray[Float]
    :param return_as_tuple: If ``True``, return coefficient vectors as a tuple
        instead of a stacked matrix.
    :type return_as_tuple: bool
    :return: Either scalar coefficients ``(b, c, d, e, f)``, a coefficient
        matrix, or a tuple of coefficient vectors.
    :rtype: tuple[Float, Float, Float, Float, Float] | NDArray[Float] | tuple[NDArray[Float], NDArray[Float], NDArray[Float], NDArray[Float], NDArray[Float]]
    """
    sin_theta = up.sin(theta)
    cs = up.cos(phi) * sin_theta

    b = up.power(distance, 2)
    c = 2 * distance * cs
    d = cs / b
    e = 0.5 * up.power(synchronicity, 2) * (1 + mass_ratio) * up.power(sin_theta, 2)
    f = 0.5 - 0.5 * mass_ratio

    if np.isscalar(phi):
        return b, c, d, e, f

    phi_array = np.asarray(phi, dtype=FLOAT)
    c_array = np.asarray(c, dtype=FLOAT)
    d_array = np.asarray(d, dtype=FLOAT)
    e_array = np.asarray(e, dtype=FLOAT)
    b_array = b * np.ones(np.shape(phi_array), dtype=FLOAT)
    f_array = f * np.ones(np.shape(phi_array), dtype=FLOAT)

    if return_as_tuple:
        return b_array, c_array, d_array, e_array, f_array

    return np.column_stack((b_array, c_array, d_array, e_array, f_array))


def radial_primary_potential_derivative_cylindrical(
    radius: NumericLike,
    *args: Float,
) -> NumericLike:
    """Calculate the radial derivative of the primary cylindrical potential.

    The expected argument layout is::

        (mass_ratio, a, b, c, d, e)

    such that::

        dPsi1 / dr = -r / (a + r^2)^(3/2) - r q / (b + r^2)^(3/2) + 2 d e r

    :param radius: Radius of the evaluation point in cylindrical coordinates.
    :type radius: Float | NDArray[Float]
    :param args: Precomputed coefficients in the order
        ``(mass_ratio, a, b, c, d, e)``.
    :type args: Float
    :return: Radial derivative value.
    :rtype: Float | NDArray[Float]
    """
    mass_ratio, a, b, _c, d, e = args
    radius2 = up.power(radius, 2)

    return -radius / up.power(a + radius2, 1.5) - radius * mass_ratio / up.power(b + radius2, 1.5) + 2 * d * e * radius


def radial_secondary_potential_derivative_cylindrical(
    radius: NumericLike,
    *args: Float,
) -> NumericLike:
    """Calculate the radial derivative of the secondary cylindrical potential.

    The expected argument layout is::

        (mass_ratio, a, b, c, d, e, f)

    such that::

        dPsi2 / dr = -q r / (a + r^2)^(3/2) - r / (b + r^2)^(3/2) + 2 d e r

    :param radius: Radius of the evaluation point in cylindrical coordinates.
    :type radius: Float | NDArray[Float]
    :param args: Precomputed coefficients in the order
        ``(mass_ratio, a, b, c, d, e, f)``.
    :type args: Float
    :return: Radial derivative value.
    :rtype: Float | NDArray[Float]
    """
    mass_ratio, a, b, _c, d, e, _f = args
    radius2 = up.power(radius, 2)

    return -radius * mass_ratio / up.power(a + radius2, 1.5) - radius / up.power(b + radius2, 1.5) + 2 * d * e * radius


@overload
def pre_calculate_for_potential_value_primary_cylindrical(
    synchronicity: Float,
    mass_ratio: Float,
    phi: Float,
    z: Float,
    distance: Float,
    *,
    return_as_tuple: Literal[False] = False,
) -> tuple[Float, Float, Float, Float, Float]: ...


@overload
def pre_calculate_for_potential_value_primary_cylindrical(
    synchronicity: Float,
    mass_ratio: Float,
    phi: NDArray[Float],
    z: NDArray[Float],
    distance: Float,
    *,
    return_as_tuple: Literal[False] = False,
) -> NDArray[Float]: ...


@overload
def pre_calculate_for_potential_value_primary_cylindrical(
    synchronicity: Float,
    mass_ratio: Float,
    phi: NDArray[Float],
    z: NDArray[Float],
    distance: Float,
    *,
    return_as_tuple: Literal[True],
) -> tuple[
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
]: ...


def pre_calculate_for_potential_value_primary_cylindrical(
    synchronicity: Float,
    mass_ratio: Float,
    phi: Float | NDArray[Float],
    z: Float | NDArray[Float],
    distance: Float,
    *,
    return_as_tuple: bool = False,
) -> (
    tuple[Float, Float, Float, Float, Float]
    | NDArray[Float]
    | tuple[
        NDArray[Float],
        NDArray[Float],
        NDArray[Float],
        NDArray[Float],
        NDArray[Float],
    ]
):
    """Precompute auxiliary values for the primary cylindrical potential.

    The coefficients are used to avoid repeated work during iterative solving.

    The returned coefficients satisfy::

        Psi1 = 1 / sqrt(a + r^2) + q / sqrt(b + r^2) - c + d (a + e r^2)

    :param synchronicity: Component synchronicity.
    :type synchronicity: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param phi: Azimuth angle in radians.
    :type phi: Float | NDArray[Float]
    :param z: Cylindrical ``z_n`` coordinate, identical with Cartesian ``x``.
    :type z: Float | NDArray[Float]
    :param distance: Distance between components.
    :type distance: Float
    :param return_as_tuple: If ``True``, return coefficient vectors as a tuple
        instead of a stacked matrix.
    :type return_as_tuple: bool
    :return: Either scalar coefficients ``(a, b, c, d, e)``, a coefficient
        matrix, or a tuple of coefficient vectors.
    :rtype: tuple[Float, Float, Float, Float, Float] | NDArray[Float] | tuple[NDArray[Float], NDArray[Float], NDArray[Float], NDArray[Float], NDArray[float]]
    """  # noqa: E501
    a = up.power(z, 2)
    b = up.power(distance - z, 2)
    c = mass_ratio * z / up.power(distance, 2)
    d = 0.5 * up.power(synchronicity, 2) * (1 + mass_ratio)
    e = up.power(up.sin(phi), 2)

    if np.isscalar(phi):
        return a, b, c, d, e

    phi_array = np.asarray(phi, dtype=FLOAT)
    a_array = np.asarray(a, dtype=FLOAT)
    b_array = np.asarray(b, dtype=FLOAT)
    c_array = np.asarray(c, dtype=FLOAT)
    e_array = np.asarray(e, dtype=FLOAT)
    d_array = d * np.ones(np.shape(phi_array), dtype=FLOAT)

    if return_as_tuple:
        return a_array, b_array, c_array, d_array, e_array

    return np.column_stack((a_array, b_array, c_array, d_array, e_array))


@overload
def pre_calculate_for_potential_value_secondary_cylindrical(
    synchronicity: Float,
    mass_ratio: Float,
    phi: Float,
    z: Float,
    distance: Float,
    *,
    return_as_tuple: Literal[False] = False,
) -> tuple[Float, Float, Float, Float, Float, Float]: ...


@overload
def pre_calculate_for_potential_value_secondary_cylindrical(
    synchronicity: Float,
    mass_ratio: Float,
    phi: NDArray[Float],
    z: NDArray[Float],
    distance: Float,
    *,
    return_as_tuple: Literal[False] = False,
) -> NDArray[Float]: ...


@overload
def pre_calculate_for_potential_value_secondary_cylindrical(
    synchronicity: Float,
    mass_ratio: Float,
    phi: NDArray[Float],
    z: NDArray[Float],
    distance: Float,
    *,
    return_as_tuple: Literal[True],
) -> tuple[
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
]: ...


def pre_calculate_for_potential_value_secondary_cylindrical(
    synchronicity: Float,
    mass_ratio: Float,
    phi: Float | NDArray[Float],
    z: Float | NDArray[Float],
    distance: Float,
    *,
    return_as_tuple: bool = False,
) -> (
    tuple[Float, Float, Float, Float, Float, Float]
    | NDArray[Float]
    | tuple[
        NDArray[Float],
        NDArray[Float],
        NDArray[Float],
        NDArray[Float],
        NDArray[Float],
        NDArray[Float],
    ]
):
    """Precompute auxiliary values for the secondary cylindrical potential.

    The coefficients are used to avoid repeated work during iterative solving.

    The returned coefficients satisfy::

        Psi2 = q / sqrt(a + r^2) + 1 / sqrt(b + r^2) - c + d (a + e r^2) + f

    :param synchronicity: Component synchronicity.
    :type synchronicity: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param phi: Azimuth angle in radians.
    :type phi: Float | NDArray[Float]
    :param z: Cylindrical ``z_n`` coordinate, identical with Cartesian ``x``.
    :type z: Float | NDArray[Float]
    :param distance: Distance between components.
    :type distance: Float
    :param return_as_tuple: If ``True``, return coefficient vectors as a tuple
        instead of a stacked matrix.
    :type return_as_tuple: bool
    :return: Either scalar coefficients ``(a, b, c, d, e, f)``, a coefficient
        matrix, or a tuple of coefficient vectors.
    :rtype: tuple[Float, Float, Float, Float, Float, Float] | NDArray[Float] | tuple[NDArray[Float], NDArray[Float], NDArray[Float], NDArray[Float], NDArray[Float], NDArray[float]]
    """  # noqa: E501
    a = up.power(z, 2)
    b = up.power(distance - z, 2)
    c = z / up.power(distance, 2)
    d = 0.5 * up.power(synchronicity, 2) * (1 + mass_ratio)
    e = up.power(up.sin(phi), 2)
    f = 0.5 * (1 - mass_ratio)

    if np.isscalar(phi):
        return a, b, c, d, e, f

    phi_array = np.asarray(phi, dtype=FLOAT)
    a_array = np.asarray(a, dtype=FLOAT)
    b_array = np.asarray(b, dtype=FLOAT)
    c_array = np.asarray(c, dtype=FLOAT)
    e_array = np.asarray(e, dtype=FLOAT)
    d_array = d * np.ones(np.shape(phi_array), dtype=FLOAT)
    f_array = f * np.ones(np.shape(phi_array), dtype=FLOAT)

    if return_as_tuple:
        return a_array, b_array, c_array, d_array, e_array, f_array

    return np.column_stack((a_array, b_array, c_array, d_array, e_array, f_array))


def primary_potential_derivative_x(x: NumericLike, *args: Float) -> NumericLike:
    """Calculate the primary potential derivative along the x-axis.

    The expected argument layout is::

        (synchronicity, mass_ratio, distance)

    :param x: Coordinate along the x-axis.
    :type x: Float | NDArray[Float]
    :param args: Coefficients in the order
        ``(synchronicity, mass_ratio, distance)``.
    :type args: Float
    :return: Derivative value.
    :rtype: Float | NDArray[Float]
    """
    synchronicity, mass_ratio, distance = args
    r_sqr = x**2
    rw_sqr = (distance - x) ** 2
    return (
        -(x / r_sqr ** (3.0 / 2.0))
        + (mass_ratio * (distance - x)) / rw_sqr ** (3.0 / 2.0)
        + synchronicity**2 * (mass_ratio + 1) * x
        - mass_ratio / distance**2
    )


def secondary_potential_derivative_x(x: NumericLike, *args: Float) -> NumericLike:
    """Calculate the secondary potential derivative along the x-axis.

    The expected argument layout is::

        (synchronicity, mass_ratio, distance)

    :param x: Coordinate along the x-axis.
    :type x: Float | NDArray[Float]
    :param args: Coefficients in the order
        ``(synchronicity, mass_ratio, distance)``.
    :type args: Float
    :return: Derivative value.
    :rtype: Float | NDArray[Float]
    """
    synchronicity, mass_ratio, distance = args
    r_sqr = x**2
    rw_sqr = (distance - x) ** 2
    return (
        -(x / r_sqr ** (3.0 / 2.0))
        + (mass_ratio * (distance - x)) / rw_sqr ** (3.0 / 2.0)
        - synchronicity**2 * (mass_ratio + 1) * (distance - x)
        + 1.0 / distance**2
    )
