"""Pulsation mode spherical harmonics and amplitude calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import const, settings
from elisa.base.body import Body
from elisa.logger import getLogger
from elisa.umpy import sph_harm

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.pulse.mode import PulsationMode

logger = getLogger("pulse.pulsations")


def spherical_harmonics(
    mode: PulsationMode,
    points: NDArray,
    time_exponential: complex,
    *,
    order: int | None = None,
    degree: int | None = None,
) -> NDArray:
    """Calculate normalized spherical harmonics with RMS = 1.

    Computes spherical harmonics Y_m^l evaluated at given spherical coordinate
    points, normalized such that the root-mean-square is 1. The harmonics are
    multiplied by a time-dependent exponential factor.

    :param mode: Pulsation mode object containing degree and order information.
    :type mode: object
    :param points: Spherical coordinates (r, phi, theta) where harmonics are evaluated,
                   shape (n_points, 3).
    :type points: NDArray
    :param time_exponential: Time-dependent complex exponential factor.
    :type time_exponential: complex
    :param order: Azimuthal order m of the mode (overrides mode.m if provided).
    :type order: int | None
    :param degree: Spherical harmonic degree l (overrides mode.l if provided).
    :type degree: int | None
    :returns: Normalized spherical harmonics at input points with shape (n_points,).
    :rtype: NDArray
    """
    l = mode.l if degree is None else degree  # noqa: E741
    m = mode.m if order is None else order
    return (
        mode.renorm_const
        * sph_harm(m, l, points[:, 1], points[:, 2])
        * time_exponential
    )


def diff_spherical_harmonics_by_phi(
    mode: PulsationMode,
    harmonics: list[NDArray],
) -> NDArray:
    """Calculate azimuthal derivative of spherical harmonics.

    Computes the partial derivative of spherical harmonics with respect to
    the azimuthal coordinate phi: d Y_m^l / d phi.

    :param mode: Pulsation mode object containing azimuthal order information.
    :type mode: object
    :param harmonics: List of harmonics [Y_l^m, Y_l^(m+1)].
    :type harmonics: list[NDArray]
    :returns: Azimuthal derivative of spherical harmonics with shape (n_points,).
    :rtype: NDArray
    """
    return (0 + 1j) * mode.m * harmonics[0]


def diff_spherical_harmonics_by_theta(
    mode: PulsationMode,
    harmonics: list[NDArray],
    phis: NDArray,
    thetas: NDArray,
) -> NDArray:
    """Calculate latitudinal derivative of spherical harmonics.

    Computes the partial derivative of spherical harmonics with respect to
    the latitudinal (polar) coordinate theta: d Y_m^l / d theta.

    :param mode: Pulsation mode object containing degree and order information.
    :type mode: object
    :param harmonics: List of harmonics [Y_l^m, Y_l^(m+1)].
    :type harmonics: list[NDArray]
    :param phis: Azimuthal coordinates where derivative is evaluated, shape (n_points,).
    :type phis: NDArray
    :param thetas: Latitudinal coordinates where derivative is evaluated, shape (n_points,).
    :type thetas: NDArray
    :returns: Latitudinal derivative of spherical harmonics with shape (n_points,).
    :rtype: NDArray
    """
    theta_test = np.logical_and(thetas != 0.0, thetas != const.PI)
    derivative = np.zeros(phis.shape, dtype=np.complex128)
    derivative[theta_test] = (
        mode.m * harmonics[0][theta_test] / np.tan(thetas[theta_test])
        + np.sqrt((mode.l - mode.m) * (mode.l + mode.m + 1))
        * np.exp((0 - 1j) * phis[theta_test])
        * harmonics[1][theta_test]
    )
    return derivative


def horizontal_displacement_normalization(
    derivatives: list[NDArray],
    harmonics: list[NDArray],
) -> float:
    """Calculate normalization constant for horizontal displacement.

    Normalizes the root-mean-square of horizontal displacement of a given
    pulsation mode to unity. The horizontal displacement is the combination
    of azimuthal and latitudinal displacement components.

    :param derivatives: List of harmonic derivatives [dY/d_phi, dY/d_theta].
    :type derivatives: list[NDArray]
    :param harmonics: List of harmonics [Y_l^m, Y_l^(m+1)].
    :type harmonics: list[NDArray]
    :returns: Normalization constant for horizontal displacement.
    :rtype: float
    """
    numerator = np.sum(np.power(np.abs(harmonics[0]), 2))
    denominator = np.sum(
        np.power(np.abs(derivatives[0]), 2)
        + np.power(np.abs(derivatives[1]), 2),
    )
    # noinspection PyUnresolvedReferences
    return np.sqrt(numerator / denominator)


def assign_amplitudes(
    star_container: StarContainer | Body,
    normalization_constant: float = 1.0,
) -> None:
    """Assign radial and horizontal displacement amplitudes to pulsation modes.

    Calculates and assigns the radial and horizontal (non-radial) displacement
    amplitudes for each pulsation mode based on the mode parameters and stellar
    properties. For modes without an explicit horizontal-to-radial amplitude ratio,
    it is calculated from the mode properties. Temperature amplitude factors are
    computed for modes where not explicitly provided.

    :param star_container: Star container object with pulsation modes.
    :type star_container: object
    :param normalization_constant: Scaling factor for amplitudes. For binary systems,
                                   this is the semi-major axis; for single stars, it
                                   should remain 1.0. Defaults to 1.0.
    :type normalization_constant: float
    :raises ValueError: If a radial mode (l=0) or mode with only radial motion
                        lacks an explicit temperature_amplitude_factor.
    """
    r_equiv = star_container.equivalent_radius * normalization_constant
    mult = const.G * star_container.mass / r_equiv**3

    for mode_index, mode in star_container.pulsations.items():
        # Calculate horizontal/radial amplitude ratio if not provided (Aerts 2010, p. 198)
        if mode.horizontal_to_radial_amplitude_ratio is None:
            amp_ratio = (
                np.sqrt(mode.l * (mode.l + 1))
                * mult
                / mode.angular_frequency**2
            )
            mode.horizontal_to_radial_amplitude_ratio = amp_ratio

        amplitude = mode.amplitude / mode.angular_frequency

        # Calculate radial and horizontal amplitudes
        mode.radial_amplitude = amplitude / np.sqrt(
            mode.horizontal_to_radial_amplitude_ratio**2 + 1,
        )
        mode.horizontal_amplitude = (
            mode.horizontal_to_radial_amplitude_ratio * mode.radial_amplitude
        )

        # Assign or calculate temperature amplitude factor
        if mode.temperature_amplitude_factor is None:
            if mode.l == 0 or mode.horizontal_to_radial_amplitude_ratio == 0.0:
                error_msg = (
                    "Parameter `temperature_amplitude_factor` needs to be "
                    "supplied in case of radial modes or in case of modes "
                    "with radial motion."
                )
                raise ValueError(error_msg)
            mode.temperature_amplitude_factor = (
                temp_amplitude(mode) * mode.radial_amplitude / r_equiv
            )

        # Check for excessive surface displacement
        surf_ampl = mode.horizontal_amplitude / r_equiv
        if surf_ampl > settings.SURFACE_DISPLACEMENT_TOL:
            prec = int(-np.log10(surf_ampl) + 2)
            if not settings.SUPPRESS_WARNINGS:
                warning_msg = (
                    f"Relative horizontal surface displacement amplitude "
                    f"({round(surf_ampl, prec)}) for the mode {mode_index} "
                    f"exceeded safe tolerances ({settings.SURFACE_DISPLACEMENT_TOL}) "
                    f"given by the use of linear approximation. This can lead to "
                    f"invalid surface discretization. Use this result with caution."
                )
                logger.warning(warning_msg)


def temp_amplitude(mode: PulsationMode) -> float:
    """Calculate temperature perturbation amplitude for a pulsation mode.

    Computes the temperature perturbation amplitude according to equation 22
    in Townsend 2003:

        delta T = temp_amplitude * (delta r / r) * T
        temp_amplitude = nabla_ad * (K*l*(l+1) - 4 - 1/K)

    where K is the horizontal-to-radial amplitude ratio and nabla_ad is the
    adiabatic temperature gradient.

    :param mode: Pulsation mode object containing degree and amplitude ratio.
    :type mode: object
    :returns: Temperature perturbation amplitude factor.
    :rtype: float
    :raises ZeroDivisionError: If horizontal_to_radial_amplitude_ratio is zero.
    """
    return const.IDEAL_ADIABATIC_GRADIENT * (
        mode.horizontal_to_radial_amplitude_ratio * mode.l * (mode.l + 1)
        - 4
        - 1 / mode.horizontal_to_radial_amplitude_ratio
    )


