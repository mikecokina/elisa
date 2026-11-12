"""Pulsation mode spherical harmonics and amplitude calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import const, settings
from elisa.base.error import raise_elisa_pulsation_mode_error
from elisa.logger import getLogger
from elisa.umpy import sph_harm

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.body import Body
    from elisa.base.container import StarContainer
    from elisa.pulse.mode import PulsationMode

logger = getLogger(__name__)


def spherical_harmonics(
    mode: PulsationMode,
    points: NDArray,
    time_exponential: complex,
    *,
    order: int | None = None,
    degree: int | None = None,
) -> NDArray:
    """Evaluate scalar harmonics with ELISA's full-sphere RMS normalization.

    Multiplication by sqrt(4*pi) converts the standard unit-integral harmonic
    to unit mean squared magnitude over a sphere. This is an analytic spherical
    normalization, not a discrete mesh RMS. The backend includes the
    Condon--Shortley phase; the supplied complex time factor is applied last.

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

    The input amplitude V is a velocity coefficient scale in m/s. For angular
    frequency omega and signed horizontal/radial ratio q, the physical length
    scales are A_r=V/(omega*sqrt(1+q**2)) and A_h=q*A_r, both in metres.
    A_r is not in general the sampled radial RMS or a pointwise maximum.
    Horizontal mesh RMS normalization is performed later in kinematics.

    :param star_container: Star container object with pulsation modes.
    :type star_container: object
    :param normalization_constant: Scaling factor for amplitudes. For binary systems,
                                   this is the semi-major axis; for single stars, it
                                   should remain 1.0. Defaults to 1.0.
    :type normalization_constant: float
    :raises PulsationModeError: If a radial mode (l=0) or mode with only radial motion
                        lacks an explicit temperature_amplitude_factor.
    """
    r_equiv = star_container.equivalent_radius * normalization_constant
    if r_equiv == 0:
        error_msg = "Pulsation amplitudes require a nonzero equivalent radius."
        raise_elisa_pulsation_mode_error(error_msg)
    mult = const.G * star_container.mass / r_equiv**3

    for mode_index, mode in star_container.pulsations.items():
        if mode.angular_frequency == 0:
            error_msg = "Pulsation amplitudes require a nonzero frequency."
            raise_elisa_pulsation_mode_error(error_msg)
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
                raise_elisa_pulsation_mode_error(error_msg)
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
    :raises PulsationModeError: If horizontal_to_radial_amplitude_ratio is zero.
    """
    # The prescription contains 1/K. Amplitude assignment requires an explicit
    # temperature factor for radial modes or K=0 rather than evaluating it.
    if mode.horizontal_to_radial_amplitude_ratio == 0:
        error_msg = "Temperature perturbation requires a nonzero horizontal-to-radial amplitude ratio."
        raise_elisa_pulsation_mode_error(error_msg)

    return const.IDEAL_ADIABATIC_GRADIENT * (
        mode.horizontal_to_radial_amplitude_ratio * mode.l * (mode.l + 1)
        - 4
        - 1 / mode.horizontal_to_radial_amplitude_ratio
    )


def cartesian_harmonics(mode: PulsationMode, directions: NDArray,
                        time_exponential: complex) -> tuple[NDArray, NDArray]:
    """Evaluate Y and its dimensionless Cartesian tangent gradient on unit n.

    Regular solid harmonics S_lm=r**l Y_lm are homogeneous Cartesian
    polynomials. Starting at S_00=1, build S_mm with x+i*y, S_(m+1,m) with z,
    then advance degree with the normalized z*S_(l-1,m), r**2*S_(l-2,m)
    recurrence. The derivative array follows the same recurrence by the product
    rule. Its coefficients include ELISA's sqrt(4*pi) harmonic normalization
    and the Condon--Shortley sign in the sectoral seed.

    At unit n, Euler's identity n.grad(S)=l*S gives the dimensionless tangent
    gradient g=grad(S)-(n.grad(S))*n. It is tangent to the equilibrium radial
    sphere, not necessarily tangent to the distorted stellar surface.
    Polynomial evaluation needs neither angular division nor a chosen pole
    meridian: the same Cartesian expression applies at and near both poles.
    On a sphere of physical radius R, the spatial surface gradient is g/R;
    kinematics uses the dimensionless g in its radius-weighted displacement law.

    Negative orders use Y_(l,-m)=(-1)**m*conj(Y_(l,m)) for the spatial harmonic
    and its gradient. Apply the supplied exp(-i*(omega*t+start_phase)) afterward;
    conjugating that temporal factor would reverse the mode's time convention.

    :param mode: Mode supplying degree l and signed order m.
    :param directions: Unit Cartesian directions, shape (n_points, 3), in
                       pulsation axes. Radius is deliberately excluded.
    :param time_exponential: Common complex temporal factor at this snapshot.
    :returns: Complex harmonic values (n_points,) and tangent gradients
              (n_points, 3), both dimensionless and in the input frame.
    """
    x, y, z = directions.T
    order = abs(mode.m)
    value = np.ones(len(x), dtype=complex)
    derivative = np.zeros((len(x), 3), dtype=complex)
    transverse = x + 1j*y
    transverse_gradient = np.array([1., 1j, 0.])
    for m in range(1, order+1):
        factor = -np.sqrt((2*m+1)/(2*m))
        derivative = factor*(transverse[:, None]*derivative
                             + value[:, None]*transverse_gradient)
        value = factor*transverse*value
    # The adjacent-degree seed starts the recurrence at fixed |m|.
    if mode.l > order:
        previous, previous_derivative = value, derivative
        factor = np.sqrt(2*order+3)
        derivative = factor*z[:, None]*derivative
        derivative[:, 2] += factor*value
        value = factor*z*value
        # Retain r**2 as a polynomial even though evaluation uses unit vectors:
        # its derivative contributes 2*n*S_(l-2,m) to the spatial gradient.
        radius_squared = np.sum(directions**2, axis=1)
        for degree in range(order+2, mode.l+1):
            a = np.sqrt((4*degree**2-1)/(degree**2-order**2))
            b = np.sqrt((2*degree+1)*((degree-1)**2-order**2)
                        / ((2*degree-3)*(degree**2-order**2)))
            next_derivative = (a*z[:, None]*derivative
                               - b*(radius_squared[:, None]*previous_derivative
                                    + 2*directions*previous[:, None]))
            next_derivative[:, 2] += a*value
            previous, value = value, a*z*value-b*radius_squared*previous
            previous_derivative, derivative = derivative, next_derivative
    # Project using the computed radial derivative (Euler: n.grad(S)=l*S).
    # This avoids leaving cancellation residuals in the tangent field at high l.
    gradient = derivative - np.sum(derivative*directions, axis=1)[:, None]*directions
    # Spatial conjugacy precedes the shared temporal factor for both outputs.
    if mode.m < 0:
        value = (-1)**order*value.conj()
        gradient = (-1)**order*gradient.conj()
    return time_exponential*value, time_exponential*gradient
