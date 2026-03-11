"""Pulsation surface displacement, velocity, and acceleration calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import settings
from elisa.pulse.utils import generate_phase_shift

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.pulse.mode import PulsationMode


def calculate_horizontal_displacements(
    mode: PulsationMode,
    thetas: NDArray,
    harmonics_derivatives: NDArray,
    radius: float,
    scale: float,
) -> tuple[NDArray, NDArray]:
    """Calculate angular horizontal components of displacement.

    Computes the azimuthal (phi) and latitudinal (theta) components of
    horizontal displacement due to pulsation mode. The horizontal
    displacement is calculated from the derivatives of spherical harmonics
    and normalized to the horizontal amplitude.

    :param mode: Pulsation mode object with amplitude and degree information.
    :type mode: elisa.pulse.mode.PulsationMode
    :param thetas: Latitudinal angles (theta) in spherical coordinates.
    :type thetas: NDArray
    :param harmonics_derivatives: Spherical harmonic derivatives [dY/dphi, dY/dtheta].
    :type harmonics_derivatives: NDArray
    :param radius: Equivalent radius of the stellar surface.
    :type radius: float
    :param scale: Scaling factor for the system (semi-major axis for binary systems).
    :type scale: float
    :returns: Tuple of azimuthal and latitudinal displacement components (both complex).
    :rtype: tuple[NDArray, NDArray]
    """
    if mode.l == 0:
        return np.zeros(thetas.shape[0], dtype=np.complex128), np.zeros(
            thetas.shape[0], dtype=np.complex128,
        )

    # Calculate angular distances along phi and theta coordinates
    sin_theta = np.sin(thetas)
    theta_test = sin_theta != 0

    # Lambda: distance along phi coordinate (azimuthal direction)
    # Nu: distance along theta coordinate (latitudinal direction)
    phi_displacement = np.zeros(thetas.shape, dtype=np.complex128)
    phi_displacement[theta_test] = harmonics_derivatives[0, theta_test] / np.power(
        sin_theta[theta_test], 2,
    )

    d_lambda = radius * sin_theta * np.abs(phi_displacement)
    theta_amp = np.abs(harmonics_derivatives[1])
    d_nu = radius * theta_amp

    # Calculate normalization factor to match desired horizontal amplitude
    dr = np.sqrt(np.mean(np.power(d_lambda, 2) + np.power(d_nu, 2)))
    corr_factor = mode.horizontal_amplitude / (dr * scale)

    # Apply normalization and return displacements
    phi_retval = np.zeros(thetas.shape, dtype=np.complex128)
    phi_retval[theta_test] = corr_factor * phi_displacement[theta_test]
    return phi_retval, corr_factor * harmonics_derivatives[1]


def calculate_displacement_coordinates(
    mode: PulsationMode,
    points: NDArray,
    harmonics: NDArray,
    harmonics_derivatives: NDArray,
    radius: float,
    scale: float = 1.0,
) -> NDArray:
    """Calculate complete surface displacement caused by pulsation mode.

    Computes the radial and horizontal (azimuthal and latitudinal) components
    of surface displacement due to a single pulsation mode in complex form.
    The displacement follows the uniform pulsation model.

    :param mode: Pulsation mode object with amplitude and harmonic information.
    :type mode: elisa.pulse.mode.PulsationMode
    :param points: Surface points in spherical coordinates (r, phi, theta).
    :type points: NDArray
    :param harmonics: Spherical harmonic values Y_l^m at surface points.
    :type harmonics: NDArray
    :param harmonics_derivatives: Derivatives of harmonics [dY/dphi, dY/dtheta].
    :type harmonics_derivatives: NDArray
    :param radius: Equivalent radius of the stellar surface.
    :type radius: float
    :param scale: Scaling factor for the system. Defaults to 1.0.
    :type scale: float
    :returns: Complex displacement vector with shape (n_points, 3) containing
              [radial, azimuthal, latitudinal] components.
    :rtype: NDArray
    :raises NotImplementedError: If PULSATION_MODEL setting is not 'uniform'.
    """
    if settings.PULSATION_MODEL == "uniform":
        radial_displacement = calculate_radial_displacement(mode, harmonics) / scale
        phi_displacement, theta_displacement = calculate_horizontal_displacements(
            mode, points[:, 2], harmonics_derivatives, radius, scale,
        )

        return np.column_stack((radial_displacement, phi_displacement, theta_displacement))

    error_msg = (
        f"Pulsation model: {settings.PULSATION_MODEL} is not implemented."
    )
    raise NotImplementedError(error_msg)


def calculate_mode_angular_displacement(displacement: NDArray) -> NDArray:
    """Extract angular displacement from complex displacement vector.

    Extracts the real part of the complex displacement, which represents
    the actual physical angular displacement on the stellar surface.

    :param displacement: Complex displacement vector or array.
    :type displacement: NDArray
    :returns: Real part of displacement (actual surface deformation).
    :rtype: NDArray
    """
    return np.real(displacement)


def calculate_radial_displacement(mode: PulsationMode, harmonics: NDArray) -> NDArray:
    """Calculate radial displacement of surface points.

    Computes the radial component of pulsation-induced surface displacement
    by multiplying the radial amplitude by the spherical harmonic values.

    :param mode: Pulsation mode object with radial amplitude.
    :type mode: elisa.pulse.mode.PulsationMode
    :param harmonics: Spherical harmonic values Y_l^m at surface points.
    :type harmonics: NDArray
    :returns: Complex radial displacement at each surface point.
    :rtype: NDArray
    """
    return mode.radial_amplitude * harmonics


def calculate_mode_derivatives(
    displacement: NDArray,
    angular_frequency: float,
) -> NDArray:
    """Calculate velocity from complex displacement.

    Computes the time derivative of displacement, which gives the pulsation-induced velocity.
    Velocity is the imaginary part of the complex displacement
    multiplied by angular frequency.

    :param displacement: Complex displacement vector.
    :type displacement: NDArray
    :param angular_frequency: Angular frequency of pulsation (omega = 2*pi*frequency).
    :type angular_frequency: float
    :returns: Velocity perturbation at each point.
    :rtype: NDArray
    """
    return angular_frequency * np.imag(displacement)


def calculate_mode_second_derivatives(
    displacement: NDArray,
    angular_frequency: float,
) -> NDArray:
    """Calculate acceleration from complex displacement.

    Computes the second time derivative of displacement, which gives the
    pulsation-induced acceleration. Acceleration is the negative real part
    of the complex displacement multiplied by angular frequency squared.

    :param displacement: Complex displacement vector.
    :type displacement: NDArray
    :param angular_frequency: Angular frequency of pulsation (omega = 2*pi*frequency).
    :type angular_frequency: float
    :returns: Acceleration perturbation at each point.
    :rtype: NDArray
    """
    return -(angular_frequency**2) * np.real(displacement)


def calculate_temperature_pert_factor(
    mode: PulsationMode,
    scale: float,
) -> NDArray:
    """Calculate temperature perturbation factor for pulsating star surface.

    Computes the temperature perturbation following Townsend (2003) treatment,
    which relates temperature changes to radial displacement and accounts for
    phase shift between geometric and temperature perturbations.

    :param mode: Pulsation mode with displacement and temperature data.
    :type mode: elisa.pulse.mode.PulsationMode
    :param scale: Scaling factor for the system (semi-major axis for binary systems).
    :type scale: float
    :returns: Temperature perturbation factor (delta_T = factor * T_eff).
    :rtype: NDArray
    """
    # Apply phase shift between geometry and temperature perturbations
    hrm_shift = np.real(
        generate_phase_shift(mode.temperature_perturbation_phase_shift)
        * mode.complex_displacement[:, 0],
    )

    # Calculate temperature perturbation following Townsend (2003)
    return (
        mode.temperature_amplitude_factor
        * hrm_shift
        * scale
        / mode.radial_amplitude
    )
