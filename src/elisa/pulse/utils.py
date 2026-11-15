"""Utilities for pulsation mode calculations and transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import const, utils

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.pulse.mode import PulsationMode
    from elisa.types import Float


def phase_correction(phase: Float, synchronicity: Float) -> Float:
    """Calculate phase correction for mode axis drift.

    :param phase: Rotation phase of the star.
    :type phase: float
    :param synchronicity: Synchronicity parameter for the mode.
    :type synchronicity: float
    :returns: Phase correction value in radians.
    :rtype: float
    """
    if not np.isnan(synchronicity):
        return (synchronicity - 1) * phase * const.FULL_ARC
    return phase * const.FULL_ARC


def generate_tilt_coordinates(
    star_container: StarContainer,
    phase: Float,
) -> tuple[Float, Float]:
    """Generate tilt coordinates of pulsation modes.

    Returns the azimuth and polar angle describing the orientation
    of the pulsation mode axis in the stellar component frame. Theta is the
    polar angle from +z, not geographic latitude. Mode 0 supplies the shared
    axis and locking convention for every mode. Locking suppresses the axis
    drift correction; it does not lock the harmonic's temporal phase.

    :param star_container: Container object representing the star.
    :type star_container: StarContainer
    :param phase: Rotational/orbital phase supplied by the system wrapper.
    :type phase: float
    :returns: Tuple of (azimuthal_angle, polar_angle) in radians.
    :rtype: tuple[float, float]
    """
    # Shared mode-0 orientation is an existing convention, independent of each
    # mode's own frequency and start phase.
    if star_container.pulsations[0].tidally_locked:
        phi_corr = 0.0
    else:
        phi_corr = phase_correction(phase, star_container.synchronicity)

    phi = star_container.pulsations[0].mode_axis_phi + phi_corr
    theta = star_container.pulsations[0].mode_axis_theta
    return phi, theta


def generate_time_exponential(mode: PulsationMode, time: Float) -> complex:
    """Generate exp(-i*(omega*time+start_phase)) for the intrinsic oscillation.

    This factor is independent of axis-drift phase. Time is in seconds and the
    angular frequency in radians per second; frame rotations are applied later.

    :param mode: Pulsation mode object containing frequency and phase information.
    :type mode: PulsationMode
    :param time: Time at which to evaluate the exponential.
    :type time: float
    :returns: Complex time-dependent exponential factor.
    :rtype: complex
    """
    exponent = mode.angular_frequency * time + mode.start_phase
    return np.exp(complex(0, -exponent))


def generate_phase_shift(shift: Float) -> complex:
    """Generate a phase shift factor for complex displacement transformation.

    :param shift: Angular phase shift in radians.
    :type shift: float
    :returns: Complex phase shift factor that can be applied to displacements.
    :rtype: complex
    """
    return np.exp(complex(0, -shift))


def tilt_mode_coordinates(
    points: NDArray,
    phi: Float,
    theta: Float,
) -> NDArray:
    """Tilt spherical coordinates to desired position described by phi and theta.

    This general point-coordinate utility expresses points in the tilted frame.
    It is not the transformation of a displacement vector or an angular rate;
    Cartesian pulsation vectors use mode_rotation instead.

    :param points: Array of spherical coordinates with shape (n_points, 3)
                   where columns are [r, azimuth, polar angle from +z].
    :type points: NDArray
    :param phi: Azimuthal coordinate of the new polar axis in radians.
    :type phi: float
    :param theta: Polar angle of the new axis from stellar +z, in radians.
    :type theta: float
    :returns: Tilted spherical coordinates with same shape as input.
    :rtype: NDArray
    """
    if theta != 0 or phi != 0:
        tilted_phi, tilted_theta = utils.rotation_in_spherical(
            points[:, 1],
            points[:, 2],
            phi,
            theta,
        )
        return np.column_stack((points[:, 0], tilted_phi, tilted_theta))
    return points


def derotate_surface_points(
    points_to_derotate: NDArray,
    phi: Float,
    theta: Float,
) -> NDArray:
    """Transform surface points from tilted to base coordinate system.

    This is the inverse point-coordinate operation of tilt_mode_coordinates.
    It is available for coordinate diagnostics; physical pulsation vectors are
    transformed directly by the Cartesian mode rotation.

    :param points_to_derotate: Surface points in tilted spherical coordinates,
                               shape (n_points, 3) with columns [r, azimuth, polar angle].
    :type points_to_derotate: NDArray
    :param phi: Azimuthal tilt of the input coordinate system in radians.
    :type phi: float
    :param theta: Polar tilt of the input coordinate system from stellar +z, in radians.
    :type theta: float
    :returns: Derotated points in spherical coordinates aligned with rotation axis.
    :rtype: NDArray
    """
    if theta != 0 or phi != 0:
        derot_phi, derot_theta = utils.derotation_in_spherical(
            points_to_derotate[:, 1],
            points_to_derotate[:, 2],
            phi,
            theta,
        )
        return np.column_stack((points_to_derotate[:, 0], derot_phi, derot_theta))
    return points_to_derotate


def mode_rotation(phi: float, theta: float) -> NDArray:
    """Return Q=R_z(phi)*R_y(theta), from mode axes to component Cartesian axes.

    Angles are in radians; theta is the axis polar angle measured from stellar
    +z. Q's third column is the pulsation-axis direction in component axes.
    Arrays hold row vectors: use v_mode @ Q.T to return to component axes,
    and n_component @ Q to express equilibrium directions in mode axes.
    Rotating vectors preserves their physical units and avoids treating finite
    angular coordinate differences as linear vector components.
    """
    cp, sp, ct, st = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta)
    return np.array([[cp*ct, -sp, cp*st], [sp*ct, cp, sp*st], [-st, 0., ct]])


def spherical_vector_components(vectors: NDArray, points: NDArray) -> NDArray:
    """Project Cartesian vectors onto orthonormal (r, phi, theta) directions.

    All components retain the vector's physical units. Tangent components at a
    pole depend on the chart meridian, while the vector and tangent norm do not.
    These are not spherical coordinate increments or angular rates. In the
    returned (r, phi, theta) order, theta points toward increasing polar angle
    (away from +z), and phi toward increasing azimuth. Both inputs have shape
    (n_points, 3); points contains (radius, azimuth, polar angle) in the same
    component frame, with angles in radians. Radius does not scale a projection.
    """
    phi, theta = points[:, 1], points[:, 2]
    cp, sp, ct, st = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta)
    radial = np.column_stack((st*cp, st*sp, ct))
    azimuth = np.column_stack((-sp, cp, np.zeros_like(phi)))
    polar = np.column_stack((ct*cp, ct*sp, -st))
    return np.column_stack([np.sum(vectors*basis, axis=1) for basis in (radial, azimuth, polar)])
