"""Utilities for pulsation mode calculations and transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import const, utils
from elisa.base.types import INT

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

    Returns the azimuthal and latitudinal coordinates describing the orientation
    of the pulsation mode axis in the stellar frame. For tidally locked modes,
    the correction term is zero.

    :param star_container: Container object representing the star.
    :type star_container: StarContainer
    :param phase: Rotational orbital phase of the star (0 to 1).
    :type phase: float
    :returns: Tuple of (azimuthal_angle, latitudinal_angle) in radians.
    :rtype: tuple[float, float]
    """
    # Presume all modes have the same tilt
    if star_container.pulsations[0].tidally_locked:
        phi_corr = 0.0
    else:
        phi_corr = phase_correction(phase, star_container.synchronicity)

    phi = star_container.pulsations[0].mode_axis_phi + phi_corr
    theta = star_container.pulsations[0].mode_axis_theta
    return phi, theta


def generate_time_exponential(mode: PulsationMode, time: Float) -> complex:
    """Generate time-dependent exponential for spherical harmonics calculation.

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

    Applies a rotation transformation to spherical coordinates (r, phi, theta)
    to align with a new coordinate system tilted by the specified angles.

    :param points: Array of spherical coordinates with shape (n_points, 3)
                   where columns are [r, azimuth, latitude].
    :type points: NDArray
    :param phi: Azimuthal coordinate of the new polar axis in radians.
    :type phi: float
    :param theta: Latitudinal coordinate of the new polar axis in radians.
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

    Derotates surface points into the base coordinate system after surface
    displacement for misaligned modes has been calculated.

    :param points_to_derotate: Surface points in tilted spherical coordinates,
                               shape (n_points, 3) with columns [r, azimuth, latitude].
    :type points_to_derotate: NDArray
    :param phi: Azimuthal tilt of the input coordinate system in radians.
    :type phi: float
    :param theta: Latitudinal tilt of the input coordinate system in radians.
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


def derotate_surface_displacements(
    velocity: NDArray,
    tilted_points: NDArray,
    points: NDArray,
    axis_phi: Float,
    axis_theta: Float,
) -> NDArray:
    """Transform spherical perturbations from tilted to rotation-aligned coordinates.

    Transforms velocity/displacement components from tilted coordinate system
    to system aligned with the stellar rotation axis.

    :param velocity: Velocity in tilted spherical coordinates,
                     shape (n_points, 3) with columns [v_r, v_phi, v_theta].
    :type velocity: NDArray
    :param tilted_points: Spherical coordinates of surface points (unperturbed)
                          in tilted coordinates, shape (n_points, 3).
    :type tilted_points: NDArray
    :param points: Unperturbed surface points in spherical coordinates,
                   shape (n_points, 3).
    :type points: NDArray
    :param axis_phi: Azimuthal rotation angle in radians.
    :type axis_phi: float
    :param axis_theta: Latitudinal rotation angle in radians.
    :type axis_theta: float
    :returns: Perturbations in spherical coordinates aligned with rotation axis.
    :rtype: NDArray
    """
    if axis_theta != 0 or axis_phi != 0:
        pert_phis, pert_thetas = utils.derotation_in_spherical(
            phi=tilted_points[:, 1] + velocity[:, 1],
            theta=tilted_points[:, 2] + velocity[:, 2],
            phi_rotation=axis_phi,
            theta_rotation=axis_theta,
        )

        crit_amplitude = const.PI
        d_phi = pert_phis - points[:, 1]
        d_phi[d_phi > crit_amplitude] -= const.FULL_ARC

        d_theta = pert_thetas - points[:, 2]

        return np.column_stack((velocity[:, 0], d_phi, d_theta))
    return velocity


def transform_spherical_displacement_to_cartesian(
    sph_displacement: NDArray,
    surf_points: NDArray,
    com_x: Float,
) -> NDArray:
    """Transform spherical to Cartesian displacement components.

    Transforms displacement components (d_r, d_phi, d_theta) from spherical
    coordinates to Cartesian coordinates (d_x, d_y, d_z).

    :param sph_displacement: Displacement in spherical coordinates,
                             shape (n_points, 3) with columns [d_r, d_phi, d_theta].
    :type sph_displacement: NDArray
    :param surf_points: Surface points in equilibrium in Cartesian coordinates
                        from container, shape (n_points, 3).
    :type surf_points: NDArray
    :param com_x: X coordinate of center of mass, assuming com = [com_x, 0, 0].
    :type com_x: float
    :returns: Cartesian displacement with shape (n_points, 3) and columns [d_x, d_y, d_z].
    :rtype: NDArray
    """
    points = surf_points - np.array([com_x, 0, 0])[None, :]
    r_xy2 = np.sum(np.power(points[:, :-1], 2), axis=1)
    r_xy = np.sqrt(r_xy2)
    r = np.sqrt(r_xy2 + np.power(points[:, 2], 2))

    # z/(x^2+y^2)^0.5
    z_rxy = np.zeros(r_xy.shape)
    non_zero = r_xy != 0
    z_rxy[non_zero] = points[non_zero, 2] / r_xy[non_zero]

    matrix = np.empty((r.shape[0], 3, 3))
    matrix[:, 0, 0], matrix[:, 1, 0], matrix[:, 2, 0] = (
        points[:, 0] / r,
        -points[:, 1],
        points[:, 0] * z_rxy,
    )
    matrix[:, 0, 1], matrix[:, 1, 1], matrix[:, 2, 1] = (
        points[:, 1] / r,
        points[:, 0],
        points[:, 1] * z_rxy,
    )
    matrix[:, 0, 2], matrix[:, 1, 2], matrix[:, 2, 2] = (
        points[:, 2] / r,
        0.0,
        -r_xy,
    )

    return np.sum(matrix * sph_displacement[:, :, None], axis=1)


def horizontal_component(
    displacement: NDArray,
    points: NDArray,
    *,
    treat_poles: bool = False,
) -> NDArray:
    """Calculate the absolute value of horizontal (non-radial) displacement.

    Computes the horizontal distance as the combined effect of azimuthal and
    latitudinal displacement components. The horizontal component is the
    displacement perpendicular to the radial direction.

    :param displacement: Spherical displacement components,
                         shape (n_points, 3) with columns [d_r, d_phi, d_theta].
    :type displacement: NDArray
    :param points: Spherical coordinates of points,
                   shape (n_points, 3) with columns [r, azimuth, latitude].
    :type points: NDArray
    :param treat_poles: If True, remove invalid/extreme values for faces in contact
                        with poles by capping them to the mean value. Defaults to False.
    :type treat_poles: bool
    :returns: Array of horizontal displacement distances with shape (n_points,).
    :rtype: NDArray
    """
    # Distance in azimuthal direction (lambda)
    d_lambda = points[:, 0] * np.sin(points[:, 2]) * displacement[:, 1]
    # Distance in latitudinal direction (nu)
    d_nu = points[:, 0] * displacement[:, 2]

    distance = np.sqrt(np.power(d_lambda, 2) + np.power(d_nu, 2))
    if treat_poles:
        distance[distance >= 10 * distance.mean()] = distance.mean()

    return distance


def pole_neighbours(star: StarContainer) -> None:
    """Find indices of both poles and their neighboring surface points.

    Identifies the location of the north and south poles on the stellar surface
    mesh and finds the nearest neighboring point to each pole. This information
    is stored as attributes on the star container.

    :param star: Container object representing the star with surface mesh information.
    :type star: StarContainer
    """
    poles = np.array(
        [
            star.points_spherical[:, 2].argmax(),
            star.points_spherical[:, 2].argmin(),
        ],
        dtype=INT,
    )
    neighbour_idx = np.empty(2, dtype=INT)
    for ii, pole in enumerate(poles):
        in_face = (pole == star.faces).any(axis=1)
        polar_face = (star.faces[in_face])[0]
        neighbour_idx[ii] = (polar_face[pole != polar_face])[0]

    star.pole_idx = poles
    star.pole_idx_neighbour = neighbour_idx
