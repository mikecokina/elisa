"""Pulsation perturbation calculations for star container surface properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import const, utils
from elisa.base.surface.faces import calculate_normals, set_all_surface_centres
from elisa.base.types import COMPLEX
from elisa.pulse import pulsations
from elisa.pulse import utils as putils
from elisa.pulse.surface import kinematics

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer


def generate_harmonics(
    star_container: StarContainer,
    com_x: float,
    phase: float,
    time: float,
) -> StarContainer:
    """Generate spherical harmonics and derivatives for pulsation modes.

    Calculates spherical harmonics Y_l^m and Y_l^(m+1) at surface points
    in tilted coordinates, along with their derivatives with respect to
    azimuthal and latitudinal angles. These harmonics are used for
    calculating perturbed surface properties due to pulsations.

    :param star_container: Star container with surface mesh.
    :type star_container: elisa.base.container.StarContainer
    :param com_x: X-coordinate of component centre of mass.
    :type com_x: float
    :param phase: Rotational/orbital phase of the system.
    :type phase: float
    :param time: Time of observation.
    :type time: float
    :returns: Star container with updated harmonics on all pulsation modes.
    :rtype: elisa.base.container.StarContainer
    :raises ValueError: If star container is not flattened.
    """
    error_msg = "Pulsations can be calculated only on flattened container."
    if not star_container.is_flat():
        raise ValueError(error_msg)

    # Transform surface points to spherical coordinates
    coords_kwargs = {"kind": "points", "com_x": com_x}
    star_container.points_spherical = star_container.transform_points_to_spherical_coordinates(**coords_kwargs)

    # Calculate tilt angles for pulsation mode axis
    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star_container, phase)
    star_container.pulsations[0].tilt_phi = tilt_phi
    star_container.pulsations[0].tilt_theta = tilt_theta

    # Apply tilt rotation to points
    tilted_points = putils.tilt_mode_coordinates(
        star_container.points_spherical,
        star_container.pulsations[0].tilt_phi,
        star_container.pulsations[0].tilt_theta,
    )

    # Store tilted points in first mode (other modes share the same coordinates)
    star_container.pulsations[0].points = tilted_points

    # Calculate harmonics and derivatives for each mode
    for mode in star_container.pulsations.values():
        # Generate time-dependent exponential factor
        exponential = putils.generate_time_exponential(mode, time)

        # Calculate spherical harmonics Y_l^m and Y_l^(m+1)
        harmonics = np.zeros((2, tilted_points.shape[0]), dtype=COMPLEX)
        harmonics[0] = pulsations.spherical_harmonics(mode, tilted_points, exponential)

        if mode.m != mode.l:
            harmonics[1] = pulsations.spherical_harmonics(
                mode,
                tilted_points,
                exponential,
                order=mode.m + 1,
                degree=mode.l,
            )

        # Calculate derivatives d/d_phi and d/d_theta
        derivatives = np.empty((2, tilted_points.shape[0]), dtype=COMPLEX)
        derivatives[0] = pulsations.diff_spherical_harmonics_by_phi(mode, list(harmonics))
        derivatives[1] = pulsations.diff_spherical_harmonics_by_theta(
            mode,
            list(harmonics),
            tilted_points[:, 1],
            tilted_points[:, 2],
        )

        # Normalize horizontal amplitude to unity for non-radial modes
        if mode.l > 0:
            norm_constant = pulsations.horizontal_displacement_normalization(
                list(derivatives),
                list(harmonics),
            )
            derivatives *= norm_constant

        # Store harmonics and derivatives on mode instance
        mode.point_harmonics = harmonics[0]
        mode.point_harmonics_derivatives = derivatives

    return star_container


def incorporate_pulsations_to_model(
    star_container: StarContainer,
    com_x: float,
    scale: float = 1.0,
) -> StarContainer:
    """Apply pulsation perturbations to surface mesh and properties.

    Calculates and incorporates perturbations to surface positions,
    velocities, accelerations, and temperatures due to pulsation modes.
    Recomputes surface normals and areas after deformation.

    :param star_container: Star container with surface mesh and pulsation modes.
    :type star_container: elisa.base.container.StarContainer
    :param com_x: X-coordinate of component centre of mass.
    :type com_x: float
    :param scale: Scaling factor (semi-major axis for binary systems). Defaults to 1.0.
    :type scale: float
    :returns: Star container with updated surface properties.
    :rtype: elisa.base.container.StarContainer
    """
    # Calculate complex displacement amplitudes for all modes
    complex_displacement(star_container, scale=scale)

    # Identify and handle polar region singularities
    putils.pole_neighbours(star_container)

    # Apply perturbations to surface properties
    position_perturbation(
        star_container,
        com_x=com_x,
        update_container=True,
        return_perturbation=False,
    )
    velocity_perturbation(
        star_container,
        scale=scale,
        update_container=True,
        return_perturbation=False,
    )
    gravity_acc_perturbation(
        star_container,
        scale=scale,
        update_container=True,
        return_perturbation=False,
    )
    temp_perturbation(
        star_container,
        scale=scale,
        update_container=True,
        return_perturbation=False,
    )

    # Recalculate surface geometry properties
    set_all_surface_centres(star_container)
    normals_args = (
        star_container.points,
        star_container.faces,
        star_container.face_centres,
        com_x,
    )
    star_container.normals = calculate_normals(*normals_args)
    star_container.calculate_all_areas()

    return star_container


def complex_displacement(star: StarContainer, scale: float) -> None:
    """Calculate complex displacement vector for surface points.

    Assigns complex displacement amplitudes at surface points in tilted
    spherical coordinates. These displacements are subsequently used to
    compute kinematic quantities (position, velocity, acceleration).

    :param star: Star container with flattened surface mesh.
    :type star: elisa.base.container.StarContainer
    :param scale: Scaling factor for amplitudes.
    :type scale: float
    :raises ValueError: If star container is not flattened.
    """
    error_msg = "Pulsations can be calculated only on flattened container."
    if not star.is_flat():
        raise ValueError(error_msg)

    for mode in star.pulsations.values():
        mode.complex_displacement = kinematics.calculate_displacement_coordinates(
            mode,
            star.pulsations[0].points,
            mode.point_harmonics,
            mode.point_harmonics_derivatives,
            star.points_spherical[:, 0],
            scale=scale,
        )


def position_perturbation(
    star: StarContainer,
    com_x: float = 0.0,
    *,
    update_container: bool = True,
    return_perturbation: bool = False,
    spherical_perturbation: bool = False,
) -> NDArray | None:
    """Calculate surface mesh deformation due to pulsations.

    Computes the displacement of surface points due to pulsation-induced
    radial and horizontal motion. Can return displacements in either
    spherical or Cartesian coordinates.

    :param star: Star container with surface mesh and pulsations.
    :type star: elisa.base.container.StarContainer
    :param com_x: X-coordinate of component centre of mass. Defaults to 0.0.
    :type com_x: float
    :param update_container: If True, update star.points with perturbed positions.
                             Defaults to True.
    :type update_container: bool
    :param return_perturbation: If True, return displacement perturbation. Defaults to False.
    :type return_perturbation: bool
    :param spherical_perturbation: If True, return displacement in spherical coordinates.
                                   Defaults to False.
    :type spherical_perturbation: bool
    :returns: Displacement perturbation array if return_perturbation is True, else None.
    :rtype: np.ndarray | None
    """
    displacement = None

    # Sum displacements from all pulsation modes
    tilt_displacement_sph = np.sum(
        [
            kinematics.calculate_mode_angular_displacement(mode.complex_displacement)
            for mode in star.pulsations.values()
        ],
        axis=0,
    )

    # Derotate displacement back to original coordinate frame
    points_spherical = putils.derotate_surface_points(
        star.pulsations[0].points + tilt_displacement_sph,
        star.pulsations[0].tilt_phi,
        star.pulsations[0].tilt_theta,
    )

    points = utils.spherical_to_cartesian(points_spherical)

    if return_perturbation:
        if spherical_perturbation:
            displacement = points_spherical - star.points_spherical
            displacement[displacement[:, 1] > const.PI, 1] -= const.FULL_ARC
        elif not spherical_perturbation:
            displacement = points - utils.spherical_to_cartesian(star.points_spherical)

    if update_container:
        com = np.array([com_x, 0.0, 0.0])
        star.points = points + com[None, :]

    return displacement if return_perturbation else None


def velocity_perturbation(
    star: StarContainer,
    scale: float,
    *,
    update_container: bool = False,
    return_perturbation: bool = False,
    spherical_perturbation: bool = False,
    point_perturbations: bool = False,
) -> NDArray | None:
    """Calculate velocity perturbation on pulsating star surface.

    Computes the velocity perturbation due to pulsation-induced motion
    at surface points or face-averaged locations. Can return values in
    either spherical or Cartesian coordinates.

    :param star: Star container with surface mesh and pulsations.
    :type star: elisa.base.container.StarContainer
    :param scale: Scaling factor (semi-major axis for binary systems).
    :type scale: float
    :param update_container: If True, add perturbation to star.velocities.
                             Defaults to False.
    :type update_container: bool
    :param return_perturbation: If True, return velocity perturbation. Defaults to False.
    :type return_perturbation: bool
    :param spherical_perturbation: If True, return in spherical coordinates.
                                   Defaults to False.
    :type spherical_perturbation: bool
    :param point_perturbations: If True, return point values (not face-averaged).
                                Defaults to False.
    :type point_perturbations: bool
    :returns: Velocity perturbation array if return_perturbation is True, else None.
    :rtype: np.ndarray | None
    """
    # Sum velocity contributions from all modes
    tilt_velocity_sph = np.sum(
        [
            kinematics.calculate_mode_derivatives(
                displacement=mode.complex_displacement,
                angular_frequency=mode.angular_frequency,
            )
            for mode in star.pulsations.values()
        ],
        axis=0,
    )

    # Derotate velocity back to original frame
    velocity_pert_sph = putils.derotate_surface_displacements(
        tilt_velocity_sph,
        star.pulsations[0].points,
        star.points_spherical,
        star.pulsations[0].tilt_phi,
        star.pulsations[0].tilt_theta,
    )

    # Handle pole singularities
    velocity_pert_sph[star.pole_idx] = velocity_pert_sph[star.pole_idx_neighbour]

    # Transform to Cartesian coordinates and scale
    points_cartesian = utils.spherical_to_cartesian(star.points_spherical)
    velocity_pert = putils.transform_spherical_displacement_to_cartesian(
        velocity_pert_sph,
        points_cartesian,
        0.0,
    )
    velocity_pert *= scale

    # Average to face values
    velocity_pert_face = velocity_pert[star.faces].mean(axis=1)

    if update_container:
        star.velocities += velocity_pert_face

    if return_perturbation:
        if spherical_perturbation:
            velocity_pert_sph[:, 0] *= scale
            if point_perturbations:
                return velocity_pert_sph
            return velocity_pert_sph[star.faces].mean(axis=1)
        if point_perturbations:
            return velocity_pert
        return velocity_pert_face

    return None


def gravity_acc_perturbation(
    star: StarContainer,
    scale: float,
    *,
    update_container: bool = False,
    return_perturbation: bool = False,
    spherical_perturbation: bool = False,
    point_perturbations: bool = False,
) -> NDArray | None:
    """Calculate gravitational acceleration perturbation on pulsating star.

    Computes the acceleration perturbation due to pulsation-induced motion
    at surface points. Updates surface gravity if requested.

    :param star: Star container with surface mesh and pulsations.
    :type star: elisa.base.container.StarContainer
    :param scale: Scaling factor (semi-major axis for binary systems).
    :type scale: float
    :param update_container: If True, update star.log_g. Defaults to False.
    :type update_container: bool
    :param return_perturbation: If True, return acceleration perturbation. Defaults to False.
    :type return_perturbation: bool
    :param spherical_perturbation: If True, return in spherical coordinates.
                                   Defaults to False.
    :type spherical_perturbation: bool
    :param point_perturbations: If True, return point values (not face-averaged).
                                Defaults to False.
    :type point_perturbations: bool
    :returns: Acceleration perturbation array if return_perturbation is True, else None.
    :rtype: np.ndarray | None
    """
    # Sum acceleration contributions from all modes
    tilt_acc_sph = np.sum(
        [
            kinematics.calculate_mode_second_derivatives(
                displacement=mode.complex_displacement,
                angular_frequency=mode.angular_frequency,
            )
            for mode in star.pulsations.values()
        ],
        axis=0,
    )

    # Derotate acceleration back to original frame
    acc_pert_sph = putils.derotate_surface_displacements(
        tilt_acc_sph,
        star.pulsations[0].points,
        star.points_spherical,
        star.pulsations[0].tilt_phi,
        star.pulsations[0].tilt_theta,
    )

    # Handle pole singularities
    acc_pert_sph[star.pole_idx] = acc_pert_sph[star.pole_idx_neighbour]

    # Transform to Cartesian coordinates and scale
    points_cartesian = utils.spherical_to_cartesian(star.points_spherical)
    acc_pert = putils.transform_spherical_displacement_to_cartesian(
        acc_pert_sph,
        points_cartesian,
        0.0,
    )
    acc_pert *= scale

    # Handle pole singularities in Cartesian frame
    acc_pert[star.pole_idx] = acc_pert[star.pole_idx_neighbour]

    # Average to face values
    acc_pert_face = acc_pert[star.faces].mean(axis=1)

    if update_container:
        # Calculate equilibrium gravity
        g_eq = -np.power(10, star.log_g)[:, None] * star.normals
        # Total acceleration magnitude
        total_acc = np.linalg.norm(g_eq + acc_pert_face, axis=1)
        star.log_g = np.log10(total_acc)

    if return_perturbation:
        if spherical_perturbation:
            acc_pert_sph[:, 0] *= scale
            if point_perturbations:
                return acc_pert_sph
            return acc_pert_sph[star.faces].mean(axis=1)
        if point_perturbations:
            return acc_pert
        return acc_pert_face

    return None


def temp_perturbation(
    star: StarContainer,
    scale: float = 1.0,
    *,
    update_container: bool = False,
    return_perturbation: bool = False,
    point_perturbations: bool = False,
) -> NDArray | None:
    """Calculate temperature perturbation on pulsating star surface.

    Computes the temperature perturbation due to pulsation-induced
    adiabatic compression and dynamical heating effects.

    :param star: Star container with surface mesh and pulsations.
    :type star: elisa.base.container.StarContainer
    :param scale: Scaling factor (semi-major axis for binary systems). Defaults to 1.0.
    :type scale: float
    :param update_container: If True, add perturbation to star.temperatures.
                             Defaults to False.
    :type update_container: bool
    :param return_perturbation: If True, return temperature perturbation. Defaults to False.
    :type return_perturbation: bool
    :param point_perturbations: If True, return point values (not face-averaged).
                                Defaults to False.
    :type point_perturbations: bool
    :returns: Temperature perturbation array if return_perturbation is True, else None.
    :rtype: np.ndarray | None
    """
    # Sum temperature perturbations from all modes
    temp_pert = np.sum(
        [kinematics.calculate_temperature_pert_factor(mode, scale) for mode in star.pulsations.values()],
        axis=0,
    )

    # Average to face values and scale by stellar temperature
    temp_pert_face = temp_pert[star.faces].mean(axis=1) * star.temperatures

    if update_container:
        star.temperatures += temp_pert_face

    if return_perturbation:
        if point_perturbations:
            return temp_pert * star.t_eff
        return temp_pert_face

    return None


