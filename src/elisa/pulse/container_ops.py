import numpy as np

from . import utils as putils
from . surface import kinematics


def incorporate_pulsations_to_model(star_container, com_x, phase, scale=1.0):
    """
    Function adds perturbation to the surface mesh due to pulsations.

    :param phase: numpy.float; (0, 1)
    :param star_container: base.container.StarContainer;
    :param com_x: float;
    :param scale: numpy.float; scale of the perturbations
    :return: base.container.StarContainer;
    """
    tilted_points, tilted_points_spot = star_container.pulsations[0].points, star_container.pulsations[0].spot_points
    # angular coordinate of pulsation axis

    # calculating kinematics quantities
    for mode_index, mode in star_container.pulsations.items():
        mode.complex_displacement = kinematics.calculate_displacement_coordinates(
            mode, tilted_points, mode.point_harmonics, mode.point_harmonics_derivatives, scale=scale
        )
        for spot_idx, spoints in tilted_points_spot.items():
            mode.spot_complex_displacement[spot_idx] = kinematics.calculate_displacement_coordinates(
                mode, spoints, mode.spot_point_harmonics[spot_idx], mode.spot_point_harmonics_derivatives[spot_idx],
                scale=scale
            )

    position_perturbation(star_container, com_x, phase, update_container=True, return_displacement=False)
    velocity_perturbation(star_container, phase, update_container=True, return_displacement=False)
    return star_container


def position_perturbation(star, com_x, phase, update_container=False, return_displacement=False):
    # initializing cumulative variables for displacement
    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star, phase)
    displacement, spot_displacement = None, dict()

    tilt_displacement_sph = np.zeros(star.pulsations[0].points.shape, dtype=np.float64)
    tilt_displacement_spots_sph = {spot_idx: np.zeros(spot.shape, dtype=np.float64)
                                   for spot_idx, spot in star.pulsations[0].spot_points.items()}

    for mode_index, mode in star.pulsations.items():
        tilt_displacement_sph += kinematics.calculate_mode_angular_displacement(mode.complex_displacement)

        for spot_idx, spoints in star.pulsations[0].spot_points.items():
            tilt_displacement_spots_sph[spot_idx] += kinematics.calculate_mode_angular_displacement(
                mode.spot_complex_displacement[spot_idx]
            )

    points = putils.derotate_surface_points(
        star.pulsations[0].points + tilt_displacement_sph,
        tilt_phi, tilt_theta, com_x
    )
    if return_displacement:
        displacement = points - getattr(star, 'points')
    if update_container:
        setattr(star, 'points', points)

    spot_points = dict()
    for spot_idx, spot in star.spots.items():
        spot_points[spot_idx] = putils.derotate_surface_points(
            star.pulsations[0].spot_points[spot_idx] + tilt_displacement_spots_sph[spot_idx],
            tilt_phi, tilt_theta, com_x)

        if return_displacement:
            spot_displacement[spot_idx] = spot_points[spot_idx] - getattr(spot, 'points')
        if update_container:
            setattr(spot, 'points', spot_points[spot_idx])

    return displacement, spot_displacement if return_displacement else None


def velocity_perturbation(star, phase, update_container=False, return_displacement=False):
    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star, phase)
    spot_velocity_pert = dict()

    tilt_velocity_sph = np.zeros(star.pulsations[0].points.shape, dtype=np.float64)
    tilt_velocity_spots_sph = {spot_idx: np.zeros(spot.shape, dtype=np.float64)
                               for spot_idx, spot in star.pulsations[0].spot_points.items()}

    # calculating kinematics quantities
    for mode_index, mode in star.pulsations.items():
        tilt_velocity_sph += kinematics.calculate_mode_angular_derivatives(
            displacement=mode.complex_displacement, angular_frequency=mode.angular_frequency
        )
        for spot_idx, spoints in star.pulsations[0].spot_points.items():
            tilt_velocity_spots_sph[spot_idx] += kinematics.calculate_mode_angular_derivatives(
                displacement=mode.spot_complex_displacement[spot_idx], angular_frequency=mode.angular_frequency
            )

    velocity_pert = putils.derotate_surface_displacements(
        tilt_velocity_sph, star.pulsations[0].points, star.points_spherical, tilt_phi, tilt_theta
    )
    velocity_pert = velocity_pert[star.faces].mean(axis=1)

    if update_container:
        star.velocities += velocity_pert

    for spot_idx, spot in star.spots.items():
        spot_velocity_pert[spot_idx] = putils.derotate_surface_displacements(
            tilt_velocity_spots_sph[spot_idx], star.pulsations[0].spot_points[spot_idx],
            star.spots[spot_idx].points_spherical, tilt_phi, tilt_theta
        )
        spot_velocity_pert[spot_idx] = spot_velocity_pert[spot_idx][spot.faces].mean(axis=1)

        if update_container:
            spot.velocities += spot_velocity_pert[spot_idx]

    return velocity_pert, spot_velocity_pert if return_displacement else None
