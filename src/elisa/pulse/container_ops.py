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
    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star_container, phase)

    # initializing cumulative variables for displacement, velocity,
    tilt_displacement_sph = np.zeros(tilted_points.shape, dtype=np.float64)
    tilt_displacement_spots_sph = {spot_idx: np.zeros(spot.shape, dtype=np.float64)
                                   for spot_idx, spot in tilted_points_spot.items()}
    tilt_velocity_sph = np.zeros(tilted_points.shape, dtype=np.float64)
    tilt_velocity_spots_sph = {spot_idx: np.zeros(spot.shape, dtype=np.float64)
                               for spot_idx, spot in tilted_points_spot.items()}

    # calculating kinematics quantities
    for mode_index, mode in star_container.pulsations.items():
        mode.complex_displacement = kinematics.calculate_displacement_coordinates(
            mode, tilted_points, mode.point_harmonics, mode.point_harmonics_derivatives, scale=scale
        )
        tilt_displacement_sph += kinematics.calculate_mode_angular_displacement(mode.complex_displacement)
        tilt_velocity_sph += kinematics.calculate_mode_angular_derivatives(
            displacement=mode.complex_displacement, angular_frequency=mode.angular_frequency
        )

        for spot_idx, spoints in tilted_points_spot.items():
            spot_complex_displacement = kinematics.calculate_displacement_coordinates(
                mode, spoints, mode.spot_point_harmonics[spot_idx], mode.spot_point_harmonics_derivatives[spot_idx],
                scale=scale
            )
            tilt_displacement_spots_sph[spot_idx] += kinematics.calculate_mode_angular_displacement(
                mode.spot_complex_displacement
            )
            tilt_velocity_spots_sph[spot_idx] += kinematics.calculate_mode_angular_derivatives(
                displacement=spot_complex_displacement, angular_frequency=mode.angular_frequency
            )

    # velocity
    incorporate_velocity(tilt_velocity_sph, tilted_points, star_container, tilt_phi, tilt_theta, com_x)

    # displacement
    star_container.points = putils.derotate_surface_points(
        tilted_points + tilt_displacement_sph,
        tilt_phi, tilt_theta, com_x
    )

    for spot_idx, spot in star_container.spots.items():
        # velocity
        incorporate_velocity(
            tilt_velocity_spots_sph[spot_idx], tilted_points_spot[spot_idx], spot, tilt_phi, tilt_theta, com_x
        )

        # displacement
        setattr(spot, 'points',
                putils.derotate_surface_points(tilted_points_spot[spot_idx] + tilt_displacement_spots_sph[spot_idx],
                                               tilt_phi, tilt_theta, com_x))

    return star_container


def calculate_displacement(star, com_x, phase, update_container=False, return_displacement=False):
    # initializing cumulative variables for displacement
    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star, phase)

    tilt_displacement_sph = np.zeros(star.pulsations[0].points.shape, dtype=np.float64)
    tilt_displacement_spots_sph = {spot_idx: np.zeros(star.pulsations[0].spot_points.shape, dtype=np.float64)
                                   for spot_idx, spot in star.pulsations[0].spot_points.items()}

    for mode_index, mode in star.pulsations.items():
        tilt_displacement_sph += kinematics.calculate_mode_angular_displacement(mode.complex_displacement)

        for spot_idx, spoints in star.pulsations[0].spot_points.items():
            tilt_displacement_spots_sph[spot_idx] += kinematics.calculate_mode_angular_displacement(
                mode.spot_complex_displacement
            )

    points = putils.derotate_surface_points(
        star.pulsations[0].points + tilt_displacement_sph,
        tilt_phi, tilt_theta, com_x
    )
    if update_container:
        setattr(star, 'points', points)

    spot_points = dict()
    for spot_idx, spot in star.spots.items():
        spoints = putils.derotate_surface_points(
            star.pulsations[0].spot_points[spot_idx] + tilt_displacement_spots_sph[spot_idx],
            tilt_phi, tilt_theta, com_x)

        if update_container:
            setattr(spot, 'points', spoints)


def incorporate_velocity(tilt_velocity_sph, tilted_points, container, axis_phi, axis_theta, com_x):
    transformed_velocity = putils.derotate_surface_displacements(
        tilt_velocity_sph, tilted_points, container.points_spherical, axis_phi, axis_theta
    )

    vel = putils.transform_spherical_displacement_to_cartesian(transformed_velocity, container.points, com_x)
    vel = vel[container.faces].mean(axis=1)
    container.velocities += vel
