import numpy as np

from . import utils as putils
from .surface import kinematics
from . import pulsations


def generate_harmonics(star_container, com_x, phase, time):
    """
    Generating spherical harmonics Y_l^m in shapes(2, n_points) and (2, n_faces) and its derivatives to be subsequently
    used for calculation of perturbed properties.

    :param star_container: elisa.base.container.StarContainer;
    :param com_x: float; centre of mass for the component
    :param phase: float; rotational/orbital phase
    :param time: float; time of the observation
    :return: elisa.base.container.StarContainer; Star container with updated harmonics
    """
    star_container.points_spherical, points_spot = \
        star_container.transform_points_to_spherical_coordinates(kind='points', com_x=com_x)

    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star_container, phase)
    tilted_points, tilted_points_spot = putils.tilt_mode_coordinates(
        star_container.points_spherical, points_spot, tilt_phi, tilt_theta
    )

    # assigning tilted points in spherical coordinates only to the first mode (the rest will share the same points)
    star_container.pulsations[0].points = tilted_points

    exponential = dict()
    norm_constant = dict()
    for mode_index, mode in star_container.pulsations.items():
        exponential[mode_index] = putils.generate_time_exponential(mode, time)

        # generating harmonics Y_m^l and Y_m+1^l for star and spot points
        harmonics = np.zeros((2, tilted_points.shape[0]), dtype=np.complex)
        harmonics[0] = pulsations.spherical_harmonics(mode, tilted_points, exponential[mode_index])
        if mode.m != mode.l:
            harmonics[1] = pulsations.spherical_harmonics(mode, tilted_points, exponential[mode_index],
                                                          order=mode.m + 1, degree=mode.l)

        # generating derivatives of spherical harmonics by phi an theta
        derivatives = np.empty((2, tilted_points.shape[0]), dtype=np.complex)
        derivatives[0] = pulsations.diff_spherical_harmonics_by_phi(mode, harmonics)
        derivatives[1] = pulsations.diff_spherical_harmonics_by_theta(mode, harmonics, tilted_points[:, 1],
                                                                      tilted_points[:, 2])

        # renormalizing horizontal amplitude to 1
        norm_constant[mode_index] = pulsations.horizontal_displacement_normalization(derivatives, harmonics)
        derivatives *= norm_constant[mode_index]

        # assignment of harmonics to mode instance variables
        mode.point_harmonics = harmonics[0]
        mode.face_harmonics = np.mean(mode.point_harmonics[star_container.faces], axis=1)

        mode.point_harmonics_derivatives = derivatives
        mode.face_harmonics_derivatives = np.mean(derivatives[:, star_container.faces], axis=1)

    if not star_container.is_flat():
        for spot_idx, spot in star_container.spots.items():
            spot.points_spherical = points_spot[spot_idx]

        # assigning tilted points in spherical coordinates only to the first mode (the rest will share the same points)
        star_container.pulsations[0].spot_points = tilted_points_spot

        for mode_index, mode in star_container.pulsations.items():
            spot_harmonics, spot_harmonics_derivatives = dict(), dict()
            for spot_idx, spoints in tilted_points_spot.items():
                # generating harmonics Y_m^l and Y_m+1^l for star and spot points
                spot_harmonics[spot_idx] = np.zeros((2, spoints.shape[0]), dtype=np.complex)
                spot_harmonics[spot_idx][0] = pulsations.spherical_harmonics(mode, spoints, exponential[mode_index])

                if mode.m != mode.l:
                    spot_harmonics[spot_idx][1] = pulsations.spherical_harmonics(
                        mode, spoints, exponential[mode_index], order=mode.m + 1, degree=mode.l)

                # generating derivatives of spherical harmonics by phi an theta
                spot_harmonics_derivatives[spot_idx] = np.zeros((2, spoints.shape[0]), dtype=np.complex)
                spot_harmonics_derivatives[spot_idx][0] = \
                    pulsations.diff_spherical_harmonics_by_phi(mode, spot_harmonics[spot_idx])
                spot_harmonics_derivatives[spot_idx][1] = \
                    pulsations.diff_spherical_harmonics_by_theta(
                        mode, spot_harmonics[spot_idx], spoints[:, 1], spoints[:, 2]
                    )

                # renormalizing horizontal amplitude to 1
                spot_harmonics_derivatives[spot_idx] *= norm_constant[mode_index]

            # assignment of harmonics to mode instance variables
            mode.spot_point_harmonics = {spot_idx: hrm[0] for spot_idx, hrm in spot_harmonics.items()}
            mode.spot_face_harmonics = {spot_idx: np.mean(spoth[star_container.spots[spot_idx].faces], axis=1)
                                        for spot_idx, spoth in mode.spot_point_harmonics.items()}

            mode.spot_point_harmonics_derivatives = spot_harmonics_derivatives
            mode.spot_face_harmonics_derivatives = {
                spot_idx: np.mean(spoth[:, star_container.spots[spot_idx].faces], axis=1)
                for spot_idx, spoth in spot_harmonics_derivatives.items()
            }

    return star_container


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

    position_perturbation(star_container, com_x, phase, update_container=True, return_perturbation=False)
    velocity_perturbation(star_container, phase, update_container=True, return_perturbation=False)
    return star_container


def position_perturbation(star, com_x, phase, update_container=False, return_perturbation=False):
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
    if return_perturbation:
        displacement = points - getattr(star, 'points')
    if update_container:
        setattr(star, 'points', points)

    spot_points = dict()
    for spot_idx, spot in star.spots.items():
        spot_points[spot_idx] = putils.derotate_surface_points(
            star.pulsations[0].spot_points[spot_idx] + tilt_displacement_spots_sph[spot_idx],
            tilt_phi, tilt_theta, com_x)

        if return_perturbation:
            spot_displacement[spot_idx] = spot_points[spot_idx] - getattr(spot, 'points')
        if update_container:
            setattr(spot, 'points', spot_points[spot_idx])

    return displacement, spot_displacement if return_perturbation else None


def velocity_perturbation(star, phase, update_container=False, return_perturbation=False):
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

    return velocity_pert, spot_velocity_pert if return_perturbation else None
