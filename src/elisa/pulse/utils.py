import numpy as np
from .. import const, utils


def phase_correction(phase):
    """
    Calculate phase correction for mode axis drift.

    :param phase: float; rotation phase of the star
    :return: float;
    """
    # return (synchronicity - 1) * phase * const.FULL_ARC if synchronicity is not np.nan else phase * const.FULL_ARC
    return phase * const.FULL_ARC


def generate_tilt_coordinates(star_container, phase):
    """
    Returns tilt coordinates of pulsation modes.

    :param star_container: StarContainer;
    :param phase: float; rotational orbital phase of the star
    :return: Tuple[float, float];
    """
    phi_corr = phase_correction(phase)
    # we presume that all modes have the same tilt
    phi = star_container.pulsations[0].mode_axis_phi + phi_corr
    theta = star_container.pulsations[0].mode_axis_theta
    return phi, theta


def generate_time_exponential(mode, time):
    """
    Returns time dependent exponential used in `generate_spherical_harmonics`.

    :param mode: PulsationMode; PulsationMode; mode used to generate sph. harmonics
    :param time: float; time at which evaluate spherical harmonics
    :return: float; complex time dependent exponential
    """
    exponent = mode.angular_frequency * time + mode.start_phase
    return np.exp(complex(0, -exponent))


def tilt_mode_coordinates(points, spot_points, phi, theta):
    """
    Function tilts spherical coordinates to desired position described by `phi`, `theta`.

    :param points: numpy.array;
    :param spot_points: Dict;
    :param phi: float; azimuthal coordinate of the new polar axis
    :param theta: float; latitude of the new polar axis
    :return: Tuple;
    """
    if theta != 0 or phi != 0:
        tilted_phi, tilted_theta = utils.rotation_in_spherical(points[:, 1], points[:, 2], phi, theta)
        ret_points = np.column_stack((points[:, 0], tilted_phi, tilted_theta))

        ret_spot_points = dict()
        for spot_idx, spoints in spot_points.items():
            tilted_phi, tilted_theta = utils.rotation_in_spherical(spoints[:, 1], spoints[:, 2], phi, theta)
            ret_spot_points[spot_idx] = np.column_stack((spoints[:, 0], tilted_phi, tilted_theta))
        return ret_points, ret_spot_points
    else:
        return points, spot_points


def derotate_surface_points(points_to_derotate, phi, theta, com_x):
    """
    Derotating surface points  into the base coordinate system after surface displacement for misalligned mode is
    calculated.

    :param points_to_derotate: numpy.array; surface points in tilted spherical coordinates
    :param phi: float; azimuthal tilt of the input coordinate system
    :param theta: float; latitudinal tilt of the input coordinate system
    :param com_x: float;
    :return: numpy.array;
    """
    derot_phi, derot_theta = \
        utils.derotation_in_spherical(points_to_derotate[:, 1],
                                      points_to_derotate[:, 2],
                                      phi, theta)
    derot_points = np.column_stack((points_to_derotate[:, 0], derot_phi, derot_theta))
    points = utils.spherical_to_cartesian(derot_points)
    points[:, 0] += com_x

    return points


def derotate_surface_displacements(velocity, tilted_points, points, axis_phi, axis_theta):
    """
    Transform spherical perturbations from tilted coordinates to system alligned with rotation axis.

    :param velocity: numpy.array; velocity in tilted spherical coordinates
    :param tilted_points: numpy.array; spherical coordinates of surface points (unperturbed) in tilted coordinates
    :param points: numpy.array; unperturbed surface points in spherical coordinates
    :param axis_theta: float;
    :param axis_phi: float;
    :return: numpy.array;
    """
    pert_phis, pert_thetas = utils.derotation_in_spherical(
        phi=tilted_points[:, 1] + velocity[:, 1],
        theta=tilted_points[:, 2] + velocity[:, 2],
        phi_rotation=axis_phi,
        theta_rotation=axis_theta
    )

    crit_amplitude = const.PI
    d_phi = pert_phis - points[:, 1]
    d_phi[d_phi > crit_amplitude] -= const.FULL_ARC
    d_theta = (pert_thetas - points[:, 2])

    return np.column_stack((velocity[:, 0], d_phi, d_theta))


def transform_spherical_displacement_to_cartesian(sph_displacement, surf_points, com_x):
    """
    Transforms displacement d_r, d_phi, d_theta into spherical cartesian displacement d_x, d_y, d_z

    :param sph_displacement: numpy.array; [[d_r1, d_phi1, d_theta1], ...]
    :param surf_points: numpy.array; surface points in equilibrium in cartesian coordinates (from container)
    :param com_x: numpy.float; x coordinate of centre of mass, assuming com = [com_x, 0, 0]
    :return: numpy.array; [[d_x1, d_y1, d_z1], ...]
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
    matrix[:, 0, 0], matrix[:, 1, 0], matrix[:, 2, 0] = points[:, 0] / r, -points[:, 1], points[:, 0] * z_rxy
    matrix[:, 0, 1], matrix[:, 1, 1], matrix[:, 2, 1] = points[:, 1] / r,  points[:, 0], points[:, 1] * z_rxy
    matrix[:, 0, 2], matrix[:, 1, 2], matrix[:, 2, 2] = points[:, 2] / r, 0.0, -r_xy

    return np.sum(matrix * sph_displacement[:, :, None], axis=1)
