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
    :param spot_points: dict;
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
