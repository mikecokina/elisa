import numpy as np

from elisa import utils, const, umpy as up
from scipy.special import sph_harm, factorial

"""
File containing functions dealing with pulsations.
"""


def phase_correction(phase):
    """
    Calculate phase correction for mode axis drift.

    :param phase: rotation phase of the star
    :return:
    """
    # return (synchronicity - 1) * phase * const.FULL_ARC if synchronicity is not np.nan else phase * const.FULL_ARC
    return phase * const.FULL_ARC


def incorporate_pulsations_to_mesh(star_container, com_x, phase, time):
    """
    adds pulsation perturbation to the mesh

    :param star_container: elisa.base.container.StarContainer;
    :param com_x: float; centre of mass of the star
    :param phase: float; rotational phase of the star
    :param time: float; time elapsed since
    :return: StarContainer;
    """
    points, points_spot = star_container.transform_points_to_spherical_coordinates(kind='points', com_x=com_x)

    tilt_phi, tilt_theta = generate_tilt_coordinates(star_container, phase)
    tilted_points, tilted_points_spot = tilt_mode_coordinates(points, points_spot, tilt_phi, tilt_theta)

    displacement = up.zeros(points.shape)
    displacement_spots = {spot_idx: up.zeros(spot.shape) for spot_idx, spot in points_spot.items()}

    for mode_index, mode in star_container.pulsations.items():
        exponent = mode.angular_frequency * time + mode.start_phase
        exponential = up.exp(complex(0, -exponent))
        rals = mode.renorm_const * sph_harm(mode.m, mode.l, tilted_points[:, 1], tilted_points[:, 2]) * exponential
        rals_spots = {spot_idx: mode.renorm_const * sph_harm(mode.m, mode.l, spotp[:, 1], spotp[:, 2]) * exponential
                      for spot_idx, spotp in tilted_points_spot.items()}

        displacement += calculate_mode_displacement(mode, points, rals)
        for spot_idx, spoints in points_spot.items():
            displacement_spots[spot_idx] += calculate_mode_displacement(mode, spoints, rals_spots[spot_idx])

    star_container.points = utils.spherical_to_cartesian(points + displacement)
    star_container.points[:, 0] += com_x

    for spot_idx, spot in star_container.spots.items():
        spot.points = utils.spherical_to_cartesian(points_spot[spot_idx] + displacement_spots[spot_idx])
        spot.points[:, 0] += com_x

    return star_container


def tilt_mode_coordinates(points, spot_points, phi, theta):
    """
    Function tilts spherical coordinates to desired position described by `phi`, `theta`.

    :param points: numpy.array;
    :param spot_points: dict;
    :param phi: float; azimuthal coordinate of the new polar axis
    :param theta: float; latitude of the new polar axis
    :return: Tuple;
    """
    if theta != 0 and phi != 0:
        tilted_phi, tilted_theta = utils.rotation_in_spherical(points[:, 1], points[:, 2], phi, theta)
        ret_points = np.column_stack((points[:, 0], tilted_phi, tilted_theta))

        ret_spot_points = dict()
        for spot_idx, spoints in spot_points.items():
            tilted_phi, tilted_theta = utils.rotation_in_spherical(spoints[:, 1], spoints[:, 2], phi, theta)
            ret_spot_points[spot_idx] = np.column_stack((spoints[:, 0], tilted_phi, tilted_theta))
        return ret_points, ret_spot_points
    else:
        return points, spot_points


def generate_tilt_coordinates(star_container, phase):
    """
    Returns tilt coordinates of pulsation modes.

    :param star_container: StarContainer;
    :param phase: float; rotational orbital phase of the star
    :return:
    """
    phi_corr = phase_correction(phase)
    # we presume that all modes have the same tilt
    phi = star_container.pulsations[0].mode_axis_phi + phi_corr
    theta = star_container.pulsations[0].mode_axis_theta
    return phi, theta


def assign_amplitudes(star_container, normalization_constant=1.0):
    """
    Assigns amplitude of radial and horizontal motion for given modes.

    :param normalization_constant: factor to adjust amplitudes, in cas of Binary system it is semi major axis, in case
                                   of single system it should stay 1.0
    :param star_container: StarContainer;
    :return:
    """
    r_polar = star_container.polar_radius * normalization_constant
    mult = const.G * star_container.mass / (r_polar)**3
    for mode_index, mode in star_container.pulsations.items():
        mode.radial_relative_amplitude = mode.amplitude / (r_polar * mode.angular_frequency)
        mode.horizontal_relative_amplitude = \
            np.sqrt(mode.l*(mode.l+1)) * mult / mode.angular_frequency**2


def calculate_radial_displacement(relative_amplitude, radii, rals):
    """
    Calculates radial displacement of surface points.

    :param rals: numpy.array;
    :param relative_amplitude: float; relative radial amplitude of the pulsation mode
                               see: PulsationMode.radial_relative_amplitude
    :param radii: numpy.array;
    :return: numpy.array
    """
    return relative_amplitude * radii * np.real(rals)


def calculate_phi_displacement(radial_amplitude, relative_amplitude, radii, thetas, m, rals):
    """
    Displacement of azimuthal coordinates.

    :param radial_amplitude: float;
    :param relative_amplitude: float; relative amplitude of horizontal displacement
    :param radii: numpy.array;
    :param thetas: numpy.array
    :param m: int;
    :param rals: numpy.array;
    :return: numpy.array;
    """
    sin_thetas = np.sin(thetas)
    sin_test = sin_thetas != 0.0
    retval = np.zeros(radii.shape)
    retval[sin_test] = \
        radial_amplitude * relative_amplitude * m * np.imag(- rals[sin_test]) / sin_thetas[sin_test]
    return retval


def calculate_theta_displacement(radial_amplitude, relative_amplitude, radii, phis, thetas, l, m):
    """
    Displacement in latitude.

    :param radial_amplitude: float;
    :param relative_amplitude: float; relative amplitude of horizontal displacement
    :param radii: numpy.array;
    :param thetas: numpy.array;
    :param phis: numpy.array;
    :param l: int;
    :param m: int;
    :return: numpy.array;
    """
    def alp_derivative(order):
        """works for m > 0"""
        derivative = \
            0.5 * np.sqrt((l + order) * (l - order + 1)) * \
            np.real(np.exp(1j*phis) * sph_harm(order-1, l, phis, thetas))
        derivative += 0 if order == l else \
            0.5 * np.sqrt((l - order) * (l + order + 1)) * \
            np.real(np.exp(-1j*phis) * sph_harm(order+1, l, phis, thetas))
        return derivative

    if m > 0:
        return radial_amplitude * relative_amplitude * alp_derivative(m)
    elif m == 0:
        return - radial_amplitude * relative_amplitude * np.real(sph_harm(1, l, phis, thetas))
    elif m < 0:
        coeff = np.power(-1, -m) * factorial(l+m) / factorial(l-m)
        return radial_amplitude * relative_amplitude * coeff * alp_derivative(-m)


def calculate_mode_displacement(mode, points, rals):
    """
    Calculates surface displacement caused by given `mode`.

    :param mode: elisa.pulse.mode.Mode;
    :param points: numpy.array;
    :param rals: numpy.array;
    :return:
    """
    radial_displacement = calculate_radial_displacement(mode.radial_relative_amplitude, points[:, 0], rals)
    phi_displacement = calculate_phi_displacement(mode.radial_relative_amplitude, mode.horizontal_relative_amplitude,
                                                  points[:, 0], points[:, 2],
                                                  mode.m, rals)
    theta_displacement = calculate_theta_displacement(mode.radial_relative_amplitude,
                                                      mode.horizontal_relative_amplitude,
                                                      points[:, 0], points[:, 1], points[:, 2],
                                                      mode.l, mode.m)
    return np.column_stack((radial_displacement, phi_displacement, theta_displacement))


def incorporate_temperature_perturbations(star_container, com_x, phase, time):
    """
    Introduces temperature perturbations to star container.

    :param star_container: elisa.base.container.StarContainer
    :param com_x: float; centre of mass
    :param phase: float;
    :param time: float;
    :return:
    """
    centres, spot_centres = star_container.transform_points_to_spherical_coordinates(kind='face_centres', com_x=com_x)

    tilt_phi, tilt_theta = generate_tilt_coordinates(star_container, phase)
    tilted_centres, tilted_spot_centres = tilt_mode_coordinates(centres, spot_centres, tilt_phi, tilt_theta)

    t_pert = up.zeros(centres.shape[0])
    t_pert_spots = {spot_idx: up.zeros(spot.shape[0]) for spot_idx, spot in spot_centres.items()}

    for mode_index, mode in star_container.pulsations.items():
        exponent = mode.angular_frequency * time + mode.start_phase + mode.temperature_perturbation_phase_shift
        exponential = up.exp(complex(0, -exponent))
        rals = mode.renorm_const * sph_harm(mode.m, mode.l, tilted_centres[:, 1], tilted_centres[:, 2]) * exponential
        rals_spots = {spot_idx: mode.renorm_const * sph_harm(mode.m, mode.l, spotp[:, 1], spotp[:, 2]) * exponential
                      for spot_idx, spotp in tilted_spot_centres.items()}

        t_pert += calculate_temperature_perturbation(mode, star_container.temperatures, rals)
        for spot_idx, t_s in t_pert_spots.items():
            t_s += calculate_temperature_perturbation(mode, star_container.spots[spot_idx].temperatures,
                                                      rals_spots[spot_idx])

    star_container.temperatures += t_pert
    for spot_idx, spot in star_container.spots.items():
        spot.temperatures += t_pert_spots[spot_idx]
    return star_container


def calculate_temperature_perturbation(mode, temperatures, rals, adiabatic_gradient=None):
    """
    Calculates temperature perturbation caused by given `mode`.

    :param mode: elisa.pulse.mode.Mode;
    :param temperatures: numpy.array;
    :param rals: numpy.array;
    :param adiabatic_gradient: Union[None, float]; if None default value from constants is used
    :return:
    """
    ad_g = const.IDEAL_ADIABATIC_GRADIENT if adiabatic_gradient is None else float(adiabatic_gradient)
    l = mode.l
    h, eps = mode.horizontal_relative_amplitude, mode.radial_relative_amplitude

    return ad_g * temperatures * (h * l * (l + 1) - 4 - (1 / h)) * eps * np.real(rals)
