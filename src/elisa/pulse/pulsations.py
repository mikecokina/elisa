import numpy as np

from elisa import utils, const, units, umpy as up

from scipy.special import sph_harm, factorial
from copy import copy

"""file containing functions dealing with pulsations"""


def set_ralp(star_container, phase, com_x=None):
    """
    Function calculates and sets time independent, dimensionless part of the pulsation modes. They are calculated as
    renormalized associated Legendre polynomials (rALP). This function needs to be evaluated only once except for the
    case of assynchronously rotating component and missaligned mode (see `set_misaligned_rals()`).

    :param com_x: float - centre of mass
    :param star_container: StarContainer;
    :return:
    """
    centres_cartesian = copy(star_container.face_centres)
    centres_spot_cartesian = {spot_idx: spot.face_centres for spot_idx, spot in star_container.spots.items()}

    # this is here to calculate surface centres in ref frame of the given component
    if com_x is not None:
        centres_cartesian[:, 0] -= com_x
        for spot_index, spot in star_container.spots.items():
            centres_spot_cartesian[spot_index][:, 0] -= com_x

    # conversion to spherical system in which ALS works
    centres = utils.cartesian_to_spherical(centres_cartesian)
    centres_spot = {spot_idx: utils.cartesian_to_spherical(spot_centres) for spot_idx, spot_centres in
                    centres_spot_cartesian.items()}

    phi_corr = phase_correction(phase)
    for mode_index, mode in star_container.pulsations.items():
        phi_spot, theta_spot = {}, {}

        # setting spherical variables in case the pulsation axis is parallel with orbital axis
        phi, theta = centres[:, 1], centres[:, 2] if mode.mode_axis_theta == 0.0 else \
            utils.rotation_in_spherical(centres[:, 1], centres[:, 2],
                                        mode.mode_axis_phi + phi_corr, mode.mode_axis_theta)
        if star_container.has_spots():
            for spot_idx, spot in star_container.spots.items():
                phi_spot[spot_idx], theta_spot[spot_idx] = centres_spot[spot_idx][:, 1], centres_spot[spot_idx][:, 2] \
                    if mode.mode_axis_theta == 0.0 else utils.rotation_in_spherical(centres_spot[spot_idx][:, 1],
                                                                                    centres_spot[spot_idx][:, 2],
                                                                                    mode.mode_axis_phi + phi_corr,
                                                                                    mode.mode_axis_theta)
        else:
            continue

        # renormalization constant for this mode
        constant = utils.spherical_harmonics_renormalization_constant(mode.l, mode.m)
        mode.rals_constant = constant
        # calculating rALS for surface faces
        surface_rals = constant * sph_harm(mode.m, mode.l, phi, theta)
        # calculating rALS for spots (complex values)
        if star_container.has_spots():
            spot_rals = {spot_idx: constant * sph_harm(mode.m, mode.l, phi_spot[spot_idx], theta_spot[spot_idx])
                         for spot_idx, spot in star_container.spots.items()}
            mode.rals = surface_rals, spot_rals
        else:
            mode.rals = surface_rals, {}


def recalculate_rals(container, phi_corr, centres, mode, mode_index):
    # this correction factor takes into account orbital/rotational phase when calculating drift of the mode
    # axis
    phi, theta = utils.rotation_in_spherical(centres[:, 1], centres[:, 2],
                                             mode.mode_axis_phi + phi_corr, mode.mode_axis_theta)

    container.rals[mode_index] = mode.rals_constant * sph_harm(mode.m, mode.l, phi, theta)


def calc_temp_pert_on_container(star_instance, container, phase, rot_period, com_x=None):
    """
    calculate temperature perturbation on EasyContainer

    :param com_x: float - centre of mass
    :param star_instance:
    :param container:
    :param phase:
    :param rot_period:
    :return:
    """
    centres = utils.cartesian_to_spherical(container.face_centres)
    # this is here to calculate surface centres in ref frame of the given component
    if com_x is not None:
        centres[:, 0] -= com_x
    phi_corr = phase_correction(star_instance.synchronicity, phase)
    temp_pert = up.zeros(np.shape(container.face_centres)[0])
    for mode_index, mode in star_instance.pulsations.items():
        if mode.mode_axis_theta != 0.0:
            recalculate_rals(container, phi_corr, centres, mode, mode_index)

        freq = (mode.frequency * units.FREQUENCY_UNIT).to(1/units.d).value
        exponent = const.FULL_ARC * freq * rot_period * phase - mode.start_phase
        exponential = up.exp(complex(0, -exponent))
        temp_pert_cmplx = mode.amplitude * container.rals[mode_index] * exponential
        temp_pert += temp_pert_cmplx.real
    return temp_pert


def calc_temp_pert(star_instance, phase, rot_period):
    """
    calculate temperature perturbation on star instance

    :param star_instance:
    :param phase:
    :param rot_period:
    :return:
    """
    temp_pert = np.zeros(np.shape(star_instance.face_centres)[0])
    temp_pert_spot = {spot_idx: np.zeros(np.shape(spot.face_centres)[0])
                      for spot_idx, spot in star_instance.spots.items()}

    for mode_index, mode in star_instance.pulsations.items():
        freq = (mode.frequency * units.FREQUENCY_UNIT).to(1 / units.d).value
        exponent = const.FULL_ARC * freq * rot_period * phase - mode.start_phase
        exponential = np.exp(complex(0, -exponent))
        temp_pert_cmplx = mode.amplitude * mode.rals[0] * exponential
        temp_pert += temp_pert_cmplx.real
        for spot_idx, spot in star_instance.spots.items():
            temp_pert_cmplx = mode.amplitude * mode.rals[1][spot_idx] * exponential
            temp_pert_spot[spot_idx] += temp_pert_cmplx.real
    return temp_pert, temp_pert_spot


def phase_correction(phase):
    """
    calculate phase correction for mode axis drift

    :param phase: rotation phase of the star
    :return:
    """
    # return (synchronicity - 1) * phase * const.FULL_ARC if synchronicity is not np.nan else phase * const.FULL_ARC
    return phase * const.FULL_ARC


def incorporate_pulsations_to_mesh(star_container, com_x, phase, time):
    """
    adds pulsation perturbation to the mesh

    :param star_container: StarContainer;
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
        exponent = mode.angular_frequency * time
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
    function tilts spherical coordinates to desired position described by `phi`, `theta`

    :param points: numpy.array;
    :param spot_points: dict;
    :param phi: float; azimuthal coordinate of the new polar axis
    :param theta: float; latitude of the new polar axis
    :return: tuple;
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
    returns tilt coordinates of pulsation modes

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
    assigns amplitude of radial and horizontal motion for given modes

    :param normalization_constant: factor to adjust amplitudes, in cas of Binary system it is semi major axis, in case
    of single system it should stay 1.0
    :param star_container: StarContainer;
    :return:
    """
    mult = const.G * star_container.mass / (star_container.polar_radius * normalization_constant)**3
    for mode_index, mode in star_container.pulsations.items():
        mode.radial_relative_amplitude = mode.amplitude / (star_container.polar_radius * mode.angular_frequency) / \
                                         normalization_constant
        mode.horizontal_relative_amplitude = \
            mode.radial_relative_amplitude * np.sqrt(mode.l*(mode.l+1)) * mult * mode.angular_frequency**2


def calculate_radial_displacement(relative_amplitude, radii, rals):
    """
    calculates radial displacement of surface points

    :param rals: numpy.array;
    :param relative_amplitude: float; relative radial amplitude of the pulsation mode
    see: PulsationMode.radial_relative_amplitude
    :param radii: numpy.array;
    :return: numpy.array
    """
    return relative_amplitude * radii * np.real(rals)


def calculate_phi_displacement(relative_amplitude, radii, thetas, m, rals):
    """
    displacement of azimuthal coordinates

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
        relative_amplitude * radii[sin_test] * m * np.imag(- rals[sin_test]) / sin_thetas[sin_test]
    return retval


def calculate_theta_displacement(relative_amplitude, radii, phis, thetas, l, m):
    """
    displacement in latitude

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
        return relative_amplitude * radii * alp_derivative(m)
    elif m == 0:
        return - relative_amplitude * radii * np.real(sph_harm(1, l, phis, thetas))
    elif m < 0:
        coeff = np.power(-1, -m) * factorial(l+m) / factorial(l-m)
        return relative_amplitude * radii * coeff * alp_derivative(-m)


def calculate_mode_displacement(mode, points, rals):
    radial_displacement = calculate_radial_displacement(mode.radial_relative_amplitude, points[:, 0], rals)
    phi_displacement = calculate_phi_displacement(mode.horizontal_relative_amplitude, points[:, 0], points[:, 2],
                                                  mode.m, rals)
    theta_displacement = calculate_theta_displacement(mode.horizontal_relative_amplitude, points[:, 0], points[:, 1],
                                                      points[:, 2], mode.l, mode.m)
    return np.column_stack((radial_displacement, phi_displacement, theta_displacement))
