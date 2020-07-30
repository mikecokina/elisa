import numpy as np

from elisa import utils, const, umpy as up
from scipy.special import sph_harm
from elisa.conf import config
from elisa.logger import getLogger


logger = getLogger('pulse.pulsations')

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


def generate_time_exponential(mode, time):
    """
    Returns time dependent exponential used in `generate_spherical_harmonics`.

    :param mode: PulsationMode; PulsationMode; mode used to generate sph. harmonics
    :param time: float; time at which evaluate spherical harmonics
    :return: np.float; complex time dependent exponential
    """
    exponent = mode.angular_frequency * time + mode.start_phase
    return up.exp(complex(0, -exponent))


def spherical_harmonics(mode, points, time_exponential, order=None, degree=None):
    """
    Returns spherical harmonics normalized such that its rms = 1.

    :param mode: PulsationMode; mode used to generate sph. harmonics
    :param points: np.array; points in spherical coordinates on which to calculate spherical harmonics value
    :param time_exponential: float; time at which evaluate spherical harmonics
    :param degree: int;
    :param order: int;
    :return: np.array; array of spherical harmonics for  given `points`
    """
    l = mode.l if degree is None else degree
    m = mode.m if order is None else order
    return mode.renorm_const * sph_harm(m, l, points[:, 1], points[:, 2]) * time_exponential


def incorporate_pulsations_to_mesh(star_container, com_x):
    """
    adds pulsation perturbation to the mesh

    :param star_container: elisa.base.container.StarContainer;
    :param com_x: float; centre of mass of the star
    :return: StarContainer;
    """
    points, points_spot = star_container.transform_points_to_spherical_coordinates(kind='points', com_x=com_x)

    tilted_points, tilted_points_spot = star_container.pulsations[0].points, star_container.pulsations[0].spot_points

    displacement = up.zeros(tilted_points.shape)
    displacement_spots = {spot_idx: up.zeros(spot.shape) for spot_idx, spot in tilted_points_spot.items()}

    for mode_index, mode in star_container.pulsations.items():
        displacement += calculate_mode_displacement(mode, tilted_points, mode.point_harmonics)
        for spot_idx, spoints in tilted_points_spot.items():
            displacement_spots[spot_idx] += \
                calculate_mode_displacement(mode, spoints, mode.spot_point_harmonics[spot_idx])

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
    r_equiv = star_container.equivalent_radius * normalization_constant
    mult = const.G * star_container.mass / r_equiv**3
    for mode_index, mode in star_container.pulsations.items():
        mode.radial_amplitude = mode.amplitude / (r_equiv * mode.angular_frequency)
        mode.horizontal_amplitude = np.sqrt(mode.l*(mode.l+1)) * mult / mode.angular_frequency**2

        surf_ampl = mode.radial_amplitude * mode.horizontal_amplitude
        if surf_ampl > config.SURFACE_DISPLACEMENT_TOL:
            prec = int(- np.log10(surf_ampl) + 2)
            logger.warning(f'Surface displacement amplitude ({round(surf_ampl, prec)}) for the mode {mode_index} '
                           f'exceeded safe tolerances ({config.SURFACE_DISPLACEMENT_TOL}) given by the use of linear '
                           f'approximation. This can lead to invalid surface discretization. Use this result with '
                           f'caution.')


def calculate_radial_displacement(mode, radii, spherical_harmonics):
    """
    Calculates radial displacement of surface points.

    :param mode: PulsationMode;
    :param radii: numpy.array;
    :param spherical_harmonics: numpy.array; Y_l^m
    :return: numpy.array
    """
    return mode.radial_amplitude * radii * np.real(spherical_harmonics)


def calculate_phi_displacement(mode, thetas, harmonics):
    """
    Displacement of azimuthal coordinates.

    :param mode: PulsationMode;
    :param thetas: numpy.array
    :param harmonics: numpy.array; Y_l^m
    :return: numpy.array;
    """
    sin_thetas = np.sin(thetas)
    sin_test = sin_thetas != 0.0
    retval = np.zeros(thetas.shape)
    retval[sin_test] = - mode.m * mode.radial_amplitude * mode.horizontal_amplitude * np.imag(harmonics[sin_test]) \
         / sin_thetas[sin_test]
    return retval


def calculate_theta_displacement(mode, phis, thetas, harmonics):
    """
    Displacement in latitude.

    :param mode: PulsationMode;
    :param thetas: numpy.array;
    :param phis: numpy.array;
    :param harmonics: np.array;
    :return: numpy.array;
    """

    def spherical_harm_derivative():
        """works for m > 0"""
        theta_test = np.logical_and(thetas != 0.0, thetas != const.PI)

        derivative = np.zeros(phis.shape)
        derivative[theta_test] = mode.m * np.real(harmonics[0][theta_test] / np.tan(thetas[theta_test])) + \
                                 np.sqrt((mode.l - mode.m) * (mode.l + mode.m + 1)) * \
                                 np.real(np.exp((0 - 1j) * phis[theta_test]) * harmonics[1][theta_test])

        return derivative

    mult = mode.radial_amplitude * mode.horizontal_amplitude
    return mult * spherical_harm_derivative()


def calculate_mode_displacement(mode, points, spherical_harmonics):
    """
    Calculates surface displacement caused by given `mode`.

    :param mode: elisa.pulse.mode.Mode;
    :param points: numpy.array;
    :param spherical_harmonics: list; [Y_l^m, Y_l^m+1]
    :return:
    """
    radial_displacement = calculate_radial_displacement(mode, points[:, 0], spherical_harmonics[0])
    phi_displacement = calculate_phi_displacement(mode, points[:, 2], spherical_harmonics[0])
    theta_displacement = calculate_theta_displacement(mode, points[:, 1], points[:, 2], spherical_harmonics)

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
        exponential = generate_time_exponential(mode, time)
        harmonics = spherical_harmonics(mode, tilted_centres, exponential)
        spot_harmonics = {spot_idx: spherical_harmonics(mode, spotp, exponential)
                          for spot_idx, spotp in tilted_spot_centres.items()}

        t_pert += calculate_temperature_perturbation(mode, star_container.temperatures, harmonics)
        for spot_idx, t_s in t_pert_spots.items():
            t_s += calculate_temperature_perturbation(mode, star_container.spots[spot_idx].temperatures,
                                                      spot_harmonics[spot_idx])

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
    h, eps = mode.horizontal_amplitude, mode.radial_amplitude

    return ad_g * temperatures * (h * l * (l + 1) - 4 - (1 / h)) * eps * np.real(rals)


def generate_harmonics(star_container, com_x, phase, time):
    """
    Generating spherical harmonics (Y_l^m, Y_l^m+1) in shapes(2, n_points) and (2, n_faces) to be subsequently used for
    calculation of perturbed properties.

    :param star_container: elisa.base.container.StarContainer;
    :param com_x: float; centre of mass for the component
    :param phase: float; rotational/orbital phase
    :param time: float; time of the observation
    :return: elisa.base.container.StarContainer; Star container with updated harmonics
    """
    points, points_spot = star_container.transform_points_to_spherical_coordinates(kind='points', com_x=com_x)

    tilt_phi, tilt_theta = generate_tilt_coordinates(star_container, phase)
    tilted_points, tilted_points_spot = tilt_mode_coordinates(points, points_spot, tilt_phi, tilt_theta)

    # assigning tilted points in spherical coordinates only to the first mode (the rest will share the same points)
    star_container.pulsations[0].points = tilted_points
    star_container.pulsations[0].spot_points = tilted_points_spot

    for mode_index, mode in star_container.pulsations.items():
        exponential = generate_time_exponential(mode, time)

        harmonics = np.zeros((2, tilted_points.shape[0]), dtype=np.complex)
        spot_harmonics = {spot_idx: np.zeros((2, spoints.shape[0]), dtype=np.complex)
                          for spot_idx, spoints in tilted_points_spot.items()}

        harmonics[0] = spherical_harmonics(mode, tilted_points, exponential)
        for spot_idx, spotp in tilted_points_spot.items():
            spot_harmonics[spot_idx][0] = spherical_harmonics(mode, spotp, exponential)

        if mode.m != mode.l:
            harmonics[1] = spherical_harmonics(mode, tilted_points, exponential,
                                               order=mode.m + 1, degree=mode.l)
            for spot_idx, spotp in tilted_points_spot.items():
                spot_harmonics[spot_idx][1] = spherical_harmonics(mode, spotp, exponential,
                                                                  order=mode.m + 1, degree=mode.l)

        mode.point_harmonics = harmonics
        mode.spot_point_harmonics = spot_harmonics

        mode.face_harmonics = np.mean(harmonics[:, star_container.faces], axis=2)
        mode.spot_face_harmonics = {spot_idx: np.mean(spoth[:, star_container.spots[spot_idx].faces], axis=2)
                                    for spot_idx, spoth in spot_harmonics.items()}

    return star_container

