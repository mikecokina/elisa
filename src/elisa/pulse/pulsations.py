import numpy as np

from .. import utils, const, umpy as up
from .. umpy import sph_harm
from .. import settings
from .. logger import getLogger
from . import utils as putils
from . surface.kinematics import calculate_displacement_coordinates

logger = getLogger('pulse.pulsations')

"""
File containing functions dealing with pulsations.
"""


def spherical_harmonics(mode, points, time_exponential, order=None, degree=None):
    """
    Returns spherical harmonics normalized such that its rms = 1.

    :param mode: PulsationMode; mode used to generate sph. harmonics
    :param points: np.array; points in spherical coordinates on which to calculate spherical harmonics value
    :param time_exponential: float; time at which evaluate spherical harmonics
    :param degree: int;
    :param order: int;
    :return: numpy.array; array of spherical harmonics for  given `points`
    """
    l = mode.l if degree is None else degree
    m = mode.m if order is None else order
    return mode.renorm_const * sph_harm(m, l, points[:, 1], points[:, 2]) * time_exponential


def diff_spherical_harmonics_by_phi(mode, harmonics):
    """
    Returns d Y_m^l / d phi

    :param mode: PulsationMode; mode used to generate sph. harmonics
    :param harmonics: list; [Y_l^m, Y_l^m+1]
    :return: numpy.array;
    """
    retval = (0 + 1j) * mode.m * harmonics[0]
    norm = np.power(np.mean(np.abs(retval)**2), 0.5)
    return retval/norm


def diff_spherical_harmonics_by_theta(mode, harmonics, phis, thetas):
    """
    Returns d Y_m^l / d theta

    :param mode: PulsationMode; mode used to generate sph. harmonics
    :param harmonics: list; [Y_l^m, Y_l^m+1]
    :param phis: numpy.array;
    :param thetas: numpy.array;
    :return: numpy.array;
    """
    theta_test = np.logical_and(thetas != 0.0, thetas != const.PI)
    derivative = np.zeros(phis.shape)
    derivative[theta_test] = mode.m * np.real(harmonics[0][theta_test] / np.tan(thetas[theta_test])) + \
                             np.sqrt((mode.l - mode.m) * (mode.l + mode.m + 1)) * \
                             np.real(np.exp((0 - 1j) * phis[theta_test]) * harmonics[1][theta_test])
    norm = np.power(np.mean(np.abs(derivative)**2), 0.5)
    return derivative / norm


def incorporate_gravity_perturbation(star_container, g_acc_vector, g_acc_vector_spot, phase):
    """

    :param star_container: base.container.StarContainer;
    :param g_acc_vector: numpy.array;
    :param g_acc_vector_spot: Dict;
    :param phase: float;
    :return: Tuple;
    """
    g_sph = utils.cartesian_to_spherical(g_acc_vector)
    g_sph_spot = {spot_idx: utils.cartesian_to_spherical(g_acc) for spot_idx, g_acc in g_acc_vector_spot.items()}

    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star_container, phase)
    tilted_acc, tilted_acc_spot = \
        putils.tilt_mode_coordinates(g_sph, g_sph_spot, tilt_phi, tilt_theta)

    tilted_points, tilted_points_spot = star_container.pulsations[0].points, star_container.pulsations[0].spot_points

    g_pert = up.zeros(tilted_points.shape)
    g_pert_spot = {spot_idx: up.zeros(spot.shape) for spot_idx, spot in tilted_points_spot.items()}

    for mode_index, mode in star_container.pulsations.items():
        g_pert += calculate_acc_pert(mode, tilted_points, mode.point_harmonics, mode.point_harmonics_derivatives)

        for spot_idx, spoints in tilted_points_spot.items():
            g_pert_spot[spot_idx] += calculate_acc_pert(mode, tilted_points_spot[spot_idx],
                                                        mode.spot_point_harmonics[spot_idx],
                                                        mode.spot_point_harmonics_derivatives[spot_idx])

    g_acc_vector = putils.derotate_surface_points(tilted_acc + g_pert,
                                                  star_container.pulsations[0].mode_axis_phi,
                                                  star_container.pulsations[0].mode_axis_theta,
                                                  com_x=0.0)

    for spot_idx, spot in star_container.spots.items():
        g_acc_vector_spot[spot_idx] = \
            putils.derotate_surface_points(tilted_acc_spot[spot_idx] + g_pert_spot[spot_idx],
                                           star_container.pulsations[0].mode_axis_phi,
                                           star_container.pulsations[0].mode_axis_theta,
                                           com_x=0.0)

    return g_acc_vector, g_acc_vector_spot


def assign_amplitudes(star_container, normalization_constant=1.0):
    """
    Assigns amplitude of radial and horizontal motion for given modes.

    :param normalization_constant: factor to adjust amplitudes, in cas of Binary system it is semi major axis, in case
                                   of single system it should stay 1.0
    :param star_container: StarContainer;
    """
    r_equiv = star_container.equivalent_radius * normalization_constant
    mult = const.G * star_container.mass / r_equiv ** 3
    for mode_index, mode in star_container.pulsations.items():
        mode.radial_amplitude = mode.amplitude / mode.angular_frequency

        # horizontal/radial amplitude (Aerts 2010), p. 198
        mode.horizontal_to_radial_amplitude_ratio = \
            np.sqrt(mode.l * (mode.l + 1)) * mult / mode.angular_frequency ** 2 \
                if mode.horizontal_to_radial_amplitude_ratio is None else mode.horizontal_to_radial_amplitude_ratio
        mode.horizontal_amplitude = mode.horizontal_to_radial_amplitude_ratio * mode.radial_amplitude / r_equiv

        surf_ampl = mode.horizontal_amplitude
        if surf_ampl > settings.SURFACE_DISPLACEMENT_TOL:
            prec = int(- np.log10(surf_ampl) + 2)
            logger.warning(f'Relative horizontal surface displacement amplitude ({round(surf_ampl, prec)}) for the mode'
                           f' {mode_index} exceeded safe tolerances ({settings.SURFACE_DISPLACEMENT_TOL}) given by the'
                           f' use of linear approximation. This can lead to invalid surface discretization. Use this'
                           f' result with caution.')


def calculate_acc_pert(mode, points, harmonics, harmonics_derivatives):
    """
    Calculate perturbation of surface acceleration.

    :param mode: elisa.pulse.mode.Mode;
    :param points: numpy.array;
    :param harmonics: numpy.array; Y_l^m
    :param harmonics_derivatives: numpy.array; [dY/dphi, dY/dtheta]
    :return: numpy.array;
    """
    return - mode.angular_frequency ** 2 * calculate_displacement_coordinates(mode, points, harmonics, harmonics_derivatives)


def incorporate_temperature_perturbations(star_container, com_x, phase, time):
    """
    Introduces temperature perturbations to star container.

    :param star_container: elisa.base.container.StarContainer
    :param com_x: float; centre of mass
    :param phase: float;
    :param time: float;
    :return: float;
    """
    centres, spot_centres = star_container.transform_points_to_spherical_coordinates(kind='face_centres', com_x=com_x)

    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star_container, phase)
    tilted_centres, tilted_spot_centres = putils.tilt_mode_coordinates(centres, spot_centres, tilt_phi, tilt_theta)

    t_pert = up.zeros(centres.shape[0])
    t_pert_spots = {spot_idx: up.zeros(spot.shape[0]) for spot_idx, spot in spot_centres.items()}

    for mode_index, mode in star_container.pulsations.items():
        exponential = putils.generate_time_exponential(mode, time)
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
    l_val = mode.l
    h, eps = mode.horizontal_amplitude, mode.radial_amplitude

    return ad_g * temperatures * (h * l_val * (l_val + 1) - 4 - (1 / h)) * eps * np.real(rals)


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
    points, points_spot = star_container.transform_points_to_spherical_coordinates(kind='points', com_x=com_x)

    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star_container, phase)
    tilted_points, tilted_points_spot = putils.tilt_mode_coordinates(points, points_spot, tilt_phi, tilt_theta)

    # assigning tilted points in spherical coordinates only to the first mode (the rest will share the same points)
    star_container.pulsations[0].points = tilted_points
    star_container.pulsations[0].spot_points = tilted_points_spot

    for mode_index, mode in star_container.pulsations.items():
        exponential = putils.generate_time_exponential(mode, time)

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

        # generating derivatives of spherical harmonics by phi an theta
        derivatives = np.empty((2, tilted_points.shape[0]), dtype=np.complex)
        derivatives[0] = diff_spherical_harmonics_by_phi(mode, harmonics)
        derivatives[1] = diff_spherical_harmonics_by_theta(mode, harmonics, tilted_points[:, 1], tilted_points[:, 2])

        spot_harmonics_derivatives = {spot_idx: np.zeros((2, spoints.shape[0]), dtype=np.complex)
                                      for spot_idx, spoints in tilted_points_spot.items()}
        for spot_idx, spotp in tilted_points_spot.items():
            spot_harmonics_derivatives[spot_idx][0] = diff_spherical_harmonics_by_phi(mode, spot_harmonics[spot_idx])
            spot_harmonics_derivatives[spot_idx][1] = \
                diff_spherical_harmonics_by_theta(mode, spot_harmonics[spot_idx], spotp[:, 1], spotp[:, 2])

        # assignment of harmonics to mode instance variables
        mode.point_harmonics = harmonics[0]
        mode.spot_point_harmonics = {spot_idx: hrm[0] for spot_idx, hrm in spot_harmonics.items()}

        mode.face_harmonics = np.mean(mode.point_harmonics[star_container.faces], axis=1)
        mode.spot_face_harmonics = {spot_idx: np.mean(spoth[star_container.spots[spot_idx].faces], axis=1)
                                    for spot_idx, spoth in mode.spot_point_harmonics.items()}

        mode.point_harmonics_derivatives = derivatives
        mode.spot_point_harmonics_derivatives = spot_harmonics_derivatives

        mode.face_harmonics_derivatives = np.mean(derivatives[:, star_container.faces], axis=1)
        mode.spot_face_harmonics_derivatives = {
            spot_idx: np.mean(spoth[:, star_container.spots[spot_idx].faces], axis=1)
            for spot_idx, spoth in spot_harmonics_derivatives.items()
        }
    return star_container

