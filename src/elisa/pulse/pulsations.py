import numpy as np

from .. import const
from .. umpy import sph_harm
from .. import settings
from .. logger import getLogger

logger = getLogger('pulse.pulsations')


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
    :param harmonics: List; [Y_l^m, Y_l^m+1]
    :return: numpy.array;
    """
    return (0 + 1j) * mode.m * harmonics[0]


def diff_spherical_harmonics_by_theta(mode, harmonics, phis, thetas):
    """
    Returns d Y_m^l / d theta

    :param mode: PulsationMode; mode used to generate sph. harmonics
    :param harmonics: List; [Y_l^m, Y_l^m+1]
    :param phis: numpy.array;
    :param thetas: numpy.array;
    :return: numpy.array;
    """
    theta_test = np.logical_and(thetas != 0.0, thetas != const.PI)
    derivative = np.zeros(phis.shape, dtype=np.complex128)
    derivative[theta_test] = mode.m * harmonics[0][theta_test] / np.tan(thetas[theta_test]) + \
                             np.sqrt((mode.l - mode.m) * (mode.l + mode.m + 1)) * \
                             np.exp((0 - 1j) * phis[theta_test]) * harmonics[1][theta_test]
    return derivative


def horizontal_displacement_normalization(derivatives, harmonics):
    """
    Normalizes the RMS of horizontal displacement of the given pulsation to 1.

    :param derivatives: numpy.array; 2*N - dY/d_phi, dY/d_theta
    :param harmonics: numpy.array; 2*N - Y^l_m, Y^l_m+1
    :return: float;
    """
    return np.sqrt(np.sum(np.power(np.abs(harmonics[0]), 2)) /
                   np.sum(np.power(np.abs(derivatives[0]), 2) + np.power(np.abs(derivatives[1]), 2)))


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

        # horizontal/radial amplitude (Aerts 2010), p. 198
        if mode.horizontal_to_radial_amplitude_ratio is None:
            amp_ratio = np.sqrt(mode.l * (mode.l + 1)) * mult / mode.angular_frequency ** 2
            mode.horizontal_to_radial_amplitude_ratio = amp_ratio

        amplitude = mode.amplitude / mode.angular_frequency

        mode.radial_amplitude = amplitude / np.sqrt(mode.horizontal_to_radial_amplitude_ratio**2 + 1)
        mode.horizontal_amplitude = mode.horizontal_to_radial_amplitude_ratio * mode.radial_amplitude

        if mode.temperature_amplitude_factor is None:
            if mode.l == 0 or mode.horizontal_to_radial_amplitude_ratio == 0.0:
                raise ValueError('Parameter `temperature_amplitude_factor` needs to be supplied in '
                                 'case of radial modes or in case of modes with radial motion.')
            mode.temperature_amplitude_factor = temp_amplitude(mode) * mode.radial_amplitude / r_equiv

        surf_ampl = mode.horizontal_amplitude / r_equiv
        if surf_ampl > settings.SURFACE_DISPLACEMENT_TOL:
            prec = int(- np.log10(surf_ampl) + 2)
            if not settings.SUPPRESS_WARNINGS:
                logger.warning(f'Relative horizontal surface displacement amplitude '
                               f'({round(surf_ampl, prec)}) for the mode {mode_index} '
                               f'exceeded safe tolerances ({settings.SURFACE_DISPLACEMENT_TOL}) given by the'
                               f' use of linear approximation. This can lead to invalid surface '
                               f'discretization. Use this result with caution.')


def temp_amplitude(mode):
    """
    Returns temperature perturbation amplitude in form of the scalar therm in eq 22 in Townsend 2003::

        delta T = temp_amplitude * (delta r / r) * T;
        temp_amplitude = nabla_ad * (Kl(l+1) - 4 - 1/K)

    where K is our horizontal to radial amplitude ratio.

    :param mode: PulsationMode;
    :return: float;
    """
    return const.IDEAL_ADIABATIC_GRADIENT * (
        mode.horizontal_to_radial_amplitude_ratio * mode.l * (mode.l + 1) - 4 -
        1 / mode.horizontal_to_radial_amplitude_ratio
    )
