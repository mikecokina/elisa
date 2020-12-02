import numpy as np

from elisa import settings


def calculate_mode_displacement(mode, points, harmonics, harmonics_derivatives, scale=1.0):
    """
    Calculates surface displacement caused by given `mode`.

    :param mode: elisa.pulse.mode.Mode;
    :param points: numpy.array; in spherical coordinates
    :param harmonics: numpy.array; Y_l^m
    :param harmonics_derivatives: numpy.array; [dY/dphi, dY/dtheta]
    :param scale: numpy.float; scale of the perturbations
    :return: numpy.array;
    """
    if settings.PULSATION_MODEL == 'uniform':
        radial_displacement = calculate_radial_displacement(mode, harmonics) / scale
        phi_displacement = calculate_phi_displacement(mode, points[:, 2], harmonics_derivatives[0])
        theta_displacement = calculate_theta_displacement(mode, harmonics_derivatives[1])

        return np.column_stack((radial_displacement, phi_displacement, theta_displacement))
    else:
        raise NotImplementedError(f'Pulsation model: {settings.PULSATION_MODEL} not implemented.')


def calculate_radial_displacement(mode, harmonics):
    """
    Calculates radial displacement of surface points.

    :param mode: PulsationMode;
    :param harmonics: numpy.array; Y_l^m
    :return: numpy.array;
    """
    return mode.radial_amplitude * np.real(harmonics)


def calculate_phi_displacement(mode, thetas, harmonics_derivatives):
    """
    Displacement of azimuthal coordinates.

    :param mode: PulsationMode;
    :param thetas: numpy.array
    :param harmonics_derivatives: numpy.array; dY/dphi
    :return: numpy.array;
    """
    sin2_thetas = np.power(np.sin(thetas), 2)
    sin_test = sin2_thetas != 0.0
    retval = np.zeros(thetas.shape)
    retval[sin_test] = \
        mode.horizontal_amplitude * np.real(harmonics_derivatives[sin_test]) \
        / sin2_thetas[sin_test]
    return retval


def calculate_theta_displacement(mode, harmonics_derivatives):
    """
    Displacement in latitude.

    :param harmonics_derivatives: numpy.array; dY/dtheta
    :param mode: PulsationMode;
    :return: numpy.array;
    """
    return mode.horizontal_amplitude * np.real(harmonics_derivatives)