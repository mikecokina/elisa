import numpy as np

from ... import settings


# ________________________complex angular coordinates_______________________
def calculate_displacement_coordinates(mode, points, harmonics, harmonics_derivatives, radius, scale=1.0):
    """
    Calculates surface displacement caused by given `mode`.

    :param mode: elisa.pulse.mode.Mode;
    :param points: numpy.array; in spherical coordinates
    :param harmonics: numpy.array; Y_l^m
    :param harmonics_derivatives: numpy.array; [dY/dphi, dY/dtheta]
    :param radius: float; equivalent radius of the component
    :param scale: numpy.float; scale of the perturbations
    :return: numpy.array; complex
    """
    if settings.PULSATION_MODEL == 'uniform':
        radial_displacement = calculate_radial_displacement(mode, harmonics) / scale
        phi_displacement, theta_displacement = \
            calculate_horizontal_displacements(mode, points[:, 2], harmonics_derivatives, radius)

        return np.column_stack((radial_displacement, phi_displacement, theta_displacement))
    else:
        raise NotImplementedError(f'Pulsation model: {settings.PULSATION_MODEL} not implemented.')


def calculate_mode_angular_displacement(displacement):
    """
    Calculates angular displacement from complex angular coordinates.

    :param displacement: numpy.array; complex
    :return: numpy.array;
    """
    return np.real(displacement)


def calculate_radial_displacement(mode, harmonics):
    """
    Calculates radial displacement of surface points.

    :param mode: PulsationMode;
    :param harmonics: numpy.array; Y_l^m
    :return: numpy.array; complex
    """
    return mode.radial_amplitude * harmonics


def calculate_horizontal_displacements(mode, thetas, harmonics_derivatives, radius):
    """
    Calculate angular horizontal components of displacement.

    :param mode: PulsationMode;
    :param thetas: numpy.array; 1D
    :param harmonics_derivatives: numpy.array; column-wise - dY/dphi, dY/dtheta
    :param radius:
    :return:
    """
    sin_theta = np.sin(thetas)
    # lambda - distance along phi
    # nu - distance along theta coordinate
    phi_amp = np.abs(harmonics_derivatives[0])
    d_lamda = radius * sin_theta * phi_amp
    theta_amp = np.abs(harmonics_derivatives[1])
    d_nu = radius * theta_amp

    dr = np.sqrt(np.mean(np.power(d_lamda, 2) + np.power(d_nu, 2)))
    corr_factor = mode.horizontal_amplitude / dr

    theta_test = sin_theta != 0
    phi_retval = np.zeros(thetas.shape, dtype=np.complex128)
    phi_retval[theta_test] = corr_factor * harmonics_derivatives[0, theta_test] / sin_theta[theta_test]
    return phi_retval, corr_factor * harmonics_derivatives[1]


# ________________________velocity coordinates_______________________
def calculate_mode_angular_derivatives(displacement, angular_frequency):
    """
    Calculates derivatives of angular displacement.

    :param displacement: numpy.array; complex
    :param angular_frequency: np.float;
    :return: numpy.array;
    """
    return - angular_frequency * np.imag(displacement)
