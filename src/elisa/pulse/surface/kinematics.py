import numpy as np

from elisa import settings


# ________________________complex angular coordinates_______________________
def calculate_displacement_coordinates(mode, points, harmonics, harmonics_derivatives, scale=1.0):
    """
    Calculates surface displacement caused by given `mode`.

    :param mode: elisa.pulse.mode.Mode;
    :param points: numpy.array; in spherical coordinates
    :param harmonics: numpy.array; Y_l^m
    :param harmonics_derivatives: numpy.array; [dY/dphi, dY/dtheta]
    :param scale: numpy.float; scale of the perturbations
    :return: numpy.array; complex
    """
    if settings.PULSATION_MODEL == 'uniform':
        radial_displacement = calculate_radial_displacement(mode, harmonics) / scale
        phi_displacement = calculate_phi_displacement(mode, points[:, 2], harmonics_derivatives[0])
        theta_displacement = calculate_theta_displacement(mode, harmonics_derivatives[1])

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


def calculate_phi_displacement(mode, thetas, harmonics_derivatives):
    """
    Displacement of azimuthal coordinates.

    :param mode: PulsationMode;
    :param thetas: numpy.array
    :param harmonics_derivatives: numpy.array; dY/dphi
    :return: numpy.array; complex
    """
    sin_thetas = np.sin(thetas)
    sin_test = sin_thetas != 0.0
    retval = np.zeros(thetas.shape, dtype=np.complex128)
    retval[sin_test] = \
        mode.horizontal_amplitude * harmonics_derivatives[sin_test] \
        / sin_thetas[sin_test]
    return retval


def calculate_theta_displacement(mode, harmonics_derivatives):
    """
    Displacement in latitude.

    :param harmonics_derivatives: numpy.array; dY/dtheta
    :param mode: PulsationMode;
    :return: numpy.array; complex
    """
    return mode.horizontal_amplitude * harmonics_derivatives


# ________________________velocity coordinates_______________________
def calculate_mode_angular_derivatives(displacement, angular_frequency):
    """
    Calculates derivatives of angular displacement.

    :param displacement: numpy.array; complex
    :param angular_frequency: np.float;
    :return: numpy.array;
    """
    return - angular_frequency * np.imag(displacement)
