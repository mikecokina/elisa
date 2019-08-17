import numpy as np

from elisa import utils, const, units

from scipy.special import sph_harm
from astropy import units as u
from copy import copy

"""file containing functions dealing with pulsations"""


def set_rals(self, com_x=None):
    """
    Function calculates and sets time independent, dimensionless part of the pulsation modes. They are calculated as
    renormalized associated Legendre polynomials (rALS). This function needs to be evaluated only once except for the
    case of assynchronously rotating component and missaligned mode (see `set_misaligned_rals()`).

    :param com_x: float - centre of mass
    :param self: Star instance
    :return:
    """
    centres_cartesian = copy(self.face_centres)
    centres_spot_cartesian = {spot_idx: spot.face_centres for spot_idx, spot in self.spots.items()}

    # this is here to calculate surface centres in ref frame of the given component
    if com_x is not None:
        centres_cartesian[:, 0] -= com_x
        for spot_index, spot in self.spots.items():
            centres_spot_cartesian[spot_index][:, 0] -= com_x

    # conversion to spherical system in which ALS works
    centres = utils.cartesian_to_spherical(centres_cartesian)
    centres_spot = {spot_idx: utils.cartesian_to_spherical(spot) for spot_idx, spot in self.spots.items()}

    for mode_index, mode in self.pulsations.items():
        phi_spot, theta_spot = {}, {}
        if mode.mode_axis_theta == 0.0:
            # setting spherical variables in case the pulsation axis is parallel with orbital axis
            phi, theta = centres[:, 1], centres[:, 2]
            if self.has_spots():
                for spot_idx, spot in self.spots.items():
                    phi_spot[spot_idx] = centres_spot[spot_idx][:, 1]
                    theta_spot[spot_idx] = centres_spot[spot_idx][:, 2]
        else:
            continue

        # renormalization constant for this mode
        constant = utils.spherical_harmonics_renormalization_constant(mode.l, mode.m)
        mode.rals_constant = constant
        # calculating rALS for surface faces
        surface_rals = constant * sph_harm(mode.m, mode.l, phi, theta)
        # calculating rALS for spots (complex values)
        if self.has_spots():
            spot_rals = {spot_idx: constant * sph_harm(mode.m, mode.l, phi_spot[spot_idx], theta_spot[spot_idx])
                         for spot_idx, spot in self.spots.items()}
            mode.rals = surface_rals, spot_rals
        else:
            mode.rals = surface_rals, {}


def set_misaligned_rals(star_instance, phase, com_x=None):
    """
    Function calculates and sets time independent, dimensionless part of the pulsation modes. They are calculated as
    renormalized associated Legendre polynomials (rALS). This function deals with a case of assynchronously rotating
    component and missaligned mode. In such case, during all phases, the drift of the mode axis needs to be taken
    into account and rALS needs to be recalculated.

    :param phase:
    :param com_x: float - centre of mass
    :type star_instance: Star instance
    :return:
    """
    centres_cartesian = copy(star_instance.face_centres)
    centres_spot_cartesian = {spot_idx: spot.face_centres for spot_idx, spot in star_instance.spots.items()}

    # this is here to calculate surface centres in ref frame of the given component
    if com_x is not None:
        centres_cartesian[:, 0] -= com_x
        for spot_index, spot in star_instance.spots.items():
            centres_spot_cartesian[spot_index][:, 0] -= com_x

    # conversion to spherical system in which ALS works
    centres = utils.cartesian_to_spherical(centres_cartesian)
    centres_spot = {spot_idx: utils.cartesian_to_spherical(spot) for spot_idx, spot in star_instance.spots.items()}

    for mode_index, mode in star_instance.pulsations.items():
        phi_spot, theta_spot = {}, {}
        if mode.mode_axis_theta == 0.0:
            continue
        else:
            # this correction factor takes into account orbital/rotational phase when calculating drift of the mode
            # axis
            phi_corr = phase_correction(star_instance, phase)
            # rotating spherical variable in case of misaligned mode
            phi, theta = utils.rotation_in_spherical(centres[:, 1], centres[:, 2],
                                                     mode.mode_axis_phi + phi_corr, mode.mode_axis_theta)
            if star_instance.has_spots():
                for spot_idx, spot in star_instance.spots.items():
                    phi_spot[spot_idx], theta_spot[spot_idx] = \
                        utils.rotation_in_spherical(centres_spot[spot_idx][:, 1], centres_spot[spot_idx][:, 2],
                                                    mode.mode_axis_phi + phi_corr, mode.mode_axis_theta)

        # renormalization constant for this mode
        constant = utils.spherical_harmonics_renormalization_constant(mode.l, mode.m)
        mode.rals_constant = constant
        # calculating rALS for surface faces
        surface_rals = constant * sph_harm(mode.m, mode.l, phi, theta)
        # calculating rALS for spots (complex values)
        if star_instance.has_spots():
            spot_rals = {spot_idx: constant * sph_harm(mode.m, mode.l, phi_spot[spot_idx], theta[spot_idx])
                         for spot_idx, spot in star_instance.spots.items()}
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
    phi_corr = phase_correction(star_instance, phase)
    temp_pert = np.zeros(np.shape(container.face_centres)[0])
    for mode_index, mode in star_instance.pulsations.items():
        if mode.mode_axis_theta != 0.0:
            recalculate_rals(container, phi_corr, centres, mode, mode_index)

        freq = (mode.frequency * units.FREQUENCY_UNIT).to(1/u.d).value
        exponent = const.FULL_ARC * freq * rot_period * phase - mode.start_phase
        exponential = np.exp(complex(0, -exponent))
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
        freq = (mode.frequency * units.FREQUENCY_UNIT).to(1 / u.d).value
        exponent = const.FULL_ARC * freq * rot_period * phase - mode.start_phase
        exponential = np.exp(complex(0, -exponent))
        temp_pert_cmplx = mode.amplitude * mode.rals[0] * exponential
        temp_pert += temp_pert_cmplx.real
        if star_instance.has_spots():
            for spot_idx, spot in star_instance.spots.items():
                temp_pert_cmplx = mode.amplitude * mode.rals[1][spot_idx] * exponential
                temp_pert_spot[spot_idx] += temp_pert_cmplx.real
    return temp_pert, temp_pert_spot


def phase_correction(self, phase):
    """
    calculate phase correction for mode axis drift

    :param self: Star instance
    :param phase: orbital/rotation phase
    :return:
    """
    return (self.synchronicity - 1) * phase * const.FULL_ARC if self.synchronicity is not np.nan else \
        phase * const.FULL_ARC
