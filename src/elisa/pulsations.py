import numpy as np

from elisa import utils, const

from scipy.special import sph_harm

"""file containing functions dealing with pulsations"""


def set_rals(self, phase=None):
    """
    Function calculates and sets time independent, dimensionless part of the pulsation modes. They are calculated as
    renormalized associated Legendre polynomials (rALS). This function needs to be evaluated only once except for the
    case of assynchronously rotating component and missaligned mode. In such case, during all phases, the drift of
    the mode axis needs to be taken into account and rALS needs to be recalculated.

    :param self: Star instance
    :param phase: float
    :return:
    """
    phase = 0.0 if phase is None else phase

    # conversion to spherical system in which ALS works
    centres = utils.cartesian_to_spherical(self.face_centres)
    centres_spot = {spot_idx: utils.cartesian_to_spherical(spot.face_centres) for spot_idx, spot in self.spots}

    for mode_index, mode in self.pulsations.items():
        phi_spot, theta_spot = {}, {}
        if mode.mode_axis_phi == 0.0 and mode.mode_axis_theta == 0.0:
            # setting spherical variables in case the pulsation axis is parallel with orbital axis
            phi, theta = centres[:, 1], centres[:, 2]
            if self.has_spots():
                for spot_idx, spot in self.spots.items():
                    phi_spot[spot_idx] = centres_spot[spot_idx][:, 1]
                    theta_spot[spot_idx] = centres_spot[spot_idx][:, 2]
        else:
            # this correction factor takes into account orbital/rotational phase when calculating drift of the mode
            # axis
            phi_corr = (self.synchronicity - 1) * phase * const.FULL_ARC if self.synchronicity is not np.nan else \
                phase * const.FULL_ARC
            # rotating spherical variable in case of misaligned mode
            phi, theta = utils.rotation_in_spherical(centres[:, 1], centres[:, 2],
                                                     mode.mode_axis_phi + phi_corr, mode.mode_axis_theta)
            if self.has_spots():
                for spot_idx, spot in self.spots.items():
                    phi_spot[spot_idx], theta_spot[spot_idx] = \
                        utils.rotation_in_spherical(centres_spot[spot_idx][:, 1], centres_spot[spot_idx][:, 2],
                                                    mode.mode_axis_phi + phi_corr, mode.mode_axis_theta)

        # renormalization constant for this mode
        constant = utils.spherical_harmonics_renormalization_constant(mode.l, mode.m)
        # calculating rALS for surface faces
        surface_rals = constant * sph_harm(mode.m, mode.l, phi, theta)
        # calculating rALS for spots (complex values)
        if self.has_spots():
            spot_rals = {spot_idx: constant * sph_harm(mode.m, mode.l, phi_spot[spot_idx], theta[spot_idx])
                         for spot_idx, spot in self.spots.items()}
            mode.rals = surface_rals, spot_rals
        else:
            mode.rals = surface_rals, {}


def calculate_temperature_perturbation(self, time):
    pass
