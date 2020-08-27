import numpy as np

from elisa import ld
from elisa.conf import config
from elisa.binary_system.curves import curves


def compute_circ_sync_rv_at_pos(velocities, pos_idx, crv_labels, system):
    """
    Calculates rv points for given orbital position in case of circular orbit and synchronous rotation.

    :param velocities: Dict; {str; component : numpy.array; rvs, ...}
    :param pos_idx: int; position in `band_curves` to which calculated lc points will be assigned
    :param crv_labels: list; list of components for which to calculate rvs
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :return: Dict; updated {str; passband : numpy.array; rvs, ...}
    """
    # calculating cosines between face normals and line of sight
    ld_law_cfs_column = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]
    for component in crv_labels:
        star = getattr(system, component)
        visibility_indices = star.indices
        cosines = star.los_cosines[star.indices]

        ld_cors = \
            ld.limb_darkening_factor(
                coefficients=star.ld_cfs['rv_band'][ld_law_cfs_column].values[visibility_indices],
                limb_darkening_law=config.LIMB_DARKENING_LAW,
                cos_theta=cosines)

        flux = star.normal_radiance['rv_band'][visibility_indices] * cosines * star.coverage[visibility_indices] * \
               ld_cors

        velocities[component][pos_idx] = np.sum(star.velocities[visibility_indices][:, 0] * flux) / np.sum(flux) \
            if np.sum(flux) != 0 else np.NaN

    return velocities


def compute_circ_spotty_async_rv_at_pos(velocities, pos_idx, crv_labels, system):
    """
    Calculates rv points for given orbital position in case of circular orbit and synchronous rotation.

    :param velocities: Dict; {str; component : numpy.array; rv curve, ...} result will be written to the
                              corresponding `pos_idx` position
    :param pos_idx: int; position in `velocities` to which calculated rv points will be assigned
    :param crv_labels: list; list of components for rv calculation
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :return: Dict; updated curve {str; passband : numpy.array; light curve, ...}
    """
    for component in crv_labels:
        velocities[component][pos_idx] = curves.calculate_rv_point(getattr(system, component))

    return velocities
