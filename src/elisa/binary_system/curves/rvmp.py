import numpy as np

from elisa.binary_system.curves import (
    utils as crv_utils
)


def calculate_rv_point(star):
    """
    Calculates point on the rv curve for given component.

    :param star: elisa.base.container.StarContainer; star container with all necessary parameters pre-calculated
    :return: Union[numpy.float, numpy.nan];
    """
    indices = getattr(star, 'indices')
    velocities = getattr(star, 'velocities')[indices]

    fluxes = crv_utils.calculate_surface_element_fluxes('rv_band', star)

    return np.sum(velocities[:, 0] * fluxes) / np.sum(fluxes) if np.sum(fluxes) != 0 else np.NaN


def compute_circ_sync_rv_at_pos(velocities, pos_idx, crv_labels, system):
    """
    Calculates rv points for given orbital position in case of circular orbit and synchronous rotation.

    :param velocities: Dict; {str; component : numpy.array; rvs, ...}
    :param pos_idx: int; position in `band_curves` to which calculated lc points will be assigned
    :param crv_labels: list; list of components for which to calculate rvs
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :return: Dict; updated {str; passband : numpy.array; rvs, ...}
    """
    for component in crv_labels:
        velocities[component][pos_idx] = calculate_rv_point(getattr(system, component))

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
        velocities[component][pos_idx] = calculate_rv_point(getattr(system, component))

    return velocities
