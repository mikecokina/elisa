from ... import umpy as np
from elisa.base.curves import utils as crv_utils


def _calculate_rv_point(star):
    """
    Calculates point on the rv curve for given component.

    :param star: elisa.base.container.StarContainer; star container with all necessary parameters pre-calculated
    :return: Union[numpy.float, numpy.nan];
    """
    indices = getattr(star, 'indices')
    velocities = getattr(star, 'velocities')[indices]
    fluxes = crv_utils.calculate_surface_element_fluxes('rv_band', star)
    return np.sum(velocities[:, 0] * fluxes) / np.sum(fluxes) \
        if np.sum(fluxes) != 0 else np.NaN


def compute_rv_at_pos(velocities, pos_idx, crv_labels, system):
    """
    Calculates rv points for given orbital position.

    :param velocities: Dict; {str; component : numpy.array; rvs, ...}
    :param pos_idx: int; position in `band_curves` to which calculated lc points will be assigned
    :param crv_labels: List; list of components for which to calculate rvs
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :return: Dict; updated {str; passband : numpy.array; rvs, ...}
    """
    for component in crv_labels:
        velocities[component][pos_idx] = _calculate_rv_point(getattr(system, component))
    return velocities


def _calculate_lsf_point(star, velocity_grid):
    """
    Calculates line spread function (LSF) array for given component.

    :param star: elisa.base.container.StarContainer; star container with all necessary parameters pre-calculated
    :param velocity_grid: numpy.array; grid of velocities for which to calculate LSF array
    :return: numpy.array;
    """
    indices = getattr(star, 'indices')
    velocities = getattr(star, 'velocities')[indices][:, 0]  # shape (N,)
    fluxes = crv_utils.calculate_surface_element_fluxes('rv_band', star)  # shape (N,)
    lsf_array = np.zeros_like(velocity_grid)

    # Calculate bin edges from velocity_grid
    bin_edges = np.concatenate([
        [velocity_grid[0] - (velocity_grid[1] - velocity_grid[0]) / 2],
        (velocity_grid[:-1] + velocity_grid[1:]) / 2,
        [velocity_grid[-1] + (velocity_grid[-1] - velocity_grid[-2]) / 2]
    ])

    # Use numpy.histogram to accumulate fluxes into bins
    lsf_array, _ = np.histogram(velocities, bins=bin_edges, weights=fluxes)

    return lsf_array


def compute_lsf_at_pos(velocities, pos_idx, crv_labels, system, velocity_grid):
    """
    Calculates line spread function (LSF) array for given orbital position.

    :param velocities: Dict; {str; component : numpy.array; rvs, ...}
    :param pos_idx: int; position in `band_curves` to which calculated lc points will be assigned
    :param crv_labels: List; list of components for which to calculate rvs
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param velocity_grid: numpy.array; grid of velocities for which to calculate lsf points
    :raises ValueError: if velocity_grid is not provided
    :return: Dict; updated {str; passband : numpy.array; lsfs, ...}
    """
    if velocity_grid is None:
        raise ValueError("velocity_grid must be provided")

    for component in crv_labels:
        velocities[component][pos_idx] = _calculate_lsf_point(getattr(system, component), velocity_grid)
    return velocities