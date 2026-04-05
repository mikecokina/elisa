from __future__ import annotations

from typing import TYPE_CHECKING

from elisa import umpy as up
from elisa.base.curves import utils as crv_utils
from elisa.base.types import FLOAT

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.types import Float


def _calculate_rv_point(star: StarContainer) -> Float:
    """Calculate the radial-velocity (RV) point for a single component.

    This function computes a flux-weighted radial velocity for the provided
    star container. If the total flux is zero the function returns NaN.

    :param star: Star container with precomputed indices, velocities and other
        per-surface-element data.
    :type star: elisa.base.container.StarContainer
    :returns: Flux-weighted radial velocity or NaN when total flux is zero.
    :rtype: elisa.types.Float
    """
    indices = star.indices
    velocities: NDArray = star.velocities[indices]
    fluxes: NDArray = crv_utils.calculate_surface_element_fluxes("rv_band", star)

    total_flux = up.sum(fluxes)
    if total_flux == 0:
        return FLOAT(up.NaN)

    # velocities[:, 0] selects the line-of-sight velocity component
    # noinspection PyUnresolvedReferences
    rv_value = up.sum(velocities[:, 0] * fluxes) / total_flux
    return FLOAT(rv_value)


def compute_rv_at_pos(
        velocities: dict[str, NDArray],
        pos_idx: int,
        crv_labels: list[str],
        system: OrbitalPositionContainer,
) -> dict[str, NDArray]:
    """Compute RV points for all requested components at a given orbital position.

    This updates the provided ``velocities`` mapping in-place by assigning the
    computed RV value for each component at index ``pos_idx``. The mapping is
    returned for convenience.

    :param velocities: Mapping from component label to an array of RV values.
    :type velocities: dict[str, numpy.typing.NDArray]
    :param pos_idx: Index in the phase/time series at which to store the value.
    :type pos_idx: int
    :param crv_labels: List of component labels to compute (e.g. ["primary", "secondary"]).
    :type crv_labels: list[str]
    :param system: Orbital position container providing per-component star containers.
    :type system: elisa.binary_system.container.OrbitalPositionContainer
    :returns: The updated ``velocities`` mapping.
    :rtype: dict[str, numpy.typing.NDArray]
    """
    for component in crv_labels:
        star = getattr(system, component)
        velocities[component][pos_idx] = _calculate_rv_point(star)
    return velocities


from numba import njit
import math


@njit(fastmath=True)
def accumulate_analytical_lsf(velocities, fluxes, velocity_grid, sigmas):
    n_out = len(velocity_grid)
    n_stars = len(velocities)
    lsf = up.zeros(n_out)
    
    dv = velocity_grid[1] - velocity_grid[0]
    grid_start_edge = velocity_grid[0] - 0.5 * dv
    
    sqrt2 = 1.41421356
    
    for i in range(n_stars):
        v_c = velocities[i]
        flux = fluxes[i]
        sigma = sigmas[i]
        
        calc_sigma = sigma if sigma > 0.1 * dv else 0.1 * dv
        
        v_min = v_c - 4.0 * calc_sigma
        v_max = v_c + 4.0 * calc_sigma
        
        idx_start = int(up.floor((v_min - grid_start_edge) / dv))
        idx_end = int(up.floor((v_max - grid_start_edge) / dv)) + 1
        
        if idx_start < 0: idx_start = 0
        if idx_end > n_out: idx_end = n_out
            
        for j in range(idx_start, idx_end):
            edge_left = grid_start_edge + j * dv
            edge_right = edge_left + dv
            
            denom = 1.0 / (sigma * sqrt2 + 1e-20) 
            
            z_right = (edge_right - v_c) * denom
            z_left = (edge_left - v_c) * denom
            
            weight = 0.5 * (math.erf(z_right) - math.erf(z_left))
            
            lsf[j] += flux * weight  
    return lsf


global indtest_lsf
indtest_lsf = 0


def _calculate_lsf_point(star, velocity_grid):
    indices = getattr(star, 'indices')
    velocities = getattr(star, 'velocities')[indices][:, 0]
    fluxes = crv_utils.calculate_surface_element_fluxes('rv_band', star)
    
    max_v = up.max(velocities)
    min_v = up.min(velocities)
    vsini_approx = (max_v - min_v) / 2

    N = len(getattr(star, 'velocities')[:, 0])
    median_RV = up.median(getattr(star, 'velocities')[:, 0])
    base_dispersion = (2 * vsini_approx) / up.sqrt(N) 
    
    ratio = up.abs(velocities-median_RV) / vsini_approx
    ratio = up.clip(ratio, 0.0, 1.0)
    
    local_sigmas = base_dispersion * up.sqrt(1.0 - ratio**2)

    return accumulate_analytical_lsf(velocities, fluxes, velocity_grid, local_sigmas)
    
    # funclst = [
    #     _calculate_lsf_point_simple,
    #     accumulate_variable_lsf,
    #     accumulate_analytical_lsf,
    # ]

    # global indtest_lsf
    # ind_sel = int(indtest_lsf/2) % 3
    # indtest_lsf = indtest_lsf + 1
    # print('ind_sel =', ind_sel)
    # if ind_sel == 0:
    #     profile = funclst[0](star, velocity_grid)
    # elif ind_sel == 1:
    #     profile = funclst[1](velocities, fluxes, velocity_grid, local_sigmas)
    # else:
    #     profile = funclst[2](velocities, fluxes, velocity_grid, local_sigmas)
    # return profile
    

def _calculate_lsf_point_simple(star, velocity_grid):
    """
    Calculates line spread function (LSF) array for given component.

    :param star: elisa.base.container.StarContainer; star container with all necessary parameters pre-calculated
    :param velocity_grid: numpy.array; grid of velocities for which to calculate LSF array
    :return: numpy.array;
    """
    indices = getattr(star, 'indices')
    velocities = getattr(star, 'velocities')[indices][:, 0]  # shape (N,)
    fluxes = crv_utils.calculate_surface_element_fluxes('rv_band', star)  # shape (N,)

    # Calculate bin edges from velocity_grid
    bin_edges = numpy.concatenate([
        [velocity_grid[0] - (velocity_grid[1] - velocity_grid[0]) / 2],
        (velocity_grid[:-1] + velocity_grid[1:]) / 2,
        [velocity_grid[-1] + (velocity_grid[-1] - velocity_grid[-2]) / 2]
    ])

    # Use numpy.histogram to accumulate fluxes into bins
    lsf_array, _ = up.histogram(velocities, bins=bin_edges, weights=fluxes)

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
    # print('velocity_grid =', velocity_grid)
    if velocity_grid is None:
        raise ValueError("velocity_grid must be provided")

    for component in crv_labels:
        velocities[component][pos_idx] = _calculate_lsf_point(getattr(system, component), velocity_grid)
    return velocities