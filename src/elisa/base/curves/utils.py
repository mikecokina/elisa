from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import ld, settings
from elisa import umpy as up
from elisa.base.types import FLOAT, INT
from elisa.observer.passband import init_rv_passband

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.types import Float


@lru_cache(maxsize=1)
def _get_rv_passband() -> tuple[Any, Any, Any]:
    """Return cached RV passband tuple (passband, right_bw, left_bw).

    Caches the result of :func:`init_rv_passband` using an LRU cache so
    repeated callers avoid reinitialization. This avoids module-level
    mutable globals while providing the same performance benefit.
    """
    return init_rv_passband()


def _ndarray_to_hashable(arr: NDArray) -> tuple[float, ...]:
    """Convert a numpy array to a stable, hashable tuple for caching.

    Round to 9 decimal places to avoid tiny floating point noise causing
    unnecessary cache misses. Use vectorized numpy ops to minimize Python-level
    overhead for large arrays.
    """
    a = np.asarray(arr).ravel()
    if a.size == 0:
        return ()
    rounded = np.round(a, 9).astype(float)
    return tuple(rounded.tolist())


@lru_cache(maxsize=128)
def _interpolate_on_ld_grid_cached(
    temps_key: tuple[float, ...],
    logg_key: tuple[float, ...],
    metallicity: float,
    passbands_key: tuple[str, ...],
) -> dict[str, NDArray]:
    """Cache result of :func:`ld.interpolate_on_ld_grid` for repeated inputs.

    Parameters are passed as hashable keys (tuples) to allow caching.
    """
    temps = np.asarray(temps_key, dtype=float)
    logg = np.asarray(logg_key, dtype=float)
    return ld.interpolate_on_ld_grid(
        temperature=temps,
        log_g=logg,
        metallicity=metallicity,
        passband=list(passbands_key),
    )


def include_passband_data_to_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Include a radial-velocity passband object and bandwidths into kwargs.

    Initialize a minimal RV passband and insert the passband instance and its
    left/right bandwidth values into :attr:`kwargs` under the standardized
    keys ``passband``, ``left_bandwidth`` and ``right_bandwidth``.
    """
    psbnd, right_bandwidth, left_bandwidth = _get_rv_passband()
    kwargs.update(
        {
            "passband": {"rv_band": psbnd},
            "left_bandwidth": left_bandwidth,
            "right_bandwidth": right_bandwidth,
        },
    )
    return kwargs


def calculate_surface_element_fluxes(band: str, star: StarContainer) -> NDArray:
    """Generate outgoing flux for each surface element of a star in a band.

    The returned array contains the flux contribution of each surface element
    computed as radiance * cos(theta) * coverage * limb-darkening correction.
    """
    indices = star.indices
    radiance = star.normal_radiance[band][indices]
    ld_cfs = star.ld_cfs[band][indices]
    cosines = star.los_cosines[indices]
    coverage = star.coverage[indices]

    ld_cors = ld.limb_darkening_factor(
        coefficients=ld_cfs,
        limb_darkening_law=settings.LIMB_DARKENING_LAW,
        cos_theta=cosines,
    )

    # Combine multiplicative terms to reduce temporary allocations:
    prod = cosines * coverage
    prod *= ld_cors
    return radiance * prod


def flux_from_star_container(band: str, star: StarContainer) -> Float:
    """Compute the integrated flux from a star container in a given band.

    The function sums per-surface-element flux contributions computed by
    :func:`calculate_surface_element_fluxes` and returns the total as the
    project's FLOAT type.
    """
    total = up.sum(calculate_surface_element_fluxes(band, star))
    return FLOAT(total)


def generate_teff_logg_for_ld_cfs(
    component_instance: StarContainer,
    *,
    symmetry_test: bool,
) -> tuple[NDArray, NDArray]:
    """Return temperature and log(g) arrays for limb-darkening interpolation.

    Depending on configuration and the symmetry test this returns either a
    single temperature/log_g pair (when single coefficients are used) or
    arrays corresponding to the star's surface faces.
    """
    if settings.USE_SINGLE_LD_COEFFICIENTS:
        temperatures = np.array([component_instance.t_eff])
        log_g = np.array([np.max(component_instance.log_g)])
    elif symmetry_test:
        temperatures = component_instance.symmetry_faces(component_instance.temperatures)
        log_g = component_instance.symmetry_faces(component_instance.log_g)
    else:
        temperatures = component_instance.temperatures
        log_g = component_instance.log_g

    return temperatures, log_g


def get_component_limbdarkening_cfs(
    component_instance: StarContainer,
    passbands: list[str],
    *,
    symmetry_test: bool,
) -> dict[str, NDArray]:
    """Return limb-darkening coefficients for a component across passbands.

    If the component provides explicit limb-darkening coefficients, they are
    tiled or mirrored to match the number of surface faces. Otherwise, the
    coefficients are interpolated from the limb-darkening grid.
    """
    if component_instance.limb_darkening_coefficients is not None:
        desired_repeats = (component_instance.temperatures.shape[0], 1)
        try:
            # tile per-passband arrays to faces count (avoid unnecessary copies)
            base_ld = component_instance.limb_darkening_coefficients
            faces = desired_repeats[0]
            ld_cfs = {}
            for passband in passbands:
                arr = np.asarray(base_ld[passband])
                # If arr is 1-D (coeffs,), reshape to (1, coeffs) and broadcast to (faces, coeffs)
                if arr.ndim == 1:
                    cols = arr.shape[0]
                    ld_cfs[passband] = np.broadcast_to(arr.reshape(1, cols), (faces, cols))
                else:
                    # fallback to tile for non-1D inputs to preserve existing behaviour
                    ld_cfs[passband] = np.tile(arr, desired_repeats)
        except KeyError as err:
            ld_passband = component_instance.limb_darkening_coefficients.keys()
            missing_passbands = list(set(passbands) - set(ld_passband))
            msg = f"Please supply limb-darkening factors for {missing_passbands} pasband(s) as well."
            raise KeyError(msg) from err
    else:
        temperatures, log_g = generate_teff_logg_for_ld_cfs(component_instance, symmetry_test=symmetry_test)

        # Use cached interpolation helper to avoid repeated expensive interpolation
        temps_key = _ndarray_to_hashable(temperatures)
        logg_key = _ndarray_to_hashable(log_g)
        passbands_key = tuple(passbands)
        ld_cfs = _interpolate_on_ld_grid_cached(
            temps_key,
            logg_key,
            float(component_instance.metallicity),
            passbands_key,
        )

        if symmetry_test:
            if settings.USE_SINGLE_LD_COEFFICIENTS:
                _zeros_idx = np.zeros(component_instance.temperatures.shape, dtype=INT)
                ld_cfs = {fltr: vals[_zeros_idx] for fltr, vals in ld_cfs.items()}
            else:
                ld_cfs = {fltr: component_instance.mirror_face_values(vals) for fltr, vals in ld_cfs.items()}

    return ld_cfs
