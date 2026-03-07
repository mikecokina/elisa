from __future__ import annotations

import json
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from elisa import settings
from elisa import umpy as up
from elisa.logger import getLogger

logger = getLogger("base.surface.temperature")

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from elisa.base.container import StarContainer
    from elisa.types import Float


def calculate_effective_temperatures(
        star_container: StarContainer,
        gradient_magnitudes: ArrayLike | NDArray,
) -> NDArray[np.floating]:
    """Calculate effective temperatures from gradient magnitudes.

    If a symmetry is present on the mesh, returned temperatures are
    expanded to the full surface using :meth:`StarContainer.mirror_face_values`.

    :param star_container: StarContainer instance describing the surface.
    :type star_container: elisa.base.container.StarContainer
    :param gradient_magnitudes: Per-face gradient magnitudes used to compute
        local effective temperatures.
    :type gradient_magnitudes: numpy.typing.ArrayLike | numpy.typing.NDArray
    :returns: Per-face effective temperatures (expanded to full surface if symmetry).
    :rtype: numpy.typing.NDArray
    """
    sc: StarContainer = star_container
    t_eff_polar = calculate_polar_effective_temperature(star_container)
    t_eff = t_eff_polar * up.power(
        gradient_magnitudes / sc.polar_potential_gradient_magnitude,
        0.25 * sc.gravity_darkening,
    )

    if star_container.symmetry_test():
        return star_container.mirror_face_values(np.asarray(t_eff))

    return np.asarray(t_eff)


def calculate_polar_effective_temperature(star_container: StarContainer) -> Float:
    """Compute the polar effective temperature of a star (scalar).

    The polar effective temperature is obtained by conserving the total
    bolometric flux over the stellar surface (including spots when
    present) following the gravity-darkening prescription.

    :param star_container: StarContainer instance.
    :type star_container: elisa.base.container.StarContainer
    :returns: Polar effective temperature (scalar).
    :rtype: elisa.types.Float
    """
    sc: StarContainer = star_container
    areas = copy(sc.areas)
    potential_gradient_magnitudes = sc.potential_gradient_magnitudes
    if sc.has_spots():
        for spot in sc.spots.values():
            areas = up.concatenate((areas, spot.areas), axis=0)
            potential_gradient_magnitudes = up.concatenate(
                (potential_gradient_magnitudes, spot.potential_gradient_magnitudes),
                axis=0,
            )

    numerator = np.sum(areas)
    denominator = np.sum(
        areas
        * up.power(
            potential_gradient_magnitudes / sc.polar_potential_gradient_magnitude,
            sc.gravity_darkening,
        ),
    )
    # avoid division by zero
    if denominator == 0:
        msg = "Division by zero encountered while computing polar effective temperature"
        raise ZeroDivisionError(msg)

    return sc.t_eff * up.power(numerator / denominator, 0.25)


def renormalize_temperatures(star: StarContainer) -> None:
    """Renormalize surface temperatures so that the global effective temperature is conserved.

    When spots are present the total radiated flux may change; this function
    rescales per-face temperatures to ensure the star's bolometric
    effective temperature (``star.t_eff``) is preserved.

    :param star: StarContainer instance to renormalize in-place.
    :type star: elisa.base.container.StarContainer
    :returns: None
    :rtype: None
    """
    sc: StarContainer = star
    total_surface = np.sum(sc.areas)
    if sc.has_spots():
        for spot in sc.spots.values():
            total_surface += np.sum(spot.areas)

    desired_flux_value = total_surface * up.power(sc.t_eff, 4)

    current_flux = np.sum(sc.areas * up.power(sc.temperatures, 4))
    if sc.spots:
        for spot in sc.spots.values():
            current_flux += np.sum(spot.areas * up.power(spot.temperatures, 4))

    # avoid division by zero
    if current_flux == 0:
        msg = "Current bolometric flux is zero, cannot renormalize temperatures"
        raise ZeroDivisionError(msg)

    coefficient = up.power(desired_flux_value / current_flux, 0.25)
    logger.debug("surface temperature map renormalized by a factor %s", coefficient)

    sc.temperatures *= coefficient
    if sc.spots:
        for spot in sc.spots.values():
            spot.temperatures *= coefficient


def interpolate_bolometric_gravity_darkening(temperature: Float) -> Float:
    """Interpolate bolometric gravity-darkening exponent beta from tabulated data.

    A small table (log10(T) vs beta) is stored in JSON at
    :data:`elisa.settings.PATH_TO_BETA` and is used to interpolate a
    continuous beta(T) following Claret (2003, A&A 406, 623-628).

    :param temperature: Effective temperature (K).
    :type temperature: elisa.types.Float
    :returns: Interpolated gravity-darkening exponent beta.
    :rtype: elisa.types.Float
    :raises ValueError: If ``temperature`` is not positive.
    :raises RuntimeError: If the beta data file cannot be read or parsed.
    """
    if temperature <= 0:
        msg = "Negative or zero temperature encountered"
        raise ValueError(msg)

    try:
        content = Path(settings.PATH_TO_BETA).read_text()
        data = json.loads(content)
    except Exception as err:
        msg = f"Failed to read or parse gravity-darkening table: {err}"
        raise RuntimeError(msg) from err

    interp_temps = data.get("x")
    interp_betas = data.get("y")

    # basic validation of data
    if interp_temps is None or interp_betas is None:
        msg = "Gravity-darkening table is missing required keys 'x'/'y'"
        raise RuntimeError(msg)

    return float(
        np.interp(
            np.log10(float(temperature)),
            np.asarray(interp_temps),
            np.asarray(interp_betas),
        ),
    )
