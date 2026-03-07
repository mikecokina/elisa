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
