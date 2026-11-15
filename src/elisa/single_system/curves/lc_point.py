from __future__ import annotations

from typing import TYPE_CHECKING

from elisa.base.curves import utils as crv_utils

# TYPE_CHECKING block at the end of import header
if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.single_system.container import SinglePositionContainer  # pragma: no cover
    from elisa.types import Float, Int  # pragma: no cover


def _calculate_lc_point(band: str, system: SinglePositionContainer) -> Float:
    """Calculate a single point on the light curve for a given band.

    :param band: Name of the photometric band compatible with supported
        names in configuration.
    :type band: str
    :param system: System container at a given orbital position.
    :type system: elisa.single_system.container.SinglePositionContainer

    :returns: Flux value corresponding to the provided band for this system
              position.
    :rtype: elisa.types.Float
    """
    star = system.star
    return crv_utils.flux_from_star_container(band, star)


def compute_lc_on_pos(
    band_curves: dict[str, NDArray[Float]],
    pos_idx: Int,
    crv_labels: list[str],
    system: SinglePositionContainer,
) -> dict[str, NDArray[Float]]:
    """Calculate light-curve points for a given orbital position.

    The function writes computed fluxes into the pre-allocated arrays in
    ``band_curves`` at index ``pos_idx`` for each passband listed in
    ``crv_labels`` and returns the updated mapping.

    :param band_curves: Mapping from passband name to numpy arrays that hold
        the light curve values. Arrays are modified in-place.
    :type band_curves: dict[str, numpy.ndarray]
    :param pos_idx: Index in the arrays corresponding to the current orbital
        position.
    :type pos_idx: elisa.types.Int
    :param crv_labels: Ordered list of passband names to iterate.
    :type crv_labels: list[str]
    :param system: System container at a given orbital position.
    :type system: elisa.single_system.container.SinglePositionContainer

    :returns: The same ``band_curves`` mapping with updated values at
              ``pos_idx``.
    :rtype: dict[str, numpy.ndarray]
    """
    for band in crv_labels:
        band_curves[band][pos_idx] = _calculate_lc_point(band, system)
    return band_curves
