from __future__ import annotations

from typing import TYPE_CHECKING

from elisa import settings
from elisa.base.curves import utils as crv_utils
from elisa.base.types import FLOAT

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.types import Float, Int


def _calculate_lc_point(
    band: str,
    system: OrbitalPositionContainer,
) -> Float:
    """Calculate a single light-curve point for the given photometric band.

    :param band: Name of the photometric band compatible with configuration.
    :type band: str
    :param system: Orbital-position container.
    :type system: OrbitalPositionContainer
    :return: Integrated flux value.
    :rtype: Float
    """
    flux: Float = FLOAT(0.0)
    for component in settings.BINARY_COUNTERPARTS:
        star = getattr(system, component)
        flux += crv_utils.flux_from_star_container(band, star)
    return flux


def compute_lc_on_pos(
    band_curves: dict[str, NDArray[Float]],
    pos_idx: Int,
    passbands: list[str],
    system: OrbitalPositionContainer,
) -> dict[str, NDArray[Float]]:
    """Calculate light-curve points for a given orbital position.

    The calculated fluxes are written into ``band_curves`` at index
    ``pos_idx`` for each passband.

    :param band_curves: Mapping of passband name to light-curve array.
    :type band_curves: dict[str, NDArray[Float]]
    :param pos_idx: Index in ``band_curves`` where results are stored.
    :type pos_idx: Int
    :param passbands: List of passband names.
    :type passbands: list[str]
    :param system: Orbital-position container.
    :type system: OrbitalPositionContainer
    :return: Updated mapping of passband curves.
    :rtype: dict[str, NDArray[Float]]
    """
    for band in passbands:
        band_curves[band][pos_idx] = _calculate_lc_point(band, system)
    return band_curves
