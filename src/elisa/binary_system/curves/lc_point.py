from ... base.curves import utils as crv_utils
from ... import settings


def _calculate_lc_point(band, system):
    """
    Calculates point on the light curve for given band.

    :param band: str; name of the photometric band compatibile with supported names in config
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :return: float;
    """
    flux = 0.0
    for component in settings.BINARY_COUNTERPARTS:
        star = getattr(system, component)
        flux += crv_utils.flux_from_star_container(band, star)
    return flux


def compute_lc_on_pos(band_curves, pos_idx, passbands, system):
    """
    Calculates lc points for given orbital position.

    :param band_curves: Dict; {str; passband : numpy.array; light curve, ...} result will be written to the
                              corresponding `pos_idx` position
    :param pos_idx: int; position in `band_curves` to which calculated lc points will be assigned
    :param passbands: List; list of passbands
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :return: Dict; updated {str; passband : numpy.array; light curve, ...}
    """
    # integrating resulting flux
    for band in passbands:
        band_curves[band][pos_idx] = _calculate_lc_point(band, system)
    return band_curves
