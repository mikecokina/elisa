from elisa.base.curves import utils as crv_utils


def _calculate_lc_point(band, system):
    """
    Calculates point on the light curve for given band.

    :param band: str; name of the photometric band compatible with supported names in config
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :return: float;
    """
    star = getattr(system, 'star')
    return crv_utils.flux_from_star_container(band, star)


def compute_lc_on_pos(band_curves, pos_idx, crv_labels, system):
    """
    Calculates lc points for given orbital position.

    :param band_curves: Dict; {str; passband : numpy.array; light curve, ...} result will be written to the
                              corresponding `pos_idx` position
    :param pos_idx: int; position in `band_curves` to which calculated lc points will be assigned
    :param crv_labels: List; list of passbands
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :return: Dict; updated {str; passband : numpy.array; light curve, ...}
    """
    # integrating resulting flux
    for band in crv_labels:
        band_curves[band][pos_idx] = _calculate_lc_point(band, system)
    return band_curves
