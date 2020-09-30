def _calculate_lc_point(band, system):
    """
    Calculates point on the light curve for given band.

    :param band: str; name of the photometric band compatibile with supported names in config
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :return: float;
    """
    star = getattr(system, 'star')
    # return crv_utils.flux_from_star_container(band, star)