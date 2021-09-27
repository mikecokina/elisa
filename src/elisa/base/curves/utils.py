from ... import settings, ld, umpy as up
from ... observer.passband import init_rv_passband


def include_passband_data_to_kwargs(**kwargs):
    """
    Including dummy passband from which radiometric radial velocities will be calculated.

    :param kwargs: Tuple;
    :return: Tuple;
    """
    psbnd, right_bandwidth, left_bandwidth = init_rv_passband()
    kwargs.update({
        'passband': {'rv_band': psbnd},
        'left_bandwidth': left_bandwidth,
        'right_bandwidth': right_bandwidth
    })
    return kwargs


def calculate_surface_element_fluxes(band, star):
    """
    Function generates outgoing flux from each surface element of given star container in certain band.

    :param star: elisa.base.container.StarContainer; star container with all necessary parameters pre-calculated
    :param band: str; name of the photometric band compatible with supported names in config
    :return: numpy.array
    """
    indices = star.indices
    radiance = star.normal_radiance[band][indices]
    ld_cfs = star.ld_cfs[band][indices]
    cosines = star.los_cosines[indices]
    coverage = star.coverage[indices]

    ld_cors = ld.limb_darkening_factor(coefficients=ld_cfs,
                                       limb_darkening_law=settings.LIMB_DARKENING_LAW,
                                       cos_theta=cosines)

    return radiance * cosines * coverage * ld_cors


def flux_from_star_container(band, star):
    """
    Function generates outgoing flux from given star container in certain band.

    :param star: elisa.base.container.StarContainer; star container with all necessary parameters pre-calculated
    :param band: str; name of the photometric band compatible with supported names in config,
    :return: numpy.array;
    """

    return up.sum(calculate_surface_element_fluxes(band, star))
