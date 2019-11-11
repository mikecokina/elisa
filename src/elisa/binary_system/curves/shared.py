import numpy as np

from elisa.conf import config
from elisa import (
    atm,
    ld
)


def get_limbdarkening_cfs(system, component="all", **kwargs):
    """
    Returns limb darkening coefficients for each face of each component.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param component: str;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float
        * ** atlas ** * - str
    :return: Dict[str, numpy.array];
    """
    if component in ["all", "both"]:
        return {
            component:
                ld.interpolate_on_ld_grid(
                    temperature=getattr(system, component).temperatures,
                    log_g=getattr(system, component).log_g,
                    metallicity=getattr(system, component).metallicity,
                    passband=kwargs["passband"]
                ) for component in config.BINARY_COUNTERPARTS.keys()
        }
    elif component in config.BINARY_COUNTERPARTS.keys():
        return ld.interpolate_on_ld_grid(
            temperature=getattr(system, component).temperatures,
            log_g=getattr(system, component).log_g,
            metallicity=getattr(system, component).metallicity,
            passband=kwargs["passband"]
        )
    else:
        raise ValueError('Invalid value of `component` argument. '
                         'Available parameters are `primary`, `secondary` or `all`.')


def get_normal_radiance(system, component="all", **kwargs):
    """
    Compute normal radiance for all faces and all components in SingleOrbitalPositionContainer.

    :param component: str;
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param kwargs: Dict; arguments to be passed into light curve generator functions
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float
        * ** atlas ** * - str
    :return: Dict[String, numpy.array]
    """
    if component in ["all", "both"]:
        return {
            component:
                atm.NaiveInterpolatedAtm.radiance(
                    **dict(
                        temperature=getattr(system, component).temperatures,
                        log_g=getattr(system, component).log_g,
                        metallicity=getattr(system, component).metallicity,
                        **kwargs
                    )
                ) for component in config.BINARY_COUNTERPARTS
        }
    elif component in config.BINARY_COUNTERPARTS:
        return atm.NaiveInterpolatedAtm.radiance(
            **dict(
                temperature=getattr(system, component).temperatures,
                log_g=getattr(system, component).log_g,
                metallicity=getattr(system, component).metallicity,
                **kwargs
            )
        )
    else:
        raise ValueError('Invalid value of `component` argument.\n'
                         'Available parameters are `primary`, `secondary` or `all`.')


def prep_surface_params(system, **kwargs):
    """
    Prepares normal radiances and limb darkening coefficients variables.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float
        * ** atlas ** * - str
    :return:
    """

    if not system.has_pulsations():
        # compute normal radiance for each face and each component
        normal_radiance = get_normal_radiance(system, **kwargs)
        # obtain limb darkening factor for each face
        ld_cfs = get_limbdarkening_cfs(system, **kwargs)
    elif not system.primary.has_pulsations():
        normal_radiance = {'primary': get_normal_radiance(system, 'primary', **kwargs)}
        ld_cfs = {'primary': get_limbdarkening_cfs(system, 'primary', **kwargs)}
    elif not system.secondary.has_pulsations():
        normal_radiance = {'secondary': get_normal_radiance(system, 'secondary', **kwargs)}
        ld_cfs = {'secondary': get_limbdarkening_cfs(system, 'secondary', **kwargs)}
    else:
        raise NotImplemented("Pulsations are not fully implemented")
    return normal_radiance, ld_cfs


def calculate_lc_point(band, ld_cfs, normal_radiance, coverage, cosines):
    """
    Calculates point on the light curve for given band.

    :param band: str; name of the photometric band
    :param ld_cfs: Dict[str, Dict[str, pandas.DataFrame]];
    :param normal_radiance: Dict[str, Dict[str, numpy.array]];
    :param coverage: Dict[str, Dict[str, numpy.array]];
    :param cosines: Dict[str, Dict[str, numpy.array]];
    :return: float;
    """
    ld_law_cfs_columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]
    ld_cors = {
        component: ld.limb_darkening_factor(coefficients=ld_cfs[component][band][ld_law_cfs_columns].values,
                                            limb_darkening_law=config.LIMB_DARKENING_LAW,
                                            cos_theta=cosines[component])
        for component in config.BINARY_COUNTERPARTS
    }
    flux = {
        component:
            np.sum(normal_radiance[component][band] * cosines[component] * coverage[component] * ld_cors[component])
        for component in config.BINARY_COUNTERPARTS
    }
    flux = flux['primary'] + flux['secondary']
    return flux
