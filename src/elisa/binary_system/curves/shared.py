import numpy as np

from elisa.conf import config
from elisa import (
    atm,
    ld,
    const
)
from elisa.binary_system import utils as butils
from elisa.observer.utils import init_bolometric_passband


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
    :return: Dict[String, dict]
    """
    components = butils.component_to_list(component)
    symmetry_test = {cmpnt: not getattr(system, cmpnt).has_spots() and not getattr(system, cmpnt).has_pulsations() for
                     cmpnt in components}
    temperatures, log_g = {}, {}

    # utilizing surface symmetry in case of a clear surface
    for cmpnt in components:
        component_instance = getattr(system, cmpnt)
        if symmetry_test[cmpnt]:
            temperatures[cmpnt] = component_instance.temperatures[:component_instance.base_symmetry_faces_number]
            log_g[cmpnt] = component_instance.log_g[:component_instance.base_symmetry_faces_number]
        else:
            temperatures[cmpnt] = component_instance.temperatures
            log_g[cmpnt] = component_instance.log_g

    retval = {
        cpmnt:
            atm.NaiveInterpolatedAtm.radiance(
                **dict(
                    temperature=temperatures[cpmnt],
                    log_g=log_g[cpmnt],
                    metallicity=getattr(system, cpmnt).metallicity,
                    **kwargs
                )
            ) for cpmnt in components
    }

    # mirroring symmetrical part back to the rest of the surface
    for cpmnt in components:
        if symmetry_test[cpmnt]:
            retval[cpmnt] = {filter: vals[getattr(system, cpmnt).face_symmetry_vector] for
                             filter, vals in retval[cpmnt].items()}

    return retval


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
    # obtain limb darkening factor for each face
    ld_cfs = get_limbdarkening_cfs(system, **kwargs)
    # compute normal radiance for each face and each component
    normal_radiance = get_normal_radiance(system, **kwargs)

    # checking if `bolometric`filter is already used
    if 'bolometric' in ld_cfs['primary'].keys():
        bol_ld_cfs = {component: {'bolometric': ld_cfs[component]['bolometric']} for component in
                      config.BINARY_COUNTERPARTS.keys()}
    else:
        passband, left_bandwidth, right_bandwidth = init_bolometric_passband()
        bol_kwargs = {
            'passband': {'bolometric': passband},
            'left_bandwidth': left_bandwidth,
            'right_bandwith': right_bandwidth,
            'atlas': 'whatever'
        }
        bol_ld_cfs = get_limbdarkening_cfs(system, **bol_kwargs)

    normal_radiance = atm.correct_normal_radiance_to_optical_depth(normal_radiance, bol_ld_cfs)
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

    # parameter 1 / PI converts to astrophysical flux
    flux = (flux['primary'] + flux['secondary']) / const.PI
    return flux
