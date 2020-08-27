import numpy as np

from elisa import atm, ld
from elisa.binary_system import (
    utils as butils,
)
from elisa.conf import config
from elisa.observer.passband import init_bolometric_passband


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
    components = butils.component_to_list(component)

    retval = {}
    for cmpnt in components:
        retval[cmpnt] = ld.interpolate_on_ld_grid(
                    temperature=getattr(system, cmpnt).temperatures,
                    log_g=getattr(system, cmpnt).log_g,
                    metallicity=getattr(system, cmpnt).metallicity,
                    passband=kwargs["passband"]
                )

    return retval


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
            retval[cpmnt] = {fltr: vals[getattr(system, cpmnt).face_symmetry_vector] for
                             fltr, vals in retval[cpmnt].items()}

    return retval


def calculate_surface_element_fluxes(band, star):
    """
        Function generates outgoing flux from each surface element of given star container in certain band.

        :param star: elisa.base.container.StarContainer; star container with all necessary parameters pre-calculated
        :param band: str; name of the photometric band compatibile with supported names in config
        :return: numpy.array
    """
    ld_law_cfs_columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]
    indices = getattr(star, 'indices')
    radiance = getattr(star, 'normal_radiance')[band][indices]
    ld_cfs = getattr(star, 'ld_cfs')[band][ld_law_cfs_columns].values[indices]
    cosines = getattr(star, 'los_cosines')[indices]
    coverage = getattr(star, 'coverage')[indices]

    ld_cors = ld.limb_darkening_factor(coefficients=ld_cfs,
                                       limb_darkening_law=config.LIMB_DARKENING_LAW,
                                       cos_theta=cosines)

    return radiance * cosines * coverage * ld_cors


def flux_from_star_container(band, star):
    """
    Function generates outgoing flux from given star container in certain band.

    :param star: elisa.base.container.StarContainer; star container with all necessary parameters pre-calculated
    :param band: str; name of the photometric band compatibile with supported names in config
    :return:
    """

    return np.sum(calculate_surface_element_fluxes(band, star))


def prep_surface_params(system, return_values=True, write_to_containers=False, **kwargs):
    """
    Prepares normal radiances and limb darkening coefficients variables.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param return_values: bool; return normal radiances and limb darkening coefficients
    :param write_to_containers: bool; calculated values will be assigned to `system` container
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

    if write_to_containers:
        for component in config.BINARY_COUNTERPARTS.keys():
            star = getattr(system, component)
            setattr(star, 'normal_radiance', normal_radiance[component])
            setattr(star, 'ld_cfs', ld_cfs[component])

    return normal_radiance, ld_cfs if return_values else None