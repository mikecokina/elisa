from elisa import (
    atm,
    ld
)


def prep_surface_params(system, **kwargs):
    """
    Prepares normal radiances and limb darkening coefficients variables.

    :param system: elisa.single_system.container.SystemContainer;
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
    else:
        raise NotImplemented("Pulsations are not fully implemented")
    return normal_radiance, ld_cfs


def get_normal_radiance(system, **kwargs):
    """
    Compute normal radiance for all faces in SingleOrbitalPositionContainer.

    :param system: elisa.single_system.container.SystemContainer;
    :param kwargs: Dict; arguments to be passed into light curve generator functions
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float
        * ** atlas ** * - str
    :return: Dict[String, numpy.array]
    """
    star_container = system.star
    return {
        'star': atm.NaiveInterpolatedAtm.radiance(
                    **dict(
                        temperature=star_container.temperatures,
                        log_g=star_container.log_g,
                        metallicity=star_container.metallicity,
                        **kwargs
                    )
                )
    }


def get_limbdarkening_cfs(system, **kwargs):
    """
    Returns limb darkening coefficients for each face.

    :param system: elisa.single_system.container.SystemContainer;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float
        * ** atlas ** * - str
    :return: Dict[str, numpy.array];
    """
    star_container = system.star

    return {
        'star': ld.interpolate_on_ld_grid(
                    temperature=star_container.temperatures,
                    log_g=star_container.log_g,
                    metallicity=star_container.metallicity,
                    passband=kwargs["passband"]
                )
    }
