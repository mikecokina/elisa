from ... import atm, ld
from ... observer.passband import init_bolometric_passband


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
    # obtain limb darkening factor for each face
    ld_cfs = get_limbdarkening_cfs(system, **kwargs)

    # compute normal radiance for each face and each component
    normal_radiance = get_normal_radiance(system, **kwargs)

    # checking if `bolometric`filter is already used
    if 'bolometric' in ld_cfs['star'].keys():
        bol_ld_cfs = {'star': {'bolometric': ld_cfs['star']['bolometric']}}
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
    star = system.star
    symmetry_test = not system.has_spots() and not system.has_pulsations()
    # symmetry_test = False

    # utilizing surface symmetry in case of a clear surface
    if symmetry_test:
        temperatures = star.temperatures[:star.base_symmetry_faces_number]
        log_g = star.log_g[:star.base_symmetry_faces_number]
    else:
        temperatures = star.temperatures
        log_g = star.log_g

    retval = {
        'star': atm.NaiveInterpolatedAtm.radiance(
                    **dict(
                        temperature=temperatures,
                        log_g=log_g,
                        metallicity=star.metallicity,
                        **kwargs
                    )
                )
    }

    if symmetry_test:
        retval['star'] = {filter: vals[star.face_symmetry_vector] for
                          filter, vals in retval['star'].items()}

    return retval


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
