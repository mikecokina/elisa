from ... import atm, ld
from ... import settings
from ... observer.passband import init_bolometric_passband


def prep_surface_params(system, return_values=True, write_to_containers=False, **kwargs):
    """
    Prepares normal radiance and limb darkening coefficients variables.

    :param system: elisa.single_system.container.SystemContainer;
    :param return_values: bool; return normal radiances and limb darkening coefficients
    :param write_to_containers: bool; calculated values will be assigned to `system` container
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float

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

    if write_to_containers:
        star = getattr(system, 'star')
        setattr(star, 'normal_radiance', normal_radiance['star'])
        setattr(star, 'ld_cfs', ld_cfs['star'])

    return normal_radiance, ld_cfs if return_values else None


def get_normal_radiance(system, **kwargs):
    """
    Compute normal radiance for all faces in SingleOrbitalPositionContainer.

    :param system: elisa.single_system.container.SystemContainer;
    :param kwargs: Dict; arguments to be passed into light curve generator functions
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float

    :return: Dict[String, numpy.array]
    """
    star = system.star
    symmetry_test = not system.has_spots() and not system.has_pulsations()

    # utilizing surface symmetry in case of a clear surface
    if symmetry_test:
        temperatures = star.symmetry_faces(star.temperatures)
        log_g = star.symmetry_faces(star.log_g)
    else:
        temperatures = star.temperatures
        log_g = star.log_g

    retval = {
        'star': atm.NaiveInterpolatedAtm.radiance(
                    **dict(
                        temperature=temperatures,
                        log_g=log_g,
                        metallicity=star.metallicity,
                        atlas=star.atmosphere or settings.ATM_ATLAS,
                        **kwargs
                    )
                )
    }

    if symmetry_test:
        retval['star'] = {band: star.mirror_face_values(vals) for band, vals in retval['star'].items()}

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
    :return: Dict[str, numpy.array];
    """
    star_container = system.star
    symmetry_test = not system.has_spots() and not system.has_pulsations()

    # utilizing surface symmetry in case of a clear surface
    if symmetry_test:
        temperatures = star_container.symmetry_faces(star_container.temperatures)
        log_g = star_container.symmetry_faces(star_container.log_g)
    else:
        temperatures = star_container.temperatures
        log_g = star_container.log_g

    retval = {
        'star': ld.interpolate_on_ld_grid(
                    temperature=temperatures,
                    log_g=log_g,
                    metallicity=star_container.metallicity,
                    passband=kwargs["passband"]
                )
    }

    if symmetry_test:
        retval['star'] = {band: star_container.mirror_face_values(vals) for band, vals in retval['star'].items()}

    return retval
