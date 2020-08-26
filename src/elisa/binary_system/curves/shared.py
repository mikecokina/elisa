import numpy as np
from copy import copy

from elisa.conf import config
from elisa import atm, ld, const, utils
from elisa.observer.passband import init_bolometric_passband
from elisa.logger import getLogger
from elisa.binary_system import (
    utils as butils,
    dynamic,
    surface
)
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.observer.mp import manage_observations


logger = getLogger('binary_system.curves.shared')


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
            retval[cpmnt] = {fltr: vals[getattr(system, cpmnt).face_symmetry_vector] for
                             fltr, vals in retval[cpmnt].items()}

    return retval


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


def calculate_lc_point(band, system):
    """
    Calculates point on the light curve for given band.

    :param band: str; name of the photometric band compatibile with supported names in config
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :return: float;
    """
    flux = 0.0
    for component in config.BINARY_COUNTERPARTS.keys():
        star = getattr(system, component)
        flux += flux_from_star_container(band, star)

    return flux


def _calculate_lc_point(band, ld_cfs, normal_radiance, coverage, cosines):
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

    flux = (flux['primary'] + flux['secondary'])
    return flux


def calculate_rv_point(star):
    """
    Calculates point on the rv curve for given component.

    :param star: elisa.base.container.StarContainer; star container with all necessary parameters pre-calculated
    :return: Union[numpy.float, numpy.nan];
    """
    indices = getattr(star, 'indices')
    velocities = getattr(star, 'velocities')[indices]

    fluxes = calculate_surface_element_fluxes('rv_band', star)

    return np.sum(velocities[:, 0] * fluxes) / np.sum(fluxes) if np.sum(fluxes) != 0 else np.NaN


def resolve_curve_method(system, fn_array):
    """
    Resolves which curve calculating method to use based on the type of the system.

    :param system: elisa.binary_system.BinarySystem;
    :param fn_array: tuple; list of curve calculating functions in specific order
    (circular synchronous or circular assynchronous without spots,
     circular assynchronous with spots,
     eccentric synchronous or eccentric assynchronous without spots,
     eccentric assynchronous with spots)
    :return: curve calculating method chosen from `fn_array`
    """
    is_circular = system.eccentricity == 0
    is_eccentric = 1 > system.eccentricity > 0
    assynchronous_spotty_p = system.primary.synchronicity != 1 and system.primary.has_spots()
    assynchronous_spotty_s = system.secondary.synchronicity != 1 and system.secondary.has_spots()
    assynchronous_spotty_test = assynchronous_spotty_p or assynchronous_spotty_s

    spotty_test_eccentric = system.primary.has_spots() or system.secondary.has_spots()

    if is_circular:
        if not assynchronous_spotty_test and not system.has_pulsations():
            logger.debug('Calculating curve for circular binary system without pulsations and without '
                         'asynchronous spotty components.')
            return fn_array[0]
        else:
            logger.debug('Calculating curve for circular binary system with pulsations or with asynchronous '
                         'spotty components.')
            return fn_array[1]
    elif is_eccentric:
        if spotty_test_eccentric:
            logger.debug('Calculating curve for eccentric binary system with spotty components.')
            return fn_array[2]
        else:
            logger.debug('Calculating curve for eccentric binary system without spotty '
                         'components.')
            return fn_array[3]

    raise NotImplementedError("Orbit type not implemented or invalid")


def prep_initial_system(binary):
    """
    Prepares base binary system from which curves will be calculated in case of circular synchronous binaries.

    :param binary: elisa.binary_system.system.BinarySystem
    :return: elisa.binary_system.container.OrbitalPositionContainer
    """
    from_this = dict(binary_system=binary, position=const.Position(0, 1.0, 0.0, 0.0, 0.0))
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)
    initial_system.build(components_distance=1.0)

    return initial_system


def produce_circ_sync_curves(binary, initial_system, phases, curve_fn, crv_labels, **kwargs):
    """
    Auxiliary function to produce curve from circular synchronous binary system

    :param binary: elisa.binary_system.system.BinarySystem;
    :param initial_system: elisa.binary_system.container.OrbitalPositionContainer
    :param phases: numpy.array
    :param curve_fn: function to calculate given type of the curve
    :param crv_labels: labels of the calculated curves (passbands, components,...)
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
            * ** position_method** * - function definition; to evaluate orbital positions
            * ** phases ** * - numpy.array
    :return: dict; calculated curves
    """

    prep_surface_params(initial_system.flatt_it(), return_values=False, write_to_containers=True, **kwargs)

    fn_args = (binary, initial_system, crv_labels, curve_fn)

    curves = manage_observations(fn=produce_circ_sync_curves_mp,
                                 fn_args=fn_args,
                                 position=phases,
                                 **kwargs)

    return curves


def produce_circ_sync_curves_mp(*args):
    """
    Curve generator function for circular synchronous systems.

    :param args: Tuple;

    ::

        Tuple[
                binary: elisa.binary_system.BinarySystem,
                initial_system: elisa.binary_system.container.OrbitalPositionContainer, system container with built
                geometry
                phase_batch: numpy.array; phases at which to calculate curves,
                normal_radiance: Dict; {component: numpy.array; normal radiances for each surface element},
                ld_cfs: Dict;
                crv_labels: List;
                curves_fn: function to calculate curve points at given orbital positions,
                kwargs: Dict,
            ]
    :return:
    """
    binary, initial_system, phase_batch, crv_labels, curves_fn, kwargs = args

    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from='phase')
    # is in eclipse test eval
    ecl_boundaries = dynamic.get_eclipse_boundaries(binary, 1.0)
    azimuths = [position.azimuth for position in orbital_motion]
    in_eclipse = dynamic.in_eclipse_test(azimuths, ecl_boundaries)

    curves = {key: np.zeros(phase_batch.shape) for key in crv_labels}

    for pos_idx, position in enumerate(orbital_motion):
        on_pos = butils.move_sys_onpos(initial_system, position)

        surface.coverage.compute_surface_coverage(on_pos, binary.semi_major_axis, in_eclipse=in_eclipse[pos_idx],
                                                  return_values=False, write_to_containers=True)

        curves = curves_fn(curves, pos_idx, crv_labels, on_pos)

    return curves


def produce_circ_spotty_async_curves(binary, curve_fn, crv_labels, **kwargs):
    """
    Function returns curve of assynchronous systems with circular orbits and spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param curve_fn: curve function
    :param crv_labels: labels of the calculated curves (passbands, components,...)
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa .observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float
        * ** atlas ** * - str
    :return: Dict; fluxes for each filter
    """
    phases = kwargs.pop("phases")
    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phases, return_nparray=False, calculate_from='phase')
    ecl_boundaries = dynamic.get_eclipse_boundaries(binary, 1.0)

    from_this = dict(binary_system=binary, position=const.Position(0, 1.0, 0.0, 0.0, 0.0))
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)

    points = dict()
    for component in config.BINARY_COUNTERPARTS:
        star = getattr(initial_system, component)
        _a, _b, _c, _d = surface.mesh.mesh_detached(initial_system, 1.0, component, symmetry_output=True)
        points[component] = _a
        setattr(star, "points", copy(_a))
        setattr(star, "point_symmetry_vector", _b)
        setattr(star, "base_symmetry_points_number", _c)
        setattr(star, "inverse_point_symmetry_matrix", _d)

    fn_args = binary, initial_system, points, ecl_boundaries, crv_labels, curve_fn

    band_curves = manage_observations(fn=produce_circ_spotty_async_curves_mp,
                                      fn_args=fn_args,
                                      position=orbital_motion,
                                      **kwargs)

    return band_curves


def produce_circ_spotty_async_curves_mp(*args):
    """
    Curve generator function for circular asynchronous spotty systems.

    :param args: Tuple;

    ::

        Tuple[
                binary: elisa.binary_system.BinarySystem,
                initial_system: elisa.binary_system.container.OrbitalPositionContainer, system container with built
                geometry: surface points of a clean system
                phase_batch: numpy.array; phases at which to calculate curves,
                ecl_boundaries: boundaries for both eclipses
                crv_labels: List;
                curves_fn: function to calculate curve points at given orbital positions,
                kwargs: Dict,
            ]

    :return:
    """
    binary, initial_system, motion_batch, base_points, ecl_boundaries, crv_labels, curve_fn, kwargs = args

    # pre-calculate the longitudes of each spot for each phase
    phases = np.array([val.phase for val in motion_batch])
    in_eclipse = dynamic.in_eclipse_test([position.azimuth for position in motion_batch], ecl_boundaries)
    spots_longitudes = dynamic.calculate_spot_longitudes(binary, phases, component="all", correct_libration=False)
    pulsation_tests = {'primary': binary.primary.has_pulsations(),
                       'secondary': binary.secondary.has_pulsations()}
    primary_reducer, secondary_reducer = \
        dynamic.resolve_spots_geometry_update(spots_longitudes, len(phases), pulsation_tests)
    combined_reducer = primary_reducer & secondary_reducer

    # calculating lc with spots gradually shifting their positions in each phase
    curves = {key: np.empty(len(motion_batch)) for key in crv_labels}
    normal_radiance, ld_cfs = None, None
    for pos_idx, orbital_position in enumerate(motion_batch):
        initial_system.set_on_position_params(position=orbital_position)
        initial_system.time = initial_system.set_time()
        # setup component necessary to build/rebuild

        require_build = "all" if combined_reducer[pos_idx] \
            else "primary" if primary_reducer[pos_idx] \
            else "secondary" if secondary_reducer[pos_idx] \
            else None

        # use clear system surface points as a starting place to save a time
        # if reducers for related component is set to False, previous build will be used

        if primary_reducer[pos_idx]:
            initial_system.primary.points = copy(base_points['primary'])
        if secondary_reducer[pos_idx]:
            initial_system.secondary.points = copy(base_points['secondary'])

        # assigning new longitudes for each spot
        dynamic.assign_spot_longitudes(initial_system, spots_longitudes, index=pos_idx, component="all")

        # build the spots points
        surface.mesh.add_spots_to_mesh(initial_system, orbital_position.distance, component=require_build)
        # build the rest of the surface based on preset surface points
        initial_system.build_from_points(components_distance=orbital_position.distance, component=require_build)

        on_pos = butils.move_sys_onpos(initial_system, orbital_position, on_copy=True)

        # if None of components has to be rebuilt, use previously computed radiances and limbdarkening when available
        if require_build is not None:
            normal_radiance, ld_cfs = \
                prep_surface_params(on_pos, return_values=True, write_to_containers=True, **kwargs)
        else:
            for component in config.BINARY_COUNTERPARTS.keys():
                star = getattr(on_pos, component)
                setattr(star, 'normal_radiance', normal_radiance[component])
                setattr(star, 'ld_cfs', ld_cfs[component])

        surface.coverage.compute_surface_coverage(on_pos, binary.semi_major_axis, in_eclipse=in_eclipse[pos_idx],
                                                  return_values=False, write_to_containers=True)

        curves = curve_fn(curves, pos_idx, crv_labels, on_pos)

    return curves
