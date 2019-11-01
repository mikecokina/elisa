import numpy as np
from elisa import const, atm, ld
from elisa.base.container import StarContainer
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.orbit.container import OrbitalSupplements
from elisa.pulse import pulsations
from elisa.binary_system import utils as bsutils
from elisa.conf import config
from elisa.conf.config import BINARY_COUNTERPARTS
from elisa.const import BINARY_POSITION_PLACEHOLDER
from scipy.interpolate import Akima1DInterpolator
from copy import copy, deepcopy
from elisa import utils

from elisa import logger

config.set_up_logging()
__logger__ = logger.getLogger(__name__)



def compute_circular_synchronous_lightcurve(system, **kwargs):
    """
    Compute light curve, exactly, from position to position, for synchronous circular
    binary system.

    :param system: elisa.binary_system.system.BinarySystem
    :param kwargs: Dict;
    * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
            * ** position_method** * - function definition; to evaluate orbital positions
    :return: Dict[str, numpy.array]
    """
    ecl_boundaries = geo.get_eclipse_boundaries(system, 1.0)

    orbital_position_container = OrbitalPositionContainer(
        primary=StarContainer.from_properties_container(system.primary.to_properties_container()),
        secondary=StarContainer.from_properties_container(system.secondary.to_properties_container()),
        position=BINARY_POSITION_PLACEHOLDER(*(0, 1.0, 0.0, 0.0, 0.0)),
        **system.properties_serializer()

    )
    system.build(components_distance=1.0)
    # in case of LC for spotless surface without pulsations unique phase interval is only (0, 0.5)
    phases = kwargs.pop("phases")
    unique_phase_interval, reverse_phase_map = _phase_crv_symmetry(system, phases)

    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=unique_phase_interval, return_nparray=False, calculate_from='phase')

    initial_props_container = geo.SingleOrbitalPositionContainer(system.primary, system.secondary)
    initial_props_container.setup_position(BINARY_POSITION_PLACEHOLDER(*(0, 1.0, 0.0, 0.0, 0.0)), system.inclination)

    ld_law_cfs_columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]

    params = dict(orbital_motion=orbital_motion, ecl_boundaries=ecl_boundaries)
    system_positions_container = system.prepare_system_positions_container(**params)
    system_positions_container = system_positions_container.darkside_filter()

    pulsations_test = {'primary': system.primary.has_pulsations(), 'secondary': system.secondary.has_pulsations()}
    normal_radiance, ld_cfs = prep_surface_params(initial_props_container, pulsations_test, **kwargs)

    band_curves = {key: np.zeros(unique_phase_interval.shape) for key in kwargs["passband"].keys()}
    for idx, container in enumerate(system_positions_container):
        # dict of components
        star_containers = {component: getattr(container, component) for component in BINARY_COUNTERPARTS}

        coverage = compute_surface_coverage(container, system.semi_major_axis, in_eclipse=system_positions_container.in_eclipse[idx])

        # calculating cosines between face normals and line of sight
        cosines, visibility_test = dict(), dict()
        for component, star_container_instance in star_containers.items():
            cosines[component] = utils.calculate_cos_theta_los_x(star_container_instance.normals)
            visibility_test[component] = cosines[component] > 0
            cosines[component] = cosines[component][visibility_test[component]]

            # calculating temperature perturbation due to pulsations
            if pulsations_test[component]:
                com_x = None if component == 'primary' else 1.0
                component_instance = getattr(system, component)
                star_container_instance.temperatures += \
                    pulsations.calc_temp_pert_on_container(component_instance,
                                                           star_container_instance,
                                                           orbital_motion[idx].phase,
                                                           system.period,
                                                           com_x=com_x)
                normal_radiance[component] = get_normal_radiance(container, component=component, **kwargs)
                ld_cfs[component] = get_limbdarkening_cfs(container, component=component, **kwargs)

        # integrating resulting flux
        for band in kwargs["passband"].keys():
            flux, ld_cors = np.empty(2), {}
            for ii, component in enumerate(config.BINARY_COUNTERPARTS.keys()):
                ld_cors[component] = \
                    ld.limb_darkening_factor(
                        coefficients=ld_cfs[component][band][ld_law_cfs_columns].values[visibility_test[component]],
                        limb_darkening_law=config.LIMB_DARKENING_LAW,
                        cos_theta=cosines[component])

                flux[ii] = np.sum(normal_radiance[component][band][visibility_test[component]] *
                                  cosines[component] *
                                  coverage[component][visibility_test[component]] *
                                  ld_cors[component])
            band_curves[band][idx] = np.sum(flux)

    band_curves = {band: band_curves[band][reverse_phase_map] for band in band_curves}

    return band_curves


def _look_for_approximation(phases_span_test, not_pulsations_test):
    """
    This condition checks if even to attempt to utilize apsidal line symmetry approximations.

    :param not_pulsations_test: bool
    :param phases_span_test: bool
    :return: bool
    """
    return config.POINTS_ON_ECC_ORBIT > 0 and config.POINTS_ON_ECC_ORBIT is not None \
           and phases_span_test and not_pulsations_test


def _eval_approximation_one(self, phases):
    """
    Test if it is appropriate to compute eccentric binary system with approximation approax one.

    :param self: elisa.binary_system.system.BinaryStar
    :param phases: numpy.array
    :return: bool
    """
    if len(phases) > config.POINTS_ON_ECC_ORBIT and self.is_synchronous():
        return True
    return False


def _eval_approximation_two(self, rel_d):
    """
    Test if it is appropriate to compute eccentric binary system with approximation approax two.

    :param self: elisa.binary_system.system.BinaryStar
    :param rel_d: numpy.array
    :return: bool
    """
    # defined bodies/objects/tempaltes in orbital supplements instance are sorted by distance,
    # what means that also radii `rel_d` computed from such values have to be already sorted by
    # their own size (radius changes based on components distance and it is monotonic function)

    if np.max(rel_d[:, 1:]) < config.MAX_RELATIVE_D_R_POINT and self.is_synchronous():
        return True
    return False


def _compute_rel_d_radii(self, orbital_supplements):
    """
    Requires `orbital_supplements` sorted by distance.

    :param self: elisa.binary_system.system.BinarySystem
    :param orbital_supplements:
    :return: numpy.array
    """
    # note: defined bodies/objects/templates in orbital supplements instance are sorted by distance (line above),
    # what means that also radii computed from such values have to be already sorted by their own size (radius changes
    # based on components distance and it is, on the half of orbit defined by apsidal line, monotonic function)
    fwd_radii = self.calculate_all_forward_radii(orbital_supplements.body[:, 1], components='all')
    fwd_radii = np.array(list(fwd_radii.values()))
    return np.abs(fwd_radii[:, 1:] - fwd_radii[:, :-1]) / fwd_radii[:, 1:]


def _resolve_ecc_approximation_method(self, phases, position_method, try_to_find_appx, **kwargs):
    """
    Resolve and return approximation method to compute lightcurve in case of eccentric orbit.
    Return value is lambda function with already prepared params.

    :param self: elisa.binary_system.system.BinarySystem
    :param phases: numpy.array
    :param position_method: function
    :param try_to_find_appx: bool
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return: lambda
    """
    params = dict(input_argument=phases, return_nparray=True, calculate_from='phase')
    all_orbital_pos_arr = position_method(**params)
    all_orbital_pos = utils.convert_binary_orbital_motion_arr_to_positions(all_orbital_pos_arr)

    azimuths = all_orbital_pos_arr[:, 2]
    reduced_phase_ids, counterpart_postion_arr, reduced_phase_mask = _prepare_geosymmetric_orbit(self, azimuths, phases)

    # spliting orbital motion into two separate groups on different sides of apsidal line
    reduced_orbit_arr, reduced_orbit_supplement_arr = _split_orbit_by_apse_line(all_orbital_pos_arr, reduced_phase_mask)

    # APPX ZERO ********************************************************************************************************
    if not try_to_find_appx:
        return 'zero', \
               lambda: _integrate_eccentric_lc_exactly(self, all_orbital_pos, phases, None, **kwargs)








    # APPX ONE *********************************************************************************************************
    appx_one = _eval_approximation_one(self, phases)

    if appx_one:
        orbital_supplements = OrbitalSupplements(body=reduced_orbit_arr, mirror=counterpart_postion_arr)
        orbital_supplements.sort(by='distance')
        rel_d_radii = _compute_rel_d_radii(self, orbital_supplements)
        new_geometry_mask = _resolve_object_geometry_update(self.has_spots(), orbital_supplements.size(), rel_d_radii)

        return 'one', \
               lambda: _integrate_eccentric_lc_appx_one(self, phases, orbital_supplements, new_geometry_mask, **kwargs)

    # APPX TWO *********************************************************************************************************

    # create object of separated objects and supplements to bodies
    orbital_supplements = find_apsidally_corresponding_positions(reduced_orbit_arr[:, 1],
                                                                 reduced_orbit_arr,
                                                                 reduced_orbit_supplement_arr[:, 1],
                                                                 reduced_orbit_supplement_arr,
                                                                 tol=config.MAX_SUPPLEMENTAR_D_DISTANCE)

    orbital_supplements.sort(by='distance')
    rel_d_radii = _compute_rel_d_radii(self, orbital_supplements)
    appx_two = _eval_approximation_two(self, rel_d_radii)
    new_geometry_mask = _resolve_object_geometry_update(self.has_spots(), orbital_supplements.size(), rel_d_radii)

    if appx_two:
        return 'two', \
               lambda: _integrate_eccentric_lc_appx_two(self, phases, orbital_supplements, new_geometry_mask, **kwargs)
    # APPX ZERO once again *********************************************************************************************
    else:
        return 'zero', \
               lambda: _integrate_eccentric_lc_exactly(self, all_orbital_pos, phases, ecl_boundaries=None, **kwargs)


def compute_eccentric_lightcurve(self, **kwargs):
    """
    Top-level helper method to compute eccentric lightcurve.

    :param self: elisa.binary_star.system.BinarySystem
    :param kwargs: Dict;

    :return: Dict[str, numpy.array]
    """
    phases = kwargs.pop("phases")
    phases_span_test = np.max(phases) - np.min(phases) >= 0.8

    position_method = kwargs.pop("position_method")

    # this condition checks if even to attempt to utilize apsidal line symmetry approximations
    # curve has to have enough point on orbit and have to span at least in 0.8 phase

    try_to_find_appx = _look_for_approximation(phases_span_test, not self.has_pulsations())
    appx_uid, run = _resolve_ecc_approximation_method(self, phases, position_method, try_to_find_appx, **kwargs)

    logger_messages = {
        'zero': 'lc will be calculated in a rigorous `phase to phase manner` without approximations',
        'one': 'one half of the points on LC on the one side of the apsidal line will be interpolated',
        'two': 'geometry of the stellar surface on one half of the apsidal '
               'line will be copied from their symmetrical counterparts'
    }

    self._logger.info(logger_messages.get(appx_uid))
    return run()


# todo: unittest this method

def get_onpos_container(self, orbital_position, ecl_boundaries):
    """
    Prepares a postion container for given orbital position
    where visibe/non visible faces are calculated and metallicities are assigned.

    :param self: elisa.binary_system.system.BinarySystem
    :param orbital_position: collections.namedtuple; elisa.const.Position
    :param ecl_boundaries: numpy.array; orbital azimuths of eclipses
    :return: container; elisa.binary_system.geo.SingleOrbitalPositionContainer
    """
    system_positions_container = self.prepare_system_positions_container(orbital_motion=[orbital_position],
                                                                         ecl_boundaries=ecl_boundaries)
    system_positions_container = system_positions_container.darkside_filter()
    container = next(iter(system_positions_container))

    # injected attributes
    setattr(container.primary, 'metallicity', self.primary.metallicity)
    setattr(container.secondary, 'metallicity', self.secondary.metallicity)
    return container


def _integrate_eccentric_lc_appx_one(self, phases, orbital_supplements, new_geometry_mask, **kwargs):
    """
    Function calculates light curves for eccentric orbits for selected filters using approximation
    where light curve points on the one side of the apsidal line are calculated exactly and the second
    half of the light curve points are calculated by mirroring the surface geometries of the first
    half of the points to the other side of the apsidal line. Since those mirrored
    points are no alligned with desired phases, the fluxes for each phase is interpolated if missing.

    :param self: elisa.binary_system.system.BinarySystem
    :param phases: numpy.array
    :param orbital_supplements: elisa.binary_system.geo.OrbitalSupplements
    :param new_geometry_mask: numpy.array
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return: Dict[str, numpy.array]
    """
    band_curves = {key: list() for key in kwargs["passband"]}
    band_curves_body, band_curves_mirror = deepcopy(band_curves), deepcopy(band_curves)

    # surface potentials with constant volume of components
    potentials = self.correct_potentials(orbital_supplements.body[:, 4], component="all", iterations=2)

    # both, body and mirror should be defined in this approximation (those points are created in way to be mirrored
    # one to another), if it is not defined, there is most likely issue with method `_prepare_geosymmetric_orbit`
    for idx, position_pair in enumerate(orbital_supplements):
        body, mirror = position_pair
        body_orb_pos, mirror_orb_pos = utils.convert_binary_orbital_motion_arr_to_positions([body, mirror])

        require_geometry_rebuild = new_geometry_mask[idx]

        self.primary.surface_potential = potentials['primary'][idx]
        self.secondary.surface_potential = potentials['secondary'][idx]

        self = _update_surface_in_ecc_orbits(self, body_orb_pos, require_geometry_rebuild)

        container_body = get_onpos_container(self, body_orb_pos, ecl_boundaries=None)
        container_mirror = get_onpos_container(self, mirror_orb_pos, ecl_boundaries=None)

        normal_radiance = get_normal_radiance(container_body, **kwargs)
        ld_cfs = get_limbdarkening_cfs(container_body, **kwargs)

        container_body.coverage, \
            container_body.cosines = calculate_surface_parameters(container_body, self.semi_major_axis, in_eclipse=True)
        container_mirror.coverage, \
            container_mirror.cosines = calculate_surface_parameters(container_mirror, self.semi_major_axis,
                                                                    in_eclipse=True)

        for band in kwargs["passband"].keys():
            band_curves_body[band].append(calculate_lc_point(container_body, band, ld_cfs, normal_radiance))
            band_curves_mirror[band].append(calculate_lc_point(container_mirror, band, ld_cfs, normal_radiance))

    # interpolation of the points in the second half of the light curves using splines
    x = np.concatenate((orbital_supplements.body[:, 4], orbital_supplements.mirror[:, 4] % 1))
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    x = np.concatenate(([x[-1] - 1], x, [x[0] + 1]))

    for band in kwargs["passband"]:
        y = np.concatenate((band_curves_body[band], band_curves_mirror[band]))
        y = y[sort_idx]
        y = np.concatenate(([y[-1]], y, [y[0]]))

        i = Akima1DInterpolator(x, y)
        f = i(phases)
        band_curves[band] = f

    return band_curves


def _integrate_eccentric_lc_appx_two(self, phases, orbital_supplements, new_geometry_mask, **kwargs):
    """
    Function calculates light curve for eccentric orbit for selected filters using
    approximation where to each OrbitalPosition on one side of the apsidal line,
    the closest counterpart OrbitalPosition is assigned and the same surface geometry is
    assumed for both of them.

    :param self: elisa.binary_system.system.BinarySystem
    :param phases: numpy.array
    :param orbital_supplements: elisa.binary_system.geo.OrbitalSupplements
    :param new_geometry_mask: numpy.array
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return: Dict[str, numpy.array]
    """
    def _onpos_params(orbital_position):
        """
        Helper function

        :param orbital_position: collections.tamedtuple; elisa.const.BINARY_POSITION_PLACEHOLDER
        :return: Tuple
        """
        _container = get_onpos_container(self, orbital_position, ecl_boundaries=None)
        _normal_radiance = get_normal_radiance(_container, **kwargs)
        _ld_cfs = get_limbdarkening_cfs(_container, **kwargs)
        _container.coverage, _container.cosines = calculate_surface_parameters(_container, self.semi_major_axis,
                                                                               in_eclipse=True)
        return _container, _normal_radiance, _ld_cfs

    def _incont_lc_point(container, ldc, n_radiance, orbital_position):
        """
        Helper function

        :param container: elisa.binary_system.geo.SingleOrbitalPosition
        :param ldc: Dict[str, Dict[str, pandas.DataFrame]]
        :param n_radiance: Dict[str, Dict[str, pandas.DataFrame]]
        :param orbital_position: collections.tamedtuple; elisa.const.BINARY_POSITION_PLACEHOLDER
        :return:
        """
        for band in kwargs["passband"]:
            band_curves[band][int(orbital_position.idx)] = \
                calculate_lc_point(container, band, ldc, n_radiance)

    # this array `used_phases` is used to check, whether flux on given phase was already computed
    # it is necessary to do it due to orbital supplementes tolarance what can leads
    # to several same phases in bodies but still different phases in mirrors
    used_phases = []
    band_curves = {key: np.zeros(phases.shape) for key in kwargs["passband"]}

    # surface potentials with constant volume of components
    # todo: compute only correction on orbital_supplements.body[:, 4][new_geometry_mask] and repopulate array
    phases_to_correct = orbital_supplements.body[:, 4]
    potentials = self.correct_potentials(phases_to_correct, component="all", iterations=2)

    for idx, position_pair in enumerate(orbital_supplements):
        body, mirror = position_pair
        body_orb_pos, mirror_orb_pos = utils.convert_binary_orbital_motion_arr_to_positions([body, mirror])
        require_geometry_rebuild = new_geometry_mask[idx]

        self.primary.surface_potential = potentials['primary'][idx]
        self.secondary.surface_potential = potentials['secondary'][idx]

        self = _update_surface_in_ecc_orbits(self, body_orb_pos, require_geometry_rebuild)

        if body_orb_pos.phase not in used_phases:
            container_body, normal_radiance, ld_cfs = _onpos_params(body_orb_pos)
            _incont_lc_point(container_body, ld_cfs, normal_radiance, body_orb_pos)
            used_phases += [body_orb_pos.phase]

        if (not OrbitalSupplements.is_empty(mirror)) and (mirror_orb_pos.phase not in used_phases):
            container_mirror, normal_radiance, ld_cfs = _onpos_params(mirror_orb_pos)
            _incont_lc_point(container_mirror, ld_cfs, normal_radiance, mirror_orb_pos)
            used_phases += [mirror_orb_pos.phase]

    return band_curves


def _integrate_eccentric_lc_exactly(self, orbital_motion, phases, ecl_boundaries, **kwargs):
    """
    Function calculates LC for eccentric orbit for selected filters.
    LC is calculated exactly for each OrbitalPosition.
    It is very slow and it should be used only as a benchmark.

    :param self: elisa.binary_system.system.BinarySystem; instance
    :param orbital_motion: list of all OrbitalPositions at which LC will be calculated
    :param ecl_boundaries: list of phase boundaries of eclipses
    :param phases: phases in which the phase curve will be calculated
    :param kwargs: kwargs taken from `compute_eccentric_lightcurve` function
    :return: dictionary of fluxes for each filter
    """
    # surface potentials with constant volume of components
    potentials = self.correct_potentials(phases, component="all", iterations=2)

    band_curves = {key: np.empty(phases.shape) for key in kwargs["passband"]}
    for idx, orbital_position in enumerate(orbital_motion):
        self.primary.surface_potential = potentials['primary'][idx]
        self.secondary.surface_potential = potentials['secondary'][idx]

        self.build(components_distance=orbital_position.distance)
        container = get_onpos_container(self, orbital_position, ecl_boundaries)

        pulsations_test = {'primary': self.primary.has_pulsations(), 'secondary': self.secondary.has_pulsations()}
        if pulsations_test['primary'] or pulsations_test['secondary']:
            star_containers = {component: getattr(container, component) for component in config.BINARY_COUNTERPARTS}
            for component, star_container_instance in star_containers.items():
                if pulsations_test[component]:
                    com_x = None if component == 'primary' else orbital_position.distance
                    component_instance = getattr(self, component)
                    star_container_instance.temperatures += \
                        pulsations.calc_temp_pert_on_container(component_instance,
                                                               star_container_instance,
                                                               orbital_motion[idx].phase,
                                                               self.period,
                                                               com_x=com_x)
        normal_radiance = get_normal_radiance(container, **kwargs)
        ld_cfs = get_limbdarkening_cfs(container, **kwargs)

        container.coverage, container.cosines = calculate_surface_parameters(container, self.semi_major_axis,
                                                                             in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][int(orbital_position.idx)] = \
                calculate_lc_point(container, band, ld_cfs, normal_radiance)

    return band_curves


def _update_surface_in_ecc_orbits(self, orbital_position, new_geometry_test):
    """
    Function decides how to update surface properties with respect to the degree of change
    in surface geometry given by new_geometry test.
    If true, only points and normals are recalculated, otherwise surface is calculated from scratch.

    :param self: elisa.binary_system.system.BinarySystem
    :param orbital_position:  OrbitalPosition list
    :param new_geometry_test: bool; test that will decide, how the following phase will be calculated
    :return: elisa.binary_system.system.BinarySystem; instance with updated geometry
    # fixme: we don't need to return self, since values have been already updated and it has been reflected everywhere
    """
    if new_geometry_test:
        self.build(components_distance=orbital_position.distance)
    else:
        self.build_mesh(component="all", components_distance=orbital_position.distance)
        self.build_surface_areas(component="all")
        self.build_faces_orientation(component="all", components_distance=orbital_position.distance)

    return self


def compute_circular_spotty_asynchronous_lightcurve(self, **kwargs):
    """
    Function returns light curve of assynchronous systems with circular orbits and spots.
    # todo: add params types

    :param self: elisa.binary_system.system.BinarySystem
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return: Dict; fluxes for each filter
    """
    ecl_boundaries = geo.get_eclipse_boundaries(self, 1.0)
    points = {}
    for component in config.BINARY_COUNTERPARTS:
        component_instance = getattr(self, component)
        _a, _b, _c, _d = self.mesh_detached(component=component, components_distance=1.0, symmetry_output=True)
        points[component] = _a
        component_instance.points = copy(_a)
        component_instance.point_symmetry_vector = _b
        component_instance.base_symmetry_points_number = _c
        component_instance.inverse_point_symmetry_matrix = _d

    phases = kwargs.pop("phases")
    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phases, return_nparray=False, calculate_from='phase')

    # pre-calculate the longitudes of each spot for each phase
    spots_longitudes = geo.calculate_spot_longitudes(self, phases, component="all")
    primary_reducer, secondary_reducer = _resolve_spots_geometry_update(spots_longitudes)
    combined_reducer = primary_reducer & secondary_reducer

    # calculating lc with spots gradually shifting their positions in each phase
    band_curves = {key: np.empty(phases.shape) for key in kwargs["passband"]}
    for pos_index, orbital_position in enumerate(orbital_motion):
        # setup component necessary to build/rebuild
        require_build = "all" if combined_reducer[pos_index] \
            else "primary" if primary_reducer[pos_index] \
            else "secondary" if secondary_reducer[pos_index] \
            else None

        # use clear system surface points as a starting place to save a time
        # if reducers for related component is set to False, previous build will be used

        # todo/fixme: we can remove `reset_spots_properties` when methods will work as expected
        if primary_reducer[pos_index]:
            self.primary.points = copy(points['primary'])
            self.primary.reset_spots_properties()
        if secondary_reducer[pos_index]:
            self.secondary.points = copy(points['secondary'])
            self.secondary.reset_spots_properties()

        # assigning new longitudes for each spot
        geo.assign_spot_longitudes(self, spots_longitudes, index=pos_index, component="all")

        # build the spots points
        build.add_spots_to_mesh(self, orbital_position.distance, component=require_build)
        # build the rest of the surface based on preset surface points
        self.build_from_points(components_distance=orbital_position.distance, do_pulsations=True,
                               phase=orbital_position.phase, component=require_build)

        container = get_onpos_container(self, orbital_position, ecl_boundaries)
        normal_radiance = get_normal_radiance(container, **kwargs)
        ld_cfs = get_limbdarkening_cfs(container, **kwargs)

        container.coverage, container.cosines = calculate_surface_parameters(container, self.semi_major_axis,
                                                                             in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][int(orbital_position.idx)] = calculate_lc_point(container, band, ld_cfs, normal_radiance)

    return band_curves


def compute_eccentric_spotty_asynchronous_lightcurve(self, **kwargs):
    """
    Function returns light curve of assynchronous systems with eccentric orbits and spots.
    fixme: add params types

    :param self:
    :param kwargs:
    :return: dictionary of fluxes for each filter
    """

    phases = kwargs.pop("phases")
    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phases, return_nparray=False, calculate_from='phase')

    # pre-calculate the longitudes of each spot for each phase
    spots_longitudes = geo.calculate_spot_longitudes(self, phases, component="all")

    # calculating lc with spots gradually shifting their positions in each phase
    band_curves = {key: np.empty(phases.shape) for key in kwargs["passband"]}

    # surface potentials with constant volume of components
    potentials = self.correct_potentials(phases, component="all", iterations=2)

    for ii, orbital_position in enumerate(orbital_motion):
        self.primary.surface_potential = potentials['primary'][ii]
        self.secondary.surface_potential = potentials['secondary'][ii]

        # assigning new longitudes for each spot
        geo.assign_spot_longitudes(self, spots_longitudes, index=ii, component="all")

        self.build(components_distance=orbital_position.distance)

        container = get_onpos_container(self, orbital_position, ecl_boundaries=None)

        normal_radiance = get_normal_radiance(container, **kwargs)
        ld_cfs = get_limbdarkening_cfs(container, **kwargs)

        container.coverage, container.cosines = calculate_surface_parameters(container, self.semi_major_axis,
                                                                             in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][int(orbital_position.idx)] = \
                calculate_lc_point(container, band, ld_cfs, normal_radiance)

    return band_curves


if __name__ == "__main__":
    pass
