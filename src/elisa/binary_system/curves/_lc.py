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
