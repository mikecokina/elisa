import numpy as np
import logging
import matplotlib.path as mpltpath

from scipy.spatial.qhull import ConvexHull
from elisa import utils, const, atm, ld, pulsations
from elisa.binary_system import geo, build
from elisa.conf import config
from elisa.const import BINARY_POSITION_PLACEHOLDER
from scipy.interpolate import Akima1DInterpolator
from copy import copy, deepcopy

__logger__ = logging.getLogger(__name__)


def partial_visible_faces_surface_coverage(points, faces, normals, hull):
    """
    Compute surface coverage of partialy visible faces.

    :param points: numpy.array
    :param faces: numpy.array
    :param normals: numpy.array
    :param hull: numpy.array; sorted clockwise to create
    matplotlib.path.Path; path of points boundary of infront component projection
    :return: numpy.array
    """
    pypex_hull = geo.hull_to_pypex_poly(hull)
    pypex_faces = geo.faces_to_pypex_poly(points[faces])
    # it is possible to None happens in intersection, tkae care about it latter
    pypex_intersection = geo.pypex_poly_hull_intersection(pypex_faces, pypex_hull)

    # think about surface normalisation like and avoid surface areas like 1e-6 which lead to loss in precission
    pypex_polys_surface_area = np.array(geo.pypex_poly_surface_area(pypex_intersection), dtype=np.float)

    inplane_points_3d = np.concatenate((points.T, [[0.0] * len(points)])).T
    inplane_surface_area = utils.triangle_areas(triangles=faces, points=inplane_points_3d)
    correction_cosine = utils.calculate_cos_theta_los_x(normals)
    retval = (inplane_surface_area - pypex_polys_surface_area) / correction_cosine
    return retval


def get_visible_projection(obj):
    """
    Returns yz projection of nearside points.

    :param obj:
    :return:
    """
    return geo.plane_projection(
        obj.points[
            np.unique(obj.faces[obj.indices])
        ], "yz"
    )


def get_eclipse_boundary_path(hull):
    """
    Return `matplotlib.path.Path` object which represents boundary of component projection
    to plane `yz`.

    :param hull: numpy.array
    :return: matplotlib.path.Path
    """
    cover_bound = ConvexHull(hull)
    hull_points = hull[cover_bound.vertices]
    bb_path = mpltpath.Path(hull_points)
    return bb_path


def compute_surface_coverage(container: geo.SingleOrbitalPositionContainer, in_eclipse=True):
    # todo: add unittests
    """
    Compute surface coverage of faces for given orbital position
    defined by container/SingleOrbitalPositionContainer.

    :param container: elisa.binary_system.geo.SingleOrbitalPositionContainer
    :param in_eclipse: bool
    :return: Dict
    """
    __logger__.debug(f"computing surface coverage for {container.position}")
    cover_component = 'secondary' if 0.0 < container.position.azimuth < const.PI else 'primary'
    cover_object = getattr(container, cover_component)
    undercover_object = getattr(container, config.BINARY_COUNTERPARTS[cover_component])
    undercover_visible_point_indices = np.unique(undercover_object.faces[undercover_object.indices])

    cover_object_obs_visible_projection = get_visible_projection(cover_object)
    undercover_object_obs_visible_projection = get_visible_projection(undercover_object)
    # get matplotlib boudary path defined by hull of projection
    if in_eclipse:
        bb_path = get_eclipse_boundary_path(cover_object_obs_visible_projection)
        # obtain points out of eclipse (out of boundary defined by hull of 'infront' object)
        out_of_bound = np.invert(bb_path.contains_points(undercover_object_obs_visible_projection))
        # undercover_visible_point_indices = undercover_visible_point_indices[out_of_bound]
    else:
        out_of_bound = np.ones(undercover_object_obs_visible_projection.shape[0], dtype=np.bool)

    undercover_visible_point_indices = undercover_visible_point_indices[out_of_bound]
    undercover_faces = np.array([const.FALSE_FACE_PLACEHOLDER] * np.shape(undercover_object.normals)[0])
    undercover_faces[undercover_object.indices] = undercover_object.faces[undercover_object.indices]
    eclipse_faces_visibility = np.isin(undercover_faces, undercover_visible_point_indices)

    # get indices of full visible, invisible and partial visible faces
    full_visible = np.all(eclipse_faces_visibility, axis=1)
    invisible = np.all(np.invert(eclipse_faces_visibility), axis=1)
    partial_visible = np.invert(full_visible | invisible)

    # process partial and full visible faces (get surface area of 3d polygon) of undercover object
    partial_visible_faces = undercover_object.faces[partial_visible]
    partial_visible_normals = undercover_object.normals[partial_visible]
    undercover_object_pts_projection = geo.plane_projection(undercover_object.points, "yz", keep_3d=False)
    if in_eclipse:
        partial_coverage = partial_visible_faces_surface_coverage(
            points=undercover_object_pts_projection,
            faces=partial_visible_faces,
            normals=partial_visible_normals,
            hull=bb_path.vertices
        )
    else:
        partial_coverage = None

    visible_coverage = utils.poly_areas(undercover_object.points[undercover_object.faces[full_visible]])

    undercover_obj_coverage = geo.surface_area_coverage(
        size=np.shape(undercover_object.normals)[0],
        visible=full_visible, visible_coverage=visible_coverage,
        partial=partial_visible, partial_coverage=partial_coverage
    )

    visible_coverage = utils.poly_areas(cover_object.points[cover_object.faces[cover_object.indices]])
    cover_obj_coverage = geo.surface_area_coverage(len(cover_object.faces), cover_object.indices, visible_coverage)

    return {
        cover_component: cover_obj_coverage,
        config.BINARY_COUNTERPARTS[cover_component]: undercover_obj_coverage
    }


def get_normal_radiance(single_orbital_position_container, component=None, **kwargs):
    """
    Compute normal radiance for all faces and all components in SingleOrbitalPositionContainer.

    :param component: str
    :param single_orbital_position_container: elisa.binary_system.geo.SingleOrbitalPositionContainer
    :param kwargs: Dict; arguments to be passed into light curve generator functions
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return: Dict[String, numpy.array]
    """
    if component is None:
        return {
            component:
                atm.NaiveInterpolatedAtm.radiance(
                    **dict(
                        temperature=getattr(single_orbital_position_container, component).temperatures,
                        log_g=getattr(single_orbital_position_container, component).log_g,
                        metallicity=getattr(single_orbital_position_container, component).metallicity,
                        **kwargs
                    )
                ) for component in config.BINARY_COUNTERPARTS.keys()
        }
    elif component in config.BINARY_COUNTERPARTS.keys():
        return atm.NaiveInterpolatedAtm.radiance(
            **dict(
                temperature=getattr(single_orbital_position_container, component).temperatures,
                log_g=getattr(single_orbital_position_container, component).log_g,
                metallicity=getattr(single_orbital_position_container, component).metallicity,
                **kwargs
            )
        )
    else:
        raise ValueError('Invalid value of `component` argument. '
                         'Available parameters are `primary`, `secondary` or None.')


def get_limbdarkening_cfs(self, component=None, **kwargs):
    """
    Returns limb darkening coefficients for each face of each component.

    :param component: str
    :param self:
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return: Dict[str, numpy.array]
    """
    if component is None:
        return {
            component:
                ld.interpolate_on_ld_grid(
                    temperature=getattr(self, component).temperatures,
                    log_g=getattr(self, component).log_g,
                    metallicity=getattr(self, component).metallicity,
                    passband=kwargs["passband"]
                ) for component in config.BINARY_COUNTERPARTS.keys()
        }
    elif component in config.BINARY_COUNTERPARTS.keys():
        return ld.interpolate_on_ld_grid(
            temperature=getattr(self, component).temperatures,
            log_g=getattr(self, component).log_g,
            metallicity=getattr(self, component).metallicity,
            passband=kwargs["passband"]
        )
    else:
        raise ValueError('Invalid value of `component` argument. '
                         'Available parameters are `primary`, `secondary` or None.')


def prep_surface_params(initial_props_container, pulsations_test, **kwargs):
    """
    Prepares normal radiances and limb darkening coefficients variables.

    :param initial_props_container: SingleOrbitalPosition
    :param pulsations_test: dict {component: bool - has_pulsations, ...}
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return:
    """
    has_pulsations = pulsations_test['primary'] or pulsations_test['secondary']
    if not has_pulsations:
        # compute normal radiance for each face and each component
        normal_radiance = get_normal_radiance(initial_props_container, **kwargs)
        # obtain limb darkening factor for each face
        ld_cfs = get_limbdarkening_cfs(initial_props_container, **kwargs)
    elif not pulsations_test['primary']:
        normal_radiance = {'primary': get_normal_radiance(initial_props_container, component='primary', **kwargs)}
        ld_cfs = {'primary': get_limbdarkening_cfs(initial_props_container, component='primary', **kwargs)}
    elif not pulsations_test['secondary']:
        normal_radiance = {'secondary': get_normal_radiance(initial_props_container, component='secondary', **kwargs)}
        ld_cfs = {'secondary': get_limbdarkening_cfs(initial_props_container, component='secondary', **kwargs)}
    else:
        normal_radiance, ld_cfs = dict(), dict()
    return normal_radiance, ld_cfs


def compute_circular_synchronous_lightcurve(self, **kwargs):
    """
    Compute light curve, exactly, from position to position, for synchronous circular
    binary system.

    :param self: elisa.binary_system.system.BinarySystem
    :param kwargs: Dict;
    * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
            * ** position_method** * - function definition; to evaluate orbital positions
    :return: Dict[str, numpy.array]
    """
    self.build(components_distance=1.0)

    ecl_boundaries = geo.get_eclipse_boundaries(self, 1.0)

    # in case of LC for spotless surface without pulsations unique phase interval is only (0, 0.5)
    phases = kwargs.pop("phases")
    unique_phase_interval, reverse_phase_map = _phase_crv_symmetry(self, phases)

    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=unique_phase_interval, return_nparray=False, calculate_from='phase')

    initial_props_container = geo.SingleOrbitalPositionContainer(self.primary, self.secondary)
    initial_props_container.setup_position(BINARY_POSITION_PLACEHOLDER(*(0, 1.0, 0.0, 0.0, 0.0)), self.inclination)

    # injected attributes
    setattr(initial_props_container.primary, 'metallicity', self.primary.metallicity)
    setattr(initial_props_container.secondary, 'metallicity', self.secondary.metallicity)
    ld_law_cfs_columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]

    system_positions_container = self.prepare_system_positions_container(orbital_motion=orbital_motion,
                                                                         ecl_boundaries=ecl_boundaries)
    system_positions_container = system_positions_container.darkside_filter()

    pulsations_test = {'primary': self.primary.has_pulsations(), 'secondary': self.secondary.has_pulsations()}
    normal_radiance, ld_cfs = prep_surface_params(initial_props_container, pulsations_test, **kwargs)

    band_curves = {key: np.zeros(unique_phase_interval.shape) for key in kwargs["passband"].keys()}
    for idx, container in enumerate(system_positions_container):
        # dict of components
        star_containers = {component: getattr(container, component) for component in config.BINARY_COUNTERPARTS.keys()}

        coverage = compute_surface_coverage(container, in_eclipse=system_positions_container.in_eclipse[idx])

        # calculating cosines between face normals and line of sight
        cosines, visibility_test = {}, {}
        for component, star_container_instance in star_containers.items():
            cosines[component] = utils.calculate_cos_theta_los_x(star_container_instance.normals)
            visibility_test[component] = cosines[component] > 0
            cosines[component] = cosines[component][visibility_test[component]]

            # calculating temperature perturbation due to pulsations
            if pulsations_test[component]:
                com_x = None if component == 'primary' else 1.0
                component_instance = getattr(self, component)
                star_container_instance.temperatures += \
                    pulsations.calc_temp_pert_on_container(component_instance,
                                                           star_container_instance,
                                                           orbital_motion[idx].phase,
                                                           self.period,
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


def _phase_crv_symmetry(self, phase):
    """
    Utilizing symmetry of circular systems without spots and pulastions where you need to evaluate only half
    of the phases. Function finds such redundant phases and returns only unique phases.
    Expects phases from 0 to 1.0.

    :param self: elisa.binary_system.system.BinarySystem
    :param phase: numpy.array
    :return: Tuple[numpy.array, numpy.array]
    """
    # keep those fucking methods imutable
    phase = phase.copy()
    if (not self.has_pulsations()) & (not self.has_spots()):
        symmetrical_counterpart = phase > 0.5
        # phase[symmetrical_counterpart] = 0.5 - (phase[symmetrical_counterpart] - 0.5)
        phase[symmetrical_counterpart] = np.round(1.0 - phase[symmetrical_counterpart], 9)
        res_phases, reverse_idx = np.unique(phase, return_inverse=True)
        return res_phases, reverse_idx
    else:
        return phase, np.arange(phase.shape[0])


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


def _split_orbit_by_apse_line(orbital_motion, orbital_mask):
    """
    Split orbital positions represented by `orbital_motion` array on two groups separated by line of apsides.
    Separation is defined by `orbital_mask`

    :param orbital_motion: numpy.array; arraywhcih represents orbital positions
    :param orbital_mask: numpy.array[bool]; mask which defines separation (True is one side and False is other side)
    :return: Tuple[numpy.array, numpy.array]
    """
    reduced_orbit_arr = orbital_motion[orbital_mask]
    supplement_to_reduced_arr = orbital_motion[~orbital_mask]
    return reduced_orbit_arr, supplement_to_reduced_arr


def _resolve_geometry_update(self, size, rel_d):
    """
    Evaluate where on orbital position is necessary to fully update geometry.

    :param self: elisa.binary_system.system.BinarySystem
    :param size: int
    :param rel_d: numpy.array
    :return: numpy.array[bool]
    """
    # in case of spots, the boundary points will cause problems if you want to use the same geometry
    if self.has_spots():
        return np.ones(size, dtype=np.bool)

    require_new_geo = np.ones(size, dtype=np.bool)

    cumulative_sum = np.array([0.0, 0.0])
    for i in range(1, size):
        cumulative_sum += rel_d[:, i - 1]
        if (cumulative_sum <= config.MAX_RELATIVE_D_R_POINT).all():
            require_new_geo[i] = False
        else:
            require_new_geo[i] = True
            cumulative_sum = np.array([0.0, 0.0])

    return require_new_geo


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
    fwd_radii = self.calculate_all_forward_radii(orbital_supplements.body[:, 1], components=None)
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
        return 'zero', lambda: _integrate_lc_exactly(self, all_orbital_pos, phases, None, **kwargs)

    # APPX ONE *********************************************************************************************************
    appx_one = _eval_approximation_one(self, phases)

    if appx_one:
        orbital_supplements = geo.OrbitalSupplements(body=reduced_orbit_arr, mirror=counterpart_postion_arr)
        orbital_supplements.sort(by='distance')
        rel_d_radii = _compute_rel_d_radii(self, orbital_supplements)
        new_geometry_mask = _resolve_geometry_update(self, orbital_supplements.size(), rel_d_radii)

        return 'one', lambda: _integrate_lc_appx_one(self, phases, orbital_supplements, new_geometry_mask, **kwargs)

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
    new_geometry_mask = _resolve_geometry_update(self, orbital_supplements.size(), rel_d_radii)

    if appx_two:
        return 'two', lambda: _integrate_lc_appx_two(self, phases, orbital_supplements, new_geometry_mask, **kwargs)
    # APPX ZERO once again *********************************************************************************************
    else:
        return 'zero', lambda: _integrate_lc_exactly(self, all_orbital_pos, phases, ecl_boundaries=None, **kwargs)


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
def _prepare_geosymmetric_orbit(self, azimuths, phases):
    """
    Prepare set of orbital positions that are symmetrical in therms of surface geometry, where orbital position is
    mirrored via apsidal line in order to reduce time for generating the light curve.

    :param self: elisa.binary_star.system.BinarySystem
    :param azimuths: numpy.array; orbital azimuths of positions in which LC will be calculated
    :param phases: numpy.array; orbital phase of positions in which LC will be calculated
    :return: Tuple;


    shape ::
        
        (numpy.array, list, numpy.array)
        
    - unique_phase_indices - numpy.array : indices that points to the orbital positions from one half of the 
    orbital motion divided by apsidal line
    - orbital_motion_counterpart - list - Positions produced by mirroring orbital positions given by 
    indices `unique_phase_indices`
    - orbital_motion_array_counterpart - numpy.array - sa as `orbital_motion_counterpart` but in numpy.array form
    """
    azimuth_boundaries = [self.argument_of_periastron, (self.argument_of_periastron + const.PI) % const.FULL_ARC]
    unique_geometry = np.logical_and(azimuths > azimuth_boundaries[0],
                                     azimuths < azimuth_boundaries[1]) \
        if azimuth_boundaries[0] < azimuth_boundaries[1] else np.logical_xor(azimuths < azimuth_boundaries[0],
                                                                             azimuths > azimuth_boundaries[1])
    unique_phase_indices = np.arange(phases.shape[0])[unique_geometry]
    unique_geometry_azimuths = azimuths[unique_geometry]
    unique_geometry_counterazimuths = (2 * self.argument_of_periastron - unique_geometry_azimuths) % const.FULL_ARC

    orbital_motion_array_counterpart = \
        self.calculate_orbital_motion(input_argument=unique_geometry_counterazimuths,
                                      return_nparray=True,
                                      calculate_from='azimuth')

    return unique_phase_indices, orbital_motion_array_counterpart, unique_geometry


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


def calculate_surface_parameters(container, in_eclipse=True):
    """
    Function prepares surface-related parameters such as coverage(area of visibility
    of the triangles) and directional cosines towards line-of-sight vector.

    :param container: SingleOrbitalPositionContainer
    :param in_eclipse: bool; indicate if eclipse occur for given position container.
    If you are not sure leave it to True
    :return: Tuple;

    shape::

        (numpy.array, Dict[str, numpy.array])

    - coverage - numpy.array - visible area of triangles
    - p_cosines, s_cosines - Dict[str, numpy.array] - directional cosines for each face with respect
    to line-of-sight vector
    """
    coverage = compute_surface_coverage(container, in_eclipse=in_eclipse)
    p_cosines = utils.calculate_cos_theta_los_x(container.primary.normals)
    s_cosines = utils.calculate_cos_theta_los_x(container.secondary.normals)
    cosines = {'primary': p_cosines, 'secondary': s_cosines}
    return coverage, cosines


def calculate_lc_point(container, band, ld_cfs, normal_radiance):
    """
    Calculates point on the light curve for given band.

    :param container: SingleOrbitalPositionContainer
    :param band: str; name of the photometric band
    :param ld_cfs: Dict[str, Dict[str, pandas.DataFrame]]
    :param normal_radiance: Dict[str, Dict[str, numpy.array]]
    :return: float
    """
    ld_law_cfs_columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]
    ld_cors = {
        component: ld.limb_darkening_factor(coefficients=ld_cfs[component][band][ld_law_cfs_columns].values,
                                            limb_darkening_law=config.LIMB_DARKENING_LAW,
                                            cos_theta=container.cosines[component])
        for component in config.BINARY_COUNTERPARTS
    }
    # fixme: add all missing multiplicators (at least is missing semi_major_axis^2 in physical units)
    flux = {
        component:
            np.sum(normal_radiance[component][band] * container.cosines[component] *
                   container.coverage[component] * ld_cors[component])
        for component in config.BINARY_COUNTERPARTS.keys()
    }
    flux = flux['primary'] + flux['secondary']
    return flux


def _integrate_lc_appx_one(self, phases, orbital_supplements, new_geometry_mask, **kwargs):
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
    potentials = self.correct_potentials(orbital_supplements.body[:, 4], component=None, iterations=2)

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
            container_body.cosines = calculate_surface_parameters(container_body, in_eclipse=True)
        container_mirror.coverage, \
            container_mirror.cosines = calculate_surface_parameters(container_mirror, in_eclipse=True)

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


def _integrate_lc_appx_two(self, phases, orbital_supplements, new_geometry_mask, **kwargs):
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
        _container.coverage, _container.cosines = calculate_surface_parameters(_container, in_eclipse=True)
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
    potentials = self.correct_potentials(phases_to_correct, component=None, iterations=2)

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

        if (not geo.OrbitalSupplements.is_empty(mirror)) and (mirror_orb_pos.phase not in used_phases):
            container_mirror, normal_radiance, ld_cfs = _onpos_params(mirror_orb_pos)
            _incont_lc_point(container_mirror, ld_cfs, normal_radiance, mirror_orb_pos)
            used_phases += [mirror_orb_pos.phase]

    return band_curves


def _integrate_lc_exactly(self, orbital_motion, phases, ecl_boundaries, **kwargs):
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
    potentials = self.correct_potentials(phases, component=None, iterations=2)

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

        container.coverage, container.cosines = calculate_surface_parameters(container, in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][int(orbital_position.idx)] = \
                calculate_lc_point(container, band, ld_cfs, normal_radiance)

    return band_curves


def calculate_new_geometry(self, orbit_template_arr, rel_d_radii):
    """
    This function chcecks at which OrbitalPositions it is necessary to recalculate geometry.

    :param self: elisa.binary_system.system.BinarySystem
    :param orbit_template_arr: numpy.array; array of orbital positions from one side of the apsidal
    line used as the symmetry template
    :param rel_d_radii: numpy.array; shape(2 x len(orbit_template arr) - relative changes in radii of each component
    with respect to the previous OrbitalPosition, excluding the first postition.
    :return: numpy.array[bool]; mask to select Orbital positions, where orbits should be calculated
    """
    # in case of spots, the boundary points will cause problems if you want to use the same geometry
    if self.has_spots():
        return np.ones(orbit_template_arr.shape[0], dtype=np.bool)

    calc_new_geometry = np.zeros(orbit_template_arr.shape[0], dtype=np.bool)
    calc_new_geometry[0] = True
    cumulative_sum = np.array([0.0, 0.0])
    for ii in range(1, orbit_template_arr.shape[0]):
        cumulative_sum += rel_d_radii[:, ii - 1]
        if (cumulative_sum <= config.MAX_RELATIVE_D_R_POINT).all():
            calc_new_geometry[ii] = False
        else:
            calc_new_geometry[ii] = True
            cumulative_sum = np.array([0.0, 0.0])

    return calc_new_geometry


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
        self.build_mesh(component=None, components_distance=orbital_position.distance)
        self.build_surface_areas(component=None)
        self.build_faces_orientation(component=None, components_distance=orbital_position.distance)

    return self


def compute_circular_spoty_asynchronous_lightcurve(self, *args, **kwargs):
    """
    Function returns light curve of assynchronous systems with circular orbits and spots.
    #todo: add params types

    :param self: BinarySystem instance
    :param args:
    :param kwargs:
    :return: dictionary of fluxes for each filter
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
    # TODO: implement minimum angular step in longitude which will result in mesh recalculation, it will save a lot of
    # TODO: time for systems with synchronicities close to one

    spots_longitudes = geo.calculate_spot_longitudes(self, phases, component=None)

    # calculating lc with spots gradually shifting their positions in each phase
    band_curves = {key: np.empty(phases.shape) for key in kwargs["passband"]}
    for ii, orbital_position in enumerate(orbital_motion):
        # use clear system surface points as a starting place to save a time
        self.primary.points = copy(points['primary'])
        self.secondary.points = copy(points['secondary'])

        # assigning new longitudes for each spot
        geo.assign_spot_longitudes(self, spots_longitudes, index=ii, component=None)

        # build the spots points
        build.add_spots_to_mesh(self, orbital_position.distance, component=None)
        # build the rest of the surface
        self.build_faces(component=None, components_distance=orbital_position.distance)
        self.build_surface_areas(component=None)
        self.build_faces_orientation(component=None, components_distance=orbital_position.distance)
        self.build_surface_gravity(component=None, components_distance=orbital_position.distance)
        self.build_temperature_distribution(component=None, components_distance=orbital_position.distance,
                                            do_pulsations=True, phase=orbital_position.phase)

        container = get_onpos_container(self, orbital_position, ecl_boundaries)

        normal_radiance = get_normal_radiance(container, **kwargs)
        ld_cfs = get_limbdarkening_cfs(container, **kwargs)

        container.coverage, container.cosines = calculate_surface_parameters(container, in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][int(orbital_position.idx)] = \
                calculate_lc_point(container, band, ld_cfs, normal_radiance)

    return band_curves


def compute_eccentric_spoty_asynchronous_lightcurve(self, *args, **kwargs):
    """
    Function returns light curve of assynchronous systems with eccentric orbits and spots.
    fixme: add params types

    :param self:
    :param args:
    :param kwargs:
    :return: dictionary of fluxes for each filter
    """
    ecl_boundaries = geo.get_eclipse_boundaries(self, 1.0)

    phases = kwargs.pop("phases")
    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phases, return_nparray=False,
                                     calculate_from='phase')

    # pre-calculate the longitudes of each spot for each phase
    spots_longitudes = geo.calculate_spot_longitudes(self, phases, component=None)

    # calculating lc with spots gradually shifting their positions in each phase
    band_curves = {key: np.empty(phases.shape) for key in kwargs["passband"]}

    # surface potentials with constant volume of components
    potentials = self.correct_potentials(phases, component=None, iterations=2)

    for ii, orbital_position in enumerate(orbital_motion):
        self.primary.surface_potential = potentials['primary'][ii]
        self.secondary.surface_potential = potentials['secondary'][ii]

        # assigning new longitudes for each spot
        geo.assign_spot_longitudes(self, spots_longitudes, index=ii, component=None)

        self.build(components_distance=orbital_position.distance)

        container = get_onpos_container(self, orbital_position, ecl_boundaries)

        normal_radiance = get_normal_radiance(container, **kwargs)
        ld_cfs = get_limbdarkening_cfs(container, **kwargs)

        container.coverage, container.cosines = calculate_surface_parameters(container, in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][int(orbital_position.idx)] = \
                calculate_lc_point(container, band, ld_cfs, normal_radiance)

    return band_curves


def find_apsidally_corresponding_positions(reduced_constraint, reduced_arr,
                                           supplement_constraint, supplement_arr,
                                           tol=1e-10, as_empty=None):
    """
    Function is inteded to look for orbital positions from reduced_arr which
    are supplementar to supplement_arr. Similarity to be a pair is based
    on constraints from input arguments, usually it is current separation of
    primary and secondary component on orbit.

    :param reduced_constraint: numpy.array
    :param reduced_arr: numpy.array
    :param supplement_constraint: numpy.array
    :param supplement_arr: numpy.array
    :param tol: float
    :param as_empty: numpy.array; e.g. [np.nan, np.nan] depends on shape of reduced_arr item
    :return: elisa.binary_system.geo.OrbitalSupplements
    """
    if as_empty is None:
        as_empty = [np.nan] * 5

    ids_of_closest_reduced_values = utils.find_idx_of_nearest(reduced_constraint, supplement_constraint)

    matrix_mask = abs(np.abs(reduced_constraint[np.newaxis, :] - supplement_constraint[:, np.newaxis])) <= tol
    is_supplement = [matrix_mask[i][idx] for i, idx in enumerate(ids_of_closest_reduced_values)]

    twin_in_reduced = np.array([-1] * len(ids_of_closest_reduced_values))
    twin_in_reduced[is_supplement] = ids_of_closest_reduced_values[is_supplement]

    supplements = geo.OrbitalSupplements()

    for id_supplement, id_reduced in enumerate(twin_in_reduced):
        args = (reduced_arr[id_reduced], supplement_arr[id_supplement]) \
            if id_reduced > -1 else (supplement_arr[id_supplement], as_empty)
        # if id_reduced > -1 else (as_empty, supplement_arr[id_supplement])

        if not utils.is_empty(args):
            supplements.append(*args)

    reduced_all_ids = np.arange(0, len(reduced_arr))
    is_not_in = ~np.isin(reduced_all_ids, twin_in_reduced)

    for is_not_in_id in reduced_all_ids[is_not_in]:
        if reduced_arr[is_not_in_id] not in supplement_arr:
            supplements.append(*(reduced_arr[is_not_in_id], as_empty))

    return supplements


if __name__ == "__main__":
    pass
