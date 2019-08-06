import numpy as np
import logging
import matplotlib.path as mpltpath

from scipy.spatial.qhull import ConvexHull

from elisa.conf import config
from elisa import logger, utils, const, atm, ld
from elisa.binary_system import geo
from elisa.const import BINARY_POSITION_PLACEHOLDER
from scipy.interpolate import interp1d

__logger__ = logging.getLogger(__name__)


def partial_visible_faces_surface_coverage(points, faces, normals, hull):
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
    returns yz projection of nearside points
    :param obj:
    :return:
    """
    return geo.plane_projection(
        obj.points[
            np.unique(obj.faces[obj.indices])
        ], "yz"
    )


def get_eclipse_boundary_path(hull):
    cover_bound = ConvexHull(hull)
    hull_points = hull[cover_bound.vertices]
    bb_path = mpltpath.Path(hull_points)
    return bb_path


def compute_surface_coverage(container: geo.SingleOrbitalPositionContainer, in_eclipse=True):
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


def get_normal_radiance(single_orbital_position_container, **kwargs):
    """
    Compute normal radiance for all faces and all components in SingleOrbitalPositionContainer

    :param single_orbital_position_container: elisa.binary_system.geo.SingleOrbitalPositionContainer
    :param kwargs: Dict; arguments to be passed into light curve generator functions
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return: Dict[String, numpy.array]
    """
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


def get_limbdarkening_cfs(self, **kwargs):
    """
    returns limg darkening coefficients for each face of each component
    :param self:
    :param kwargs: dict - {'primary': numpy.array, 'secondary': numpy.array}
    :return:
    """
    return {
        component:
        ld.interpolate_on_ld_grid(
            temperature=getattr(self, component).temperatures,
            log_g=getattr(self, component).log_g,
            metallicity=getattr(self, component).metallicity,
            passband=kwargs["passband"]
        ) for component in config.BINARY_COUNTERPARTS.keys()
    }


def compute_circular_synchronous_lightcurve(self, **kwargs):
    """
    fixme: add docstrings
    :param self:
    :param kwargs:
    :return:
    """
    self.build(components_distance=1.0)

    ecl_boundaries = geo.get_eclipse_boundaries(self, 1.0)

    # in case of LC for spotless surface without pulsations unique phase interval is only (0, 0.5)
    phases = kwargs.pop("phases")
    unique_phase_interval, reverse_phase_map = phase_crv_symmetry(self, phases)

    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=unique_phase_interval, return_nparray=False, calculate_from='phase')

    initial_props_container = geo.SingleOrbitalPositionContainer(self.primary, self.secondary)
    initial_props_container.setup_position(BINARY_POSITION_PLACEHOLDER(*(0, 1.0, 0.0, 0.0, 0.0)), self.inclination)

    # injected attributes
    setattr(initial_props_container.primary, 'metallicity', self.primary.metallicity)
    setattr(initial_props_container.secondary, 'metallicity', self.secondary.metallicity)

    # compute normal radiance for each face and each component
    normal_radiance = get_normal_radiance(initial_props_container, **kwargs)
    # obtain limb darkening factor for each face
    ld_cfs = get_limbdarkening_cfs(initial_props_container, **kwargs)
    ld_law_cfs_columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]

    system_positions_container = self.prepare_system_positions_container(orbital_motion=orbital_motion,
                                                                         ecl_boundaries=ecl_boundaries)
    system_positions_container = system_positions_container.darkside_filter()

    band_curves = {key: np.zeros(unique_phase_interval.shape) for key in kwargs["passband"].keys()}
    for idx, container in enumerate(system_positions_container):
        coverage = compute_surface_coverage(container, in_eclipse=system_positions_container.in_eclipse[idx])
        p_cosines = utils.calculate_cos_theta_los_x(container.primary.normals)
        s_cosines = utils.calculate_cos_theta_los_x(container.secondary.normals)

        for band in kwargs["passband"].keys():
            p_ld_cors = ld.limb_darkening_factor(coefficients=ld_cfs['primary'][band][ld_law_cfs_columns].values,
                                                 limb_darkening_law=config.LIMB_DARKENING_LAW, cos_theta=p_cosines)

            s_ld_cors = ld.limb_darkening_factor(coefficients=ld_cfs['secondary'][band][ld_law_cfs_columns].values,
                                                 limb_darkening_law=config.LIMB_DARKENING_LAW, cos_theta=s_cosines)
            # fixme: add all missing multiplicators (at least is missing semi_major_axis^2 in physical units)
            p_flux = np.sum(normal_radiance["primary"][band] * p_cosines * coverage["primary"] * p_ld_cors)
            s_flux = np.sum(normal_radiance["secondary"][band] * s_cosines * coverage["secondary"] * s_ld_cors)
            flux = p_flux + s_flux
            # band_curves[band].append(flux)
            band_curves[band][idx] = flux
    band_curves = {band: band_curves[band][reverse_phase_map] for band in band_curves}

    return band_curves


def phase_crv_symmetry(self, phase):
    """
    Utilizing symmetry of circular systems without spots and pulastions where you need to evaluate only half of the
    phases. Function finds such redundant phases and returns only unique phases.

    :param self: elisa.binary_system.system.BinarySystem
    :param phase: numpy.array
    :return: Tuple[numpy.array, numpy.array]
    """
    if not self.primary.has_pulsations() and not self.primary.has_pulsations() and \
            not self.primary.has_spots() and not self.secondary.has_spots():
        symmetrical_counterpart = phase > 0.5
        # phase[symmetrical_counterpart] = 0.5 - (phase[symmetrical_counterpart] - 0.5)
        phase[symmetrical_counterpart] = np.round(1.0 - phase[symmetrical_counterpart], 9)
        res_phases, reverse_idx = np.unique(phase, return_inverse=True)
        return res_phases, reverse_idx
    else:
        return phase, np.arange(phase.shape[0])


def compute_eccentric_lightcurve(self, **kwargs):
    self._logger = logger.getLogger(self.__class__.__name__, suppress=True)
    ecl_boundaries = geo.get_eclipse_boundaries(self, 1.0)

    phases = kwargs.pop("phases")

    position_method = kwargs.pop("position_method")
    orbital_motion, orbital_motion_array = position_method(input_argument=phases,
                                                           return_nparray=True, calculate_from='phase')

    # this condition checks if even to attempt to utilize apsidal line symmetry approximations
    if config.POINTS_ON_ECC_ORBIT > 0 and config.POINTS_ON_ECC_ORBIT is not None:
        # in case of clean surface or synchronous rotation (more-less), symmetry around semi-major axis can be utilized
        # mask isolating the symmetrical part of the orbit
        azimuths = orbital_motion_array[:, 2]

        # test whether mirroring around semi-major axis will be performed
        approximation_test1 = len(phases) > config.POINTS_ON_ECC_ORBIT and self.primary.synchronicity == 1.0 and \
            self.secondary.synchronicity == 1.0

        unique_phase_indices, orbital_motion_counterpart, orbital_motion_array_counterpart, uniq_geom_test = \
            construct_geometry_symmetric_azimuths(self, azimuths, phases)

        # spliting orbital motion into two separate groups on different sides of apsidal line
        orbit_template_arr = orbital_motion_array[uniq_geom_test]
        orbit_counterpart_arr = orbital_motion_array[~uniq_geom_test]

        index_of_closest = utils.find_idx_of_nearest(orbit_counterpart_arr[:, 1],
                                                     orbit_template_arr[:, 1])

        # testing whether all counterpart phases were assigned to template part of orbital motion
        isin_test = np.isin(np.arange(np.count_nonzero(~uniq_geom_test)), index_of_closest)
        # finding indices of orbit_counterpart_arr which were not assigned to any symmetricall orbital position
        missing_phases_indices = np.arange(np.count_nonzero(~uniq_geom_test))[~isin_test]

        # finding index of closest symmetrical orbital position to the missing phase
        index_of_closest_reversed = []
        if len(missing_phases_indices) > 0:
            index_of_closest_reversed = utils.find_idx_of_nearest(orbit_template_arr[:, 1],
                                                                  orbit_counterpart_arr[missing_phases_indices, 1])
            index_of_closest = np.append(index_of_closest, missing_phases_indices)
            orbit_template_arr = np.append(orbit_template_arr, orbit_template_arr[index_of_closest_reversed],
                                           axis=0)
            orbit_counterpart_arr = np.append(orbit_counterpart_arr, orbit_counterpart_arr[missing_phases_indices],
                                              axis=0)

        # changes in component distances between symmetrical couples
        d_distance = np.abs(orbit_template_arr[:, 1] -
                            orbit_counterpart_arr[index_of_closest, 1])

        forward_radii = self.calculate_all_forward_radii(orbit_template_arr[:, 1], components=None)
        # calculating change in forward radius as a indicator of change in overall geometry, not calculated for the
        # first OrbitalPosition since it is True
        forward_radii = np.array(list(forward_radii.values()))
        rel_d_radii = np.abs(forward_radii[:, 1:] - np.roll(forward_radii, shift=1, axis=1)[:, 1:]) / \
                      forward_radii[:, 1:]
        # second approximation does not interpolates the resulting light curve but assumes that geometry is the same as
        # the geometry of the found counterpart
        # testing if change in geometry will not be too severe, you should rather use changes in point radius instead
        forward_radii_sorted = np.sort(forward_radii, axis=1)
        rel_d_radii_sorted = np.abs(forward_radii_sorted - np.roll(forward_radii_sorted, shift=1, axis=1)) / \
                             forward_radii_sorted
        approximation_test2 = np.max(rel_d_radii_sorted[:, 1:]) < config.MAX_RELATIVE_D_R_POINT and \
                              self.primary.synchronicity == 1.0 and self.secondary.synchronicity == 1.0  # spots???

        # this part checks if differences between geometries of adjacent phases are small enough to assume that
        # geometries are the same.
        new_geometry_test = calculate_new_geometry(self, orbit_template_arr, rel_d_radii)

    else:
        approximation_test1 = False
        approximation_test2 = False

    # initial values of radii to be compared with
    # orig_forward_rad_p, orig_forward_rad_p = 100.0, 100.0  # 100.0 is too large value, it will always fail the first
    # test and therefore the surface will be built
    if approximation_test1:
        __logger__.info('one half of the points on LC on the one side of the apsidal line will be interpolated')
        band_curves = integrate_lc_using_approx1(self, orbital_motion, orbital_motion_counterpart, unique_phase_indices,
                                                 uniq_geom_test, ecl_boundaries, phases,
                                                 orbital_motion_array_counterpart, new_geometry_test, **kwargs)

    elif approximation_test2:
        __logger__.info('geometry of the stellar surface on one half of the apsidal '
                        'line will be copied from their symmetrical counterparts')
        band_curves = integrate_lc_using_approx2(self, orbital_motion, missing_phases_indices, index_of_closest,
                                                 index_of_closest_reversed, uniq_geom_test, ecl_boundaries, phases,
                                                 new_geometry_test, **kwargs)

    else:
        __logger__.info('lc will be calculated in a rigorous phase to phase manner without approximations')
        band_curves = integrate_lc_exactly(self, orbital_motion, ecl_boundaries, phases, **kwargs)

    return band_curves


def construct_geometry_symmetric_azimuths(self, azimuths, phases):
    """
    prepare set of orbital positions that are symmetrical in therms of surface geometry, where orbital position is
    mirrored via apsidal line in order to reduce time for generating the light curve
    :param self: elisa.binary_star.system.BinarySystem
    :param azimuths: numpy.array - orbital azimuths of positions in which LC will be calculated
    :param phases: numpy.array - orbital phase of positions in which LC will be calculated
    :return: tuple - unique_phase_indices - numpy.array : indices that points to the orbital positions from one half of the
                                                       orbital motion divided by apsidal line
                   - orbital_motion_counterpart - list - Positions produced by mirroring orbital positions given by
                                                         indices `unique_phase_indices`
                   - orbital_motion_array_counterpart - numpy.array - sa as `orbital_motion_counterpart` but in numpy.array
                                                        form
    """
    azimuth_boundaries = [self.argument_of_periastron, (self.argument_of_periastron + const.PI) % const.FULL_ARC]
    unique_geometry = np.logical_and(azimuths > azimuth_boundaries[0],
                                     azimuths < azimuth_boundaries[1]) \
        if azimuth_boundaries[0] < azimuth_boundaries[1] else np.logical_xor(azimuths < azimuth_boundaries[0],
                                                                             azimuths > azimuth_boundaries[1])
    unique_phase_indices = np.arange(phases.shape[0])[unique_geometry]
    unique_geometry_azimuths = azimuths[unique_geometry]
    unique_geometry_counterazimuths = (2 * self.argument_of_periastron - unique_geometry_azimuths) % const.FULL_ARC

    orbital_motion_counterpart, orbital_motion_array_counterpart = \
        self.calculate_orbital_motion(input_argument=unique_geometry_counterazimuths,
                                      return_nparray=True,
                                      calculate_from='azimuth')

    return unique_phase_indices, orbital_motion_counterpart, orbital_motion_array_counterpart, unique_geometry


def prepare_star_container(self, orbital_position, ecl_boundaries):
    """
    prepares a postion container for given orbital position where visibe/non visible faces are calculated and
    metallicities are assigned

    :param self: BinarySystem
    :param orbital_position: Position
    :param ecl_boundaries: numpy.array - orbital azimuths of eclipses
    :return: container - SingleOrbitalPositionContainer
    """
    system_positions_container = self.prepare_system_positions_container(orbital_motion=[orbital_position],
                                                                         ecl_boundaries=ecl_boundaries)
    system_positions_container = system_positions_container.darkside_filter()
    # for containerf in system_positions_container:
    #     pass
    container = next(iter(system_positions_container))

    # injected attributes
    setattr(container.primary, 'metallicity', self.primary.metallicity)
    setattr(container.secondary, 'metallicity', self.secondary.metallicity)
    return container


def calculate_surface_parameters(container, in_eclipse=True):
    """
    function prepares surface-related parameters such as coverage(area o visibility of the triangles), and directional
    cosines towards line-of-sight vector

    :param container: SingleOrbitalPositionContainer
    :param in_eclipse: bool - switch to indicate if in orout of eclipse calculations to use, if you are not sure leave
                              it to True
    :return: tuple - coverage - numpy.array - visible area of triangles
                   - p_cosines, s_cosines - numpy.array - directional cosines for each face with respect to line-of-sight
                                                       vector
    """
    coverage = compute_surface_coverage(container, in_eclipse=in_eclipse)
    p_cosines = utils.calculate_cos_theta_los_x(container.primary.normals)
    s_cosines = utils.calculate_cos_theta_los_x(container.secondary.normals)
    cosines = {'primary': p_cosines, 'secondary': s_cosines}
    return coverage, cosines


def calculate_lc_point(container, band, ld_cfs, normal_radiance):
    """
    calculates point on the light curve for given band

    :param container: SingleOrbitalPositionContainer
    :param band: str - name of the photometric band
    :param ld_cfs: dict - {'primary': numpy.float of ld coefficents, etc for secondary}
    :param normal_radiance: dict - {'primary': numpy.float of normal radiances, etc for secondary}
    :return:
    """
    ld_law_cfs_columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]
    ld_cors = {component: ld.limb_darkening_factor(coefficients=ld_cfs[component][band][ld_law_cfs_columns].values,
                                                   limb_darkening_law=config.LIMB_DARKENING_LAW,
                                                   cos_theta=container.cosines[component])
               for component in config.BINARY_COUNTERPARTS.keys()}
    # fixme: add all missing multiplicators (at least is missing semi_major_axis^2 in physical units)
    flux = {
        component:
            np.sum(normal_radiance[component][band] * container.cosines[component] *
               container.coverage[component] * ld_cors[component])
        for component in config.BINARY_COUNTERPARTS.keys()
    }
    flux = flux['primary'] + flux['secondary']
    return flux


def integrate_lc_using_approx1(self, orbital_motion, orbital_motion_counterpart, unique_phase_indices, uniq_geom_test,
                               ecl_boundaries, phases, orbital_motion_array_counterpart, new_geometry_test, **kwargs):
    """
    function calculates LC for eccentric orbits for selected filters using approximation where LC points on the one side
    of the apsidal line are calculated exactly and the second half of the LC points are calculated by mirroring the
    surface geometries of the first half of the points to the other side of the apsidal line. Since those mirrored
    points are no alligned with desired phases, the fluxes for each phase is interpolated if missing.

    :param new_geometry_test: bool array - mask to indicate, during which orbital position, surface geometry should be
                                           recalculated
    :param self: BinarySystem instance
    :param orbital_motion: list of all OrbitalPositions at which LC will be calculated
    :param orbital_motion_counterpart: list of OrbitalPositions on one side of the apsidal line on which approximation
    is performed
    :param unique_phase_indices: list of indices that points to OrbitalPositions which geometries will be used for their
    counterparts on the other side of apsidal line
    :param uniq_geom_test: boll array that is used as a mask to select orbital positions from one side of the apsidal
    line which LC points will be calculated exactly
    :param ecl_boundaries: list of phase boundaries of eclipses
    :param phases: phases in which the phase curve will be calculated
    :param orbital_motion_array_counterpart: array of orbital positions that will be interpolated
    :param kwargs: kwargs taken from `compute_eccentric_lightcurve` function
    :return: dictionary of fluxes for each filter
    """
    band_curves = {key: list() for key in kwargs["passband"].keys()}
    band_curves_counterpart = {key: list() for key in kwargs["passband"].keys()}
    # for orbital_position in orbital_motion:
    for counterpart_idx, unique_phase_idx in enumerate(unique_phase_indices):
        orbital_position = orbital_motion[unique_phase_idx]
        self = update_surface_in_ecc_orbits(self, orbital_position, new_geometry_test[counterpart_idx])

        container = prepare_star_container(self, orbital_position, ecl_boundaries)
        container_counterpart = prepare_star_container(self, orbital_motion_counterpart[counterpart_idx],
                                                       ecl_boundaries)

        normal_radiance = get_normal_radiance(container, **kwargs)
        ld_cfs = get_limbdarkening_cfs(container, **kwargs)

        container.coverage, container.cosines = calculate_surface_parameters(container, in_eclipse=True)
        container_counterpart.coverage, container_counterpart.cosines = \
            calculate_surface_parameters(container_counterpart, in_eclipse=True)

        for band in kwargs["passband"].keys():
            band_curves[band].append(calculate_lc_point(container, band, ld_cfs, normal_radiance))
            band_curves_counterpart[band].append(calculate_lc_point(container_counterpart, band, ld_cfs,
                                                                    normal_radiance))

    # interpolation of the points in the second half of the light curves using splines
    x = np.concatenate((phases[unique_phase_indices], orbital_motion_array_counterpart[:, 4] % 1))
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    x = np.concatenate(([x[-1] - 1], x, [x[0] + 1]))
    phases_to_interp = phases[~uniq_geom_test]
    for band in kwargs["passband"].keys():
        y = np.concatenate((band_curves[band], band_curves_counterpart[band]))
        y = y[sort_idx]
        y = np.concatenate(([y[-1]], y, [y[0]]))
        f = interp1d(x, y, kind='cubic')
        interpolated_fluxes = f(phases_to_interp)
        # band_curves[band] = np.concatenate((band_curves[band], interpolated_fluxes))
        full_crv = np.empty(phases.shape)
        full_crv[uniq_geom_test] = band_curves[band]
        full_crv[~uniq_geom_test] = interpolated_fluxes
        band_curves[band] = full_crv

    return band_curves


def integrate_lc_using_approx2(self, orbital_motion, missing_phases_indices, index_of_closest,
                               index_of_closest_reversed, uniq_geom_test, ecl_boundaries, phases, new_geometry_test,
                               **kwargs):
    """
    function calculates LC for eccentric orbit for selected filters using approximation where to each OrbitalPosition on
    one side of the apsidal line, the closest counterpart OrbitalPosition is assigned and the same surface geometry is
    assumed for both of them.

    :param new_geometry_test: bool array - mask to indicate, during which orbital position, surface geometry should be
                                           recalculated
    :param self: BinarySystem instance
    :param orbital_motion: list of all OrbitalPositions at which LC will be calculated
    :param missing_phases_indices: if the number of phase curve is odd, or due to specific alligning of the phases along
    the orbit, the projection between two groups of the points is not necessarilly bijective. In such case
    `missing_phases_indices` point to the OrbitalPositions from approximated side of the orbit that doesnt have the
    counterpart on the other side of the apsidal line yet. This issue is remedied inside the function
    :param index_of_closest: list of indices that points to the counterpart OrbitalPositions on the approximated side of
    the orbit, The n-th index points to the conterpart of the n-th Orbital position on the exactly evaluated side of the
    orbit
    :param index_of_closest_reversed: for OrbitalPositions without counterpart, the index of the closest counterpart
    from the exactly evaluated side of the orbit is supplied
    :param uniq_geom_test: boll array that is used as a mask to select orbital positions from one side of the apsidal
    line which LC points will be calculated exactly
    :param ecl_boundaries: list of phase boundaries of eclipses
    :param phases: phases in which the phase curve will be calculated
    :param kwargs: kwargs taken from `compute_eccentric_lightcurve` function
    :return: dictionary of fluxes for each filter
    """
    band_curves = {key: np.empty(phases.shape) for key in kwargs["passband"].keys()}

    template_phases_idx = np.arange(phases.shape[0])[uniq_geom_test]
    orb_motion_template = [orbital_motion[ii] for ii in template_phases_idx]
    counterpart_phases_idx = np.arange(phases.shape[0])[~uniq_geom_test]
    orb_motion_counterpart = [orbital_motion[ii] for ii in counterpart_phases_idx]

    # appending orbital motion arrays to include missing phases to complete LC
    if len(missing_phases_indices) > 0:
        for ii, idx_reversed in enumerate(index_of_closest_reversed):
            orb_motion_template.append(orb_motion_template[idx_reversed])
            orb_motion_counterpart.append(orb_motion_counterpart[missing_phases_indices[ii]])

    for counterpart_idx, orbital_position in enumerate(orb_motion_template):
        self = update_surface_in_ecc_orbits(self, orbital_position, new_geometry_test[counterpart_idx])

        orbital_position_counterpart = orb_motion_counterpart[index_of_closest[counterpart_idx]]

        container = prepare_star_container(self, orbital_position, ecl_boundaries)
        container_counterpart = prepare_star_container(self, orbital_position_counterpart, ecl_boundaries)

        normal_radiance = get_normal_radiance(container, **kwargs)
        ld_cfs = get_limbdarkening_cfs(container, **kwargs)

        container.coverage, container.cosines = calculate_surface_parameters(container, in_eclipse=True)
        container_counterpart.coverage, container_counterpart.cosines = \
            calculate_surface_parameters(container_counterpart, in_eclipse=True)

        for band in kwargs["passband"].keys():
            band_curves[band][int(orbital_position.idx)] = \
                calculate_lc_point(container, band, ld_cfs, normal_radiance)
            band_curves[band][int(orb_motion_counterpart[index_of_closest[counterpart_idx]].idx)] = \
                calculate_lc_point(container_counterpart, band, ld_cfs, normal_radiance)

    return band_curves


def integrate_lc_exactly(self, orbital_motion, ecl_boundaries, phases, **kwargs):
    """
    function calculates LC for eccentric orbit for selected filters. LC is calculated exactly for each OrbitalPosition.
    It is very slow and it should be used only as a benchmark.

    :param self: BinarySystem instance
    :param orbital_motion: list of all OrbitalPositions at which LC will be calculated
    :param ecl_boundaries: list of phase boundaries of eclipses
    :param phases: phases in which the phase curve will be calculated
    :param kwargs: kwargs taken from `compute_eccentric_lightcurve` function
    :return: dictionary of fluxes for each filter
    """
    band_curves = {key: np.empty(phases.shape) for key in kwargs["passband"].keys()}
    for orbital_position in orbital_motion:
        self.build(components_distance=orbital_position.distance)
        container = prepare_star_container(self, orbital_position, ecl_boundaries)

        normal_radiance = get_normal_radiance(container, **kwargs)
        ld_cfs = get_limbdarkening_cfs(container, **kwargs)

        container.coverage, container.cosines = calculate_surface_parameters(container, in_eclipse=True)

        for band in kwargs["passband"].keys():
            band_curves[band][int(orbital_position.idx)] = \
                calculate_lc_point(container, band, ld_cfs, normal_radiance)

    return band_curves


def calculate_new_geometry(self, orbit_template_arr, rel_d_radii):
    """
    this function chcecks at which OrbitalPositions it is necessary to recalculate geometry
    :param self: BinarySystem instance
    :param orbit_template_arr: array of orbital positions from one side of the apsidal line used as the symmetry
    template
    :param rel_d_radii: numpy.array - shape(2 x len(orbit_template arr) - relative changes in radii of each component with
    respect to the previous OrbitalPosition, excluding the first postition.
    :return: bool array - mask to select Orbital positions, where orbits should be calculated
    """
    # in case of spots, the boundary points will cause problems if you want to use the same geometry
    if self.primary.has_spots() or self.secondary.has_spots():
        return np.ones(orbit_template_arr.shape[0], dtype=np.bool)

    calc_new_geometry = np.empty(orbit_template_arr.shape[0], dtype=np.bool)
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


def update_surface_in_ecc_orbits(self, orbital_position, new_geometry_test):
    """
    function decides how to update surface properties with respect to the degree of change in surface geometry given by
    new_geometry test, if true, only points and normals are recalculated, otherwise surface is calculated from scratch
    :param self: BinarySystem instance
    :param orbital_position:  OrbitalPosition list
    :param new_geometry_test: bool - test that will decide, how the following phase will be calculated
    :return: BinarySystem instance with updated geometry
    """
    if new_geometry_test:
        self.build(components_distance=orbital_position.distance)
    else:
        self.build_mesh(component=None, components_distance=orbital_position.distance)
        self.build_surface_areas(component=None)
        self.build_faces_orientation(component=None, components_distance=orbital_position.distance)

    return self


def compute_circular_spoty_asynchronous_lightcurve(self, *args, **kwargs):
    self.build(components_distance=1.0)


if __name__ == "__main__":
    pass
