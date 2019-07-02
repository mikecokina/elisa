import numpy as np
import logging
import matplotlib.path as mpltpath

from scipy.spatial.qhull import ConvexHull

from elisa.conf import config
from elisa.engine import const, utils, atm, ld, logger
from elisa.engine.binary_system import geo
from elisa.engine.const import BINARY_POSITION_PLACEHOLDER
from scipy.interpolate import interp1d

__logger__ = logging.getLogger(__name__)


def partial_visible_faces_surface_coverage(points, faces, normals, hull):
    pypex_hull = geo.hull_to_pypex_poly(hull)
    pypex_faces = geo.faces_to_pypex_poly(points[faces])
    # it is possible to None happens in intersection, tkae care about it latter
    pypex_intersection = geo.pypex_poly_hull_intersection(pypex_faces, pypex_hull)

    # think about surface normalisation like and avoid surface areas like 1e-6 which lead to precission lose

    try:
        pypex_polys_surface_area = np.array(geo.pypex_poly_surface_area(pypex_intersection), dtype=np.float)
    except:
        print("ok")

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
    # bb_path = get_eclipse_boundary_path(cover_object_obs_visible_projection)
    # # obtain points out of eclipse (out of boundary defined by hull of 'infront' object)
    # out_of_bound = np.invert(bb_path.contains_points(undercover_object_obs_visible_projection))
    # undercover_visible_point_indices = undercover_visible_point_indices[out_of_bound]

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


def get_normal_radiance(self, **kwargs):
    return {
    'primary': atm.NaiveInterpolatedAtm.radiance(
        **dict(
            temperature=self.primary.temperatures,
            log_g=self.primary.log_g,
            metallicity=self.primary.metallicity,
            **kwargs
        )
    ),

    'secondary': atm.NaiveInterpolatedAtm.radiance(
        **dict(
            temperature=self.secondary.temperatures,
            log_g=self.secondary.log_g,
            metallicity=self.secondary.metallicity,
            **kwargs
        )
    ),
    }


def get_limbdarkening(self, **kwargs):
    """
    returns limg darkening coefficients for each face of each component
    :param self:
    :param kwargs: dict - {'primary': np.array, 'secondary': np.array}
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
    base_phases2, reverse_idx2 = phase_crv_symmetry(self, phases)

    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=base_phases2, return_nparray=False, calculate_from='phase')

    initial_props_container = geo.SingleOrbitalPositionContainer(self.primary, self.secondary)
    initial_props_container.setup_position(BINARY_POSITION_PLACEHOLDER(*(0, 1.0, 0.0, 0.0, 0.0)), self.inclination)

    # injected attributes
    setattr(initial_props_container.primary, 'metallicity', self.primary.metallicity)
    setattr(initial_props_container.secondary, 'metallicity', self.secondary.metallicity)

    normal_radiance = get_normal_radiance(initial_props_container, **kwargs)
    ld_cfs = get_limbdarkening(initial_props_container, **kwargs)
    ld_law_cfs_columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]

    system_positions_container = self.prepare_system_positions_container(orbital_motion=orbital_motion,
                                                                         ecl_boundaries=ecl_boundaries)
    system_positions_container = system_positions_container.darkside_filter()

    band_curves = {key: np.empty(base_phases2.shape) for key in kwargs["passband"].keys()}
    for idx, container in enumerate(system_positions_container):
        coverage = compute_surface_coverage(container, in_eclipse=system_positions_container.in_eclipse[idx])
        p_cosines = utils.calculate_cos_theta_los_x(container.primary.normals)
        s_cosines = utils.calculate_cos_theta_los_x(container.secondary.normals)

        for band in kwargs["passband"].keys():
            # fixme: do something with this fucking zero indexing
            p_ld_cors = ld.limb_darkening_factor(coefficients=ld_cfs["primary"][band][ld_law_cfs_columns].values.T,
                                                 limb_darkening_law=config.LIMB_DARKENING_LAW,
                                                 cos_theta=p_cosines)[0]

            s_ld_cors = ld.limb_darkening_factor(coefficients=ld_cfs["secondary"][band][ld_law_cfs_columns].values.T,
                                                 limb_darkening_law=config.LIMB_DARKENING_LAW,
                                                 cos_theta=s_cosines)[0]
            # fixme: add all missing multiplicators (at least is missing semi_major_axis^2 in physical units)
            p_flux = np.sum(normal_radiance["primary"][band] * p_cosines * coverage["primary"] * p_ld_cors)
            s_flux = np.sum(normal_radiance["secondary"][band] * s_cosines * coverage["secondary"] * s_ld_cors)
            flux = p_flux + s_flux
            # band_curves[band].append(flux)
            band_curves[band][idx] = flux
    band_curves = {band: band_curves[band][reverse_idx2] for band in band_curves}

    return band_curves


def phase_crv_symmetry(self, phase):
    """
    utilizing symmetry of circular systems without spots and pulastions where you need to evaluate only half of the
    phases. Function finds such redundant phases and returns only unique phases.
    :param self:
    :param phase:
    :return:
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
    # todo: move it to for loop
    ecl_boundaries = geo.get_eclipse_boundaries(self, 1.0)

    phases = kwargs.pop("phases")

    position_method = kwargs.pop("position_method")
    orbital_motion, orbital_motion_array = position_method(input_argument=phases,
                                                           return_nparray=True, calculate_from='phase')
    azimuths = orbital_motion_array[:, 2]

    approximation_test1 = len(phases) > config.POINTS_ON_ECC_ORBIT and self.primary.synchronicity == 1.0 and \
                        self.secondary.synchronicity == 1.0

    # in case of clean surface or synchronous rotation (moreless), symmetry around semi-major axis can be utilized
    # mask isolating the symmetrical part of the orbit
    unique_phase_indices, orbital_motion_counterpart, orbital_motion_array_counterpart, uniq_geom_test = \
        cunstruct_geometry_symmetric_azimuths(self, azimuths, phases)

    # if approximation_test1:
    #     unique_phase_indices, orbital_motion_counterpart, orbital_motion_array_counterpart, uniq_geom_test = \
    #         cunstruct_geometry_symmetric_azimuths(self, azimuths, phases)
    #     # counterpart_phases = orbital_motion_array_counterpart[:, 4]
    # else:
    #     # # calculating all forward radii
    #     # distances = orbital_motion_array[:, 1]
    #     # forward_rad = self.calculate_all_forward_radii(distances, components=None)
    #     #
    #     # # calculating relative changes in radii
    #     # rel_d_forward_radii = {component: np.abs(radii - np.roll(radii, 1)) / radii for component, radii in
    #     #                        forward_rad.items()}
    #     # max_rel_d_forward_radii = np.max([rel_d_forward_radii['primary'].max(), rel_d_forward_radii['secondary'].max()])
    #
    #     # second approximation does not interpolates the resulting light curve but assumes that geometry is the same as
    #     # the geometry of the found counterpart
    index_of_closest = utils.find_idx_of_nearest(orbital_motion_array_counterpart[:, 1],
                                                 orbital_motion_array[~uniq_geom_test, 1])
    d_distance = np.abs(orbital_motion_array[~uniq_geom_test, 1] -
                        orbital_motion_array_counterpart[index_of_closest, 1])
    approximation_test2 = max(d_distance) < config.MAX_D_DISTANCE and \
                          self.primary.synchronicity == 1.0 and self.secondary.synchronicity == 1.0

    band_curves = {key: list() for key in kwargs["passband"].keys()}

    #initial values of radii to be compared with
    orig_forward_rad_p, orig_forward_rad_p = 100.0, 100.0  # 100.0 is too large value, it will always fail the first
    # test and therefore the surface will be built
    if approximation_test1:
        band_curves_counterpart = {key: list() for key in kwargs["passband"].keys()}
        # for orbital_position in orbital_motion:
        for counterpart_idx, unique_phase_idx in enumerate(unique_phase_indices):
            orbital_position = orbital_motion[unique_phase_idx]

            self.build(components_distance=orbital_position.distance)

            container = prepare_star_container(self, orbital_position, ecl_boundaries)
            container_counterpart = prepare_star_container(self, orbital_motion_counterpart[counterpart_idx],
                                                           ecl_boundaries)

            normal_radiance = get_normal_radiance(container, **kwargs)
            ld_cfs = get_limbdarkening(container, **kwargs)

            container.coverage, container.cosines = calculate_surface_parameters(container, in_eclipse=True)
            container_counterpart.coverage, container_counterpart.cosines = \
                calculate_surface_parameters(container_counterpart, in_eclipse=True)

            for band in kwargs["passband"].keys():
                band_curves[band].append(calculate_lc_point(container, band, ld_cfs, normal_radiance))
                band_curves_counterpart[band].append(calculate_lc_point(container_counterpart, band, ld_cfs,
                                                                        normal_radiance))

    elif approximation_test2:
        band_curves_counterpart = {key: list() for key in kwargs["passband"].keys()}
        for counterpart_idx, unique_phase_idx in enumerate(unique_phase_indices):
            pass

    else:
        for orbital_position in orbital_motion:
            self.build(components_distance=orbital_position.distance)
            container = prepare_star_container(self, orbital_position, ecl_boundaries)

            normal_radiance = get_normal_radiance(container, **kwargs)
            ld_cfs = get_limbdarkening(container, **kwargs)

            container.coverage, container.cosines = calculate_surface_parameters(container, in_eclipse=True)

            for band in kwargs["passband"].keys():
                band_curves[band].append(calculate_lc_point(container, band, ld_cfs, normal_radiance))

    # LC interpolation of symmetrical part
    if approximation_test1:
        x = np.concatenate((phases[unique_phase_indices], orbital_motion_array_counterpart[:, 4] % 1))
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        x = np.concatenate(([x[-1]-1], x, [x[0]+1]))
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

    # # temporary
    # from matplotlib import pyplot as plt
    # for band, curve in band_curves.items():
    #     x = phases[unique_phase_indices]
    #     plt.scatter(x, curve[unique_phase_indices])
    #     plt.scatter(orbital_motion_array_counterpart[:, 4] % 1, band_curves_counterpart[band])
    # plt.show()

    return band_curves


def cunstruct_geometry_symmetric_azimuths(self, azimuths, phases):
    """
    prepare set of orbital positions that are symmetrical in therms of surface geometry, where orbital position is
    mirrored via apsidal line in order to reduce time for generating the light curve
    :param self: BinarySystem
    :param azimuths: np.array - orbital azimuths of positions in which LC will be calculated
    :param phases: np.array - orbital phase of positions in which LC will be calculated
    :return: tuple - unique_phase_indices - np.array : indices that points to the orbital positions from one half of the
                                                       orbital motion divided by apsidal line
                   - orbital_motion_counterpart - list - Positions produced by mirroring orbital positions given by
                                                         indices `unique_phase_indices`
                   - orbital_motion_array_counterpart - np.array - sa as `orbital_motion_counterpart` but in np.array
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
    # unique_geometry_counterazimuths = np.concatenate(([azimuth_boundaries[0]],
    #                                                   unique_geometry_counterazimuths,
    #                                                   [azimuth_boundaries[1]]))
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
    :param ecl_boundaries: np.array - orbital azimuths of eclipses
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
    :return: tuple - coverage - np.array - visible area of triangles
                   - p_cosines, s_cosines - np.array - directional cosines for each face with respect to line-of-sight
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
    :param ld_cfs: dict - {'primary': np.float of ld coefficents, etc for secondary}
    :param normal_radiance: dict - {'primary': np.float of normal radiances, etc for secondary}
    :return:
    """
    ld_law_cfs_columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]
    ld_cors = {component: ld.limb_darkening_factor(coefficients=ld_cfs[component][band][ld_law_cfs_columns].values.T,
                                                   limb_darkening_law=config.LIMB_DARKENING_LAW,
                                                   cos_theta=container.cosines[component])[0]
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


if __name__ == "__main__":
    pass
