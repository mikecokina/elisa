import numpy as np
import logging
import matplotlib.path as mpltpath

from scipy.spatial.qhull import ConvexHull

from elisa.conf import config
from elisa.engine import const, utils, atm, ld, logger
from elisa.engine.binary_system import geo
from elisa.engine.const import BINARY_POSITION_PLACEHOLDER

__logger__ = logging.getLogger(__name__)


def partial_visible_faces_surface_coverage(points, faces, normals, hull):
    pypex_hull = geo.hull_to_pypex_poly(hull)
    pypex_faces = geo.faces_to_pypex_poly(points[faces])
    # it is possible to None happens in intersection, tkae care about it latter
    pypex_intersection = geo.pypex_poly_hull_intersection(pypex_faces, pypex_hull)

    # think about surface normalisation like and avoid surface areas like 1e-6 which lead to precission lose

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


def get_normal_radiance(self, **kwargs):
    primary = atm.NaiveInterpolatedAtm.radiance(
        **dict(
            temperature=self.primary.temperatures,
            log_g=self.primary.log_g,
            metallicity=self.primary.metallicity,
            **kwargs
        )
    )

    secondary = atm.NaiveInterpolatedAtm.radiance(
        **dict(
            temperature=self.secondary.temperatures,
            log_g=self.secondary.log_g,
            metallicity=self.secondary.metallicity,
            **kwargs
        )
    )
    return primary, secondary


def get_limbdarkening(self, **kwargs):
    return [
        ld.interpolate_on_ld_grid(
            temperature=getattr(self, component).temperatures,
            log_g=getattr(self, component).log_g,
            metallicity=getattr(self, component).metallicity,
            passband=kwargs["passband"]
        ) for component in config.BINARY_COUNTERPARTS.keys()
    ]


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
    orbital_motion = position_method(phase=base_phases2)

    initial_props_container = geo.SingleOrbitalPositionContainer(self.primary, self.secondary)
    initial_props_container.setup_position(BINARY_POSITION_PLACEHOLDER(*(0, 1.0, 0.0, 0.0, 0.0)), self.inclination)

    # injected attributes
    setattr(initial_props_container.primary, 'metallicity', self.primary.metallicity)
    setattr(initial_props_container.secondary, 'metallicity', self.secondary.metallicity)

    primary_normal_radiance, secondary_normal_radiance = get_normal_radiance(initial_props_container, **kwargs)
    primary_ld_cfs, secondary_ld_cfs = get_limbdarkening(initial_props_container, **kwargs)
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
            p_ld_cors = ld.limb_darkening_factor(coefficients=primary_ld_cfs[band][ld_law_cfs_columns].values.T,
                                                 limb_darkening_law=config.LIMB_DARKENING_LAW,
                                                 cos_theta=p_cosines)[0]

            s_ld_cors = ld.limb_darkening_factor(coefficients=secondary_ld_cfs[band][ld_law_cfs_columns].values.T,
                                                 limb_darkening_law=config.LIMB_DARKENING_LAW,
                                                 cos_theta=s_cosines)[0]
            # fixme: add all missing multiplicators (at least is missing semi_major_axis^2 in physical units)
            p_flux = np.sum(primary_normal_radiance[band] * p_cosines * coverage["primary"] * p_ld_cors)
            s_flux = np.sum(secondary_normal_radiance[band] * s_cosines * coverage["secondary"] * s_ld_cors)
            flux = p_flux + s_flux
            # band_curves[band].append(flux)
            band_curves[band][idx] = flux
    band_curves = {band: band_curves[band][reverse_idx2] for band in band_curves}
    return band_curves


def phase_crv_symmetry(self, phase):
    """
    utilizing symmetry of circular systems without spots and pulastions wher e you need to evalueate only half of the
    phases. Function finds such redundant phases and returns only unique phases.
    :param self:
    :param phase:
    :return:
    """
    if self.primary.pulsations is None and self.primary.pulsations is None and \
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
    orbital_motion = kwargs.pop("positions")
    # todo: move it to for loop
    ecl_boundaries = np.array([0, const.PI, const.PI, const.FULL_ARC])

    band_curves = {key: list() for key in kwargs["passband"].keys()}
    ld_law_cfs_columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]

    for orbital_position in orbital_motion:
        self.build(components_distance=orbital_position.distance)
        system_positions_container = self.prepare_system_positions_container(orbital_motion=[orbital_position],
                                                                             ecl_boundaries=ecl_boundaries)
        system_positions_container = system_positions_container.darkside_filter()
        container = next(iter(system_positions_container))

        # injected attributes
        setattr(container.primary, 'metallicity', self.primary.metallicity)
        setattr(container.secondary, 'metallicity', self.secondary.metallicity)

        primary_normal_radiance, secondary_normal_radiance = get_normal_radiance(container, **kwargs)
        primary_ld_cfs, secondary_ld_cfs = get_limbdarkening(container, **kwargs)

        coverage = compute_surface_coverage(container, in_eclipse=True)
        p_cosines = utils.calculate_cos_theta_los_x(container.primary.normals)
        s_cosines = utils.calculate_cos_theta_los_x(container.secondary.normals)

        for band in kwargs["passband"].keys():
            p_ld_cors = ld.limb_darkening_factor(coefficients=primary_ld_cfs[band][ld_law_cfs_columns].values.T,
                                                 limb_darkening_law=config.LIMB_DARKENING_LAW,
                                                 cos_theta=p_cosines)[0]

            s_ld_cors = ld.limb_darkening_factor(coefficients=secondary_ld_cfs[band][ld_law_cfs_columns].values.T,
                                                 limb_darkening_law=config.LIMB_DARKENING_LAW,
                                                 cos_theta=s_cosines)[0]
            # fixme: add all missing multiplicators (at least is missing semi_major_axis^2 in physical units)
            p_flux = np.sum(primary_normal_radiance[band] * p_cosines * coverage["primary"] * p_ld_cors)
            s_flux = np.sum(secondary_normal_radiance[band] * s_cosines * coverage["secondary"] * s_ld_cors)
            flux = p_flux + s_flux
            band_curves[band].append(flux)

    # temporary
    from matplotlib import pyplot as plt
    for band, curve in band_curves.items():
        x = np.arange(len(curve))
        plt.scatter(x, curve)
    plt.show()

    return band_curves





if __name__ == "__main__":
    pass
