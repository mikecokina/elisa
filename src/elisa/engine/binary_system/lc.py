import numpy as np
import matplotlib.path as mpltpath

from scipy.spatial.qhull import ConvexHull

from elisa.conf import config
from elisa.engine import const, utils, atm, ld
from elisa.engine.binary_system import geo


def partial_visible_faces_surface_coverage(points, faces, normals, hull):
    # todo: pypex is too slow, need some improvement; I don't care right now
    pypex_hull = geo.hull_to_pypex_poly(hull)
    pypex_faces = geo.faces_to_pypex_poly(points[faces])
    # it is possible to None happens in intersection, tkae care about it latter
    pypex_intersection = geo.pypex_poly_hull_intersection(pypex_faces, pypex_hull)

    # think about surface normalisation like and avoid surface areas like 1e-6 which lead to precission lose
    pypex_polys_surface_area = geo.pypex_poly_surface_area(pypex_intersection)
    inplane_points_3d = np.concatenate((points.T, [[0.0] * len(points)])).T
    inplane_surface_area = utils.triangle_areas(triangles=faces, points=inplane_points_3d)
    correction_cosine = (np.dot(const.LINE_OF_SIGHT, normal) / np.linalg.norm(normal) for normal in normals)
    # todo: profile case when generator will be evaluted first, then substracted from inplane_surface area instead of
    # todo: this
    return [(c - a) / b for a, b, c in zip(pypex_polys_surface_area, correction_cosine, inplane_surface_area)]


def get_visible_projection(obj):
    return geo.plane_projection(
        obj.points[
            list(set(obj.faces[obj.indices].flatten()))
        ], "yz"
    )


def get_eclipse_boundary_path(hull):
    cover_bound = ConvexHull(hull)
    hull_points = hull[cover_bound.vertices]
    bb_path = mpltpath.Path(hull_points)
    return bb_path


def compute_surface_coverage(container: geo.SingleOrbitalPositionContainer):
    cover_component = 'secondary' if 0.0 < container.position.azimut < const.PI else 'primary'
    cover_object = getattr(container, cover_component)
    undercover_object = getattr(container, config.BINARY_COUNTERPARTS[cover_component])
    undercover_visible_point_indices = list(set(undercover_object.faces[undercover_object.indices].flatten()))

    cover_object_obs_visible_projection = get_visible_projection(cover_object)
    undercover_object_obs_visible_projection = get_visible_projection(undercover_object)

    # get matplotlib boudary path defined by hull of projection
    bb_path = get_eclipse_boundary_path(cover_object_obs_visible_projection)
    # obtain points out of eclipse (out of boundary defined by hull of 'infront' object)
    out_of_bound = np.invert(bb_path.contains_points(undercover_object_obs_visible_projection))
    undercover_visible_point_indices = np.array(undercover_visible_point_indices)[out_of_bound]
    undercover_faces = np.array([const.FALSE_FACE_PLACEHOLDER] * int(len(undercover_object.normals)))
    undercover_faces[undercover_object.indices] = undercover_object.faces[undercover_object.indices]
    eclipse_faces_visibility = np.isin(undercover_faces, undercover_visible_point_indices)

    # get indices of full visible, invisible and partial visible faces
    full_visible = np.array([np.all(face) for face in eclipse_faces_visibility])
    invisible = np.array([np.all(face) for face in np.invert(eclipse_faces_visibility)])
    partial_visible = np.invert(full_visible | invisible)

    # process partial and full visible faces (get surface area of 3d polygon) of undercover object
    partial_visible_faces = undercover_object.faces[partial_visible]
    partial_visible_normals = undercover_object.normals[partial_visible]
    undercover_object_pts_projection = geo.plane_projection(undercover_object.points, "yz", keep_3d=False)

    partial_coverage = partial_visible_faces_surface_coverage(
        points=undercover_object_pts_projection,
        faces=partial_visible_faces,
        normals=partial_visible_normals,
        hull=bb_path.vertices
    )
    visible_coverage = utils.poly_areas(undercover_object.points[undercover_object.faces[full_visible]])
    undercover_obj_coverage = geo.surface_area_coverage(
        size=len(undercover_faces),
        visible=full_visible, visible_coverage=visible_coverage,
        partial=partial_visible, partial_coverage=partial_coverage
    )

    # process full visible faces of cover object
    visible_coverage = utils.poly_areas(cover_object.points[cover_object.faces[cover_object.indices]])
    cover_obj_coverage = geo.surface_area_coverage(len(cover_object.faces), cover_object.indices, visible_coverage)

    return {
        cover_component: cover_obj_coverage,
        config.BINARY_COUNTERPARTS[cover_component]: undercover_obj_coverage
    }

    # return {
    #     "cover": {
    #         "coverage": cover_obj_coverage,
    #         "visible": full_visible,
    #         "partial_visible": partial_visible,
    #         "invisible": invisible
    #     },
    #     "undercover": {
    #
    #         "coverage": undercover_obj_coverage,
    #         # add
    #     }
    # }


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

    # import pickle
    # pickle.dump(primary, open("primary.atm.pickle", "wb"))
    # pickle.dump(secondary, open("secondary.atm.pickle", "wb"))

    # primary = pickle.load(open("primary.atm.pickle", "rb"))
    # secondary = pickle.load(open("secondary.atm.pickle", "rb"))
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

    # import pickle
    # pickle.dump(x[0], open("primary.ld.pickle", "wb"))
    # pickle.dump(x[1], open("secondary.ld.pickle", "wb"))
    # return pickle.load(open("primary.ld.pickle", "rb")), pickle.load(open("secondary.ld.pickle", "rb"))


def compute_circular_synchronous_lightcurve(self, **kwargs):
    orbital_motion = kwargs.pop("positions")
    # fixme: not used!!
    eclipses = geo.get_eclipse_boundaries(self, 1.0)

    initial_props_container = geo.SingleOrbitalPositionContainer(self.primary, self.secondary)
    initial_props_container.setup_position(geo.PositionContainer(*(0, 1.0, 0.0, 0.0, 0.0)), self.inclination)

    # injected attributes
    setattr(initial_props_container.primary, 'metallicity', self.primary.metallicity)
    setattr(initial_props_container.secondary, 'metallicity', self.secondary.metallicity)

    # primary_normal_radiance, secondary_normal_radiance = get_normal_radiance(initial_props_container, **kwargs)
    primary_ld_cfs, secondary_ld_cfs = get_limbdarkening(initial_props_container, **kwargs)
    ld_law_cfs_columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]

    system_positions_container = self.prepare_system_positions_container(orbital_motion=orbital_motion)
    system_positions_container = system_positions_container.darkside_filter()

    band_curves = {key: list() for key in kwargs["passband"].keys()}
    for container in system_positions_container:
        # container = pickle.load(open("container.pickle", "rb"))
        coverage = compute_surface_coverage(container)
        for band in kwargs["passband"].keys():
            # optimize
            p_cosines = np.array([np.dot(n, const.LINE_OF_SIGHT) / np.linalg.norm(n)
                                  for n in container.primary.normals])

            s_cosines = np.array([np.dot(n, const.LINE_OF_SIGHT) / np.linalg.norm(n)
                                  for n in container.secondary.normals])

            # fixme: do something with this fucking zero indexing
            p_ld_cors = ld.limb_darkening_factor(coefficients=primary_ld_cfs[band][ld_law_cfs_columns].values.T,
                                                 limb_darkening_law=config.LIMB_DARKENING_LAW,
                                                 cos_theta=p_cosines)[0]

            s_ld_cors = ld.limb_darkening_factor(coefficients=secondary_ld_cfs[band][ld_law_cfs_columns].values.T,
                                                 limb_darkening_law=config.LIMB_DARKENING_LAW,
                                                 cos_theta=s_cosines)[0]
            # p_band_normal_radiance = np.array([rad.intensity for rad in primary_normal_radiance[band]])
            # s_band_normal_radiance = np.array([rad.intensity for rad in secondary_normal_radiance[band]])

            # p_flux = sum(p_band_normal_radiance * p_cosines * coverage["primary"] * p_ld_cors)
            # s_flux = sum(s_band_normal_radiance * s_cosines * coverage["secondary"] * s_ld_cors)
            # flux = p_flux + s_flux
            # band_curves[band].append(flux)

    exit()
























    # faces = faces[visible]
    # points = points
    #
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    #
    # clr = 'b'
    # pts = points
    # fcs = faces
    #
    # plot = ax.plot_trisurf(
    #     pts[:, 0], pts[:, 1],
    #     pts[:, 2], triangles=fcs,
    #     antialiased=True, shade=False, color=clr)
    #
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)
    # ax.view_init(0, -np.degrees(0.09424778))
    #
    # plot.set_edgecolor('black')
    #
    # plt.show()
    #
    #
    # pass
    # geo.darkside_filter()

    # compute on filtered atmospheres (doesn't meeter how will be filtered)
    # primary_radiance = \
    #     atm.NaiveInterpolatedAtm.radiance(_temperature, _logg, self.primary.metallicity, config.ATM_ATLAS, **kwargs)

    # primary_radiance = \
    #     atm.NearestAtm.radiance(_temperature, _logg, self.primary.metallicity, config.ATM_ATLAS, **kwargs)


if __name__ == "__main__":
    pass
