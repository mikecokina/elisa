import numpy as np
import matplotlib.path as mpltpath

from scipy.spatial.qhull import ConvexHull

from elisa.conf.config import BINARY_COUNTERPARTS
from elisa.engine import const, utils
from elisa.engine.binary_system import geo


def partial_visible_faces_surface_coverage(points, faces, normals, hull):
    # todo: pypex is too slow, need some improvement; I don't care right now
    pypex_hull = geo.hull_to_pypex_poly(hull)
    pypex_faces = geo.faces_to_pypex_poly(points[faces])
    # it is possible to None happens in intersection, tkae care about it latter
    pypex_intersection = geo.pypex_poly_hull_intersection(pypex_faces, pypex_hull)

    # think about surface normalisation like and avoid surface areas like 1e-6 which lead to precission lose
    pypex_polys_surface_area = geo.pypex_poly_surface_area(pypex_intersection)
    correction_cosine = (np.dot(const.BINARY_SIGHT_OF_VIEW, normal) / np.linalg.norm(normal) for normal in normals)
    return [a / b for a, b, in zip(pypex_polys_surface_area, correction_cosine)]


def get_visible_projection(obj):
    return  geo.plane_projection(
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
    undercover_object = getattr(container, BINARY_COUNTERPARTS[cover_component])
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
        "container": container,
        "cover": {
            "coverage": cover_obj_coverage,
            "visible": full_visible,
            "partial_visible": partial_visible,
            "invisible": invisible
        },
        "undercover": {

            "coverage": undercover_obj_coverage,
            # add
        }
    }


def compute_circular_synchronous_lightcurve(self, **kwargs):
    # get orbital motion from kwargs
    # get eclipses
    # prepare rotated (orb motion, inclination) points and normals
    # apply darkside filter !!!!!!! just indices information !!!!!!!

    # NOTE: extention of system primary and secondary points due to horizont faces fractalisation
    # apply eclipse filter !!!!!!! just indices information !!!!!!!

    orbital_motion = kwargs.pop("positions")
    eclipses = geo.get_eclipse_boundaries(self, 1.0)
    # system_positions_container = self.prepare_system_positions_container(orbital_motion=orbital_motion)
    # system_positions_container = system_positions_container.darkside_filter()

    # todo: it makes more sense to do eclipse filter in same way as darkside filter
    # todo: rewrite it in the future

    # for container in system_positions_container:
    #     pass

    import pickle
    container = pickle.load(open("container.pickle", "rb"))
    coverage = compute_surface_coverage(container)

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
