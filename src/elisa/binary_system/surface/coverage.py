import numpy as np
import matplotlib.path as mpltpath

from scipy.spatial.qhull import ConvexHull

from elisa.conf import config
from elisa.logger import getLogger
from elisa.binary_system import utils as bsutils
from elisa import (
    umpy as up,
    utils,
    const
)
from elisa.base.surface import coverage as bcoverage

logger = getLogger('binary_system.curves.lcmp')


def partial_visible_faces_surface_coverage(points, faces, normals, hull):
    """
    Compute surface coverage of partialy visible faces.

    :param points: numpy.array;
    :param faces: numpy.array;
    :param normals: numpy.array;
    :param hull: numpy.array; sorted clockwise to create
                              matplotlib.path.Path; path of points boundary of infront component projection
    :return: numpy.array;
    """
    pypex_hull = bsutils.hull_to_pypex_poly(hull)
    pypex_faces = bsutils.faces_to_pypex_poly(points[faces])
    # it is possible to None happens in intersection, tkae care about it latter
    pypex_intersection = bsutils.pypex_poly_hull_intersection(pypex_faces, pypex_hull)

    # think about surface normalisation like and avoid surface areas like 1e-6 which lead to loss in precission
    pypex_polys_surface_area = np.array(bsutils.pypex_poly_surface_area(pypex_intersection), dtype=np.float)

    inplane_points_3d = up.concatenate((points.T, [[0.0] * len(points)])).T
    inplane_surface_area = utils.triangle_areas(triangles=faces, points=inplane_points_3d)
    correction_cosine = utils.calculate_cos_theta_los_x(normals)
    retval = (inplane_surface_area - pypex_polys_surface_area) / correction_cosine
    return retval


def compute_surface_coverage(system, semi_major_axis, in_eclipse=True):
    # todo: add unittests
    """
    Compute surface coverage of faces for given orbital position
    defined by container/SingleOrbitalPositionContainer.

    :param semi_major_axis: float;
    :param system: elisa.binary_system.container.OrbitalPositionContainer;;
    :param in_eclipse: bool;
    :return: Dict;
    """
    logger.debug(f"computing surface coverage for {system.position}")
    cover_component = 'secondary' if 0.0 < system.position.azimuth < const.PI else 'primary'
    cover_object = getattr(system, cover_component)
    undercover_object = getattr(system, config.BINARY_COUNTERPARTS[cover_component])
    undercover_visible_point_indices = np.unique(undercover_object.faces[undercover_object.indices])

    cover_object_obs_visible_projection = utils.get_visible_projection(cover_object)
    undercover_object_obs_visible_projection = utils.get_visible_projection(undercover_object)
    # get matplotlib boudary path defined by hull of projection
    if in_eclipse:
        bb_path = get_eclipse_boundary_path(cover_object_obs_visible_projection)
        # obtain points out of eclipse (out of boundary defined by hull of 'infront' object)
        out_of_bound = up.invert(bb_path.contains_points(undercover_object_obs_visible_projection))
        # undercover_visible_point_indices = undercover_visible_point_indices[out_of_bound]
    else:
        out_of_bound = np.ones(undercover_object_obs_visible_projection.shape[0], dtype=np.bool)

    undercover_visible_point_indices = undercover_visible_point_indices[out_of_bound]
    undercover_faces = -1 * np.ones(np.shape(undercover_object.normals), dtype=int)
    undercover_faces[undercover_object.indices] = undercover_object.faces[undercover_object.indices]
    eclipse_faces_visibility = np.isin(undercover_faces, undercover_visible_point_indices)

    # get indices of full visible, invisible and partial visible faces
    full_visible = np.all(eclipse_faces_visibility, axis=1)
    invisible = np.all(up.invert(eclipse_faces_visibility), axis=1)
    partial_visible = up.invert(full_visible | invisible)

    # process partial and full visible faces (get surface area of 3d polygon) of undercover object
    partial_visible_faces = undercover_object.faces[partial_visible]
    partial_visible_normals = undercover_object.normals[partial_visible]
    undercover_object_pts_projection = utils.plane_projection(undercover_object.points, "yz", keep_3d=False)
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

    undercover_obj_coverage = bcoverage.surface_area_coverage(
        size=np.shape(undercover_object.normals)[0],
        visible=full_visible, visible_coverage=visible_coverage,
        partial=partial_visible, partial_coverage=partial_coverage
    )

    visible_coverage = utils.poly_areas(cover_object.points[cover_object.faces[cover_object.indices]])
    cover_obj_coverage = bcoverage.surface_area_coverage(len(cover_object.faces),
                                                         cover_object.indices, visible_coverage)

    # areas are now in SMA^2, converting to SI
    cover_obj_coverage *= up.power(semi_major_axis, 2)
    undercover_obj_coverage *= up.power(semi_major_axis, 2)

    return {
        cover_component: cover_obj_coverage,
        config.BINARY_COUNTERPARTS[cover_component]: undercover_obj_coverage
    }


def get_eclipse_boundary_path(hull):
    """
    Return `matplotlib.path.Path` object which represents boundary of component projection
    to plane `yz`.

    :param hull: numpy.array;
    :return: matplotlib.path.Path;
    """
    cover_bound = ConvexHull(hull)
    hull_points = hull[cover_bound.vertices]
    bb_path = mpltpath.Path(hull_points)
    return bb_path


def calculate_coverage_with_cosines(system, semi_major_axis, in_eclipse=True):
    """
    Function prepares surface-related parameters such as coverage(area of visibility
    of the triangles) and directional cosines towards line-of-sight vector.

    :param semi_major_axis: float;
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param in_eclipse: bool; indicate if eclipse occur for given position container.
                             If you are not sure leave it to True
    :return: Tuple;

    shape::

        (numpy.array, Dict[str, numpy.array])

    coverage -- numpy.array; visible area of triangles
    p_cosines, s_cosines -- Dict[str, numpy.array]; directional cosines for each face with
    respect to line-of-sight vector
    """
    coverage = compute_surface_coverage(system, semi_major_axis=semi_major_axis, in_eclipse=in_eclipse)
    p_cosines = utils.calculate_cos_theta_los_x(system.primary.normals)
    s_cosines = utils.calculate_cos_theta_los_x(system.secondary.normals)
    cosines = {'primary': p_cosines, 'secondary': s_cosines}
    return coverage, cosines
