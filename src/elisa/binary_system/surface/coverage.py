import numpy as np
import matplotlib.path as mpltpath

from scipy.spatial.qhull import ConvexHull
from copy import copy

from .. import utils as bsutils
from ... import settings
from ... logger import getLogger
from ... import (
    umpy as up,
    utils,
    const
)
from ... base.surface import coverage as bcoverage


logger = getLogger('binary_system.surface.coverage')


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

    inplane_points_3d = np.column_stack((points, np.zeros(points.shape[0])))
    inplane_surface_area = utils.triangle_areas(triangles=faces, points=inplane_points_3d)
    correction_cosine = utils.calculate_cos_theta_los_x(normals)
    retval = (inplane_surface_area - pypex_polys_surface_area) / correction_cosine
    return retval


def calculate_centre_of_star_projection(system, component):
    """
    Returns yz projection of centre of mass of given `component`.

    :param system: elisa.binary_system.container.OrbitalPositionContainer
    :param component: str; `primary` or `secondary`
    :return: numpy.array;
    """
    if component == 'primary':
        return np.array([0.0, 0.0])
    else:
        centre_vector = np.array([system.position.distance, 0.0, 0.0])
        args = (system.position.azimuth - const.HALF_PI, centre_vector, "z", False, False)
        centre_vector = utils.around_axis_rotation(*args)

        args = (const.HALF_PI - system.inclination, centre_vector, "y", False, False)
        centre_vector = utils.around_axis_rotation(*args)

        return centre_vector[1:]


def expand_star_outline(path, system, cover_component):
    """
    Function takes outline of the cover star and expands it slightly to compensate for the loss of its area due to
    surface discretization.

    :param path: Path; outline of the eclipsing star
    :param system: elisa.binary_system.container.OrbitalPositionContainer
    :param cover_component: str; `primary` or `secondary`
    :return: Path; expanded outline of the star
    """
    cpm = calculate_centre_of_star_projection(system, cover_component)
    alpha = const.FULL_ARC / np.shape(path.vertices)[0]
    corr_factor = np.sqrt(2 - (np.sin(alpha) / alpha))
    path.vertices = corr_factor * (path.vertices - cpm[None, :]) + cpm[None, :]
    return path


def test_size_similarity(cover_object, undercover_object):
    """
    Checking whether size of the cover component is comparable to the triangle size of the undercovar component which
    requires a separate approach.

    :param cover_object: StarContainer;
    :param undercover_object: StarContainer;
    :return: bool;
    """
    cover_size = 2.0 * cover_object.equivalent_radius
    undercover_triangle_size = undercover_object.equivalent_radius * np.sin(undercover_object.discretization_factor)
    return cover_size <= undercover_triangle_size


def visibility_out_of_eclipse(undercover_object):
    """
    Decides visibility of the near side faces of the undercover component (eclipsed component) outside of eclipse.

    :param undercover_object: StarContainer; eclipsed star
    :return: tuple; full_visible, invisible, partial_visible triangles observer-facing part of eclipsed component
    """
    eclipse_faces_visibility = np.full(undercover_object.normals.shape, False, dtype=bool)
    eclipse_faces_visibility[undercover_object.indices] = True

    # get indices of full visible, invisible and partial visible faces
    full_visible = np.all(eclipse_faces_visibility, axis=1)
    placeholder = np.full(full_visible.shape, False, dtype=bool)
    return full_visible, placeholder, placeholder


def visibility_similar_objects(undercover_visible_projection, undercover_object, undercover_visible_point_indices,
                               cover_outline):
    """
    Decides visibility of the near side faces of the undercover component (eclipsed component) during eclipse in case
    of similarly sized components where cover (eclipsing) component is much larger than triangle on eclipsed component.

    :param undercover_visible_projection: numpy.array; observer-facing points of eclipsed component
    :param undercover_object: StarContainer; eclipsed component
    :param undercover_visible_point_indices: numpy.array; indices of observer-facing points of eclipsed component
    :param cover_outline: matplotlib.path.Path; hull of the eclipsing component
    :return: tuple; full_visible, invisible, partial_visible triangles observer-facing part of eclipsed component
    """
    # obtain points out of eclipse (out of boundary defined by hull of 'infront' object)
    out_of_bound = up.invert(cover_outline.contains_points(undercover_visible_projection))

    undercover_visible_point_indices = undercover_visible_point_indices[out_of_bound]
    undercover_faces = np.full(undercover_object.normals.shape, -1, dtype=np.int)
    undercover_faces[undercover_object.indices] = undercover_object.faces[undercover_object.indices]

    eclipse_faces_visibility = np.isin(undercover_faces, undercover_visible_point_indices)

    # get indices of full visible, invisible and partial visible faces
    full_visible = np.all(eclipse_faces_visibility, axis=1)
    invisible = np.all(up.invert(eclipse_faces_visibility), axis=1)
    partial_visible = up.invert(full_visible | invisible)
    return full_visible, invisible, partial_visible


def visibility_disimilar_objects(undercover_visible_projection, undercover_object, undercover_visible_point_indices,
                                 cover_outline):
    """
    Decides visibility of the near side faces of the undercover component (eclipsed component) during eclipse in case
    of significantly smaller (eclipsing) component that is comparable or smaller to triangle on eclipsed component.

    :param undercover_visible_projection: numpy.array; observer-facing points of eclipsed component
    :param undercover_object: StarContainer; eclipsed component
    :param undercover_visible_point_indices: numpy.array; indices of observer-facing points of eclipsed component
    :param cover_outline: matplotlib.path.Path; hull of the eclipsing component
    :return: tuple; full_visible, invisible, partial_visible triangles observer-facing part of eclipsed component
    """
    outline_max_coord = cover_outline.vertices.max(axis=0)
    outline_min_coord = cover_outline.vertices.min(axis=0)
    cover_centre = 0.5 * (outline_max_coord + outline_min_coord)  # centre of eclipsed component
    selection_radius = undercover_object.equivalent_radius * np.sin(undercover_object.discretization_factor)

    # square searchbox around cover component COM with half size equivalent to triangle size
    max_condition = (undercover_visible_projection < (cover_centre + selection_radius)[None, :]).all(axis=1)
    min_condition = (undercover_visible_projection > (cover_centre - selection_radius)[None, :]).all(axis=1)

    out_of_bound = ~np.logical_and(max_condition, min_condition)

    undercover_visible_point_indices = undercover_visible_point_indices[out_of_bound]
    undercover_faces = np.full(undercover_object.normals.shape, -1, dtype=np.int)
    undercover_faces[undercover_object.indices] = undercover_object.faces[undercover_object.indices]

    eclipse_faces_visibility = np.isin(undercover_faces, undercover_visible_point_indices)

    full_visible = np.all(eclipse_faces_visibility, axis=1)
    invisible = np.full(full_visible.shape, False, dtype=bool)
    partial_visible = copy(full_visible)
    partial_visible[undercover_object.indices] = ~partial_visible[undercover_object.indices]
    return full_visible, invisible, partial_visible


def compute_surface_coverage(system, semi_major_axis, in_eclipse=True, return_values=True, write_to_containers=False):
    # todo: add unittests
    """
    Compute surface coverage of faces for given orbital position
    defined by container/SingleOrbitalPositionContainer.

    :param semi_major_axis: float;
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param in_eclipse: bool;
    :param return_values: bool; return coverages
    :param write_to_containers: bool; calculated values will be assigned to `system` container
    :return: Dict;
    """
    logger.debug(f"computing surface coverage for {system.position}")
    cover_component = 'secondary' if 0.0 < system.position.azimuth < const.PI else 'primary'
    cover_object = getattr(system, cover_component)
    undercover_object = getattr(system, settings.BINARY_COUNTERPARTS[cover_component])

    # all surface values in sma unit which are smaller then following threshold are discarded (set to 0.0)
    surface_noise_threshold = (2.0 * np.pi * np.power(undercover_object.polar_radius, 2) /
                               len(undercover_object.faces)) / 1e6

    cover_object_obs_visible_projection = utils.get_visible_projection(cover_object)
    undercover_object_obs_visible_projection = utils.get_visible_projection(undercover_object)

    if in_eclipse:
        # indices of points on near side
        undercover_visible_point_indices = np.unique(undercover_object.faces[undercover_object.indices])

        # outline of the eclipsing component
        bb_path = get_eclipse_boundary_path(cover_object_obs_visible_projection)

        similar_size_test = test_size_similarity(cover_object, undercover_object)
        args = (undercover_object_obs_visible_projection, undercover_object, undercover_visible_point_indices, bb_path)
        full_visible, invisible, partial_visible = \
            visibility_disimilar_objects(*args) if similar_size_test else visibility_similar_objects(*args)
    else:
        full_visible, invisible, partial_visible = visibility_out_of_eclipse(undercover_object)

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
        partial_coverage[partial_coverage < surface_noise_threshold] = 0.0
    else:
        partial_coverage = None

    # discard values of surface which are under threshold
    visible_coverage = undercover_object.areas[full_visible]

    undercover_obj_coverage = bcoverage.surface_area_coverage(
        size=np.shape(undercover_object.normals)[0],
        visible=full_visible, visible_coverage=visible_coverage,
        partial=partial_visible, partial_coverage=partial_coverage
    )

    cover_obj_coverage = np.zeros(cover_object.areas.shape)
    cover_obj_coverage[cover_object.indices] = cover_object.areas[cover_object.indices]

    # areas are now in SMA^2, converting to SI
    cover_obj_coverage *= up.power(semi_major_axis, 2)
    undercover_obj_coverage *= up.power(semi_major_axis, 2)

    if write_to_containers:
        setattr(cover_object, 'coverage', cover_obj_coverage)
        setattr(undercover_object, 'coverage', undercover_obj_coverage)

    return {
        cover_component: cover_obj_coverage,
        settings.BINARY_COUNTERPARTS[cover_component]: undercover_obj_coverage
    } if return_values else None


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
