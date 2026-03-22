from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.path as mpltpath
import numpy as np
from scipy.spatial import ConvexHull

from elisa import const, settings, utils
from elisa import umpy as up
from elisa.base.surface import coverage as bcoverage
from elisa.base.types import BOOL, FLOAT, INT
from elisa.binary_system import utils as bsutils
from elisa.logger import getLogger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.types import NP_BOOL_, ComponentName, Float, Int, NumpyBool

logger = getLogger("binary_system.surface.coverage")



def partial_visible_faces_surface_coverage(
    points: NDArray[Float],
    faces: NDArray[Int],
    normals: NDArray[Float],
    hull: NDArray[Float],
) -> NDArray[Float]:
    """Compute surface coverage of partially visible faces.

    :param points: Projected 2D point coordinates.
    :type points: NDArray[Float]
    :param faces: Triangle vertex indices.
    :type faces: NDArray[Int]
    :param normals: Face normals.
    :type normals: NDArray[Float]
    :param hull: Boundary points of the in-front component projection, sorted
        clockwise to form a valid :class:`matplotlib.path.Path`.
    :type hull: NDArray[Float]
    :return: Surface coverage of partially visible faces.
    :rtype: NDArray[Float]
    """
    pypex_hull = bsutils.hull_to_pypex_poly(hull)
    pypex_faces = bsutils.faces_to_pypex_poly(points[faces])

    # it is possible to None happens in intersection, tkae care about it latter
    pypex_intersection = bsutils.pypex_poly_hull_intersection(
        pypex_faces,
        pypex_hull,
    )

    # think about surface normalisation like and avoid surface areas like 1e-6
    # which lead to loss in precission
    pypex_polys_surface_area = np.array(
        bsutils.pypex_poly_surface_area(pypex_intersection),
        dtype=FLOAT,
    )

    inplane_points_3d = np.column_stack((points, np.zeros(points.shape[0])))
    inplane_surface_area = utils.triangle_areas(
        triangles=faces,
        points=inplane_points_3d,
    )
    correction_cosine = utils.calculate_cos_theta_los_x(normals)
    return (inplane_surface_area - pypex_polys_surface_area) / correction_cosine


def calculate_centre_of_star_projection(
    system: OrbitalPositionContainer,
    component: ComponentName,
) -> NDArray[Float]:
    """Return yz projection of the centre of mass of the given component.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param component: Component selector.
    :type component: ComponentName
    :return: yz projection of the component centre of mass.
    :rtype: NDArray[Float]
    """
    if component == "primary":
        return np.array([0.0, 0.0], dtype=FLOAT)

    centre_vector = np.array([system.position.distance, 0.0, 0.0], dtype=FLOAT)

    args = (
        system.position.azimuth - const.HALF_PI,
        centre_vector,
        "z",
        False,
        False,
    )
    centre_vector = utils.around_axis_rotation(*args)

    args = (
        const.HALF_PI - system.inclination,
        centre_vector,
        "y",
        False,
        False,
    )
    centre_vector = utils.around_axis_rotation(*args)

    return centre_vector[1:]


def expand_star_outline(
    path: mpltpath.Path,
    system: OrbitalPositionContainer,
    cover_component: ComponentName,
) -> mpltpath.Path:
    """Expand the outline of the eclipsing component.

    The outline is expanded slightly to compensate for area loss introduced by
    surface discretization.

    :param path: Outline of the eclipsing star.
    :type path: matplotlib.path.Path
    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param cover_component: Component selector.
    :type cover_component: ComponentName
    :return: Expanded outline path.
    :rtype: matplotlib.path.Path
    """
    centre_projection = calculate_centre_of_star_projection(system, cover_component)
    alpha = const.FULL_ARC / path.vertices.shape[0]
    correction_factor = np.sqrt(2 - (np.sin(alpha) / alpha))
    path.vertices = correction_factor * (path.vertices - centre_projection) + centre_projection
    return path


def test_size_similarity(
    cover_object: StarContainer,
    undercover_object: StarContainer,
) -> NumpyBool:
    """Check whether the cover size is comparable to triangle size underneath.

    This condition determines whether a separate visibility treatment is needed
    for eclipses involving relatively small projected cover geometry.

    :param cover_object: Eclipsing component.
    :type cover_object: StarContainer
    :param undercover_object: Eclipsed component.
    :type undercover_object: StarContainer
    :return: ``True`` if the cover size is less than or equal to the
        characteristic triangle size of the eclipsed component.
    :rtype: bool
    """
    cover_size = 2.0 * cover_object.equivalent_radius
    undercover_triangle_size = undercover_object.equivalent_radius * np.sin(undercover_object.discretization_factor)
    return cover_size <= undercover_triangle_size


def visibility_out_of_eclipse(
    undercover_object: StarContainer,
) -> tuple[NDArray[NP_BOOL_], NDArray[NP_BOOL_], NDArray[NP_BOOL_]]:
    """Determine visible near-side faces outside eclipse.

    :param undercover_object: Eclipsed component.
    :type undercover_object: StarContainer
    :return: Tuple of ``(full_visible, invisible, partial_visible)`` masks for
        observer-facing triangles of the eclipsed component.
    :rtype: tuple[NDArray[bool], NDArray[bool, NDArray[bool]]
    """
    n_faces = undercover_object.normals.shape[0]
    full_visible = np.zeros(n_faces, dtype=BOOL)
    full_visible[undercover_object.indices] = True
    placeholder = np.zeros(n_faces, dtype=BOOL)
    return full_visible, placeholder, placeholder


def visibility_similar_objects(
    undercover_visible_projection: NDArray[Float],
    undercover_object: StarContainer,
    undercover_visible_point_indices: NDArray[Int],
    cover_outline: mpltpath.Path,
) -> tuple[NDArray[NP_BOOL_], NDArray[NP_BOOL_], NDArray[NP_BOOL_]]:
    """Determine face visibility during eclipse for similarly sized objects.

    This branch is used when the eclipsing component is much larger than a
    typical triangle on the eclipsed component.

    :param undercover_visible_projection: Observer-facing points of the
        eclipsed component.
    :type undercover_visible_projection: NDArray[Float]
    :param undercover_object: Eclipsed component.
    :type undercover_object: StarContainer
    :param undercover_visible_point_indices: Indices of observer-facing points
        of the eclipsed component.
    :type undercover_visible_point_indices: NDArray[Int]
    :param cover_outline: Hull of the eclipsing component.
    :type cover_outline: matplotlib.path.Path
    :return: Tuple of ``(full_visible, invisible, partial_visible)`` masks for
        observer-facing triangles of the eclipsed component.
    :rtype: tuple[NDArray[bool], NDArray[bool], NDArray[bool]]
    """
    # obtain points out of eclipse (out of boundary defined by hull of
    # 'infront' object)
    out_of_bound = up.invert(
        cover_outline.contains_points(undercover_visible_projection),
    )

    undercover_visible_point_indices = undercover_visible_point_indices[out_of_bound]
    undercover_faces = np.full(undercover_object.normals.shape, -1, dtype=INT)
    undercover_faces[undercover_object.indices] = undercover_object.faces[undercover_object.indices]

    eclipse_faces_visibility = np.isin(
        undercover_faces,
        undercover_visible_point_indices,
    )

    full_visible = np.all(eclipse_faces_visibility, axis=1)
    invisible = np.all(up.invert(eclipse_faces_visibility), axis=1)
    partial_visible = up.invert(full_visible | invisible)
    return full_visible, invisible, partial_visible


def visibility_disimilar_objects(
    undercover_visible_projection: NDArray[Float],
    undercover_object: StarContainer,
    undercover_visible_point_indices: NDArray[Int],
    cover_outline: mpltpath.Path,
) -> tuple[NDArray[NP_BOOL_], NDArray[NP_BOOL_], NDArray[NP_BOOL_]]:
    """Determine face visibility during eclipse for dissimilar-sized objects.

    This branch is used when the eclipsing component is comparable to or
    smaller than a typical triangle on the eclipsed component.

    :param undercover_visible_projection: Observer-facing points of the
        eclipsed component.
    :type undercover_visible_projection: NDArray[Float]
    :param undercover_object: Eclipsed component.
    :type undercover_object: StarContainer
    :param undercover_visible_point_indices: Indices of observer-facing points
        of the eclipsed component.
    :type undercover_visible_point_indices: NDArray[Int]
    :param cover_outline: Hull of the eclipsing component.
    :type cover_outline: matplotlib.path.Path
    :return: Tuple of ``(full_visible, invisible, partial_visible)`` masks for
        observer-facing triangles of the eclipsed component.
    :rtype: tuple[NDArray[bool], NDArray[bool], NDArray[bool]]
    """
    outline_max_coord = cover_outline.vertices.max(axis=0)
    outline_min_coord = cover_outline.vertices.min(axis=0)
    cover_centre = 0.5 * (outline_max_coord + outline_min_coord)
    selection_radius = undercover_object.equivalent_radius * np.sin(undercover_object.discretization_factor)
    cover_hi = cover_centre + selection_radius
    cover_lo = cover_centre - selection_radius

    # square searchbox around cover component COM with half size equivalent to triangle size
    in_bound = (
        (undercover_visible_projection < cover_hi).all(axis=1)
        & (undercover_visible_projection > cover_lo).all(axis=1)
    )
    out_of_bound = ~in_bound

    undercover_visible_point_indices = undercover_visible_point_indices[out_of_bound]
    undercover_faces = np.full(undercover_object.normals.shape, -1, dtype=INT)
    undercover_faces[undercover_object.indices] = undercover_object.faces[undercover_object.indices]

    eclipse_faces_visibility = np.isin(
        undercover_faces,
        undercover_visible_point_indices,
    )

    full_visible = np.all(eclipse_faces_visibility, axis=1)
    invisible = np.zeros(full_visible.shape, dtype=BOOL)
    partial_visible = full_visible.copy()
    partial_visible[undercover_object.indices] = ~partial_visible[undercover_object.indices]
    return full_visible, invisible, partial_visible


def compute_surface_coverage(
    system: OrbitalPositionContainer,
    semi_major_axis: Float,
    *,
    in_eclipse: bool = True,
    return_values: bool = True,
    write_to_containers: bool = False,
) -> dict[str, NDArray[Float]] | None:
    """Compute surface coverage of faces for a given orbital position.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param semi_major_axis: Semi-major axis in SI-compatible scaling units.
    :type semi_major_axis: Float
    :param in_eclipse: Whether eclipse should be considered for the current
        orbital position.
    :type in_eclipse: bool
    :param return_values: Whether computed coverages should be returned.
    :type return_values: bool
    :param write_to_containers: Whether computed values should be assigned to
        the component containers.
    :type write_to_containers: bool
    :return: Coverage arrays keyed by component name, or ``None``.
    :rtype: dict[str, NDArray[Float]] | None
    """
    # TODO: add unittests  # noqa: FIX002, TD002, TD003
    logger.debug("computing surface coverage for %s", system.position)

    bb_path = None
    cover_component: ComponentName = "secondary" if 0.0 < system.position.azimuth < const.PI else "primary"
    cover_object = getattr(system, cover_component)
    undercover_component = settings.BINARY_COUNTERPARTS[cover_component]
    undercover_object = getattr(system, undercover_component)

    # all surface values in sma unit which are smaller than following threshold
    # are discarded (set to 0.0)
    surface_noise_threshold = (
        2.0 * np.pi * undercover_object.polar_radius**2 / len(undercover_object.faces)
    ) / 1e6

    cover_object_obs_visible_projection = utils.get_visible_projection(cover_object)
    undercover_object_obs_visible_projection = utils.get_visible_projection(
        undercover_object,
    )

    if in_eclipse:
        # indices of points on near side
        undercover_visible_point_indices = np.unique(
            undercover_object.faces[undercover_object.indices],
        )

        # outline of the eclipsing component
        bb_path = get_eclipse_boundary_path(cover_object_obs_visible_projection)

        similar_size_test = test_size_similarity(cover_object, undercover_object)
        args = (
            undercover_object_obs_visible_projection,
            undercover_object,
            undercover_visible_point_indices,
            bb_path,
        )
        full_visible, _invisible, partial_visible = (
            visibility_disimilar_objects(*args) if similar_size_test else visibility_similar_objects(*args)
        )
    else:
        full_visible, _invisible, partial_visible = visibility_out_of_eclipse(
            undercover_object,
        )

    # process partial and full visible faces (get surface area of 3d polygon)
    # of undercover object
    partial_visible_faces = undercover_object.faces[partial_visible]
    partial_visible_normals = undercover_object.normals[partial_visible]
    undercover_object_pts_projection = utils.plane_projection(
        undercover_object.points,
        "yz",
        keep_3d=False,
    )

    if in_eclipse:
        if bb_path is None:
            msg = "Boundary path should have been defined at this point, check the code logic"
            raise RuntimeError(msg)

        partial_coverage = partial_visible_faces_surface_coverage(
            points=undercover_object_pts_projection,
            faces=partial_visible_faces,
            normals=partial_visible_normals,
            hull=bb_path.vertices,
        )
        partial_coverage[partial_coverage < surface_noise_threshold] = 0.0
    else:
        partial_coverage = None

    # discard values of surface which are under threshold
    visible_coverage = undercover_object.areas[full_visible]

    undercover_obj_coverage = bcoverage.surface_area_coverage(
        size=undercover_object.normals.shape[0],
        visible=full_visible,
        visible_coverage=visible_coverage,
        partial=partial_visible,
        partial_coverage=partial_coverage,
    )

    cover_obj_coverage = np.zeros(cover_object.areas.shape, dtype=FLOAT)
    cover_obj_coverage[cover_object.indices] = cover_object.areas[cover_object.indices]

    # areas are now in SMA^2, converting to SI
    scale = semi_major_axis**2
    cover_obj_coverage *= scale
    undercover_obj_coverage *= scale

    if write_to_containers:
        cover_object.coverage = cover_obj_coverage
        undercover_object.coverage = undercover_obj_coverage

    if not return_values:
        return None

    return {
        cover_component: cover_obj_coverage,
        undercover_component: undercover_obj_coverage,
    }


def get_eclipse_boundary_path(hull: NDArray[Float]) -> mpltpath.Path:
    """Return boundary path of a component projection in the yz plane.

    :param hull: Projected hull points.
    :type hull: NDArray[Float]
    :return: Boundary path of the component projection.
    :rtype: matplotlib.path.Path
    """
    cover_bound = ConvexHull(hull)
    hull_points = hull[cover_bound.vertices]
    return mpltpath.Path(hull_points)


def calculate_coverage_with_cosines(
    system: OrbitalPositionContainer,
    semi_major_axis: Float,
    *,
    in_eclipse: bool = True,
) -> tuple[dict[str, NDArray[Float]], dict[str, NDArray[Float]]]:
    """Prepare surface coverage and line-of-sight directional cosines.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param semi_major_axis: Semi-major axis in SI-compatible scaling units.
    :type semi_major_axis: Float
    :param in_eclipse: Whether eclipse should be considered for the current
        orbital position. If unsure, leave as ``True``.
    :type in_eclipse: bool
    :return: Tuple ``(coverage, cosines)`` where ``coverage`` contains visible
        face areas and ``cosines`` contains directional cosines for each face
        with respect to the line-of-sight vector.
    :rtype: tuple[dict[str, NDArray[Float]], dict[str, NDArray[Float]]]
    """
    coverage = compute_surface_coverage(
        system,
        semi_major_axis=semi_major_axis,
        in_eclipse=in_eclipse,
    )
    p_cosines = utils.calculate_cos_theta_los_x(system.primary.normals)
    s_cosines = utils.calculate_cos_theta_los_x(system.secondary.normals)
    cosines = {
        "primary": p_cosines,
        "secondary": s_cosines,
    }
    return coverage, cosines
