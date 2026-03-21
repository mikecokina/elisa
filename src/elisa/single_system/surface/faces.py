from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import Delaunay

from elisa.base import spot
from elisa.base.surface import faces as bfaces
from elisa.base.surface.faces import mirror_triangulation, set_all_surface_centres
from elisa.base.types import INT
from elisa.logger import getLogger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.single_system.container import SinglePositionContainer


logger = getLogger("single_system.surface.faces")


def build_faces(system_container: SinglePositionContainer) -> SinglePositionContainer:
    """Tessellate the stellar surface into triangular faces.

    The function chooses a path depending on whether the star has spots.

    :param system_container: Single-position container holding the star.
    :type system_container: elisa.single_system.container.SinglePositionContainer
    :returns: The same container with faces (and related data) assigned.
    :rtype: elisa.single_system.container.SinglePositionContainer
    """
    # build surface if there is no spot specified
    if not system_container.star.spots:
        build_surface_with_no_spots(system_container)
    else:
        build_surface_with_spots(system_container)

    return system_container


def build_surface_with_no_spots(system_container: SinglePositionContainer) -> SinglePositionContainer:
    """Build surface triangulation for a single star without spots.

    The function triangulates the base symmetry patch and mirrors the
    triangulation using the container symmetry matrix to produce faces for
    the whole surface.

    :param system_container: Single-position container holding the star.
    :type system_container: elisa.single_system.container.SinglePositionContainer
    :returns: The container with faces and symmetry vectors set.
    :rtype: elisa.single_system.container.SinglePositionContainer
    """
    star_container = system_container.star
    points_length = star_container.base_symmetry_points_number
    # triangulating only one eighth of the star
    points_to_triangulate = np.append(star_container.symmetry_points(), [[0.0, 0.0, 0.0]], axis=0)
    triangles = single_surface(star_container=star_container, points=points_to_triangulate)
    # removing faces from triangulation where origin point is included
    triangles = triangles[~(triangles >= points_length).any(1)]
    triangles = triangles[~((points_to_triangulate[triangles] == 0.0).all(1)).any(1)]
    # setting number of base symmetry faces
    star_container.base_symmetry_faces_number = INT(np.shape(triangles)[0])
    # exploit axial symmetry and fill the rest with the surface of the star
    star_container.faces = mirror_triangulation(triangles, star_container.inverse_point_symmetry_matrix)

    base_face_symmetry_vector = np.arange(star_container.base_symmetry_faces_number)
    star_container.face_symmetry_vector = np.concatenate([base_face_symmetry_vector for _ in range(8)])

    return system_container


def single_surface(
    star_container: SinglePositionContainer | StarContainer | None = None,
    points: NDArray | None = None,
) -> NDArray:
    """Triangulate a set of points using Delaunay and return triangle vertex indices.

    :param star_container: Optional star container used as fallback for points.
    :type star_container: elisa.single_system.container.SinglePositionContainer | None
    :param points: Optional array of points to triangulate.
    :type points: numpy.typing.NDArray | None
    :returns: Array of triangle vertex indices.
    :rtype: numpy.typing.NDArray
    """
    if points is None:
        points = star_container.points
    triangulation = Delaunay(points)
    return triangulation.convex_hull


def build_surface_with_spots(system_container: SinglePositionContainer) -> SinglePositionContainer:
    """Build surface triangulation for stars with spots.

    The function triangulates the flattened point set (star+spots), builds a
    model, splits spot and component faces, removes overlapped spots and
    remaps surface elements onto spot containers.

    :param system_container: Single-position container holding the star and spots.
    :type system_container: elisa.single_system.container.SinglePositionContainer
    :returns: The input container with faces and spot remapping applied.
    :rtype: elisa.single_system.container.SinglePositionContainer
    """
    star_container = system_container.star
    points, vertices_map = star_container.get_flatten_points_map()
    faces = single_surface(points=points)
    model, spot_candidates = bfaces.initialize_model_container(vertices_map)
    model = bfaces.split_spots_and_component_faces(
        star_container,
        points,
        faces,
        model,
        spot_candidates,
        vertices_map,
        component_com=0.0,
    )

    spot.remove_overlaped_spots_by_vertex_map(star_container, vertices_map)
    spot.remap_surface_elements(star_container, model, points)

    return system_container


def compute_all_surface_areas(system_container: SinglePositionContainer) -> SinglePositionContainer:
    """Compute surface areas for the star and its spots and assign them on the container.

    :param system_container: Single-position container.
    :type system_container: elisa.single_system.container.SinglePositionContainer
    :returns: The same container with areas computed.
    :rtype: elisa.single_system.container.SinglePositionContainer
    """
    star_container = system_container.star
    comp_name = getattr(star_container, "name", None)
    logger.debug("computing surface areas of component: %s / name: %s", star_container, comp_name)
    star_container.calculate_all_areas()

    return system_container


def build_faces_orientation(system_container: SinglePositionContainer) -> SinglePositionContainer:
    """Compute face centres and normals for the star (and spots).

    :param system_container: Single-position container.
    :type system_container: elisa.single_system.container.SinglePositionContainer
    :returns: Input container with face centres and normals assigned.
    :rtype: elisa.single_system.container.SinglePositionContainer
    """
    com_x: float = 0.0

    star = system_container.star
    set_all_surface_centres(star)
    set_all_normals(star, com=com_x)

    return system_container


def set_all_normals(star_container: StarContainer, com: float) -> StarContainer:
    """Calculate and assign face normals for a star container and its spots.

    :param star_container: StarContainer with faces and centres set.
    :type star_container: elisa.base.container.StarContainer
    :param com: Centre of mass x-offset to subtract when computing normals.
    :type com: float
    :returns: The same star container with normals assigned.
    :rtype: elisa.base.container.StarContainer
    """
    points, faces, cntrs = star_container.points, star_container.faces, star_container.face_centres
    star_container.normals = bfaces.calculate_normals(points, faces, cntrs, com)

    if star_container.has_spots():
        for spot_index in star_container.spots:
            sp = star_container.spots[spot_index]
            sp.normals = bfaces.calculate_normals(sp.points, sp.faces, sp.face_centres, com)
    return star_container


def build_velocities(system: SinglePositionContainer) -> SinglePositionContainer:
    """Calculate velocity vectors for face centres due to rotation.

    The function computes per-vertex rotational velocities with angular
    velocity taken from the container and averages them over each face.

    :param system: Single-position container.
    :type system: elisa.single_system.container.SinglePositionContainer
    :returns: Container with velocities assigned on star and spots.
    :rtype: elisa.single_system.container.SinglePositionContainer
    """
    star = system.star
    omega = np.array([0.0, 0.0, system.angular_velocity])

    # rotational velocity
    p_velocities = np.cross(star.points, omega, axisa=1)
    star.velocities = np.mean(p_velocities[star.faces], axis=1)

    if star.has_spots():
        for _spot in star.spots.values():
            p_velocities_spot = np.cross(_spot.points, omega, axisa=1)
            _spot.velocities = np.mean(p_velocities_spot[_spot.faces], axis=1)

    return system
