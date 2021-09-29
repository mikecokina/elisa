import numpy as np
from scipy.spatial.qhull import Delaunay

from ... base import spot
from ... base.surface import faces as bfaces
from ... logger import getLogger
from ... base.surface.faces import set_all_surface_centres, mirror_triangulation

logger = getLogger("single_system.surface.faces")


def build_faces(system_container):
    """
    Function tessellates the stellar surface points into a set of triangles
    covering the star without gaps and overlaps.

    :return: elisa.single_system.container.SinglePositionContainer;
    """
    # build surface if there is no spot specified
    if not system_container.star.spots:
        build_surface_with_no_spots(system_container)
    else:
        build_surface_with_spots(system_container)

    return system_container


def build_surface_with_no_spots(system_container):
    """
    function is calling surface building function for single systems without spots and assigns star's surface to
    star object as its property
    :return:
    """
    star_container = system_container.star
    points_length = star_container.base_symmetry_points_number
    # triangulating only one eighth of the star
    points_to_triangulate = np.append(star_container.symmetry_points(), [[0, 0, 0]], axis=0)
    triangles = single_surface(star_container=star_container, points=points_to_triangulate)
    # removing faces from triangulation, where origin point is included
    triangles = triangles[~(triangles >= points_length).any(1)]
    triangles = triangles[~((points_to_triangulate[triangles] == 0.).all(1)).any(1)]
    # setting number of base symmetry faces
    star_container.base_symmetry_faces_number = np.int(np.shape(triangles)[0])
    # lets exploit axial symmetry and fill the rest of the surface of the star
    star_container.faces = mirror_triangulation(triangles, star_container.inverse_point_symmetry_matrix)

    base_face_symmetry_vector = np.arange(star_container.base_symmetry_faces_number)
    star_container.face_symmetry_vector = np.concatenate([base_face_symmetry_vector for _ in range(8)])


def single_surface(star_container=None, points=None):
    """
    calculates triangulation of given set of points, if points are not given, star surface points are used. Returns
    set of triple indices of surface pints that make up given triangle

    :param star_container: StarContainer;
    :param points: np.array:

    ::

        numpy.array([[x1 y1 z1],
                     [x2 y2 z2],
                     ...
                    [xN yN zN]])

    :return: np.array():

    ::

        numpy.array([[point_index1 point_index2 point_index3],
                     [...],
                      ...
                     [...]])
    """
    if points is None:
        points = star_container.points
    triangulation = Delaunay(points)
    triangles_indices = triangulation.convex_hull
    return triangles_indices


def build_surface_with_spots(system_container):
    """
    function for triangulation of surface with spots

    :return:
    """
    star_container = system_container.star
    points, vertices_map = star_container.get_flatten_points_map()
    faces = single_surface(points=points)
    model, spot_candidates = bfaces.initialize_model_container(vertices_map)
    model = bfaces.split_spots_and_component_faces(
        star_container, points, faces, model, spot_candidates, vertices_map, component_com=0.0
    )

    spot.remove_overlaped_spots_by_vertex_map(star_container, vertices_map)
    spot.remap_surface_elements(star_container, model, points)

    return system_container


def compute_all_surface_areas(system_container):
    """
    Compute surface areas of all faces (spots included).

    :param system_container: elisa.single_system.container.SinglePositionContainer; instance
    :return: system; elisa.single_system.container.SinglePositionContainer; instance
    """
    star_container = system_container.star
    logger.debug(f'computing surface areas of component: '
                 f'{star_container} / name: {star_container.name}')
    star_container.calculate_all_areas()

    return system_container


def build_faces_orientation(system_container):
    """
    Compute face orientation (normals) for each face.

    :param system_container: elisa.single_system.container.SinglePositionContainer;
    :return: elisa.single_system.container.SinglePositionContainer;
    """
    com_x = 0.0

    star = system_container.star
    set_all_surface_centres(star)
    set_all_normals(star, com=com_x)

    return system_container


def set_all_normals(star_container, com):
    """
    Function calculates normals for each face of given body (including spots) and assign it to object.

    :param star_container: instance of container to set normals on;
    :param com: numpy.array;
    :param star_container: instance of container to set normals on;
    """
    points, faces, cntrs = star_container.points, star_container.faces, star_container.face_centres
    star_container.normals = bfaces.calculate_normals(points, faces, cntrs, com)

    if star_container.has_spots():
        for spot_index in star_container.spots:
            star_container.spots[spot_index].normals = \
                bfaces.calculate_normals(star_container.spots[spot_index].points,
                                         star_container.spots[spot_index].faces,
                                         star_container.spots[spot_index].face_centres,
                                         com)
    return star_container


def build_velocities(system):
    """
    Function calculates velocity vector for each face relative to the observer.

    :param system: elisa.single_system.container.SingleSystemContainer
    :return: elisa.single_system.container.SingleSystemContainer
    """
    star = system.star
    omega = np.array([0, 0, system.angular_velocity])

    # rotational velocity
    p_velocities = np.cross(star.points, omega, axisa=1)
    star.velocities = np.mean(p_velocities[star.faces], axis=1)

    if star.has_spots():
        for _spot in star.spots.values():
            p_velocities = np.cross(_spot.points, omega, axisa=1)
            _spot.velocities = np.mean(p_velocities[_spot.faces], axis=1)

    return system
