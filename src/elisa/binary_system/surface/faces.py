import gc
import numpy as np

from copy import copy
from scipy.spatial.qhull import Delaunay
from elisa import logger, umpy as up
from elisa.conf import config
from elisa.pulse import pulsations
from elisa.utils import is_empty
from elisa.binary_system import utils as bsutils

config.set_up_logging()
__logger__ = logger.getLogger("binary-system-faces-module")


def initialize_model_container(vertices_map):
    """
    Initializes basic data structure `model` objects that will contain faces divided by its origin (star or spots)
    and data structure containing spot candidates with its index and center point.
    Structure is based on input `verties_map`.
    Example of return Tuple

    ::

        (<class 'dict'>: {'object': [], 'spots': {0: []}}, <class 'dict'>: {'com': [], 'ix': []})
    :param vertices_map: List or ndarray; map which define refrences of index in
                         given Iterable to object (spot or Star).
                         For more info, see docstring for `incorporate_spots_mesh` method.
    :return: Tuple[Dict, Dict]
    """
    model = {"object": list(), "spots": dict()}
    spot_candidates = {"com": list(), "ix": list()}

    spots_instance_indices = list(set([vertices_map[ix]["enum"] for ix, _ in enumerate(vertices_map)
                                       if vertices_map[ix]["enum"] >= 0]))
    for spot_index in spots_instance_indices:
        model["spots"][spot_index] = list()
    return model, spot_candidates


def get_surface_builder_fn(morphology):
    """
    Returns suitable triangulation function depending on morphology.
    :return: method; method that performs generation surface faces
    """
    return over_contact_system_surface if morphology == "over-contact" else detached_system_surface


def split_spots_and_component_faces(star_container, points, faces, model, spot_candidates, vmap, component_com):
    """
    Function that sorts faces to model data structure by distinguishing if it belongs to star or spots.
    :param star_container:
    :param component_com: float; center of mass of component
    :param points: numpy.array; (N_points * 3) - all points of surface
    :param faces: numpy.array; (N_faces * 3) - all faces of the surface
    :param model: Dict; data structure for faces sorting (more in docstring of method `initialize_model_container`)
    :param spot_candidates: Dict; initialised data structure for spot candidates
    :param vmap: vertice map, for more info, see docstring for `incorporate_spots_mesh` method
    :return: Dict; same as param `model`
    """
    model, spot_candidates = resolve_obvious_spots(points, faces, model, spot_candidates, vmap)
    model = resolve_spot_candidates(star_container, model, spot_candidates, faces, component_com=component_com)
    # converting lists in model to numpy arrays
    model['object'] = np.array(model['object'])
    for spot_ix in star_container.spots:
        model['spots'][spot_ix] = np.array(model['spots'][spot_ix])
    return model


def resolve_obvious_spots(points, faces, model, spot_candidates, vmap):
    """
    Resolve those Spots/Star faces, where all tree vertices belongs to given object.
    If there are mixed vertices of any face, append it to spot candidate List.
    :param points: numpy.array; array of all points
    :param faces: numpy.array; array of all faces
    :param model: Dict; dictionary which describe object with spots as one entity
    :param spot_candidates:
    :param vmap: vertices map; for more info, see docstring for `incorporate_spots_mesh` method
    :return: Tuple[Dict, Dict]
    """
    for simplex, face_points, ix in list(zip(faces, points[faces], range(faces.shape[0]))):
        # if each point belongs to the same spot, then it is for sure face of that spot
        condition1 = vmap[simplex[0]]["enum"] == vmap[simplex[1]]["enum"] == vmap[simplex[2]]["enum"]
        if condition1:
            if 'spot' == vmap[simplex[0]]["type"]:
                model["spots"][vmap[simplex[0]]["enum"]].append(np.array(simplex))
            else:
                model["object"].append(np.array(simplex))
        else:
            spot_candidates["com"].append(np.average(face_points, axis=0))
            spot_candidates["ix"].append(ix)

    gc.collect()
    return model, spot_candidates


def resolve_spot_candidates(star_container, model, spot_candidates, faces, component_com):
    """
    Resolves spot face candidates by comparing angular distances of face cantres and spot centres.
    In case of multiple layered spots, face is assigned to the top layer.
    :param star_container:
    :param model: Dict; initialised dictionary with placeholders which will describe object with spots as one entity
    :param spot_candidates: Dict; contain indices and center of mass of each
                                  face that is candodate to be a face of spot
    :param faces: ndarray; array of indices which defines faces from points;
    :param component_com: float; center of mass of given component
    :return: Dict; filled model from input of following structure


    ::

        {
            "object": ndarray[[point_idx_i, point_idx_j, point_idx_k], ...],
            "spots": {
                spot_index: ndarray[[point_idx_u, point_idx_v, point_idx_w], ...]
            }
        }
    """
    # checking each candidate one at a time trough all spots
    com = np.array(spot_candidates["com"]) - np.array([component_com, 0.0, 0.0])
    cos_max_angle = {idx: up.cos(spot.angular_radius) for idx, spot in star_container.spots.items()}
    center = {idx: spot.center - np.array([component_com, 0.0, 0.0])
              for idx, spot in star_container.spots.items()}
    for idx, _ in enumerate(spot_candidates["com"]):
        spot_idx_to_assign = -1
        simplex_ix = spot_candidates["ix"][idx]
        for spot_ix in star_container.spots:
            cos_angle_com = up.inner(center[spot_ix], com[idx]) / \
                            (np.linalg.norm(center[spot_ix]) * np.linalg.norm(com[idx]))
            if cos_angle_com > cos_max_angle[spot_ix]:
                spot_idx_to_assign = spot_ix

        if spot_idx_to_assign == -1:
            model["object"].append(np.array(faces[simplex_ix]))
        else:
            model["spots"][spot_idx_to_assign].append(np.array(faces[simplex_ix]))

    gc.collect()
    return model


def build_faces(system_container, components_distance, component="all"):
    """
    Function creates faces of the star surface for given components. Faces are evaluated upon points that
    have to be in this time already calculated.
    :param system_container: BinarySystem; instance
    :type components_distance: float
    :param component: `primary` or `secondary` if not supplied both component are calculated
    :return:
    """
    if is_empty(component):
        __logger__.debug("no component set to build faces")
        return

    if is_empty(components_distance):
        raise ValueError('Value of `components_distance` was not provided.')

    components = bsutils.component_to_list(component)
    for component in components:
        component_instance = getattr(system_container, component)
        if component_instance.has_spots():
            build_surface_with_spots(system_container, components_distance, component)
        else:
            build_surface_with_no_spots(system_container, components_distance, component)


def build_surface_with_no_spots(system_container, components_distance, component="all"):
    """
    Function for building binary star component surfaces without spots.
    :param system_container: BinarySystem; instance
    :param components_distance: float
    :param component: str; `primary` or `secondary` if not supplied both component are calculated
    :return:
    """
    components = bsutils.component_to_list(component)

    for component in components:
        star_container = getattr(system_container, component)
        # triangulating only one quarter of the star

        if system_container.morphology != 'over-contact':
            triangulate = star_container.points[:star_container.base_symmetry_points_number, :]
            triangles = detached_system_surface(system_container, components_distance, triangulate, component)
        else:
            points = star_container.points
            neck = np.max(points[:, 0]) if component == 'primary' else np.min(points[:, 0])
            triangulate = \
                np.append(points[:star_container.base_symmetry_points_number, :], np.array([[neck, 0, 0]]), axis=0)
            triangles = over_contact_system_surface(system_container, triangulate, component)
            # filtering out triangles containing last point in `points_to_triangulate`
            triangles = triangles[np.array(triangles < star_container.base_symmetry_points_number).all(1)]

        # filtering out faces on xy an xz planes
        y0_test = ~np.isclose(triangulate[triangles][:, :, 1], 0).all(1)
        z0_test = ~np.isclose(triangulate[triangles][:, :, 2], 0).all(1)
        triangles = triangles[up.logical_and(y0_test, z0_test)]

        star_container.base_symmetry_faces_number = np.int(np.shape(triangles)[0])
        # lets exploit axial symmetry and fill the rest of the surface of the star
        all_triangles = [inv[triangles] for inv in star_container.inverse_point_symmetry_matrix]
        star_container.base_symmetry_faces = triangles
        star_container.faces = up.concatenate(all_triangles, axis=0)

        base_face_symmetry_vector = up.arange(star_container.base_symmetry_faces_number)
        star_container.face_symmetry_vector = up.concatenate([base_face_symmetry_vector for _ in range(4)])


def build_surface_with_spots(system_container, components_distance, component="all"):
    """
    Function capable of triangulation of spotty stellar surfaces.
    It merges all surface points, triangulates them and then sorts the resulting surface faces under star or spot.
    :param system_container: BinarySystem instance
    :param components_distance: float
    :param component: str `primary` or `secondary`
    :return:
    """
    components = bsutils.component_to_list(component)
    component_com = {'primary': 0.0, 'secondary': components_distance}
    for component in components:
        start_container = getattr(system_container, component)
        points, vertices_map = start_container.get_flatten_points_map()

        surface_fn = get_surface_builder_fn(system_container.morphology)
        faces = surface_fn(system_container, component=component, points=points, components_distance=components_distance)
        model, spot_candidates = initialize_model_container(vertices_map)
        model = split_spots_and_component_faces(start_container, points, faces, model,
                                                spot_candidates, vertices_map, component_com[component])
        start_container.remove_overlaped_spots_by_vertex_map(vertices_map)
        start_container.remap_surface_elements(model, points)


def detached_system_surface(system_container, components_distance, points=None, component="all"):
    """
    Calculates surface faces from the given component's points in case of detached or semi-contact system.
    :param system_container:
    :param components_distance: float
    :param points: numpy.array
    :param component: str
    :return: numpy.array; N x 3 array of vertices indices
    """
    component_instance = getattr(system_container, component)
    if points is None:
        points = component_instance.points

    if not np.any(points):
        raise ValueError(f"{component} component, with class instance name {component_instance.name} do not "
                         "contain any valid surface point to triangulate")
    # there is a problem with triangulation of near over-contact system, delaunay is not good with pointy surfaces
    critical_pot = system_container.primary.critical_surface_potential if component == 'primary' \
        else system_container.secondary.critical_surface_potential
    potential = system_container.primary.surface_potential if component == 'primary' \
        else system_container.secondary.surface_potential
    if potential - critical_pot > 0.01:
        __logger__.debug(f'triangulating surface of {component} component using standard method')
        triangulation = Delaunay(points)
        triangles_indices = triangulation.convex_hull
    else:
        __logger__.debug(f'surface of {component} component is near or at critical potential; therefore custom '
                         f'triangulation method for (near)critical potential surfaces will be used')
        # calculating closest point to the barycentre
        r_near = np.max(points[:, 0]) if component == 'primary' else np.min(points[:, 0])
        # projection of component's far side surface into ``sphere`` with radius r1

        points_to_transform = copy(points)
        if component == 'secondary':
            points_to_transform[:, 0] -= components_distance
        projected_points = \
            r_near * points_to_transform / np.linalg.norm(points_to_transform, axis=1)[:, None]
        if component == 'secondary':
            projected_points[:, 0] += components_distance

        triangulation = Delaunay(projected_points)
        triangles_indices = triangulation.convex_hull

    return triangles_indices


def over_contact_system_surface(system_container, points=None, component="all", **kwargs):
    # do not remove kwargs, keep compatible interface w/ detached where components distance has to be provided
    # in this case, components distance is sinked in kwargs and not used
    """
    Calculates surface faces from the given component's points in case of over-contact system.

    :param system_container:
    :param points: numpy.array - points to triangulate
    :param component: str; `primary` or `secondary`
    :return: numpy.array; N x 3 array of vertice indices
    """

    component_instance = getattr(system_container, component)
    if points is None:
        points = component_instance.points
    if up.isnan(points).any():
        raise ValueError(f"{component} component, with class instance name {component_instance.name} "
                         f"contain any valid point to triangulate")
    # calculating position of the neck
    neck_x = np.max(points[:, 0]) if component == 'primary' else np.min(points[:, 0])
    # parameter k is used later to transform inner surface to quasi sphere (convex object) which will be then
    # triangulated
    k = neck_x / (neck_x + 0.01) if component == 'primary' else neck_x / ((1 - neck_x) + 0.01)

    # projection of component's far side surface into ``sphere`` with radius r1
    projected_points = up.zeros(np.shape(points), dtype=float)

    # outside facing points are just inflated to match with transformed inner surface
    # condition to select outward facing points
    outside_points_test = points[:, 0] <= 0 if component == 'primary' else points[:, 0] >= 1
    outside_points = points[outside_points_test]
    if component == 'secondary':
        outside_points[:, 0] -= 1
    projected_points[outside_points_test] = neck_x * outside_points / np.linalg.norm(outside_points, axis=1)[:, None]
    if component == 'secondary':
        projected_points[:, 0] += 1

    # condition to select outward facing points
    inside_points_test = (points[:, 0] > 0)[:-1] if component == 'primary' else (points[:, 0] < 1)[:-1]
    # if auxiliary point was used than  it is not appended to list of inner points to be transformed
    # (it would cause division by zero error)
    inside_points_test = np.append(inside_points_test, False) if \
        np.array_equal(points[-1], np.array([neck_x, 0, 0])) else np.append(inside_points_test, True)
    inside_points = points[inside_points_test]
    # scaling radii for each point in cylindrical coordinates
    r = (neck_x ** 2 - (k * inside_points[:, 0]) ** 2) ** 0.5 if component == 'primary' else \
        (neck_x ** 2 - (k * (1 - inside_points[:, 0])) ** 2) ** 0.5

    length = np.linalg.norm(inside_points[:, 1:], axis=1)
    projected_points[inside_points_test, 0] = inside_points[:, 0]
    projected_points[inside_points_test, 1:] = r[:, None] * inside_points[:, 1:] / length[:, None]
    # if auxiliary point was used, than it will be appended to list of transformed points
    if np.array_equal(points[-1], np.array([neck_x, 0, 0])):
        projected_points[-1] = points[-1]

    triangulation = Delaunay(projected_points)
    triangles_indices = triangulation.convex_hull

    # removal of faces on top of the neck
    neck_test = ~(up.equal(points[triangles_indices][:, :, 0], neck_x).all(-1))
    new_triangles_indices = triangles_indices[neck_test]

    return new_triangles_indices


def compute_all_surface_areas(system_container, component):
    """
    Compute surface are of all faces (spots included).

    :param system_container: BinaryStar instance
    :param component: str `primary` or `secondary`
    :return:
    """
    if is_empty(component):
        __logger__.debug("no component set to build surface areas")
        return

    components = bsutils.component_to_list(component)
    for component in components:
        star_container = getattr(system_container, component)
        __logger__.debug(f'computing surface areas of component: '
                         f'{star_container} / name: {star_container.name}')
        star_container.calculate_all_areas()


def build_faces_orientation(system_container, components_distance, component="all"):
    """
    Compute face orientation (normals) for each face.
    If pulsations are present, than calculate renormalized associated
    Legendree polynomials (rALS) for each pulsation mode.

    :param system_container: BinarySystem instance
    :param component: str; `primary` or `secondary`
    :param components_distance: float
    orbit with misaligned pulsations, where pulsation axis drifts with star
    :return:
    """
    if is_empty(component):
        __logger__.debug("no component set to build face orientation")
        return

    component = bsutils.component_to_list(component)
    com_x = {'primary': 0.0, 'secondary': components_distance}

    for _component in component:
        star_container = getattr(system_container, _component)
        star_container.set_all_surface_centres()
        star_container.set_all_normals(com=com_x[_component])

        # here we calculate time independent part of the pulsation modes, renormalized Legendree polynomials for each
        # pulsation mode
        if star_container.has_pulsations():
            pulsations.set_ralp(star_container, com_x=com_x[_component])
