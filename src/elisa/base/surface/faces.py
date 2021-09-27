import gc
import numpy as np

from copy import copy
from ... import umpy as up


def initialize_model_container(vertices_map):
    """
    Initializes basic data structure `model` objects that will contain faces sorted by its origin (star or spots)
    and data structure containing spot candidates with its index and center point.
    Structure is based on input `verties_map`.
    Example of return Tuple

    ::

        (<class 'dict'>: {'object': [], 'spots': {0: []}}, <class 'dict'>: {'com': [], 'ix': []})

    :param vertices_map: List or ndarray; map which define refrences of index in
                         given Iterable to object (spot or Star).
                         For more info, see docstring for `incorporate_spots_mesh` method.
    :return: Tuple[Dict, Dict];
    """
    model = {"object": list(), "spots": dict()}
    spot_candidates = {"com": list(), "ix": list()}

    spots_instance_indices = list(set([vertices_map[ix]["enum"] for ix, _ in enumerate(vertices_map)
                                       if vertices_map[ix]["enum"] >= 0]))
    for spot_index in spots_instance_indices:
        model["spots"][spot_index] = list()
    return model, spot_candidates


def split_spots_and_component_faces(star, points, faces, model, spot_candidates, vmap, component_com):
    """
    Function that sorts faces to model data structure by distinguishing if it belongs to star or spots.

    :param star: elisa.base.container.StarContainer;
    :param component_com: float; center of mass of component
    :param points: numpy.array; (N_points * 3) - all points of surface
    :param faces: numpy.array; (N_faces * 3) - all faces of the surface
    :param model: Dict; data structure for faces sorting (more in docstring of method `initialize_model_container`)
    :param spot_candidates: Dict; initialised data structure for spot candidates
    :param vmap: vertice map, for more info, see docstring for `incorporate_spots_mesh` method
    :return: Dict; same as param `model`
    """
    model, spot_candidates = resolve_obvious_spots(points, faces, model, spot_candidates, vmap)
    model = resolve_spot_candidates(star, model, spot_candidates, faces, component_com=component_com)
    # converting lists in model to numpy arrays
    model['object'] = np.array(model['object'])
    for spot_ix in star.spots:
        model['spots'][spot_ix] = np.array(model['spots'][spot_ix])
    return model


def resolve_spot_candidates(star, model, spot_candidates, faces, component_com):
    """
    Resolves spot face candidates by comparing angular distances of face cantres and spot centres.
    In case of multiple layered spots, face is assigned to the top layer.

    :param star: elisa.base.container.StarContainer;
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
    cos_max_angle = {idx: up.cos(_spot.angular_radius) for idx, _spot in star.spots.items()}
    center = {idx: _spot.center - np.array([component_com, 0.0, 0.0]) for idx, _spot in star.spots.items()}
    for idx, _ in enumerate(spot_candidates["com"]):
        spot_idx_to_assign = -1
        simplex_ix = spot_candidates["ix"][idx]
        for spot_ix in star.spots:
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


def set_all_surface_centres(star):
    """
    Calculates all surface centres for given body(including spots) and assign to object as `face_centers` property
    """
    star.face_centres = calculate_surface_centres(star.points, star.faces)
    if star.has_spots() and not star.is_flat():
        for spot_index, spot_instance in star.spots.items():
            spot_instance.face_centres = calculate_surface_centres(spot_instance.points, spot_instance.faces)
    return star


def calculate_surface_centres(points, faces):
    """
    Returns centers of every surface face.

    :return: numpy.array;

    ::

        numpy.array([[center_x1, center_y1, center_z1],
                     [center_x2, center_y2, center_z2],
                      ...
                     [center_xn, center_yn, center_zn]])
    """
    return np.average(points[faces], axis=1)


def calculate_normals(points, faces, centres, com):
    """
    Returns outward facing normal unit vector for each face of stellar surface.

    :param points: numpy.array;
    :param faces: numpy.array;
    :param centres: numpy.array;
    :param com: numpy.array;
    :return: numpy.array;

    ::

        numpy.array([[normal_x1, normal_y1, normal_z1],
                     [normal_x2, normal_y2, normal_z2],
                      ...
                     [normal_xn, normal_yn, normal_zn]])
    """
    # vectors defining triangle ABC, a = B - A, b = C - A
    a = points[faces[:, 1]] - points[faces[:, 0]]
    b = points[faces[:, 2]] - points[faces[:, 0]]
    normals = np.cross(a, b)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    corr_centres = copy(centres) - np.array([com, 0, 0])[None, :]

    # making sure that normals are properly oriented near the axial planes
    sgn = up.sign(np.sum(up.multiply(normals, corr_centres), axis=1))
    return normals * sgn[:, None]


def correct_face_orientation(star_container, com=0):
    """
    Function corrects order if face indices in order to be consistent with outward facing normals.

    :param star_container: elisa.base.container.StarContainer;
    :param com: float; centre of mass x-position
    :return:
    """
    def correct_orientation(object):
        points = getattr(object, 'points')
        faces = getattr(object, 'faces')
        centres = getattr(object, 'face_centres')

        a = points[faces[:, 1]] - points[faces[:, 0]]
        b = points[faces[:, 2]] - points[faces[:, 0]]
        normals = np.cross(a, b)

        corr_centres = copy(centres) - np.array([com, 0, 0])[None, :]

        sgn = up.sign(up.sum(up.multiply(normals, corr_centres), axis=1))
        negative_sgn = sgn < 0
        faces[negative_sgn] = faces[negative_sgn][:, [1, 0, 2]]

    correct_orientation(star_container)
    if star_container.has_spots():
        for spot in star_container.spots.values():
            correct_orientation(spot)

    return star_container


def mirror_triangulation(q_triangles, inverse_point_symmetry_matrix):
    """
    This function enables for the triangulation of symmetrical part of the surface to be mirrored to the rest of
    the surface.

    :param q_triangles: numpy.array; triangles of base symmetry portion of the surface
    :param inverse_point_symmetry_matrix: numpy.array; row-wise - array that map base symmetry portion of the surface
                                                       points to the rest of the surface (quadrants or octants)
    :return: numpy.array; triangulation covering the whole star
    """
    all_triangles = [inv[q_triangles] for inv in inverse_point_symmetry_matrix]
    return np.concatenate(all_triangles, axis=0)


def mirror_face_values(values, face_symmetry_vector):
    """
    Mirroring the array values on symmetrical faces to the rest of the surface.

    :param values: numpy.array; surface parameter values corresponding to the base symmetry part of the surface
    :param face_symmetry_vector: numpy.array; array that remaps `values` to the rest of the surface
    :return: numpy.array; remapped array
    """
    return values[face_symmetry_vector]


def symmetry_face_reduction(values, base_symmetry_faces_number):
    """
    Reducing the surface distribution of surface `values` array to the symmetrical component.

    :param values: numpy.array; surface parameter distribution
    :param base_symmetry_faces_number: int; number of the first `base_symmetry_faces_numbe` from the symmetrical
                                            component
    :return: numpy.array; reduced surface distribution array
    """
    return values[:base_symmetry_faces_number]
