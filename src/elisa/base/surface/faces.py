import gc
import numpy as np

from elisa import umpy as up


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

