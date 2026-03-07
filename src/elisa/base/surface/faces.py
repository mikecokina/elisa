from __future__ import annotations

import gc
from copy import copy
from typing import TYPE_CHECKING

import numpy as np

from elisa import umpy as up

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.types import Float


def initialize_model_container(vertices_map: list[dict[str, int | str]]) -> tuple[dict, dict]:
    """Initialize containers used to group faces by origin (star or spots).

    The function inspects ``vertices_map`` to discover spot instance indices
    and returns two containers: ``model`` which will hold lists of face
    index triplets for the star and for each spot, and ``spot_candidates``
    which is initialized to collect candidate faces for later resolution.

    :param vertices_map: Sequence mapping each vertex index to its owning
        object (fields ``"enum"`` and ``"type"`` are expected).
    :type vertices_map: list[dict[str, int | str]]
    :returns: Tuple containing ``model`` and ``spot_candidates``.
    :rtype: tuple[dict, dict]
    """
    model: dict = {"object": [], "spots": {}}
    spot_candidates: dict = {"com": [], "ix": []}

    spots_instance_indices = {entry["enum"] for entry in vertices_map if entry["enum"] >= 0}
    for spot_index in spots_instance_indices:
        model["spots"][spot_index] = []
    return model, spot_candidates


def split_spots_and_component_faces(
        star: StarContainer,
        points: NDArray[np.floating],
        faces: NDArray[np.integer],
        model: dict,
        spot_candidates: dict,
        vmap: list[dict[str, int | str]],
        component_com: Float,
) -> dict:
    """Sort faces into the ``model`` structure, separating star faces and spots.

    Faces that are clearly owned by a single object are placed directly into
    ``model``; mixed faces are collected into ``spot_candidates`` and later
    resolved by proximity to spot centres.

    :param star: StarContainer instance containing spot definitions.
    :type star: elisa.base.container.StarContainer
    :param points: Array of 3D points describing the surface.
    :type points: numpy.typing.NDArray
    :param faces: Array of triangular faces (indices into ``points``).
    :type faces: numpy.typing.NDArray
    :param model: Pre-initialised model container returned by
        :func:`initialize_model_container`.
    :type model: dict
    :param spot_candidates: Candidate structure for mixed faces.
    :type spot_candidates: dict
    :param vmap: Vertices map describing ownership of vertices.
    :type vmap: list[dict[str, int | str]]
    :param component_com: X coordinate of the component centre of mass.
    :type component_com: elisa.types.Float
    :returns: The populated ``model`` mapping with arrays instead of lists.
    :rtype: dict
    """
    model, spot_candidates = resolve_obvious_spots(points, faces, model, spot_candidates, vmap)
    model = resolve_spot_candidates(star, model, spot_candidates, faces, component_com=component_com)

    # convert lists to numpy arrays for downstream consumers
    if len(model["object"]) > 0:
        model["object"] = np.array(model["object"])
    else:
        model["object"] = np.array([], dtype=int).reshape(0, 3)

    for spot_ix in list(model["spots"].keys()):
        if len(model["spots"][spot_ix]) > 0:
            model["spots"][spot_ix] = np.array(model["spots"][spot_ix])
        else:
            model["spots"][spot_ix] = np.array([], dtype=int).reshape(0, 3)

    return model


def resolve_spot_candidates(
        star: StarContainer,
        model: dict,
        spot_candidates: dict,
        faces: NDArray[np.integer],
        component_com: Float,
) -> dict:
    """Assign mixed faces (candidates) to the correct spot or to the object.

    The assignment is based on angular distance between face centres and
    spot centres; for layered spots the topmost matching spot gets the
    face.

    :param star: StarContainer instance with spot metadata (``angular_radius`` and ``center``).
    :type star: elisa.base.container.StarContainer
    :param model: Current model dictionary being filled.
    :type model: dict
    :param spot_candidates: Structure containing candidate face centres and indices.
    :type spot_candidates: dict
    :param faces: All faces array used to fetch face indices for assignment.
    :type faces: numpy.typing.NDArray
    :param component_com: X coordinate of the component centre of mass.
    :type component_com: elisa.types.Float
    :returns: Updated ``model`` dictionary with faces appended to appropriate lists.
    :rtype: dict
    """
    com = np.array(spot_candidates["com"]) - np.array([component_com, 0.0, 0.0])
    cos_max_angle = {idx: up.cos(_spot.angular_radius) for idx, _spot in star.spots.items()}
    center = {idx: _spot.center - np.array([component_com, 0.0, 0.0]) for idx, _spot in star.spots.items()}

    for idx in range(len(spot_candidates["com"])):
        spot_idx_to_assign = -1
        simplex_ix = spot_candidates["ix"][idx]
        for spot_ix in star.spots:
            # compute cosine of angle between two vectors
            denom = np.linalg.norm(center[spot_ix]) * np.linalg.norm(com[idx])
            cos_angle_com = up.inner(center[spot_ix], com[idx]) / denom if denom != 0 else -1.0
            if cos_angle_com > cos_max_angle[spot_ix]:
                spot_idx_to_assign = spot_ix

        if spot_idx_to_assign == -1:
            model["object"].append(np.array(faces[simplex_ix]))
        else:
            model["spots"][spot_idx_to_assign].append(np.array(faces[simplex_ix]))

    gc.collect()
    return model


def resolve_obvious_spots(
        points: NDArray[np.floating],
        faces: NDArray[np.integer],
        model: dict,
        spot_candidates: dict,
        vmap: list[dict[str, int | str]],
) -> tuple[dict, dict]:
    """Classify faces that clearly belong to a single object (star or spot).

    Faces whose three vertices all map to the same owning entity are
    appended directly to the corresponding ``model`` list. Mixed faces
    are recorded into ``spot_candidates`` for later resolution.

    :param points: Array of 3D points describing the surface.
    :type points: numpy.typing.NDArray
    :param faces: Array of triangular faces (indices into ``points``).
    :type faces: numpy.typing.NDArray
    :param model: Model dictionary to append classified faces to.
    :type model: dict
    :param spot_candidates: Candidate structure to collect mixed faces.
    :type spot_candidates: dict
    :param vmap: Vertices map describing ownership of vertices.
    :type vmap: list[dict[str, int | str]]
    :returns: Tuple containing updated (model, spot_candidates).
    :rtype: tuple[dict, dict]
    """
    for ix, simplex in enumerate(faces):
        face_points = points[simplex]
        same_owner = (
                vmap[simplex[0]]["enum"] == vmap[simplex[1]]["enum"] == vmap[simplex[2]]["enum"]
        )
        if same_owner:
            if vmap[simplex[0]]["type"] == "spot":
                model["spots"][vmap[simplex[0]]["enum"]].append(np.array(simplex))
            else:
                model["object"].append(np.array(simplex))
        else:
            spot_candidates["com"].append(np.average(face_points, axis=0))
            spot_candidates["ix"].append(ix)

    gc.collect()
    return model, spot_candidates


def set_all_surface_centres(star: StarContainer) -> StarContainer:
    """Compute face centres for the star and its spots (if present).

    The function assigns the computed face centres to ``star.face_centres``
    and to ``spot_instance.face_centres`` for each spot.

    :param star: StarContainer whose face centres are to be updated.
    :type star: elisa.base.container.StarContainer
    :returns: The same ``star`` instance with updated face centre attributes.
    :rtype: elisa.base.container.StarContainer
    """
    star.face_centres = calculate_surface_centres(star.points, star.faces)
    if star.has_spots() and not star.is_flat():
        for spot_instance in star.spots.values():
            spot_instance.face_centres = calculate_surface_centres(spot_instance.points, spot_instance.faces)
    return star


def calculate_surface_centres(points: NDArray[np.floating], faces: NDArray[np.integer]) -> NDArray[np.floating]:
    """Return centroids of triangular faces.

    :param points: Array of 3D points.
    :type points: numpy.typing.NDArray
    :param faces: Array of triangle indices.
    :type faces: numpy.typing.NDArray
    :returns: Array of face centroids (N_faces x 3).
    :rtype: numpy.typing.NDArray
    """
    return np.average(points[faces], axis=1)


# noinspection PyUnreachableCode
def calculate_normals(
        points: NDArray[np.floating],
        faces: NDArray[np.integer],
        centres: NDArray[np.floating],
        com: Float,
) -> NDArray[np.floating]:
    """Compute outward unit normals for triangular faces.

    The orientation of normals is adjusted so that they point outwards
    from the object centre.

    :param points: Array of 3D points.
    :type points: numpy.typing.NDArray
    :param faces: Array of triangle indices.
    :type faces: numpy.typing.NDArray
    :param centres: Array of precomputed face centres.
    :type centres: numpy.typing.NDArray
    :param com: X coordinate of object centre-of-mass.
    :type com: elisa.types.Float
    :returns: Unit normal vectors for each face.
    :rtype: numpy.typing.NDArray
    """
    a = points[faces[:, 1]] - points[faces[:, 0]]
    b = points[faces[:, 2]] - points[faces[:, 0]]
    normals = np.cross(a, b)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    corr_centres = copy(centres) - np.array([com, 0, 0])[None, :]

    sgn = up.sign(np.sum(up.multiply(normals, corr_centres), axis=1))
    return normals * sgn[:, None]


def correct_face_orientation(star_container: StarContainer, com: Float = 0) -> StarContainer:
    """Ensure face indices are ordered consistent with outward normals.

    The function flips faces where the computed sign of dot(normals, centres)
    is negative so that normals point outwards.

    :param star_container: StarContainer or spot instance.
    :type star_container: elisa.base.container.StarContainer
    :param com: X coordinate of centre-of-mass to use for orientation.
    :type com: elisa.types.Float
    :returns: The same container with possibly modified face ordering.
    :rtype: elisa.base.container.StarContainer
    """

    def _correct_orientation(obj: StarContainer) -> None:
        points = obj.points
        faces = obj.faces
        centres = obj.face_centres

        a = points[faces[:, 1]] - points[faces[:, 0]]
        b = points[faces[:, 2]] - points[faces[:, 0]]
        normals = np.cross(a, b)

        corr_centres = copy(centres) - np.array([com, 0, 0])[None, :]

        sgn = up.sign(np.sum(up.multiply(normals, corr_centres), axis=1))
        negative_sgn = sgn < 0
        faces[negative_sgn] = faces[negative_sgn][:, [1, 0, 2]]

    _correct_orientation(star_container)
    if star_container.has_spots():
        for spot in star_container.spots.values():
            _correct_orientation(spot)

    return star_container


def mirror_triangulation(
        q_triangles: NDArray[np.integer],
        inverse_point_symmetry_matrix: NDArray[np.integer],
) -> NDArray[np.integer]:
    """Mirror triangulation from base symmetry portion to the full mesh.

    :param q_triangles: Triangles of the base symmetric portion.
    :type q_triangles: numpy.typing.NDArray
    :param inverse_point_symmetry_matrix: Array of index mappings per symmetry block.
    :type inverse_point_symmetry_matrix: numpy.typing.NDArray
    :returns: Concatenated triangulation covering the full surface.
    :rtype: numpy.typing.NDArray
    """
    all_triangles = [inv[q_triangles] for inv in inverse_point_symmetry_matrix]
    return np.concatenate(all_triangles, axis=0)


def mirror_face_values(
        values: NDArray[np.floating],
        face_symmetry_vector: NDArray[np.integer],
) -> NDArray[np.floating]:
    """Map face-local values from the base symmetry block to the full surface.

    :param values: Values defined on the base symmetric faces.
    :type values: numpy.typing.NDArray
    :param face_symmetry_vector: Index map to expand values to full mesh.
    :type face_symmetry_vector: numpy.typing.NDArray
    :returns: Remapped values for the full surface.
    :rtype: numpy.typing.NDArray
    """
    return values[face_symmetry_vector]


def symmetry_face_reduction(values: NDArray[np.floating], base_symmetry_faces_number: int) -> NDArray[np.floating]:
    """Reduce a full-surface distribution to its base-symmetry subset.

    :param values: Full-surface parameter distribution.
    :type values: numpy.typing.NDArray
    :param base_symmetry_faces_number: Number of faces in the base symmetry block.
    :type base_symmetry_faces_number: int
    :returns: Reduced distribution limited to the base symmetry faces.
    :rtype: numpy.typing.NDArray
    """
    return values[:base_symmetry_faces_number]
