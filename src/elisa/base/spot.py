from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np

from elisa import umpy as up
from elisa import units as u
from elisa import utils
from elisa.base.error import SpotError
from elisa.base.transform import SpotProperties
from elisa.base.types import INT
from elisa.logger import getLogger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.types import Float, Int, VertexMap
    from elisa.units import _DefaultSpotInputUnits, _DefaultSpotUnits


logger = getLogger(__name__)

_IS_ZERO_TOLERANCE = 1e-9


class Spot:
    """Container describing a circular spot on a stellar surface.

    The spot stores its input parameters (longitude, latitude, angular radius,
    temperature factor, discretization factor) and the derived mesh data
    (points, faces, normals, areas, etc.).

    Parameters
    ----------
    longitude : float
        Longitude of the spot (degrees or astropy units accepted at input).
    latitude : float
        Latitude of the spot (degrees or astropy units accepted at input).
    angular_radius : float
        Angular radius of the spot (degrees or astropy units accepted at input).
    temperature_factor : float
        Ratio t_eff,spot / t_eff,star.
    discretization_factor : float, optional
        Mean angular size of a spot face (degrees or astropy units accepted).

    Attributes
    ----------
    boundary : numpy.ndarray
        Boundary points of the spot.
    boundary_center : float
        Geometrical centre of the spot boundary.
    center : float
        Coordinates of the spot centre.
    points : numpy.ndarray
        Surface points that belong to the spot.
    normals, faces, face_centres, areas, temperatures, log_g
        Various arrays describing the surface elements of the spot.

    """

    MANDATORY_KWARGS = ("longitude", "latitude", "angular_radius", "temperature_factor")
    OPTIONAL_KWARGS = ("discretization_factor",)
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs) -> None:
        utils.invalid_kwarg_checker(kwargs=kwargs, kwarglist=Spot.ALL_KWARGS, instance=Spot)
        utils.check_missing_kwargs(Spot.MANDATORY_KWARGS, kwargs, instance_of=Spot)
        self.kwargs = self.transform_input(**kwargs)

        # supplied parameters
        self.discretization_factor = np.nan
        self.latitude = np.nan
        self.longitude = np.nan
        self.angular_radius = np.nan
        self.temperature_factor = np.nan

        # container parameters
        self.boundary = np.array([])
        self.boundary_center = np.nan
        self.center = np.nan

        self.points = np.array([])
        self.normals = np.array([])
        self.faces = np.array([])
        self.face_centres = np.array([])
        self.points_spherical = np.array([])

        self.velocities = np.array([])

        self.areas = np.array([])
        self.potential_gradient_magnitudes = np.array([])
        self.temperatures = np.array([])
        self.log_g = np.array([])

        self.init_properties(**self.kwargs)

    @property
    def default_input_units(self) -> _DefaultSpotInputUnits:
        """Return default units used for spot input values.

        :returns: DefaultSpotInputUnits
        :rtype: elisa.units.DefaultSpotInputUnits
        """
        return u.DefaultSpotInputUnits

    @property
    def default_internal_units(self) -> _DefaultSpotUnits:
        """Return default internal units used by the spot.

        :returns: DefaultSpotUnits
        :rtype: elisa.units.DefaultSpotUnits
        """
        return u.DefaultSpotUnits

    @staticmethod
    def transform_input(**kwargs) -> dict:
        """Transform and normalise input kwargs for internal use.

        :returns: Transformed kwargs mapping
        :rtype: dict
        """
        return SpotProperties.transform_input(**kwargs)

    def calculate_areas(self) -> NDArray[Float]:
        """Compute areas of faces defined for this spot.

        :returns: Areas of each triangular face
        :rtype: numpy.typing.NDArray[numpy.float64]
        """
        return utils.triangle_areas(triangles=self.faces, points=self.points)

    def init_properties(self, **kwargs) -> None:
        """Initialise instance attributes from provided kwargs.

        This method sets attributes found in ``kwargs`` on ``self``.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def kwargs_serializer(self) -> dict:
        """Serialize mandatory spot kwargs (including units when applicable).

        :returns: Mapping of kwarg name to values (with units when applicable)
        :rtype: dict
        """
        default_units = {
            "longitude": u.DefaultSpotUnits.longitude,
            "latitude": u.DefaultSpotUnits.latitude,
            "angular_radius": u.DefaultSpotUnits.angular_radius,
            "discretization_factor": u.DefaultSpotUnits.discretization_factor,
        }

        serialized_kwargs: dict = {}
        for kwarg in self.ALL_KWARGS:
            if kwarg in default_units:
                value = getattr(self, kwarg)
                if not isinstance(value, u.Quantity):
                    value = value * default_units[kwarg]
                serialized_kwargs[kwarg] = value
            else:
                serialized_kwargs[kwarg] = getattr(self, kwarg)
        return serialized_kwargs


def split_points_of_spots_and_component(
    on_container: StarContainer,
    points: NDArray[Float],
    vertices_map: NDArray[Int],
) -> dict:
    """Split merged points array into component points and spot points.

    The function uses ``vertices_map`` where value ``-1`` denotes a component
    point and integers ``>= 0`` denote spot ownership by index.

    :param on_container: Container holding spot instances (must implement a
        ``spots`` mapping).
    :type on_container: elisa.base.container.StarContainer
    :param points: Array containing component and spot points stacked together.
    :type points: numpy.typing.NDArray[numpy.float64]
    :param vertices_map: Integer array assigning each point to either the
        component (-1) or a spot index (>= 0).
    :type vertices_map: numpy.typing.NDArray[int]
    :returns: Mapping with key ``"object"`` for component points and keys
        ``"0"``, ``"1"`` ... for spot points.
    :rtype: dict[str, numpy.typing.NDArray[numpy.float64]]
    """
    points = np.array(points)
    component_points = {"object": points[vertices_map == -1]}
    indices = np.unique(vertices_map[vertices_map > -1])

    # Remove spot definitions that would be fully overlapped (no points)
    remove_overlaped_spots_by_spot_index(on_container, indices)

    spots_points = {
        f"{i}": points[vertices_map == i]
        for i in range(len(on_container.spots))
        if len(vertices_map[vertices_map == i]) > 0
    }
    return {**component_points, **spots_points}


def setup_body_points(on_container: StarContainer, points: dict) -> None:
    """Assign component and spot points back to the container.

    :param on_container: Container with ``spots`` mapping.
    :type on_container: elisa.base.container.StarContainer
    :param points: Mapping returned by :func:`split_points_of_spots_and_component`.
    :type points: dict
    """
    on_container.points = points.pop("object")
    for spot_index, spot in points.items():
        on_container.spots[int(spot_index)].points = spot


def incorporate_spots_mesh(to_container: StarContainer, component_com: Float) -> StarContainer:
    """Incorporate spot points into component mesh and remove underlying points.

    The function collects component points, appends spot points and produces a
    vertices map that is then used to split points back to component and spots.

    :param to_container: Container with spot definitions.
    :type to_container: elisa.base.container.StarContainer
    :param component_com: X coordinate of component centre of mass.
    :type component_com: Float
    :returns: The same container with updated ``points`` and spot ``points`` set.
    :rtype: elisa.base.container.StarContainer
    """
    if not to_container.spots:
        logger.debug(
            "not spots found, skipping incorporating spots to_container mesh on component %s",
            to_container.name,
        )
        return to_container

    logger.debug("incorporating spot points to_container component %s mesh", to_container.name)

    vertices_map = np.full(to_container.points.shape[0], -1)
    all_component_points = copy(to_container.points)

    neck = np.max(np.abs(to_container.points[:, 0] - component_com))

    for spot_index, spot in to_container.spots.items():
        cos_max_angle_point = up.cos(spot.angular_radius + 0.30 * spot.discretization_factor)
        spot_center = spot.center - np.array([component_com, 0.0, 0.0])

        com_pts = all_component_points - np.array([component_com, 0.0, 0.0])[None, :]
        cos_angles = np.sum(np.multiply(spot_center[None, :], com_pts), axis=1) / (
            np.linalg.norm(spot_center) * np.linalg.norm(com_pts, axis=1)
        )

        angular_dist_cond = cos_angles < cos_max_angle_point
        neck_cond = np.abs(np.abs(com_pts[:, 0]) - neck) < _IS_ZERO_TOLERANCE
        in_condition = np.logical_or(angular_dist_cond, neck_cond)

        vertices_to_keep = np.arange(com_pts.shape[0], dtype=INT)[in_condition]
        vertices_to_keep = np.unique(vertices_to_keep)
        all_component_points = all_component_points[vertices_to_keep]
        vertices_map = vertices_map[vertices_to_keep]

        all_component_points = np.vstack((all_component_points, spot.points))
        vertices_map = np.concatenate((vertices_map, np.full(spot.points.shape[0], spot_index)))

    separated_points = split_points_of_spots_and_component(to_container, all_component_points, vertices_map)
    setup_body_points(to_container, separated_points)
    return to_container


def remap_surface_elements(
    on_container: StarContainer,
    mapper: Mapping,
    points_to_remap: NDArray[Float],
) -> StarContainer:
    """Remap points and faces arrays for the component and spots according to a mapper.

    :param on_container: Container with ``spots``.
    :type on_container: elisa.base.container.StarContainer
    :param mapper: Mapping with keys ``"object"`` and ``"spots"`` describing new indices.
    :type mapper: Mapping
    :param points_to_remap: Array with all points used in remapping.
    :type points_to_remap: numpy.typing.NDArray[numpy.float64]
    :returns: Updated container.
    :rtype: elisa.base.container.StarContainer
    """
    logger.debug("changing value of parameter points of component %s", on_container.name)
    indices = np.unique(mapper["object"])
    on_container.points = points_to_remap[indices]

    logger.debug("changing value of parameter faces of component %s", on_container.name)

    points_length = np.shape(points_to_remap)[0]
    remap_list = np.empty(points_length, dtype=INT)
    remap_list[indices] = up.arange(np.shape(indices)[0])
    on_container.faces = remap_list[mapper["object"]]

    for spot_index in list(on_container.spots.keys()):
        logger.debug("changing value of parameter points of spot %s / component %s", spot_index, on_container.name)
        indices = np.unique(mapper["spots"][spot_index])
        on_container.spots[spot_index].points = points_to_remap[indices]

        logger.debug("changing value of parameter faces of spot %s / component %s", spot_index, on_container.name)

        remap_list = np.empty(points_length, dtype=INT)
        remap_list[indices] = up.arange(np.shape(indices)[0])
        on_container.spots[spot_index].faces = remap_list[mapper["spots"][spot_index]]
    return on_container


def remove_overlaped_spots_by_spot_index(
    from_container: StarContainer,
    keep_spot_indices: NDArray[Int],
    *,
    _raise: bool = True,
) -> StarContainer:
    """Remove spot definitions that are overlapped and have no points / faces left.

    :param from_container: Container with ``spots``.
    :type from_container: elisa.base.container.StarContainer
    :param keep_spot_indices: Iterable of spot indices that have points and should be kept.
    :type keep_spot_indices: numpy.typing.NDArray[int]
    :param _raise: If True, raise :class:`SpotError` when spots would be removed.
    :type _raise: bool
    :returns: Updated container.
    :rtype: elisa.base.container.StarContainer
    """
    all_spot_indices = {int(val) for val in from_container.spots}
    spot_indices_to_remove = all_spot_indices.difference(keep_spot_indices)
    spots_meta = [
        from_container.spots[idx].kwargs_serializer() for idx in from_container.spots if idx in spot_indices_to_remove
    ]
    spots_meta_str = "\n".join([str(meta) for meta in spots_meta])
    if _raise and not utils.is_empty(spot_indices_to_remove):
        msg = (f"Spots {spots_meta_str} have no points to continue."
               f"Please, specify spots wisely.")
        raise SpotError(msg)
    for spot_index in spot_indices_to_remove:
        from_container.remove_spot(spot_index)
    return from_container


def remove_overlaped_spots_by_vertex_map(
    from_container: StarContainer,
    vertices_map: VertexMap,
) -> StarContainer:
    """Remove spots that are totally overlapped by another spot according to ``vertices_map``.

    :param from_container: Container with ``spots``.
    :type from_container: elisa.base.container.StarContainer
    :param vertices_map: Iterable of vertex descriptors; each item must provide an
        ``"enum"`` key with spot ownership index.
    :type vertices_map: elisa.types.VertexMap
    :returns: Updated container.
    :rtype: elisa.base.container.StarContainer
    """
    spots_instance_indices = list(
        {vertices_map[ix]["enum"] for ix, _ in enumerate(vertices_map) if vertices_map[ix]["enum"] >= 0},
    )
    for spot_index in list(from_container.spots.keys()):
        if spot_index not in spots_instance_indices:
            msg = f"Spot with index {spot_index} doesn't contain any face on component {from_container.name}."
            raise SpotError(msg)
    return from_container
