import gc
import numpy as np

from copy import copy
from .. base.transform import SpotProperties
from .. import units as u, settings
from .. utils import is_empty
from .. logger import getLogger
from .. import (
    utils,
    umpy as up
)

logger = getLogger("base.spots")


class Spot(object):
    """
    Spot object container. It is available as a list item of a `Star.spots` attribute after the initialization of the
    host system. This spot class is producing a circular spot at a specified coordinates with a temperature difference
    between the spot and the host star described by the `temperature_factor` defined as
    t_eff,spot/t_eff,star.

    Input parameters:

    :param longitude: Union[(numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity];
                      Expecting value in degrees or as astropy units instance.
    :param latitude: Union[(numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity];
                     Expecting value in degrees or as astropy units instance.
    :param angular_radius: Union[(numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity];
            `              Expecting value in degrees or as astropy units instance.
    :param temperature_factor: Union[(numpy.)int, (numpy.)float];
    :param discretization_factor: Union[(numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity];
                                  Spot discretization_factor (mean angular size of spot face).
                                  Expecting value in degrees or as astropy units instance.

    Output parameters that describing the spot:
    
    :boundary: numpy.array; points at the spot/star
    :boundary_center: float; geometrical centre of the spot boundary
    :center: float; coordinates of the spot centre
    :points: numpy.array; surface points belonging to spot
    :normals: numpy.array; outward facing normal vectors of surface elements belonging to the spot
    :faces: numpy.array; surface element (triangle) simplices
    :face_centres: numpy.array; centers of spot surface elements
    :areas: numpy.array; areas of spot surface elements
    :potential_gradient_magnitudes: numpy.array;
    :temperatures: numpy.array; effective temperatures of surface elements
    :log_g: numpy.array; surface gravity distribution over the surface elements

    """
    MANDATORY_KWARGS = ["longitude", "latitude", "angular_radius", "temperature_factor"]
    OPTIONAL_KWARGS = ["discretization_factor"]
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
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
    def default_input_units(self):
        """
        Returns set of default units of intialization parameters, in case, when provided without units.

        :return: elisa.units.DefaultSpotInputUnits;
        """
        return u.DefaultSpotInputUnits

    @property
    def default_internal_units(self):
        """
        Returns set of internal units of Spot parameters.

        :return: elisa.units.DefaultSpotUnits;
        """
        return u.DefaultSpotUnits

    @staticmethod
    def transform_input(**kwargs):
        return SpotProperties.transform_input(**kwargs)

    def calculate_areas(self):
        """
        Returns areas of each face of the spot build_surface.

        :return: numpy.array:

        ::

            numpy.array([area_1, ..., area_n])
        """
        return utils.triangle_areas(triangles=self.faces, points=self.points)

    def init_properties(self, **kwargs):
        for key in kwargs:
            set_val = kwargs.get(key)
            setattr(self, key, set_val)

    def kwargs_serializer(self):
        """
        Serializer and return mandatory kwargs of sefl (Spot) instance to dict.

        :return: Dict; { kwarg: value }
        """

        default_units = {
            "longitude": u.ARC_UNIT,
            "latitude": u.ARC_UNIT,
            "angular_radius": u.ARC_UNIT,
            "discretization_factor": u.ARC_UNIT
        }

        serialized_kwargs = dict()
        for kwarg in self.ALL_KWARGS:
            if kwarg in default_units:
                value = getattr(self, kwarg)
                if not isinstance(value, u.Quantity):
                    value = value * default_units[kwarg]
                serialized_kwargs[kwarg] = value
            else:
                serialized_kwargs[kwarg] = getattr(self, kwarg)
        return serialized_kwargs


def split_points_of_spots_and_component(on_container, points, vertices_map):
    """
    Based on vertices map, separates points to points which belong to Star object
    and points which belong to each defined Spot object.
    During the process remove overlapped spots.

    :param on_container: Union; instance of object to split spots on
    :param points: numpy.array; all points of object (spot points and component points together)
    :param vertices_map: Union[List, numpy.array]; map which define refrences of index in
                         given Iterable to object (Spot or Star).
    :return: Dict;

    ::

        {
            "object": numpy.array([[pointN_x, pointN_y, pointN_z], ...]),
            "spot_index": numpy.array([[pointM_x, pointM_y, pointM_z], ...]), ...
        }
    """
    points = np.array(points)
    # component_points = {"object": points[up.where(np.array(vertices_map) == {'enum': -1})[0]]}
    component_points = {"object": points[vertices_map == -1]}
    # indices = set([int(val["enum"]) for val in vertices_map if val["enum"] > -1])
    indices = np.unique(vertices_map[vertices_map > -1])
    remove_overlaped_spots_by_spot_index(on_container, indices)
    # spots_points = {
    #     f"{i}": points[up.where(np.array(vertices_map) == {'enum': i})[0]] for i in range(len(on_container.spots))
    #     if len(up.where(np.array(vertices_map) == {'enum': i})[0]) > 0
    # }
    spots_points = {
        f"{i}": points[vertices_map == i] for i in range(len(on_container.spots))
        if len(vertices_map[vertices_map == i]) > 0
    }
    return {**component_points, **spots_points}


def setup_body_points(on_container, points):
    """
    Setup points for Star instance and spots based on input `points` Dict object.
    Such `points` map looks like following

    :param on_container: Union; instance to setup spot points and body points on
    :param points: Dict[str, numpy.array];

    ::

        {
            "object": [[point0_x, point0_y, point0_z], ..., [pointN_x, pointN_y, pointN_z]],
            "0": [<points>],
            "1": [<points>]...
        },

    where `object` contain numpy.array of object points and indices points for given spot.
    """
    on_container.points = points.pop("object")
    for spot_index, spot_points in points.items():
        on_container.spots[int(spot_index)].points = points[spot_index]


def incorporate_spots_mesh(to_container, component_com):
    """
    Based on spots definitions, evaluate spot points on Star surface and remove those points of Star itself
    which are inside of any spots. Do the same operation with iteratively for each Spot instance defined previously.
    Evaluation is running from index 0, what means that spot with lower index
    is overwriten by spot with higher index if located on top of each other.

    All points are assigned to given object (Star points to Star object and Spot points to related Spot object).

    Function defines variable `vertices_map` used in other methods. Structure of this variable is following::

        [-1, -1, -1, ..., 0, ..., 1, ...]

    Value -1 means that points on given
    index position of vertices_map belongs to Star point.
    Enum indices >= 0 means that given point belongs to Spot with given index.

    :param to_container: Union; instace to incorporate spots into
    :param component_com: float; center of mass of component (it's x coordinate)
    :return: to_container: Union; instace to incorporate spots into
    """
    if not to_container.spots:
        logger.debug(f"not spots found, skipping incorporating spots "
                     f"to_container mesh on component {to_container.name}")
        return to_container
    logger.debug(f"incorporating spot points to_container component {to_container.name} mesh")

    # vertices_map = [{"enum": -1} for _ in to_container.points]
    vertices_map = np.full(to_container.points.shape[0], -1)
    # `all_component_points` do not contain points of Any spot
    all_component_points = copy(to_container.points)

    neck = np.max(np.abs(to_container.points[:, 0] - component_com))

    for spot_index, spot in to_container.spots.items():
        # average spacing in spot points
        cos_max_angle_point = up.cos(spot.angular_radius + 0.30 * spot.discretization_factor)
        spot_center = spot.center - np.array([component_com, 0., 0.])

        # removing star points in spot
        # all component points means just points of component NOT merged points + spots

        com_pts = all_component_points - np.array([component_com, 0., 0.])[None, :]
        cos_angles = np.sum(np.multiply(spot_center[None, :], com_pts), axis=1) / (
                    np.linalg.norm(spot_center) * np.linalg.norm(com_pts, axis=1))

        # skip all points of object outside of spot
        angular_dist_cond = cos_angles < cos_max_angle_point
        # skip points near neck
        neck_cond = np.abs(np.abs(com_pts[:, 0]) - neck) < 1e-9
        in_condition = np.logical_or(angular_dist_cond, neck_cond)

        vertices_to_keep = np.arange(com_pts.shape[0], dtype=np.int)[in_condition]

        # simplices of target object for testing whether point lying inside or not of spot boundary, removing
        # duplicate points on the spot border
        # vertices_to_remove = np.unique(vertices_to_remove)
        vertices_to_keep = np.unique(vertices_to_keep)
        all_component_points = all_component_points[vertices_to_keep]
        vertices_map = vertices_map[vertices_to_keep]

        all_component_points = np.row_stack((all_component_points, spot.points))
        vertices_map = np.concatenate((vertices_map, np.full(spot.points.shape[0], spot_index)))

    separated_points = split_points_of_spots_and_component(to_container, all_component_points, vertices_map)
    setup_body_points(to_container, separated_points)
    return to_container


def remap_surface_elements(on_container, mapper, points_to_remap):
    """
    Function remaps all surface points (`points_to_remap`) and faces (star and spots) according to the `model`.

    :param on_container: Union; container object with spots
    :param mapper: List; list of indices of points in `points_to_remap` divided into star and spots sublists
    :param points_to_remap: numpy.array; array of all surface points (star + points used in
                            `_split_spots_and_component_faces`)
    :return: on_container: Union; container object with spots
    """

    # remapping points and faces of star
    logger.debug(f"changing value of parameter points of component {on_container.name}")
    indices = np.unique(mapper["object"])
    on_container.points = points_to_remap[indices]

    logger.debug(f"changing value of parameter faces of component {on_container.name}")

    points_length = np.shape(points_to_remap)[0]
    remap_list = np.empty(points_length, dtype=int)
    remap_list[indices] = up.arange(np.shape(indices)[0])
    on_container.faces = remap_list[mapper["object"]]

    # remapping points and faces of spots
    for spot_index, _ in list(on_container.spots.items()):
        logger.debug(f"changing value of parameter points of spot {spot_index} / component {on_container.name}")
        # get points currently belong to the given spot
        indices = np.unique(mapper["spots"][spot_index])
        on_container.spots[spot_index].points = points_to_remap[indices]

        logger.debug(f"changing value of parameter faces of spot {spot_index} / component {on_container.name}")

        remap_list = np.empty(points_length, dtype=int)
        remap_list[indices] = up.arange(np.shape(indices)[0])
        on_container.spots[spot_index].faces = remap_list[mapper["spots"][spot_index]]
    gc.collect()
    return on_container


def remove_overlaped_spots_by_spot_index(from_container, keep_spot_indices, _raise=True):
    """
    Remove definition and instance of those spots that are overlaped
    by another one and basically has no face to work with.

    :param from_container: Union; container object with spots
    :param keep_spot_indices: List[int]; list of spot indices to keep
    :param _raise: bool;
    :return: from_container: Union; container object with spots
    """
    all_spot_indices = set([int(val) for val in from_container.spots.keys()])
    spot_indices_to_remove = all_spot_indices.difference(keep_spot_indices)
    spots_meta = [from_container.spots[idx].kwargs_serializer()
                  for idx in from_container.spots if idx in spot_indices_to_remove]
    spots_meta = '\n'.join([str(meta) for meta in spots_meta])
    if _raise and not is_empty(spot_indices_to_remove):
        raise ValueError(f"Spots {spots_meta} have no pointns to continue.\nPlease, specify spots wisely.")
    for spot_index in spot_indices_to_remove:
        from_container.remove_spot(spot_index)
    return from_container


def remove_overlaped_spots_by_vertex_map(from_container, vertices_map):
    """
    Remove spots of Start object that are totally overlapped by another spot.

    :param from_container: Union; container object with spots
    :param vertices_map: Union[List, numpy.array]
    :return: from_container: Union; container object with spots
    """
    # remove spots that are totaly overlaped
    spots_instance_indices = list(set([vertices_map[ix]["enum"] for ix, _ in enumerate(vertices_map)
                                       if vertices_map[ix]["enum"] >= 0]))
    for spot_index, _ in list(from_container.spots.items()):
        if spot_index not in spots_instance_indices:
            if not settings.SUPPRESS_WARNINGS:
                logger.warning(f"spot with index {spot_index} doesn't contain Any face "
                               f"and will be removed from component {from_container.name} spot list")
            from_container.remove_spot(spot_index=spot_index)
    gc.collect()
    return from_container
