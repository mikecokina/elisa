import numpy as np

from copy import copy
from elisa import utils, logger, umpy as up
from elisa.base.transform import SpotProperties
from elisa.conf import config
from elisa.utils import is_empty


config.set_up_logging()
__logger__ = logger.getLogger("spots-module")


class Spot(object):
    """
    Spot data container.

    :param log_g: numpy.array

    """
    MANDATORY_KWARGS = ["longitude", "latitude", "angular_radius", "temperature_factor"]
    OPTIONAL_KWARGS = ["discretization_factor"]
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs=kwargs, kwarglist=Spot.ALL_KWARGS, instance=Spot)
        utils.check_missing_kwargs(Spot.MANDATORY_KWARGS, kwargs, instance_of=Spot)
        kwargs = self.transform_input(**kwargs)

        # supplied parameters
        self.discretization_factor = np.nan
        self.latitude = np.nan
        self.longitude = np.nan
        self.angular_radius = np.nan
        self.temperature_factor = np.nan

        # container parameters
        self.boundary = np.array([])
        self.boundary_center = np.array([])
        self.center = np.array([])

        self.points = np.array([])
        self.normals = np.array([])
        self.faces = np.array([])
        self.face_centres = np.array([])

        self.areas = np.array([])
        self.potential_gradient_magnitudes = np.array([])
        self.temperatures = np.array([])
        self.log_g = np.array([])

        self.init_properties(**kwargs)

    @staticmethod
    def transform_input(**kwargs):
        return SpotProperties.transform_input(**kwargs)

    def calculate_areas(self):
        """
        Returns areas of each face of the spot build_surface.
        :return: ndarray:

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
        return {kwarg: getattr(self, kwarg) for kwarg in self.MANDATORY_KWARGS if not is_empty(getattr(self, kwarg))}


def split_points_of_spots_and_component(on, points, vertices_map):
    """
    Based on vertices map, separates points to points which belong to Star object
    and points which belong to each defined Spot object.
    During the process remove overlapped spots.
    :param on: instance of object to split spots on
    :param points: numpy.array; all points of object (spot points and component points together)
    :param vertices_map: List or numpy.array; map which define refrences of index in
                         given Iterable to object (Spot or Star).
    :return: Dict;

    ::

        {
            "object": numpy.array([[pointN_x, pointN_y, pointN_z], ...]),
            "spot_index": numpy.array([[pointM_x, pointM_y, pointM_z], ...]), ...
        }
    """
    points = np.array(points)
    component_points = {"object": points[up.where(np.array(vertices_map) == {'enum': -1})[0]]}
    on.remove_overlaped_spots_by_spot_index(set([int(val["enum"]) for val in vertices_map if val["enum"] > -1]))
    spots_points = {
        f"{i}": points[up.where(np.array(vertices_map) == {'enum': i})[0]] for i in range(len(on.spots))
        if len(up.where(np.array(vertices_map) == {'enum': i})[0]) > 0
    }
    return {**component_points, **spots_points}


def setup_body_points(on, points):
    """
    Setup points for Star instance and spots based on input `points` Dict object.
    Such `points` map looks like following

    :param on: instance to setup spot points and body points on
    :param points: Dict[str, numpy.array]
    ::

        {
            "object": [[point0_x, point0_y, point0_z], ..., [pointN_x, pointN_y, pointN_z]],
            "0": [<points>],
            "1": [<points>]...
        },
        where `object` contain numpy.array of object points and indices points for given spot.

    :return:
    """
    on.points = points.pop("object")
    for spot_index, spot_points in points.items():
        on.spots[int(spot_index)].points = points[spot_index]


def incorporate_spots_mesh(to, component_com):
    """
    Based on spots definitions, evaluate spot points on Star surface and remove those points of Star itself
    which are inside of any spots. Do the same operation with spot to each other.
    Evaluation is running from index 0, what means that spot with lower index
    is overwriten by spot with higher index.

    All points are assigned to given object (Star points to Star object and Spot points to related Spot object).

    # todo: change structure flat array like [-1, -1, -1, ..., 0, ..., 1, ...]
    Defines variable `vertices_map` used in others method. Strutcure of this variable is following::

        [{"enum": -1}, {"enum": 0}, {"enum": 1}, ..., {"enum": N}].

    Enum index -1 means that points in joined array of points on current
    index position of vertices_map belongs to Star point.
    Enum indices >= 0 means the same, but for Spot.

    :param to: isntnace to incorporate spots into
    :param component_com: center of mass of component
    :return:
    """
    if not to.spots:
        __logger__.debug(f"not spots found, skipping incorporating spots to mesh on component {to.name}")
        return
    __logger__.debug(f"incorporating spot points to component {to.name} mesh")

    vertices_map = [{"enum": -1} for _ in to.points]
    # `all_component_points` do not contain points of Any spot
    all_component_points = copy(to.points)
    neck = np.min(all_component_points[:to.base_symmetry_points_number, 0])

    for spot_index, spot in to.spots.items():
        # average spacing in spot points
        vertices_to_remove, vertices_test = [], []
        cos_max_angle_point = up.cos(spot.angular_radius + 0.30 * spot.discretization_factor)
        spot_center = spot.center - np.array([component_com, 0., 0.])

        # removing star points in spot
        # all component points means just points of component NOT merged points + spots
        for ix, pt in enumerate(all_component_points):
            surface_point = all_component_points[ix] - np.array([component_com, 0., 0.])
            cos_angle = \
                up.inner(spot_center, surface_point) / (
                    np.linalg.norm(spot_center) * np.linalg.norm(surface_point))
            # skip all points of object outside of spot
            if cos_angle < cos_max_angle_point or np.linalg.norm(pt[0] - neck) < 1e-9:
                continue
            # mark component point (NOT point of spot) for removal if is within the spot
            vertices_to_remove.append(ix)

        # simplices of target object for testing whether point lying inside or not of spot boundary, removing
        # duplicate points on the spot border
        # kedze vo vertice_map nie su body skvrny tak toto tu je zbytocne viac menej
        vertices_to_remove = list(set(vertices_to_remove))

        # points and vertices_map update
        _points, _vertices_map = list(), list()

        for ix, vertex in list(zip(range(0, len(all_component_points)), all_component_points)):
            if ix in vertices_to_remove:
                # skip point if is marked for removal
                continue

            # append only points of currrent object that do not intervent to spot
            # [current, since there should be already spot from previous iteration step]
            _points.append(vertex)
            _vertices_map.append({"enum": vertices_map[ix]["enum"]})

        for vertex in spot.points:
            _points.append(vertex)
            _vertices_map.append({"enum": spot_index})

        all_component_points = copy(_points)
        vertices_map = copy(_vertices_map)

    separated_points = split_points_of_spots_and_component(to, all_component_points, vertices_map)
    setup_body_points(to, separated_points)
