import numpy as np

from abc import abstractmethod
from copy import deepcopy, copy

from elisa import logger, umpy as up, utils
from elisa.conf import config

config.set_up_logging()
__logger__ = logger.getLogger("base-container-module")


class PropertiesContainer(object):
    def __init__(self, **kwargs):
        self.properties = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return self.properties

    def __getitem__(self, item):
        return getattr(self, item)

    def __str__(self):
        return str(self.to_dict())


class StarPropertiesContainer(PropertiesContainer):
    pass


class SystemPropertiesContainer(PropertiesContainer):
    pass


class PositionContainer(object):
    @abstractmethod
    def build(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_mesh(self, *args, **kwargs):
        pass


class StarContainer(object):
    """
    :param points: numpy.array;

    ::

        Numpy array of points that form surface of Body.
        Input dictionary has to be in shape::
            points = numpy.array([[x1 y1 z1],
                                  [x2 y2 z2],
                                   ...
                                  [xN yN zN]])
        where xi, yi, zi are cartesian coordinates of vertice i.

    :param faces: numpy.array;

    ::

        Numpy array of triangles that will create surface of body.
        Triangles are stored as list of indices of points.
            numpy.array(
            [[vertice_index_k, vertice_index_l, vertice_index_m]),
             [...]),
              ...
             [...]])

    Container carrying non-static properties of Star objecet (properties which vary from phase to phase).

    :param normals: numpy.array;
    :param temperatures: numpy.array;
    :param log_g: numpy.array;
    :param indices: numpy.array;
    :param face_centres: numpy.array; Get renormalized associated Legendre polynomials (rALS). Array of complex
    arrays for each face.
    :param metallicity: float;
    :param: properties Dict;
    """

    def __init__(self,
                 points=None,
                 normals=None,
                 indices=None,
                 faces=None,
                 temperatures=None,
                 log_g=None,
                 coverage=None,
                 rals=None,
                 face_centres=None,
                 metallicity=None,
                 areas=None):

        self.points = points
        self.normals = normals
        self.faces = faces
        self.temperatures = temperatures
        self.log_g = log_g
        self.coverage = coverage
        self.indices = indices
        self.rals = rals
        self.face_centres = face_centres
        self.metallicity = metallicity
        self.areas = areas

        self.point_symmetry_vector = np.array([])
        self.inverse_point_symmetry_matrix = np.array([])
        self.base_symmetry_points_number = 0

        self.face_symmetry_vector = np.array([])
        self.base_symmetry_faces_number = 0

        self.spots = dict()
        self.pulsations = dict()

    @staticmethod
    def from_properties_container(properties_container):
        """
        Create StarContainer from properties container.
        :param properties_container:
        :return: StarContainer
        """
        container = StarContainer()
        container.__dict__.update(properties_container.__dict__)
        return container

    def has_spots(self):
        return len(self.spots) > 0

    def has_pulsations(self):
        return len(self.pulsations) > 0

    def copy(self):
        """
        Return deepcopy of StarContainer instance.
        :return: StarContainer;
        """
        return deepcopy(self)

    def flatten(self):
        pass

    def remove_spot(self, spot_index: int):
        """
        Remove n-th spot index of object.
        :param spot_index: int
        :return:
        """
        del (self.spots[spot_index])

    def get_flatten_points_map(self):
        """
        Function returns all surface point and faces optionally with corresponding map of vertices.
        :param self: Star instance
        :return: Tuple[numpy.array, Any]: [all surface points including star and surface points, vertices map or None]
        """
        points = copy(self.points)
        for spot_index, spot_instance in self.spots.items():
            points = up.concatenate([points, spot_instance.points])

        vertices_map = [{"type": "object", "enum": -1}] * len(self.points)
        for spot_index, spot_instance in self.spots.items():
            vertices_map = up.concatenate(
                [vertices_map, [{"type": "spot", "enum": spot_index}] * len(spot_instance.points)]
            )
        return points, vertices_map

    def calculate_areas(self):
        """
        Returns areas of each face of the star surface. (spots not included)
        :return: numpy.array:

        ::

            numpy.array([area_1, ..., area_n])
        """
        if len(self.faces) == 0 or len(self.points) == 0:
            raise ValueError('Faces or/and points of object {self.name} have not been set yet.\n'
                             'Run build method first.')
        if not self.has_spots():  # temporary
            base_areas = utils.triangle_areas(self.faces[:self.base_symmetry_faces_number],
                                              self.points[:self.base_symmetry_points_number])
            return base_areas[self.face_symmetry_vector]
        else:
            return utils.triangle_areas(self.faces, self.points)

    def calculate_all_areas(self):
        """
        Calculates areas for all faces on the surface including spots and assigns values to its corresponding variables.
        :return:
        """
        self.areas = self.calculate_areas()
        if self.has_spots():
            for spot_index, spot_instance in self.spots.items():
                spot_instance.areas = spot_instance.calculate_areas()
