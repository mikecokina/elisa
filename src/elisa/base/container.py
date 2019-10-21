import numpy as np

from abc import abstractmethod
from copy import deepcopy, copy

from elisa import logger, umpy as up
from elisa.conf import config
from elisa.base import spot
from elisa.utils import is_empty

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
    Container carrying non-static properties of Star objecet (properties which vary from phase to phase).

    :param points: numpy.array;
    :param normals: numpy.array;
    :param faces: numpy.array;
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
                 metallicity=None):

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

        self.point_symmetry_vector = np.array([])
        self.inverse_point_symmetry_matrix = np.array([])
        self.base_symmetry_points_number = 0

        self.spots = dict()
        self.pulsations = dict()

    def has_spots(self):
        return len(self.spots) > 0

    def has_pulsations(self):
        return len(self.pulsations) > 0

    def copy(self):
        return deepcopy(self)

    @staticmethod
    def from_properties_container(properties_container):
        container = StarContainer()
        container.__dict__.update(properties_container.__dict__)
        return container

    def flatten(self):
        pass

    def incorporate_spots_mesh(self, component_com):
        spot.incorporate_spots_mesh(to=self, component_com=component_com)

    def remove_spot(self, spot_index: int):
        """
        Remove n-th spot index of object.
        :param spot_index: int
        :return:
        """
        del (self.spots[spot_index])

    def remove_overlaped_spots_by_spot_index(self, keep_spot_indices, _raise=True):
        """
        Remove definition and instance of those spots that are overlaped
        by another one and basically has no face to work with.
        :param keep_spot_indices: List[int]; list of spot indices to keep
        :return:
        """
        all_spot_indices = set([int(val) for val in self.spots.keys()])
        spot_indices_to_remove = all_spot_indices.difference(keep_spot_indices)
        spots_meta = [self.spots[idx].kwargs_serializer() for idx in self.spots if idx in spot_indices_to_remove]
        spots_meta = '\n'.join(spots_meta)
        if _raise and not is_empty(spot_indices_to_remove):
            raise ValueError(f"Spots {spots_meta} have no pointns to continue.\n"
                             f"Please, specify spots wisely.")
        for spot_index in spot_indices_to_remove:
            self.remove_spot(spot_index)

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
