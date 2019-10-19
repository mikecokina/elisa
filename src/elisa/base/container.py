import numpy as np

from elisa import logger
from elisa.conf import config
from copy import deepcopy
from elisa.base import spot

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

    def copy(self):
        """
        Copy self instance

        :return: self; copied self instance
        """
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

    def remove_overlaped_spots_by_spot_index(self, keep_spot_indices):
        """
        Remove definition and instance of those spots that are overlaped
        by another one and basically has no face to work with.
        :param keep_spot_indices: List[int]; list of spot indices to keep
        :return:
        """
        all_spot_indices = set([int(val) for val in self.spots.keys()])
        spot_indices_to_remove = all_spot_indices.difference(keep_spot_indices)

        for spot_index in spot_indices_to_remove:
            self.remove_spot(spot_index)
