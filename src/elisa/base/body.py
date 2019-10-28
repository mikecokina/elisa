import numpy as np

from abc import ABCMeta, abstractmethod
from copy import copy
from elisa import logger, utils, units, umpy as up
from elisa.base import spot
from elisa.utils import is_empty
from elisa.base.spot import Spot


class Body(metaclass=ABCMeta):
    """
    Abstract class that defines bodies that can be modelled by this software.
    Units are imported from astropy.units module::

        see documentation http://docs.astropy.org/en/stable/units/

    :param synchronicity: float;
    :param mass: float or astropy.quantity.Quantity;
    :param albedo: float;
    :param discretization_factor: float or astropy.quantity.Quantity;
    :param t_eff: float or astropy.quantity.Quantity;
    :param polar_radius: float or astropy.quantity.Quantity;
    :param equatorial_radius: float;
    :param spots: Dict;


    """

    ID = 1

    def __init__(self, name, logger_name=None, suppress_logger=False, **kwargs):
        """
        Properties of abstract class Body.
        """
        # initial kwargs
        self.kwargs = copy(kwargs)
        self._suppress_logger = suppress_logger
        self._logger = logger.getLogger(logger_name or self.__class__.__name__, suppress=self._suppress_logger)

        if is_empty(name):
            self.name = str(Body.ID)
            self._logger.debug(f"name of class instance {self.__class__.__name__} set to {self.name}")
            Body.ID += 1
        else:
            self.name = str(name)

        # initializing parmas to default values
        self.synchronicity = np.nan
        self.mass = np.nan
        self.albedo = np.nan
        self.discretization_factor = up.radians(3)
        self.t_eff = np.nan
        self.polar_radius = np.nan
        self._spots = dict()
        self.equatorial_radius = np.nan






        # move to container
        self.points = np.array([])
        self.faces = np.array([])
        self.normals = np.array([])
        self.areas = np.array([])
        self.temperatures = np.array([])

        self._face_centres = np.array([])
        self._point_symmetry_vector = np.array([])
        self.inverse_point_symmetry_matrix = np.array([])
        self.base_symmetry_points_number = 0
        self._face_symmetry_vector = np.array([])
        self.base_symmetry_faces_number = 0
        # those are used only if case of spots are used
        self.base_symmetry_points = np.array([])
        self.base_symmetry_faces = np.array([])

    def serialize_symmetries(self):
        return {
            "point_symmetry_vector": self._point_symmetry_vector,
            "inverse_point_symmetry_matrix": self.inverse_point_symmetry_matrix,
            "base_symmetry_points_number": self.base_symmetry_points_number,
            "face_symmetry_vector": self._face_symmetry_vector,
            "base_symmetry_faces_number": self.base_symmetry_faces_number,
            "base_symmetry_points": self.base_symmetry_points,
            "base_symmetry_faces": self.base_symmetry_faces
        }

    @abstractmethod
    def transform_input(self, *args, **kwargs):
        pass

    @property
    def spots(self):
        """
        :return: Dict[int, Spot]
        """
        return self._spots

    @spots.setter
    def spots(self, spots):
        # todo: update example
        """
        example of defined spots

        ::

            [
                 {"longitude": 90,
                  "latitude": 58,
                  "angular_radius": 15,
                  "temperature_factor": 0.9},
                 {"longitude": 85,
                  "latitude": 80,
                  "angular_radius": 30,
                  "temperature_factor": 1.05},
                 {"longitude": 45,
                  "latitude": 90,
                  "angular_radius": 30,
                  "temperature_factor": 0.95},
             ]

        :param spots: Iterable[Dict]; definition of spots for given object
        :return:
        """
        self._spots = {idx: Spot(**spot_meta) for idx, spot_meta in enumerate(spots)} if not is_empty(spots) else dict()
        for spot_idx, spot_instance in self.spots.items():
            self.setup_spot_instance_discretization_factor(spot_instance, spot_idx)

    def has_spots(self):
        """
        Find whether object has defined spots.

        :return: bool
        """
        return len(self._spots) > 0

    def remove_spot(self, spot_index: int):
        """
        Remove n-th spot index of object.

        :param spot_index: int
        :return:
        """
        del (self._spots[spot_index])

    def setup_spot_instance_discretization_factor(self, spot_instance, spot_index):
        """
        Setup discretization factor for given spot instance based on defined rules::

            - used Star discretization factor if not specified in spot
            - if spot_instance.discretization_factor > 0.5 * spot_instance.angular_diameter then factor is set to
              0.5 * spot_instance.angular_diameter
        :param spot_instance: Spot
        :param spot_index: int; spot index (has no affect on process, used for logging)
        :return:
        """
        # component_instance = getattr(self, component)
        if is_empty(spot_instance.discretization_factor):
            self._logger.debug(f'angular density of the spot {spot_index} on {self.name} component was not supplied '
                               f'and discretization factor of star {self.discretization_factor} was used.')
            spot_instance.discretization_factor = (0.9 * self.discretization_factor * units.ARC_UNIT).value
        if spot_instance.discretization_factor > spot_instance.angular_radius:
            self._logger.debug(f'angular density {self.discretization_factor} of the spot {spot_index} on {self.name} '
                               f'component was larger than its angular radius. Therefore value of angular density was '
                               f'set to be equal to 0.5 * angular diameter')
            spot_instance.discretization_factor = spot_instance.angular_radius * units.ARC_UNIT

        return spot_instance












    @property
    def face_centres(self):
        """
        Returns coordinates of centres of corresponding faces.

        :return: numpy.array

        ::

            numpy.array([[face_centrex1, face_centrey1, face_centrez1],
                           [face_centrex2, face_centrey2, face_centrez2],
                            ...,
                           [face_centrexN, face_centrey2, face_centrez2]])
        """
        return self._face_centres

    @face_centres.setter
    def face_centres(self, centres):
        """
        :param centres: numpy.array

        ::

                numpy.array([[face_centrex1, face_centrey1, face_centrez1],
                             [face_centrex2, face_centrey2, face_centrez2],
                              ...,
                             [face_centrexN, face_centrey2, face_centrez2]])
        :return:
        """
        if np.shape(centres)[0] != np.shape(self.faces)[0]:
            raise ValueError('Number of surface centres doesn`t equal to number of faces')
        self._face_centres = centres

    @property
    def point_symmetry_vector(self):
        """
        Vector of indices with the same length as body`s points.
        N-th value of point_symmetry_matrix indicates position of base symmetry point for given n-th point.

        :return: numpy.array
        """
        return self._point_symmetry_vector

    @point_symmetry_vector.setter
    def point_symmetry_vector(self, symmetry_vector):
        """
        Setter for vector of indices with the same length as body`s points, n-th value of point_symmetry_matrix
        indicates position of base symmetry point for given n-th point.

        :param symmetry_vector: numpy.array

        ::

            numpy.array([index_of_symmetry_point_for_point1, ..., index_of_symmetry_point_for_pointN])

        :return:
        """
        if np.shape(self.points)[0] != np.shape(symmetry_vector)[0] and not self.has_spots():
            raise ValueError(f'Length of symmetry vector {np.shape(symmetry_vector)[0]} is not '
                             f'the same as number of surface points {np.shape(self.points)[0]}')
        self._point_symmetry_vector = symmetry_vector

    @property
    def face_symmetry_vector(self):
        """
        Vector of indices with the same length as body`s faces, n-th value of face_symmetry_matrix indicates position
        of base symmetry face for given n-th point.

        :return: numpy.array
        """
        return self._face_symmetry_vector

    @face_symmetry_vector.setter
    def face_symmetry_vector(self, symmetry_vector):
        """
        Setter for vector of indices with the same length as body`s faces, n-th value of face_symmetry_matrix
        indicates position of base symmetry face for given n-th point.

        :param symmetry_vector: numpy.array;

        ::

            numpy.array([index_of_symmetry_face1, ..., index_of_symmetry_faceN])

        :return:
        """
        if np.shape(self.faces)[0] != np.shape(symmetry_vector)[0]:
            raise ValueError(f'Length of symmetry vector {np.shape(symmetry_vector)[0]} is not '
                             f'the same as number of surface faces {np.shape(self.faces)[0]}')
        self._face_symmetry_vector = symmetry_vector

    def calculate_areas(self):
        """
        Returns areas of each face of the star surface.

        :return: numpy.array:

        ::

            numpy.array([area_1, ..., area_n])
        """
        if len(self.faces) == 0 or len(self.points) == 0:
            raise ValueError('Faces or/and points of object {self.name} have not been set yet.\n'
                             'Run build method first.')
        # FIXME: why the hell is this temporary?????
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
            for spot_index, spot in self.spots.items():
                spot.areas = spot.calculate_areas()

    # TODO: remove
    def setup_body_points(self, points):
        spot.setup_body_points(self, points)

    # TODO: remove
    def remove_overlaped_spots_by_vertex_map(self, vertices_map):
        spot.remove_overlaped_spots_by_vertex_map(self, vertices_map)

    # TODO: remove
    def remap_surface_elements(self, model, points_to_remap):
        spot.remap_surface_elements(self, model, points_to_remap)

    # TODO: remove
    def split_points_of_spots_and_component(self, points, vertices_map):
        return spot.split_points_of_spots_and_component(self, points, vertices_map)

    # TODO: remove
    def remove_overlaped_spots_by_spot_index(self, keep_spot_indices):
        spot.remove_overlaped_spots_by_spot_index(self, keep_spot_indices)

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
