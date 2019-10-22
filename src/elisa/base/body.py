import gc
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

    def calculate_normals(self, points, faces, centres=None, com=None):
        """
        Returns outward facing normal unit vector for each face of stellar surface.

        :return: numpy.array:

        ::

            numpy.array([[normal_x1, normal_y1, normal_z1],
                         [normal_x2, normal_y2, normal_z2],
                          ...
                         [normal_xn, normal_yn, normal_zn]])
        """
        normals = np.array([np.cross(points[xx[1]] - points[xx[0]], points[xx[2]] - points[xx[0]]) for xx in faces])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        cntr = self.calculate_surface_centres(points, faces) if is_empty(centres) else copy(centres)

        corr_centres = cntr - np.array([com, 0, 0])[None, :]

        # making sure that normals are properly oriented near the axial planes
        sgn = np.sign(np.sum(np.multiply(normals, corr_centres), axis=1))

        return normals * sgn[:, None]

    def set_all_normals(self, com=None):
        """
        Function calculates normals for each face of given body (including spots) and assign it to object.

        :param com: numpy.array
        :return:
        """
        self.normals = self.calculate_normals(points=self.points, faces=self.faces, centres=self.face_centres, com=com)
        if self.has_spots():
            for spot_index in self.spots:
                self.spots[spot_index].normals = self.calculate_normals(points=self.spots[spot_index].points,
                                                                        faces=self.spots[spot_index].faces,
                                                                        centres=self.spots[spot_index].face_centres,
                                                                        com=com)

    def set_all_surface_centres(self):
        """
        Calculates all surface centres for given body(including spots) and assign to object as `face_centers` property

        :return:
        """
        self.face_centres = self.calculate_surface_centres()
        if self.has_spots():
            for spot_index, spot in self.spots.items():
                spot.face_centres = \
                    self.calculate_surface_centres(points=spot.points,
                                                   faces=spot.faces)

    def calculate_surface_centres(self, points=None, faces=None):
        """
        Returns centers of every surface face.
        If `points` is not supplied, parameter of self instance is used for both, `points` and `faces`.

        :return: numpy.array:

        ::

            numpy.array([[center_x1, center_y1, center_z1],
                         [center_x2, center_y2, center_z2],
                          ...
                         [center_xn, center_yn, center_zn]])
        """
        if is_empty(points):
            points = self.points
            faces = self.faces
        return np.average(points[faces], axis=1)

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

    def return_whole_surface(self):
        """
        Returns all points and faces of the whole star.

        :return: Tuple[numpy.array, numpy.array]
        """
        ret_points = copy(self.points)
        ret_faces = copy(self.faces)
        if self.has_spots():
            for spot_index, spot in self.spots.items():
                n_points = np.shape(ret_points)[0]
                ret_points = np.append(ret_points, spot.points, axis=0)
                ret_faces = np.append(ret_faces, spot.faces + n_points, axis=0)

        return ret_points, ret_faces

    def setup_body_points(self, points):
        spot.setup_body_points(self, points)

    def remove_overlaped_spots_by_vertex_map(self, vertices_map):
        """
        Remove spots of Start object that are totally overlapped by another spot.

        :param vertices_map: List or numpy.array
        :return:
        """
        # remove spots that are totaly overlaped
        spots_instance_indices = list(set([vertices_map[ix]["enum"] for ix, _ in enumerate(vertices_map)
                                           if vertices_map[ix]["enum"] >= 0]))
        for spot_index, _ in list(self.spots.items()):
            if spot_index not in spots_instance_indices:
                self._logger.warning(f"spot with index {spot_index} doesn't contain Any face "
                                     f"and will be removed from component {self.name} spot list")
                self.remove_spot(spot_index=spot_index)
        gc.collect()

    def remap_surface_elements(self, model, points_to_remap):
        """
        Function remaps all surface points (`points_to_remap`) and faces (star and spots) according to the `model`.

        :param model: dict - list of indices of points in `points_to_remap` divided into star and spots sublists
        :param points_to_remap: array of all surface points (star + points used in `_split_spots_and_component_faces`)
        :return:
        """
        # remapping points and faces of star
        self._logger.debug(f"changing value of parameter points of component {self.name}")
        indices = np.unique(model["object"])
        self.points = points_to_remap[indices]

        self._logger.debug(f"changing value of parameter faces of component {self.name}")

        points_length = np.shape(points_to_remap)[0]
        remap_list = np.empty(points_length, dtype=int)
        remap_list[indices] = np.arange(np.shape(indices)[0])
        self.faces = remap_list[model["object"]]

        # remapping points and faces of spots
        for spot_index, _ in list(self.spots.items()):
            self._logger.debug(f"changing value of parameter points of spot {spot_index} / component {self.name}")
            # get points currently belong to the given spot
            indices = np.unique(model["spots"][spot_index])
            self.spots[spot_index].points = points_to_remap[indices]

            self._logger.debug(f"changing value of parameter faces of spot {spot_index} / component {self.name}")

            remap_list = np.empty(points_length, dtype=int)
            remap_list[indices] = np.arange(np.shape(indices)[0])
            self.spots[spot_index].faces = remap_list[model["spots"][spot_index]]
        gc.collect()

    def incorporate_spots_mesh(self, component_com=None):
        spot.incorporate_spots_mesh(self, component_com)

    def split_points_of_spots_and_component(self, points, vertices_map):
        return spot.split_points_of_spots_and_component(self, points, vertices_map)

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
