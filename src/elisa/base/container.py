import numpy as np

from abc import abstractmethod
from copy import (
    deepcopy,
    copy
)
from elisa.conf import config
from elisa import (
    logger,
    utils,
    umpy as up
)


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
                 areas=None,
                 potential_gradient_magnitudes=None):

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
        self.potential_gradient_magnitudes = potential_gradient_magnitudes

        self.point_symmetry_vector = np.array([])
        self.inverse_point_symmetry_matrix = np.array([])
        self.base_symmetry_points_number = 0

        self.face_symmetry_vector = np.array([])
        self.base_symmetry_faces_number = 0

        # those are used only if case of spots are NOT used
        self.base_symmetry_points = np.array([])
        self.base_symmetry_faces = np.array([])

        self.spots = dict()
        self.pulsations = dict()
        self.polar_potential_gradient_magnitude = np.nan

    @classmethod
    def from_star_instance(cls, star):
        return cls.from_properties_container(star.to_properties_container())

    @classmethod
    def from_properties_container(cls, properties_container):
        """
        Create StarContainer from properties container.

        :param properties_container:
        :return: StarContainer
        """
        container = cls()
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

    def get_flatten_parameter(self, parameter):
        """
        returns flatten parameter
        :param parameter: str; name of the parameter to flatten (do not use for faces)
        :return:
        """
        if parameter in ['faces']:
            raise ValueError(f'Function is not applicable to flatten `{parameter}` attribute.')
        retval = getattr(self, parameter)
        if self.has_spots():
            for spot in self.spots.values():
                retval = up.concatenate((retval, getattr(spot, parameter)), axis=0)
        return retval

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

    def surface_serializer(self):
        """
        Returns all points and faces of the whole star.

        :return: Tuple[numpy.array, numpy.array]
        """
        points = copy(self.points)
        faces = copy(self.faces)
        if self.has_spots():
            for spot_index, spot in self.spots.items():
                n_points = np.shape(points)[0]
                points = np.append(points, spot.points, axis=0)
                faces = np.append(faces, spot.faces + n_points, axis=0)

        return points, faces

    def reset_spots_properties(self):
        """
        Reset computed spots properties
        """
        for _, spot_instance in self.spots.items():
            spot_instance.boundary = np.array([])
            spot_instance.boundary_center = np.array([])
            spot_instance.center = np.array([])

            spot_instance.points = np.array([])
            spot_instance.normals = np.array([])
            spot_instance.faces = np.array([])
            spot_instance.face_centres = np.array([])

            spot_instance.areas = np.array([])
            spot_instance.potential_gradient_magnitudes = np.array([])
            spot_instance.temperatures = np.array([])

            spot_instance.log_g = np.array([])

    def get_flatten_properties(self):
        """
        Return flatten ndarrays of points, faces, etc. from object instance and spot instances for given object.
        :return: Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]

        ::

            Tuple(points, normals, faces, temperatures, log_g, rals, face_centres)
        """
        points = self.points
        normals = self.normals
        faces = self.faces
        temperatures = self.temperatures
        log_g = self.log_g
        rals = {mode_idx: mode.rals[0] for mode_idx, mode in self.pulsations.items()}
        centres = self.face_centres

        if isinstance(self.spots, (dict,)):
            for idx, spot in self.spots.items():
                faces = up.concatenate((faces, spot.faces + len(points)), axis=0)
                points = up.concatenate((points, spot.points), axis=0)
                normals = up.concatenate((normals, spot.normals), axis=0)
                temperatures = up.concatenate((temperatures, spot.temperatures), axis=0)
                log_g = up.concatenate((log_g, spot.log_g), axis=0)
                for mode_idx, mode in self.pulsations.items():
                    rals[mode_idx] = up.concatenate((rals[mode_idx], mode.rals[1][idx]), axis=0)
                centres = up.concatenate((centres, spot.face_centres), axis=0)

        return points, normals, faces, temperatures, log_g, rals, centres

    def flatt_it(self):
        """
        Make properties "points", "normals", "faces", "temperatures", "log_g", "rals", "centers"
        of container flatt. It means all properties of start and spots are put together.

        :return: self
        """
        props_list = ["points", "normals", "faces", "temperatures", "log_g", "rals", "centers"]
        flat_props = self.get_flatten_properties()
        for prop, value in zip(props_list, flat_props):
            setattr(self, prop, value)

        return self
