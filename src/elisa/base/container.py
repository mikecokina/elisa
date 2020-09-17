import numpy as np

from abc import abstractmethod

from .. import const
from .. logger import getLogger
from copy import (
    deepcopy,
    copy
)
from .. import (
    utils,
    umpy as up
)

logger = getLogger("base.container")


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
    def __init__(self, position):
        self._flatten = False
        self._components = list()
        self.position = position
        self.inclination = np.nan
        self.period = np.nan

    @abstractmethod
    def build(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_mesh(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_faces(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_surface_areas(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_faces_orientation(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_surface_gravity(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_temperature_distribution(self, *args, **kwargs):
        pass

    def flatt_it(self):
        # naive implementation of idempotence
        if self._flatten:
            return self

        for component in self._components:
            star_container = getattr(self, component)
            if star_container.has_spots() or star_container.has_pulsations():
                star_container.flatt_it()

        self._flatten = True
        return self

    def apply_rotation(self):
        """
        Rotate quantities defined in __PROPERTIES__ in case of components defined in __PROPERTIES__.
        Rotation is made in orbital plane and inclination direction in respective order.
        Angle are defined in self.position and self.inclination.
        :return: elisa.base.PositionContainer;
        """
        __PROPERTIES__ = ["points", "normals", "velocities"]

        for component in self._components:
            star_container = getattr(self, component)
            for prop in __PROPERTIES__:
                prop_value = getattr(star_container, prop)

                args = (self.position.azimuth - const.HALF_PI, prop_value, "z", False, False)
                prop_value = utils.around_axis_rotation(*args)

                args = (const.HALF_PI - self.inclination, prop_value, "y", False, False)
                prop_value = utils.around_axis_rotation(*args)
                setattr(star_container, prop, prop_value)
        return self

    def add_secular_velocity(self):
        """
        Addition of secular radial velocity of centre of mass to convert velocieties to reference frame of observer
        """
        gamma = getattr(self, "gamma")
        for component in self._components:
            star = getattr(self, component)
            star.velocities[:, 0] += gamma
        return self

    def apply_darkside_filter(self):
        """
        Apply darkside filter on current position defined in container.
        Function iterates over components and assigns indices of visible points of star_component instance.

        :return: elisa.base.PositionContainer;
        """

        for component in self._components:
            star_container = getattr(self, component)
            cosines = getattr(star_container, "los_cosines")
            valid_indices = self.darkside_filter(cosines=cosines)
            setattr(star_container, "indices", valid_indices)
        return self

    def calculate_face_angles(self, line_of_sight):
        """
        calculates angles between normals and line_of_sight vector for all components of the system

        :param line_of_sight: np.array;
        :return:
        """
        for component in self._components:
            star_container = getattr(self, component)
            normals = getattr(star_container, "normals")
            los_cosines = self.return_cosines(normals, line_of_sight=line_of_sight)
            setattr(star_container, "los_cosines", los_cosines)

    @staticmethod
    def return_cosines(normals, line_of_sight):
        """
        returns angles between normals and line_of_sight vector

        :param normals: numpy.array;
        :param line_of_sight: numpy.array;
        :return:
        """
        if (line_of_sight == np.array([1.0, 0.0, 0.0])).all():
            return utils.calculate_cos_theta_los_x(normals=normals)
        else:
            return utils.calculate_cos_theta(normals=normals, line_of_sight_vector=np.array([1, 0, 0]))

    @staticmethod
    def darkside_filter(cosines):
        """
        Return indices for visible faces defined by given normals.
        Function assumes that `line_of_sight` ([1, 0, 0]) and `normals` are already normalized to one.

        :param cosines: numpy.array;
        :return: numpy.array;
        """
        # todo: require to resolve self shadowing in case of W UMa, but probably not here
        # recovering indices of points on near-side (from the point of view of observer)
        return up.arange(np.shape(cosines)[0])[cosines > 0]

    def copy(self):
        """
        Return deepcopy of PositionContainer instance.
        :return: elisa.base.container.PositionContainer;
        """
        return deepcopy(self)


class StarContainer(object):
    """
    Container carrying non-static properties of Star objecet (properties which vary from phase to phase) and also
    all properties set on create.
    Method `from_properties_container` or `from_star_instance` has to be used to create container properly.
    Following parameters are gathered from Star object

    'mass', 't_eff', 'synchronicity', 'albedo', 'discretization_factor', 'name', 'spots'
    'polar_radius', 'equatorial_radius', 'gravity_darkening', 'surface_potential',
    'pulsations', 'metallicity', 'polar_log_g', 'critical_surface_potential', 'side_radius'

    If you are experienced used, you can create instance directly by calling

    >>> from elisa.base.container import StarContainer
    >>> kwargs = {}
    >>> inst = StarContainer(**kwargs)

    and setup static parameter latter of not at all if not necessary for further use.
    Bellow are optional imput parameters of StarContainer.
    kwargs: Dict;
    :**kwargs options**:

        * **points** * -- numpy.array;
        * **normals** * -- numpy.array;
        * **indices** * -- numpy.array;
        * **faces** * -- numpy.array;
        * **temperatures** * -- numpy.array;
        * **log_g** * -- numpy.array;
        * **coverage** * -- numpy.array;
        * **rals** * -- numpy.array;
        * **face_centres** * -- numpy.array;
        * **metallicity** * -- float;
        * **areas** * -- numpy.array;
        * **potential_gradient_magnitudes** * -- numpy.array;


    Output parameters (obtained by applying given methods upon container).

    :points: numpy.array;

    ::

        Numpy array of points that form surface of Body.
        Input dictionary has to be in shape::
            points = numpy.array([[x1 y1 z1],
                                  [x2 y2 z2],
                                   ...
                                  [xN yN zN]])

        where xi, yi, zi are cartesian coordinates of vertice i.

    :normals: numpy.array; Array containing normalised outward facing normals
              of corresponding faces with same index

    ::

            normals = numpy_array([[normal_x1, normal_y1, normal_z1],
                                   [normal_x2, normal_y2, normal_z2],
                                    ...
                                   [normal_xn, normal_yn, normal_zn]]

    :faces: numpy.array;

    ::

        Numpy array of triangles that will create surface of body.
        Triangles are stored as list of indices of points.

            numpy.array([[vertice_index_k, vertice_index_l, vertice_index_m]),
                         [...]),
                          ...
                         [...]])

    :temperatures: numpy.array; Array of temeratures of corresponding faces.

    ::
            numpy.array([t_eff1, ..., t_effn])

    :log_g: numpy.array; Array of log_g (gravity) of corresponding faces.

    ::

        numpy.array([log_g1, ..., log_gn])

    :coverage: numpy.array;
    :indices: numpy.array; Indices of visible faces when darkside_filter is applied.
    :rals: numpy.array; Renormalized associated Legendre polynomials (rALS). Array of complex
                        arrays for each face.
    :face_centres: numpy.array;
    :metallicity: float;
    :areas: numpy.array;
    :potential_gradient_magnitudes: numpy.array;
    :point_symmetry_vector: numpy.array;
    :inverse_point_symmetry_matrix: numpy.array;
    :base_symmetry_points_number: float;
    :face_symmetry_vector: numpy.array;
    :base_symmetry_faces_number: float;
    :base_symmetry_points: numpy.array;
    :base_symmetry_points: numpy.array;
    :base_symmetry_faces: numpy.array;
    :polar_potential_gradient_magnitude: numpy.array;
    """

    def __init__(self,
                 points=None,
                 normals=None,
                 velocities=None,
                 indices=None,
                 faces=None,
                 temperatures=None,
                 log_g=None,
                 coverage=None,
                 rals=None,
                 face_centres=None,
                 metallicity=None,
                 areas=None,
                 potential_gradient_magnitudes=None,
                 ld_cfs=None,
                 normal_radiance=None,
                 los_cosines=None):

        self.points = points
        self.normals = normals
        self.faces = faces
        self.velocities = velocities
        self.temperatures = temperatures
        self.log_g = log_g
        self.coverage = coverage
        self.indices = indices
        self.rals = rals
        self.face_centres = face_centres
        self.metallicity = metallicity
        self.areas = areas
        self.potential_gradient_magnitudes = potential_gradient_magnitudes
        self.ld_cfs = ld_cfs
        self.normal_radiance = normal_radiance
        self.los_cosines = los_cosines

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

        self._flatten = False

    @classmethod
    def from_star_instance(cls, star):
        return cls.from_properties_container(star.to_properties_container())

    @classmethod
    def from_properties_container(cls, properties_container):
        """
        Create StarContainer from properties container.

        :param properties_container:
        :return: elisa.base.container.StarContainer
        """
        container = cls()
        container.__dict__.update(properties_container.__dict__)
        return container

    def has_spots(self):
        return len(self.spots) > 0

    def has_pulsations(self):
        return len(self.pulsations) > 0

    def symmetry_test(self):
        return not self.has_spots() and not self.has_pulsations()

    def copy(self):
        """
        Return deepcopy of StarContainer instance.

        :return: elisa.base.container.StarContainer;
        """
        return deepcopy(self)

    def get_flatten_parameter(self, parameter):
        """
        returns flatten parameter
        :param parameter: str; name of the parameter to flatten (do not use for faces)
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
        if self.symmetry_test():
            base_areas = utils.triangle_areas(self.faces[:self.base_symmetry_faces_number],
                                              self.points[:self.base_symmetry_points_number])
            return base_areas[self.face_symmetry_vector]
        else:
            return utils.triangle_areas(self.faces, self.points)

    def calculate_all_areas(self):
        """
        Calculates areas for all faces on the surface including spots and assigns values to its corresponding variables.
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
            spot_instance.velocities = np.array([])

            spot_instance.log_g = np.array([])

    def get_flatten_properties(self):
        """
        Return flatten ndarrays of points, faces, etc. from object instance and spot instances for given object.
        :return: Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]

        ::

            Tuple(points, normals, faces, temperatures, log_g, rals, face_centres, areas, velocities)
        """
        points = self.points
        normals = self.normals
        faces = self.faces
        temperatures = self.temperatures
        log_g = self.log_g
        rals = {mode_idx: None for mode_idx, mode in self.pulsations.items()}
        # rals = {mode_idx: mode.rals[0] for mode_idx, mode in self.pulsations.items()}
        centres = self.face_centres
        velocities = self.velocities
        areas = self.areas

        if isinstance(self.spots, (dict,)):
            for idx, spot in self.spots.items():
                faces = up.concatenate((faces, spot.faces + len(points)), axis=0)
                points = up.concatenate((points, spot.points), axis=0)
                normals = up.concatenate((normals, spot.normals), axis=0)
                temperatures = up.concatenate((temperatures, spot.temperatures), axis=0)
                log_g = up.concatenate((log_g, spot.log_g), axis=0)
                centres = up.concatenate((centres, spot.face_centres), axis=0)
                areas = up.concatenate((areas, spot.areas), axis=0)
                velocities = up.concatenate((velocities, spot.velocities), axis=0)

        return points, normals, faces, temperatures, log_g, rals, centres, areas, velocities

    def flatt_it(self):
        """
        Make properties "points", "normals", "faces", "temperatures", "log_g", "rals", "centers", "areas"
        of container flatt. It means all properties of start and spots are put together.

        :return: self
        """
        # naive implementation of idempotence
        if self._flatten:
            return self

        props_list = ["points", "normals", "faces", "temperatures", "log_g", "rals", "centers", "areas", "velocities"]
        flat_props = self.get_flatten_properties()
        for prop, value in zip(props_list, flat_props):
            setattr(self, prop, value)

        self._flatten = True
        return self

    def transform_points_to_spherical_coordinates(self, kind='points', com_x=0):
        """
        Transforming container cartesian variable to spherical coordinates in a frame of reference of given object.

        :param kind: str; `points` or `face_centres` or other variable containing cartesian
                          points in both star and spot containers
        :param com_x: float;
        :return: Tuple; spherical coordinates of star variable, dictionary of spherical coordinates of spot variable
        """
        # separating variables to convert
        centres_cartesian = copy(getattr(self, kind))
        centres_spot_cartesian = {spot_idx: copy(getattr(spot, kind)) for spot_idx, spot in self.spots.items()}

        # transforming variables
        centres_cartesian[:, 0] -= com_x
        for spot_index, spot in self.spots.items():
            centres_spot_cartesian[spot_index][:, 0] -= com_x

        # conversion
        centres = utils.cartesian_to_spherical(centres_cartesian)
        centres_spot = {spot_idx: utils.cartesian_to_spherical(spot_centres) for spot_idx, spot_centres in
                        centres_spot_cartesian.items()}

        return centres, centres_spot
