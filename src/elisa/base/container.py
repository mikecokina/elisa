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
from . surface.mesh import symmetry_point_reduction
from . surface.faces import mirror_face_values, symmetry_face_reduction

logger = getLogger("base.container")


class PropertiesContainer(object):
    """
    General object for storing attributes of modelled objects.
    """
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
    """
    Object for handling attributes of Star objects.
    """
    pass


class SystemPropertiesContainer(PropertiesContainer):
    """
    Object for handling attributes of System objects.
    """
    pass


class PositionContainer(object):
    """
    General object for handling system models at a given time or photometric phase.
    """
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

    def is_flat(self):
        return self._flatten

    def flat_it(self):
        """
        Function merges surface attributes for each system component which were previously assigned to
        component surface or spot attributes. Resulting attributes such as points, faces, temperatures. etc.
        are then stored in unified arrays as component attributes.

        :return: PositionContainer; flattened container
        """
        # naive implementation of idempotency
        if self._flatten:
            return self

        for component in self._components:
            star_container = getattr(self, component)
            if star_container.has_spots() or star_container.has_pulsations():
                star_container.flat_it()

        self._flatten = True
        return self

    def apply_rotation(self):
        """
        Rotate quantities defined in __PROPERTIES_TO_ROTATE__.
        Rotation is made in orbital plane (around z-axis) and inclination (y-axis) direction in respective order.
        Angle are defined in self.position and self.inclination.

        :return: elisa.base.PositionContainer;
        """
        __PROPERTIES_TO_ROTATE__ = ["points", "normals", "velocities", "face_centres"]

        for component in self._components:
            star_container = getattr(self, component)
            for prop in __PROPERTIES_TO_ROTATE__:
                self.rotate_property(star_container, prop)
        return self

    def rotate_property(self, container, prop):
        """
        Rotating property of StarContainer from co-rotating frame to observer's frame of reference.

        :param container: base.StarContainer;
        :param prop: str; name of the property (e.g. 'points')
        :return:
        """
        prop_value = getattr(container, prop)
        prop_value = utils.rotate_item(prop_value, self.position, self.inclination)
        setattr(container, prop, prop_value)

    def add_secular_velocity(self):
        """
        Addition of secular radial velocity of centre of mass to convert velocities to reference frame of observer
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
        Calculates angles between normals and line_of_sight vector for all components of the system.

        :param line_of_sight: np.array;
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
        return utils.calculate_cos_theta_los_x(normals=normals) if (line_of_sight == const.LINE_OF_SIGHT).all() \
            else utils.calculate_cos_theta(normals=normals, line_of_sight_vector=line_of_sight)

    @staticmethod
    def darkside_filter(cosines):
        """
        Return indices for visible faces defined by given normals.
        Function assumes that `line_of_sight` ([1, 0, 0]) and `normals` are already normalized to one.

        :param cosines: numpy.array;
        :return: numpy.array;
        """
        # todo: resolve self shadowing in case of W UMa, but probably not here
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
    Container carrying non-static properties of Star object (properties which vary from phase to phase) and also
    all properties set during creation of the parent BinarySystem.
    Method `from_properties_container` or `from_star_instance` has to be used to create container properly.
    Following parameters are gathered from Star object

    'mass', 't_eff', 'synchronicity', 'albedo', 'discretization_factor', 'name', 'spots'
    'polar_radius', 'equatorial_radius', 'gravity_darkening', 'surface_potential', 'atmosphere',
    'pulsations', 'metallicity', 'polar_log_g', 'critical_surface_potential', 'side_radius'

    If you are experienced user, you can create instance directly by calling

    >>> from elisa.base.container import StarContainer
    >>> kwargs = {}
    >>> inst = StarContainer(**kwargs)

    and setup static parameter latter of not at all if not necessary for further use. However, the StarContainer
    instance is intialized within a given PositionContainer during its initialization.

    Bellow are optional input parameters of StarContainer.
    kwargs: Dict;
    :**kwargs options**:

        * **points** * -- numpy.array;
        * **normals** * -- numpy.array;
        * **indices** * -- numpy.array;
        * **faces** * -- numpy.array;
        * **temperatures** * -- numpy.array;
        * **log_g** * -- numpy.array;
        * **coverage** * -- numpy.array;
        * **face_centres** * -- numpy.array;
        * **metallicity** * -- float;
        * **areas** * -- numpy.array;
        * **potential_gradient_magnitudes** * -- numpy.array;
        * **ld_cfs** * -- numpy.array;
        * **normal_radiance** * -- numpy.array;
        * **los_cosines** * -- numpy.array;


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

    :coverage: numpy.array; surface area of each triangle visible to the observer
    :indices: numpy.array; Indices of visible faces when darkside_filter is applied.
    :face_centres: numpy.array; row-wise geometrical centres of each triangle
    :metallicity: float;
    :areas: numpy.array; surface areas corresponing to each surface triangle
    :potential_gradient_magnitudes: numpy.array;
    :inverse_point_symmetry_matrix: numpy.array; row-wise, set of vectors that map base symmetry quadrant points
                                                 (octant in case of single star) to all others quadrants, each row
                                                 contains indices of points in given quadrant which order corresponds to
                                                 the order of points in the 1st quadrant (octant)
    :base_symmetry_points_number: int; number of first n surface points stored in StarContainer.points that are located
                                       on a symmetrical part of the surface. E.g.:

                                        ::

                                            StarContainer.points[:StarContainer.base_symmetry_points_number]

                                       will select surface points from one quarter (eighth) of the star in case of
                                       binary (single) star system
    :base_symmetry_faces_number: int; number of first n triangles stored in StarContainer.faces located in the 1st
                                      quadrant (octant). E.g.:

                                       ::

                                            StarContainer.temperatures[:StarContainer.base_symmetry_faces_number]

                                      will return surface temperatures only from triangles int the 1st quadrant
                                      (octant) in case of binary (single) star system.
    :face_symmetry_vector: numpy.array; array that maps each surface triangle to the symmetrical surface triangle
                                        located in the 1st quadrant. Array contains indices between 0 and
                                        `base_symmetry_faces_number`.
    :base_symmetry_points: numpy.array;
    :base_symmetry_faces: numpy.array;
    :polar_potential_gradient_magnitude: numpy.array;
    :ld_cfs: numpy.array;
    :normal_radiance: numpy.array;
    :los_cosines: numpy.array;
    """

    def __init__(self,
                 points=None,
                 normals=None,
                 velocities=None,
                 accelerations=None,
                 indices=None,
                 faces=None,
                 temperatures=None,
                 log_g=None,
                 coverage=None,
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
        self.accelerations = accelerations
        self.temperatures = temperatures
        self.log_g = log_g
        self.coverage = coverage
        self.indices = indices
        self.face_centres = face_centres
        self.metallicity = metallicity
        self.areas = areas
        self.potential_gradient_magnitudes = potential_gradient_magnitudes
        self.ld_cfs = ld_cfs
        self.normal_radiance = normal_radiance
        self.los_cosines = los_cosines
        self.points_spherical = np.array([])
        self.com = np.array([])

        self.inverse_point_symmetry_matrix = np.array([])
        self.base_symmetry_points_number = 0

        self.face_symmetry_vector = np.array([])
        self.base_symmetry_faces_number = 0

        # aux variables for treating singularities at poles
        self.pole_idx = np.array([])
        self.pole_idx_neighbour = np.array([])

        # those are used only if case of spots are NOT used ------------------------------------------------------------
        self.base_symmetry_points = np.array([])
        self.base_symmetry_faces = np.array([])
        self.azimuth_args = np.array([])
        # --------------------------------------------------------------------------------------------------------------

        self.spots = dict()
        self.pulsations = dict()
        self.polar_potential_gradient_magnitude = np.nan

        # all star radii in any position (set on fly) ------------------------------------------------------------------
        # set only via `assign_radii()` method
        self.polar_radius = None
        self.forward_radius = None
        self.side_radius = None
        self.backward_radius = None
        self.equatorial_radius = None
        self.equivalent_radius = None
        # --------------------------------------------------------------------------------------------------------------

        self._flatten = False

    @classmethod
    def from_star_instance(cls, star):
        """
        Initialize StarContainer from Star instance.

        :param star: elisa.base.star.Star;
        :return: StarContainer;
        """
        return cls.from_properties_container(star.to_properties_container())

    @classmethod
    def from_properties_container(cls, properties_container):
        """
        Create StarContainer from properties container.

        :param properties_container: elisa.base.container.StarPropertiesContainer;
        :return: elisa.base.container.StarContainer;
        """
        container = cls()
        container.__dict__.update(properties_container.__dict__)
        return container

    def has_spots(self):
        """
        True if StarContainer contains spots.

        :return: bool;
        """
        return len(self.spots) > 0

    def has_pulsations(self):
        """
        True if StarContainer contains pulsation modes.

        :return: bool;
        """
        return len(self.pulsations) > 0

    def symmetry_test(self):
        """
        True if stellar surface can utilize surface symmetry during the model calculation.

        :return: bool;
        """
        return not self.has_spots() and not self.has_pulsations()

    def is_flat(self):
        """
        Returns True if StarContainer is flattened.

        :return: bool;
        """
        return self._flatten

    def copy(self):
        """
        Return deepcopy of StarContainer instance.

        :return: elisa.base.container.StarContainer;
        """
        return deepcopy(self)

    def remove_spot(self, spot_index: int):
        """
        Remove n-th spot index of object.

        :param spot_index: int
        """
        del (self.spots[spot_index])

    def get_flatten_points_map(self):
        """
        Function returns all surface point and faces with corresponding map of vertices.

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
        Returns areas of each face of the star surface. (spots not included if StarContainer is not flattened)

        :return: numpy.array:

        ::

            numpy.array([area_1, ..., area_n])
        """
        if len(self.faces) == 0 or len(self.points) == 0:
            raise ValueError('Faces or/and points of object {self.name} have not been set yet.\n'
                             'Run build method first.')
        if self.symmetry_test():
            base_areas = utils.triangle_areas(self.symmetry_faces(self.faces), self.symmetry_points())
            return self.mirror_face_values(base_areas)
        return utils.triangle_areas(self.faces, self.points)

    def calculate_all_areas(self):
        """
        Calculates areas for all faces on the surface including spots and assigns values to its corresponding variables.
        """
        self.areas = self.calculate_areas()
        if self.has_spots() and not self.is_flat():
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

    def get_flatten_parameter(self, prop):
        """
        Returns flattened property of the container.

        :param prop: str; property identifier (e.g. 'points')
        :return: numpy.array; flattened property
        """
        list_to_concat = [getattr(self, prop), ]
        if not self.has_spots() or self._flatten:
            return list_to_concat[0]

        list_to_concat += [getattr(spot, prop) for spot in self.spots.values()]
        if prop == 'faces':
            lengths = np.cumsum([np.max(item) + 1 for item in list_to_concat])
            list_to_concat[1:] = [list_to_concat[index+1] + lengths[index] for index in self.spots]
        return np.concatenate(list_to_concat, axis=0)

    def flat_it(self):
        """
        Make properties "points", "normals", "faces", "temperatures", "log_g", "centers", "areas"
        of container flat. It means all properties of start and spots are put together.

        :return: self
        """
        # naive implementation of idempotency
        if self._flatten:
            return self

        props_list = ["points", "normals", "faces", "temperatures", "log_g", "face_centres", "areas", "velocities"]

        for prop in props_list:
            setattr(self, prop, self.get_flatten_parameter(prop))

        self._flatten = True
        return self

    def transform_points_to_spherical_coordinates(self, kind='points', com_x=0):
        """
        Transforming container cartesian variable to spherical coordinates in a frame of reference of given object.

        :param kind: str; `points` or `face_centres` or other variable containing cartesian
                          points in both star and spot containers
        :param com_x: float;
        :return: numpy.array; spherical coordinates of star variable
        """
        # separating variables to convert
        centres_cartesian = copy(getattr(self, kind))

        # transforming variables
        centres_cartesian[:, 0] -= com_x

        # conversion
        centres = utils.cartesian_to_spherical(centres_cartesian)
        return centres

    def assign_radii(self, radii):
        """
        Function automatically assigns radii stored in radii dict.

        :param radii: Dict; {radius_type: value, ...}
        """
        for key, value in radii.items():
            setattr(self, key, value)

    def symmetry_points(self):
        """
        Returns surface points from the symmetry base part of the star.

        :return: numpy.array;
        """
        # pulsations are not yet introduced during the creation of point cloud
        if self.has_spots():
            raise ValueError('Surface symmetry is not applicable in this case')
        return symmetry_point_reduction(self.points, self.base_symmetry_points_number)

    def mirror_face_values(self, values):
        """
        Mirroring the array values on symmetrical faces to the rest of the surface.

        :param values: numpy.array; surface parameter values corresponding to the base symmetry part of the surface
        :return: numpy.array; remapped array
        """
        if not self.symmetry_test():
            raise ValueError('Surface symmetry is not applicable in this case.')
        return mirror_face_values(values, self.face_symmetry_vector)

    def symmetry_faces(self, values):
        """
        Reducing the surface distribution of surface `values` array to a symmetrical component.

        :param values: numpy.array; surface parameter distribution
        :return: numpy.array; reduced surface distribution array
        """
        if not self.symmetry_test():
            raise ValueError('Surface symmetry is not applicable in this case.')
        return symmetry_face_reduction(values, self.base_symmetry_faces_number)
