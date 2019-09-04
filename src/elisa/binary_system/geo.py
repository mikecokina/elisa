import logging
import numpy as np

from copy import deepcopy
from elisa import utils, const
from pypex.poly2d.polygon import Polygon
from elisa.binary_system import utils as bsutils
from elisa import const
from elisa.binary_system import static
from collections.abc import Sequence
from matplotlib import pyplot as plt

from elisa.utils import is_empty

__logger__ = logging.getLogger(__name__)


def get_critical_inclination(binary, components_distance):
    """
    Get critical inclination for eclipses.

    :param binary: elisa.binary_system.system.BinarySystem
    :param components_distance: float
    :return: float
    """
    if binary.morphology != 'over-contact':
        radius1 = np.mean([binary.primary.side_radius, binary.primary.forward_radius, binary.primary.backward_radius,
                           binary.primary.polar_radius])
        radius2 = np.mean([binary.secondary.side_radius, binary.secondary.forward_radius,
                           binary.secondary.backward_radius, binary.secondary.polar_radius])
        cos_i_critical = (radius1 + radius2) / components_distance
        return np.degrees(np.arccos(cos_i_critical))


def get_eclipse_boundaries(binary, components_distance):
    """
    Calculates the ranges in orbital azimuths (for phase=0 -> azimuth=pi/2)!!!  where eclipses occur.

    :param binary: elisa.binary_system.system.BinarySystem
    :param components_distance: float
    :return: numpy.array;

    shape::

        [primary ecl_start, primary_ecl_stop, sec_ecl_start, sec_ecl_stop]
    """
    # check whether the inclination is high enough to enable eclipses
    if binary.morphology != 'over-contact':
        radius1 = np.mean([binary.primary.side_radius, binary.primary.forward_radius, binary.primary.backward_radius,
                           binary.primary.polar_radius])
        radius2 = np.mean([binary.secondary.side_radius, binary.secondary.forward_radius,
                           binary.secondary.backward_radius, binary.secondary.polar_radius])
        sin_i_critical = (radius1 + radius2) / components_distance
        sin_i = np.sin(binary.inclination)
        if sin_i < sin_i_critical:
            __logger__.debug('inclination is not sufficient to produce eclipses')
            return np.array([const.HALF_PI, const.HALF_PI, const.PI, const.PI])
        radius1 = binary.primary.forward_radius
        radius2 = binary.secondary.forward_radius
        sin_i_critical = 1.01 * (radius1 + radius2) / components_distance
        azimuth = np.arcsin(np.sqrt(np.power(sin_i_critical, 2) - np.power(np.cos(binary.inclination), 2)))
        azimuths = np.array([const.HALF_PI - azimuth, const.HALF_PI + azimuth, 1.5 * const.PI - azimuth,
                             1.5 * const.PI + azimuth]) % const.FULL_ARC
        return azimuths
    else:
        return np.array([0, const.PI, const.PI, const.FULL_ARC])


def darkside_filter(line_of_sight, normals):
    """
    Return indices for visible faces defined by given normals.
    Function assumes that `line_of_sight` ([1, 0, 0]) and `normals` are already normalized to one.

    :param line_of_sight: numpy.array
    :param normals: numpy.array
    :return: numpy.array
    """
    # todo: resolve self shadowing in case of W UMa
    # calculating normals utilizing the fact that normals and line of sight vector [1, 0, 0] are already normalized
    if (line_of_sight == np.array([1.0, 0.0, 0.0])).all():
        cosines = utils.calculate_cos_theta_los_x(normals=normals)
    else:
        cosines = utils.calculate_cos_theta(normals=normals, line_of_sight_vector=np.array([1, 0, 0]))
    # recovering indices of points on near-side (from the point of view of observer)
    return np.arange(np.shape(normals)[0])[cosines > 0]


def plane_projection(points, plane, keep_3d=False):
    """
    Function projects 3D points into given plane.

    :param keep_3d: bool; if True, the dimensions of the array is kept the same, with given column equal to zero
    :param points: numpy.array
    :param plane: str; ('xy', 'yz', 'zx')
    :return: numpy.array
    """
    rm_index = {"xy": 2, "yz": 0, "zx": 1}[plane]
    if not keep_3d:
        indices_to_keep = [0, 1, 2]
        del indices_to_keep[rm_index]
        return points[:, indices_to_keep]
    in_plane = deepcopy(points)
    in_plane[:, rm_index] = 0.0
    return in_plane


def calculate_spot_longitudes(binary_instance, phases, component=None):
    """
    function calculates the latitudes of every spot on given component(s) for every phase

    :param binary_instance: BinarySystem instance
    :param phases: numpy.array
    :param component: str; 'primary' or 'secondary', if None both will be calculated
    :return: Dict; {component: {spot_idx: np.array([....]), ...}, ...}
    """
    components_list = static.component_to_list(component)
    components = {comp: getattr(binary_instance, comp) for comp in components_list}
    spots_longitudes = {comp: {spot_index: (instance.synchronicity - 1) * phases * const.FULL_ARC + spot.longitude
                               for spot_index, spot in instance.spots.items()}
                        for comp, instance in components.items()}
    return spots_longitudes


def assign_spot_longitudes(binary_instance, spots_longitudes, index=None, component=None):
    """
    function assigns spot latitudes for each spot according to values in `spots_longitudes` in index `index`

    :param binary_instance: BinarySystem instance
    :param spots_longitudes: dict {component: {spot_idx: np.array([....]), ...}, ...}, takes output of function
    `calculate_spot_latitudes`
    :param index: index of spot longitude values to be used, if none is given, scalar values are expected in
    `spots_longitudes`
    :param component: 'primary' or 'secondary', if None both will be calculated
    :return:
    """
    components_list = static.component_to_list(component)
    components = {comp: getattr(binary_instance, comp) for comp in components_list}
    for comp, instance in components.items():
        for spot_index, spot in instance.spots.items():
            spot._longitude = spots_longitudes[comp][spot_index] if index is None else \
                spots_longitudes[comp][spot_index][index]


def surface_area_coverage(size, visible, visible_coverage, partial=None, partial_coverage=None):
    """
    Prepare array with coverage os surface areas.

    :param size: int; size of array
    :param visible: numpy.array; full visible areas (numpy fancy indexing), array like [False, True, True, False]
    :param visible_coverage: numpy.array; defines coverage of visible (coverage onTrue positions)
    :param partial: numpy.array; partial visible areas (numpy fancy indexing)
    :param partial_coverage: numpy.array; defines coverage of partial visible
    :return: numpy.array
    """
    # initialize zeros, since there is no input for invisible (it means everything what left after is invisible)
    coverage = np.zeros(size)
    coverage[visible] = visible_coverage
    if partial is not None:
        coverage[partial] = partial_coverage
    return coverage


def faces_to_pypex_poly(t_hulls):
    """
    Convert all faces defined as numpy.array to pypex Polygon class instance

    :param t_hulls: List[numpy.array]
    :return: List
    """
    return [Polygon(t_hull, _validity=False) for t_hull in t_hulls]


def pypex_poly_hull_intersection(pypex_faces_gen, pypex_hull: Polygon):
    """
    Resolve intersection of polygons defined in `pypex_faces_gen` with polyogn `pypex_hull`.

    :param pypex_faces_gen: List[pypex.poly2d.polygon.Plygon]
    :param pypex_hull: pypex.poly2d.polygon.Plygon
    :return: List[pypex.poly2d.polygon.Plygon]
    """
    return [pypex_hull.intersection(poly) for poly in pypex_faces_gen]


def pypex_poly_surface_area(pypex_polys_gen):
    """
    Compute surface areas of pypex.poly2d.polygon.Plygon's.

    :param pypex_polys_gen: List[pypex.poly2d.polygon.Plygon]
    :return: List[float]
    """
    return [poly.surface_area() if poly is not None else 0.0 for poly in pypex_polys_gen]


def hull_to_pypex_poly(hull):
    """
    Convert convex polygon defined by points in List or numpy.array to pypex.poly2d.polygon.Plygon.

    :param hull: List or numpy.array
    :return: pypex.poly2d.polygon.Plygon
    """
    return Polygon(hull, _validity=False)


def adjust_distance(points, old_distance, new_distance):
    points[:, 0] = points[:, 0] - old_distance + new_distance
    return points


class EasyObject(object):
    def __init__(self, points, normals, indices, faces=None, temperatures=None, log_g=None, coverage=None, rals=None,
                 face_centres=None, metallicity=None):
        """
        None default gives a capability to be used without such parameters

        :param points: numpy.array
        :param normals: numpy.array
        :param indices: List
        :param faces: numpy.array
        :param temperatures: numpy.array
        :param log_g: numpy.array
        :param coverage: numpy.array
        """
        self._points = deepcopy(points)
        self._normals = deepcopy(normals)
        self.indices = deepcopy(indices)
        self.coverage = deepcopy(coverage)
        self._faces = deepcopy(faces)
        self._log_g = deepcopy(log_g)
        self._temperatures = deepcopy(temperatures)
        self._rals = deepcopy(rals)
        self._face_centres = deepcopy(face_centres)
        self._metallicity = deepcopy(metallicity)

    def serialize(self):
        """
        Return all class properties at once.

        :return: Tuple
        """
        return self.points, self.normals, self.indices, self.faces, self.coverage, self.rals, self.face_centres

    def copy(self):
        """
        Copy self instance

        :return: self; copied self instance
        """
        return deepcopy(self)

    @property
    def points(self):
        """
        Return points of instance.

        :return: numpy.array
        """
        return self._points

    @points.setter
    def points(self, points):
        """
        Set points.

        :param points: numpy.array
        :return:
        """
        self._points = points

    @property
    def normals(self):
        """
        Get normals.

        :return: numpy.array
        """
        return self._normals

    @normals.setter
    def normals(self, normals):
        """
        Set normals.

        :param normals: numpy.array
        :return:
        """
        self._normals = normals

    @property
    def faces(self):
        """
        Get faces.

        :return: numpy.array
        """
        return self._faces

    @property
    def temperatures(self):
        """
        Get temperatures.

        :return: numpy.array
        """
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temperatures):
        """
        set temperatures
        :param temperatures: array
        :return:
        """
        self._temperatures = temperatures

    @property
    def log_g(self):
        """
        Get log_g

        :return: numpy.array
        """
        return self._log_g

    @property
    def rals(self):
        """
        Get renormalized associated Legendre polynomials (rALS)
        :return: array of complex arrays for each face
        """
        return self._rals

    @rals.setter
    def rals(self, rals):
        """
        Set renormalized associated Legendre polynomials (rALS)
        :param rals:
        :return:
        """
        self._rals = rals

    @property
    def face_centres(self):
        """
        Get face centres
        :return: array
        """
        return self._face_centres

    @face_centres.setter
    def face_centres(self, centres):
        """
        Set face centres
        :param centres: array
        :return:
        """
        self._face_centres = centres

    @property
    def metallicity(self):
        """
        Get star metallicity
        :return: array
        """
        return self._metallicity

    @metallicity.setter
    def metallicity(self, metallicity):
        """
        Set star metallicity
        :param metallicity: float
        :return:
        """
        self._metallicity = metallicity


class SystemOrbitalPosition(object):
    """
    Class instance will keep iterator rotated and darkside filtered orbital positions.
    """
    def __init__(self, primary, secondary, inclination, motion, ecl_boundaries=None):
        """
        :param primary: elisa.base.Star
        :param secondary: elisa.base.Star
        :param inclination: float
        :param motion: numpy.array
        :param ecl_boundaries: numpy.array
        """

        self.inclination = inclination
        self.motion = motion
        self.data = ()
        self._init_data = None
        self._coverage = None
        self._cosines = None

        args = (primary, secondary)
        setattr(self, 'init_data', args)
        self.init_positions()
        self._idx = 0
        self.in_eclipse = self.in_eclipse_test(ecl_boundaries)

    def __iter__(self):
        for single_position_container in self.data:
            yield single_position_container

    def do(self, pos):
        """
        On initial data (SingleOrbitalPositionContainer) created on the begining of init method,
        will apply::

            - `setup_position` method
            - `rotate` method

        :param pos: NamedTuple; elisa.const.BINARY_POSITION_PLACEHOLDER
        :return: SingleOrbitalPositionContainer
        """
        single_pos_sys = self.init_data.copy()
        single_pos_sys.setup_position(pos, self.inclination)
        single_pos_sys.rotate()
        return single_pos_sys

    def init_positions(self):
        """
        Initialise positions data. Call `do` method for each position.

        :return:
        """
        self.data = (self.do(pos) for pos in self.motion)

    @property
    def init_data(self):
        """
        Get init_data

        :return: SingleOrbitalPositionContainer
        """
        return self._init_data

    @init_data.setter
    def init_data(self, args):
        """
        Set init_data (create initial SingleOrbitalPositionContainer).

        :param args: Tuple[elisa.base.Star, elisa.base.Star]; (primary, secondary)
        :return:
        """
        self._init_data = SingleOrbitalPositionContainer(*args)

    def darkside_filter(self):
        """
        Rewrite `data` od self with applied darkside filter.

        :return: self
        """
        self.data = (single_position_container.darkside_filter() for single_position_container in self.data)
        return self

    def eclipse_filter(self):
        """
        Just placeholder. Maybe will be used in future.

        :return: self
        """
        self.data = ()
        return self

    @property
    def coverage(self):
        """
        returns visible area of each surface face

        :return: np.array
        """
        return self._coverage

    @coverage.setter
    def coverage(self, coverage):
        """
        setter for visible area of each surface face

        :param coverage: np.array
        :return:
        """
        self._coverage = coverage

    @property
    def cosines(self):
        """
        returns directional cosines for each surface face of both components with respect to line_of_sight
        :return: dict - {'primary': np.array, 'secondary': np.array}
        """

        return self._cosines

    @cosines.setter
    def cosines(self, value):
        """
        setter for storing directional cosines for each surface face of both components with respect to line_of_sight
        :param value: dict - {'primary': np.array, 'secondary': np.array}
        :return:
        """

        self._cosines = value

    def in_eclipse_test(self, ecl_boundaries):
        """
        Test whether in given phases eclipse occurs or not.

        :param ecl_boundaries: numpy.array
        :return: bool; numpy.array
        """

        if is_empty(ecl_boundaries):
            return np.ones(len(self.motion), dtype=bool)

        azimuths = [position.azimuth for position in self.motion]

        if ecl_boundaries[0] < 1.5 * const.PI:
            primary_ecl_test = np.logical_and((azimuths >= ecl_boundaries[0]), (azimuths <= ecl_boundaries[1]))
        else:
            primary_ecl_test = np.logical_or((azimuths >= ecl_boundaries[0]), (azimuths < ecl_boundaries[1]))

        if ecl_boundaries[2] > const.HALF_PI:
            if ecl_boundaries[3] > const.HALF_PI:
                secondary_ecl_test = np.logical_and((azimuths >= ecl_boundaries[2]), (azimuths <= ecl_boundaries[3]))
            else:
                secondary_ecl_test = np.logical_or((azimuths >= ecl_boundaries[2]), (azimuths <= ecl_boundaries[3]))
        else:
            secondary_ecl_test = np.logical_and((azimuths >= ecl_boundaries[2]), (azimuths <= ecl_boundaries[3]))

        return np.logical_or(primary_ecl_test, secondary_ecl_test)


class SingleOrbitalPositionContainer(object):
    """
    Keep parmetres of position on orbit (rotated points, normals, etc.)
    """
    __COMPONENTS__ = ["_primary", "_secondary"]
    __PROPERTIES__ = ["points", "normals"]

    def __init__(self, primary, secondary):
        """
        Initialize container

        :param primary: elisa.base.Star
        :param secondary: elisa.base.Star
        """
        _primary, _secondary = primary, secondary
        self._primary = None
        self._secondary = None
        self.position = None
        self.inclination = np.nan

        for component in self.__COMPONENTS__:
            setattr(self, component[1:], locals()[component])

    def setup_position(self, position: const.BINARY_POSITION_PLACEHOLDER, inclination):
        """
        Set geo attributes of current container.

        :param position: NamedTuple; elisa.const.BINARY_POSITION_PLACEHOLDER
        :param inclination: float
        :return:
        """
        self.position = position
        self.inclination = inclination

    def copy(self):
        """
        Deepcopy of self

        :return: self; copied self
        """
        return deepcopy(self)

    @staticmethod
    def get_flatten(component):
        """
        Get flatten parmetres og given `component` instance. (Flat points, normals, etc. of Star and Spots).


        :param component: elisa.base.Star
        :return: Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]

        ::

            Tuple(points, normals, faces, temperatures, log_g, rals, centres)
        """
        return bsutils.get_flaten_properties(component)

    @property
    def primary(self):
        """
        Get primary star.

        :return: EasyObject
        """
        return self._primary

    @primary.setter
    def primary(self, value):
        """
        Set primary paramter of self. Find flatten form of points, normalns, faces, etc. and create EasyObject.

        :param value: elisa.base.Star
        :return:
        """
        points, normals, faces, temp, log_g, rals, centres = self.get_flatten(value)
        self._primary = EasyObject(points, normals, None,
                                   faces=faces, temperatures=temp, log_g=log_g,
                                   rals=rals, face_centres=centres, metallicity=value.metallicity)

    @property
    def secondary(self):
        """
        Get parameter `secondary` of self.

        :return: EasyObject
        """
        return self._secondary

    @secondary.setter
    def secondary(self, value):
        """
        Set secondary paramter of self. Find flatten form of points, normalns, faces, etc. and create EasyObject.

        :param value: elisa.base.Star
        :return:
        """
        points, normals, faces, temp, log_g, rals, centres = self.get_flatten(value)
        self._secondary = EasyObject(points, normals, None,
                                     faces=faces, temperatures=temp, log_g=log_g,
                                     rals=rals, face_centres=centres, metallicity=value.metallicity)

    def set_indices(self, component, indices):
        """
        For given `component` set `indices`.

        :param component: `primary` or `secondary`; define EasyObject
        :param indices: numpy.array
        :return:
        """
        attr = getattr(self, component)
        setattr(attr, 'indices', indices)

    def set_coverage(self, component, coverage):
        """
        For given `component` set `indices`.

        :param component: :param component: `primary` or `secondary`; define EasyObject
        :param coverage: numpy.array
        :return:
        """
        attr = getattr(self, component)
        setattr(attr, 'coverage', coverage)

    def rotate(self):
        """
        Rotate quantities defined in cls.__PROPERTIES__ in case of components defined in cls.__PROPERTIES__.
        Rotation is made in orbital plane and inclination direction in respective order.
        Angle are defined in self.position and self.inclination.

        :return:
        """
        for component in self.__COMPONENTS__:
            easyobject_instance = getattr(self, component)
            for prop in self.__PROPERTIES__:
                prop_value = getattr(easyobject_instance, prop)

                args = (self.position.azimuth - const.HALF_PI, prop_value, "z", False, False)
                prop_value = utils.around_axis_rotation(*args)

                args = (const.HALF_PI - self.inclination, prop_value, "y", False, False)
                prop_value = utils.around_axis_rotation(*args)
                setattr(easyobject_instance, prop, prop_value)

    def darkside_filter(self):
        """
        Apply darkside filter on current position defined in container.
        Function iterates over components and assigns indices of visible points to EasyObject instance.

        :return: self
        """
        for component in self.__COMPONENTS__:
            easyobject_instance = getattr(self, component)
            normals = getattr(easyobject_instance, "normals")
            valid_indices = darkside_filter(line_of_sight=const.LINE_OF_SIGHT, normals=normals)
            self.set_indices(component=component, indices=valid_indices)
        return self

    def eclipse_filter(self):
        """
        Just placeholder.

        :return:
        """
        pass


class OrbitalSupplements(Sequence):
    """
    !!! BEWARE, THIS IS MUTABLE !!!


    """

    def __getitem__(self, index):
        return self.body[index], self.mirror[index]

    def __init__(self, body=None, mirror=None):
        if body is None and mirror is None:
            self._body = np.array([])
            self._mirror = np.array([])

        else:
            self._body = np.array(body)
            self._mirror = np.array(mirror)

    def append(self, body, mirror):
        self._body = np.vstack((self._body, body)) if not is_empty(self._body) else np.array([body])
        self._mirror = np.vstack((self._mirror, mirror)) if not is_empty(self._mirror) else np.array([mirror])

    @property
    def body(self):
        return self._body

    @property
    def mirror(self):
        return self._mirror

    @property
    def body_defined(self):
        return self.not_empty(self.body)

    @property
    def mirror_defined(self):
        return self.not_empty(self.mirror)

    @staticmethod
    def is_empty(val):
        return np.all(np.isnan(val))

    @classmethod
    def not_empty(cls, arr):
        """
        Return values where supplied array is not empty.

        :param arr: numpy.array
        :return: numpy.array
        """
        return arr[list(map(lambda x: not cls.is_empty(x), arr))]

    def sort(self, by='distance'):
        """
        Sort by given quantity.
        This method sorts bodies and mirrors based on quantity chosen on input.
        Sorting of mirrors is based on sorting of bodies.

        :param by: str
        :return: self
        """

        if by == 'index':
            by = 0
        elif by == 'distance' or by == 'radius':
            by = 1
        else:
            raise ValueError("Invalid value of `by`")

        sort_index = np.argsort(self.body[:, by])
        self._body = self.body[sort_index]
        self._mirror = self.mirror[sort_index]

        return self

    def size(self):
        return self.__len__()

    def to_orbital_position(self):
        pass

    def plot_bodies(self):
        self._plot(self.body_defined)

    def plot_mirrors(self):
        self._plot(self.mirror_defined, marker="x")

    def plot(self):
        self._plot(self.body_defined, self.mirror_defined)

    @classmethod
    def _plot(cls, arr1, arr2=None, marker="o"):

        x, y = utils.polar_to_cartesian(arr1[:, 1], arr1[:, 2] - (np.pi / 2))
        plt.scatter(x, y, marker=marker)

        if not is_empty(arr2):
            x, y = utils.polar_to_cartesian(arr2[:, 1], arr2[:, 2] - (np.pi / 2))
            plt.scatter(x, y, marker="x")

        plt.grid(True)
        plt.axes().set_aspect('equal')
        plt.show()

    def __iter__(self):
        for body, mirror in zip(self.body, self.mirror):
            yield body, mirror
        raise StopIteration

    def __len__(self):
        return len(self.body)

    def __eq__(self, other):
        return np.all(self._body == other.body) & \
               np.all((self.mirror == other.mirror)[~np.all(np.isnan(other.mirror) & np.isnan(self.mirror), axis=1)])

    def __str__(self):
        return f"{self.__class__.__name__}\nbodies: {self.body}\nmirrors: {self._mirror}"

    def __repr__(self):
        return self.__str__()
