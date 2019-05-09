import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from elisa.engine import const, utils
from pypex.poly2d.polygon import Polygon
from elisa.engine.binary_system import utils as bsutils


def get_critical_inclination(binary, components_distance: float):
    if binary.morphology != 'over-contact':
        radius1 = np.mean([binary.primary.side_radius, binary.primary.forward_radius, binary.primary.backward_radius,
                           binary.primary.polar_radius])
        radius2 = np.mean([binary.secondary.side_radius, binary.secondary.forward_radius,
                           binary.secondary.backward_radius, binary.secondary.polar_radius])
        cos_i_critical = (radius1 + radius2) / components_distance
        return np.degrees(np.arccos(cos_i_critical))


def get_eclipse_boundaries(binary, components_distance: float):
    # check whether the inclination is high enough to enable eclipses
    if binary.morphology != 'over-contact':
        radius1 = np.mean([binary.primary.side_radius, binary.primary.forward_radius, binary.primary.backward_radius,
                           binary.primary.polar_radius])
        radius2 = np.mean([binary.secondary.side_radius, binary.secondary.forward_radius,
                           binary.secondary.backward_radius, binary.secondary.polar_radius])
        sin_i_critical = (radius1 + radius2) / components_distance
        sin_i = np.sin(binary.inclination)
        if sin_i < sin_i_critical:
            binary._logger.debug('Inclination is not sufficient to produce eclipses.')
            return None
        radius1 = binary.primary.forward_radius
        radius2 = binary.secondary.forward_radius
        sin_i_critical = (radius1 + radius2) / components_distance
        azimuth = np.arcsin(np.sqrt(np.power(sin_i_critical, 2) - np.power(np.cos(binary.inclination), 2)))
        azimuths = np.array([const.FULL_ARC - azimuth, azimuth, const.PI - azimuth, const.PI + azimuth])
        return azimuths
    else:
        return np.array([const.FULL_ARC, 0.0, 0.0, const.FULL_ARC])


def darkside_filter(line_of_sight: np.array, normals: np.array):
    """
    return indices for visible faces defined by given normals, function assumes that `line_of_sight` ([1, 0, 0]) and
    `normals` are already normalized to one

    :param line_of_sight: np.array
    :param normals: np.array
    :return: np.array
    """
    # todo: resolve self shadowing in case of W UMa
    # calculating normals utilizing the fact that normals and line of sight vector [1, 0, 0] are already normalized
    # hovnokod...
    if (line_of_sight == np.array([1.0, 0.0, 0.0])).all():
        cosines = utils.calculate_cos_theta_los_x(normals=normals)
    else:
        cosines = utils.calculate_cos_theta(normals=normals, line_of_sight_vector=np.array([1, 0, 0]))

    # recovering indices of points on near-side (from the point of view of observer)
    return np.arange(np.shape(normals)[0])[cosines > 0]


def plane_projection(points, plane, keep_3d=False):
    """
    function projects 3D points into given plane

    :param keep_3d: if True, the dimensions of the array is kept the same, with given column equal to zero
    :param points:
    :param plane: str; ('xy', 'yz', 'zx')
    :return:
    """
    rm_index = {"xy": 2, "yz": 0, "zx": 1}[plane]
    if not keep_3d:
        indices_to_keep = [0, 1, 2]
        del indices_to_keep[0]
        return points[:, indices_to_keep]
    in_plane = deepcopy(points)
    in_plane[:, rm_index] = 0.0
    return in_plane


# fixme: this shouldn't be here, move it to the graphics
def to_png(x=None, y=None, x_label="y", y_label="z", c=None, fpath=None):
    plt.clf()
    ax = plt.scatter(x, y, marker="o", c=c, s=1)
    plt.xlabel(x_label)
    plt.xlabel(y_label)
    plt.axis("equal")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.savefig(fpath)

    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    #
    # clr = 'b'
    # pts = prop_value
    # ax.scatter(
    #     prop_value.T[0],
    #     prop_value.T[1],
    #     prop_value.T[2]
    #
    # )
    # plt.show()


class EasyObject(object):
    # fixme: why the hell is faces, temperatures and log_g == None???
    def __init__(self, points, normals, indices, faces=None, temperatures=None, log_g=None, coverage=None):
        self._points = deepcopy(points)
        self._normals = deepcopy(normals)
        self.indices = deepcopy(indices)
        self.coverage = deepcopy(coverage)
        self._faces = deepcopy(faces)
        self._log_g = deepcopy(log_g)
        self._temperatures = deepcopy(temperatures)

    def serialize(self):
        return self.points, self.normals, self.indices, self.faces, self.coverage

    def copy(self):
        return deepcopy(self)

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = points

    @property
    def normals(self):
        # if self.indices is not None:
        #     return self._normals[self.indices]
        return self._normals

    @normals.setter
    def normals(self, normals):
        self._normals = normals

    @property
    def faces(self):
        return self._faces

    @property
    def temperatures(self):
        return self._temperatures

    @property
    def log_g(self):
        return self._log_g


class PositionContainer(object):
    def __init__(self, idx, distance, azimut, true_anomaly, phase):
        self.position_index = idx
        self.azimut = azimut
        self.true_anomaly = true_anomaly
        self.phase = phase
        self.distance = distance


class SystemOrbitalPosition(object):
    def __init__(self, primary, secondary, inclination, motion):
        self.inclination = inclination
        self.motion = [PositionContainer(*pos) for pos in motion]
        self.data = ()
        self._init_data = None

        args = (primary, secondary)
        setattr(self, 'init_data', args)
        self.init_positions()
        self._idx = 0

    def __iter__(self):
        for single_position_container in self.data:
            yield single_position_container

    def do(self, pos):
        single_pos_sys = self.init_data.copy()
        single_pos_sys.setup_position(pos, self.inclination)
        single_pos_sys.rotate()
        return single_pos_sys

    def init_positions(self):
        self.data = (self.do(pos) for pos in self.motion)

    @property
    def init_data(self):
        return self._init_data

    @init_data.setter
    def init_data(self, args):
        self._init_data = SingleOrbitalPositionContainer(*args)

    def darkside_filter(self):
        self.data = (single_position_container.darkside_filter() for single_position_container in self.data)
        return self

    def eclipse_filter(self):
        self.data = ()
        return self


class SingleOrbitalPositionContainer(object):
    __COMPONENTS__ = ["_primary", "_secondary"]
    __PROPERTIES__ = ["points", "normals"]

    def __init__(self, primary, secondary):
        _primary, _secondary = primary, secondary
        self._primary = None
        self._secondary = None
        self.position = None
        self.inclination = None

        for component in self.__COMPONENTS__:
            setattr(self, component[1:], locals()[component])

    def setup_position(self, position: PositionContainer, inclination: float):
        self.position = position
        self.inclination = inclination

    def copy(self):
        return deepcopy(self)

    @property
    def primary(self):
        return self._primary

    @staticmethod
    def setup_component(component):
        return bsutils.get_flaten_properties(component)

    @primary.setter
    def primary(self, value):
        points, normals, faces, temp, log_g = self.setup_component(value)
        self._primary = EasyObject(points, normals, None, faces, temp, log_g)

    @property
    def secondary(self):
        return self._secondary

    @secondary.setter
    def secondary(self, value):
        points, normals, faces, temp, log_g = self.setup_component(value)
        self._secondary = EasyObject(points, normals, None, faces, temp, log_g)

    def set_indices(self, component, indices):
        attr = getattr(self, component)
        setattr(attr, 'indices', indices)

    def set_coverage(self, component, coverage):
        attr = getattr(self, component)
        setattr(attr, 'coverage', coverage)

    def rotate(self):
        """
        what is this?
        :return:
        """
        for component in self.__COMPONENTS__:
            easyobject_instance = getattr(self, component)
            for prop in self.__PROPERTIES__:
                prop_value = getattr(easyobject_instance, prop)

                args = (self.position.azimut - const.HALF_PI, prop_value, "z", False, False)
                prop_value = utils.axis_rotation(*args)

                args = (const.HALF_PI - self.inclination, prop_value, "y", False, False)
                prop_value = utils.axis_rotation(*args)
                setattr(easyobject_instance, prop, prop_value)

    def darkside_filter(self):
        """
        function iterates over components and assigns indices of visible points to easyobject instance
        :return:
        """
        for component in self.__COMPONENTS__:
            easyobject_instance = getattr(self, component)
            normals = getattr(easyobject_instance, "normals")
            valid_indices = darkside_filter(line_of_sight=const.LINE_OF_SIGHT, normals=normals)
            self.set_indices(component=component, indices=valid_indices)
        return self

    def eclipse_filter(self):
        """
        currently defined directly in lightcurve computaion function
        :return:
        """
        pass


def surface_area_coverage(size, visible, visible_coverage, partial=None, partial_coverage=None):
    coverage = np.zeros(size)
    coverage[visible] = visible_coverage
    if partial is not None:
        coverage[partial] = partial_coverage
    return coverage


def faces_to_pypex_poly(t_hulls):
    return (Polygon(t_hull) for t_hull in t_hulls)


def pypex_poly_hull_intersection(pypex_faces_gen, pypex_hull: Polygon):
    return (pypex_hull.intersection(poly) for poly in pypex_faces_gen)


def pypex_poly_surface_area(pypex_polys_gen):
    return (poly.surface_area() for poly in pypex_polys_gen)


def hull_to_pypex_poly(hull):
    return Polygon(hull)
