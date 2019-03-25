import numpy as np

import matplotlib.pyplot as plt
from copy import deepcopy
from elisa.engine import const, utils


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


def darkside_filter(sight_of_view: np.array, normals: np.array):
    """
    return indices for visible faces defined by given normals

    :param sight_of_view: np.array
    :param normals: np.array
    :return: np.array
    """
    # todo: resolve self shadowing in case of W UMa
    valid = np.array([idx for idx, normal in enumerate(normals)
                      if utils.cosine_similarity(sight_of_view, normal) > 0])
    return valid


def plane_projection(points, plane):
    """

    :param points:
    :param plane: str; (xy, yz, zx)
    :return:
    """
    # rm_index = {"xy": 2, "yz": 0, "zx": 1}[plane]
    # in_plane = np.array([l for i, l in enumerate(points.T) if i != rm_index])
    in_plane = np.array([points.T[1], points.T[2]])
    return in_plane.T


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
    def __init__(self, points, normals, indices):
        self.points = deepcopy(points)
        self.normals = deepcopy(normals)
        self.indices = deepcopy(indices)

    def serialize(self):
        return self.points, self.normals, self.indices

    def copy(self):
        return deepcopy(self)


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
        for d in self.data:
            yield d

    def do(self, pos):
        easy_sys = self.init_data.copy()
        easy_sys.setup_position(pos, self.inclination)
        easy_sys.rotate()
        return easy_sys

    def init_positions(self):
        self.data = (self.do(pos) for pos in self.motion)

    @property
    def init_data(self):
        return self._init_data

    @init_data.setter
    def init_data(self, args):
        self._init_data = SingleOrbitalPositionContainer(*args)


class SingleOrbitalPositionContainer(object):
    __COMPONENTS__ = ["_primary", "_secondary"]
    __PROPERTIES__ = ["points", "normals"]

    def __init__(self, primary, secondary):
        self._primary = None
        self._secondary = None
        self.position = None
        self.inclination = None

        setattr(self, 'primary', primary)
        setattr(self, 'secondary', secondary)

    def setup_position(self, position: PositionContainer, inclination: float):
        self.position = position
        self.inclination = inclination

    def copy(self):
        return deepcopy(self)

    @property
    def primary(self):
        return self._primary

    @primary.setter
    def primary(self, value):
        self._primary = EasyObject(
            value.points,
            value.normals,
            indices=None
        )

    @property
    def secondary(self):
        return self._secondary

    @secondary.setter
    def secondary(self, value):
        self._secondary = EasyObject(
            value.points,
            value.normals,
            indices=None
        )

    def set_indices(self, component, indices):
        attr = getattr(self, component)
        setattr(attr, 'indices', indices)

    def rotate(self):
        for component in self.__COMPONENTS__:
            easyobject_instance = getattr(self, component)
            for prop in self.__PROPERTIES__:
                prop_value = getattr(easyobject_instance, prop)

                args = (self.position.azimut - const.HALF_PI, prop_value, "z", False, False)
                prop_value = utils.axis_rotation(*args)

                # args = (const.PI - self.inclination, prop_value, "y", False, False)
                # prop_value = utils.axis_rotation(*args)
                setattr(easyobject_instance, prop, prop_value)


