import numpy as np

from pypex.base.conf import ROUND_PRECISION


def _line_side(p1, p2, a, b):
    """
    Idea is described in ``
    Help to determine, whether both point of line p lie on the same side of line ab or not.

    :param p1: numpy.array; firs point of line p
    :param p2: numpy.array; second point of line p
    :param a: numpy.array; first point of line p'
    :param b: numpy.array; second point of line p'
    :return: float
    """
    p1, p2, a, b = np.array(p1), np.array(p2), np.array(a), np.array(b)
    cp1 = np.cross(b - a, p1 - a)
    cp2 = np.cross(b - a, p2 - a)
    return np.dot(cp1, cp2)


# /* Check whether p1 and p2 lie on the same side of line ab */
# todo: reconsider to use edge definition instead of single point in arguments
def same_side(p1, p2, a, b):
    """
    Idea is described in ``
    Determine, whether both point of line p lie on the same side of line ab or not.

    :param p1: numpy.array; firs point of line p
    :param p2: numpy.array; second point of line p
    :param a: numpy.array; first point of line p'
    :param b: numpy.array; second point of line p'
    :return: float
    """
    # todo: add possibility to deside whther on edge point is inside or outside
    return True if _line_side(p1, p2, a, b) >= 0 else False


def is_point_in_polygon(point, polygon):
    """
    test wether point lie in polygon

    :param point: pypex.poly2d.point.Point
    :param polygon: pypex.poly2d.polygon.Polygon
    :return:
    """
    if len(polygon) < 3:
        raise ValueError("invalid polygon shape, expected at least 3 corners polygon")
    return polygon.mplpath.contains_point(point.to_array())


class _Point(object):
    def __init__(self, i, x, y):
        self.i = i
        self.x = x
        self.y = y

    def __key(self):
        return self.x, self.y

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return (other.x == self.x) & (other.y == self.y)


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "Point [{}, {}]".format(self.x, self.y)

    def __repr__(self):
        return "Point [{}, {}]".format(self.x, self.y)

    def __eq__(self, other):
        return (other.x == self.x) & (other.y == self.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    @staticmethod
    def set(points, round_tol=ROUND_PRECISION):
        """
        Naive implementaion `set` like function in python.
        This method relly on tolerance. Points are same up to supplied tolerance.

        :param points: pypex.poly2d.point.Point
        :param round_tol: int
        :return: numpy.arra; list of Point instances
        """
        _points = [_Point(i, round(point.x, round_tol), round(point.y, round_tol)) for i, point in enumerate(points)]
        indices = [_point.i for _point in set(_points)]
        return np.array(points)[indices]

    def is_inside_polygon(self, polygon):
        """
        Test whether current (`self`) Point is inside or outside of polygon
        :param polygon: pypex.poly2d.point.Point
        :return: bool
        """
        return is_point_in_polygon(self, polygon)

    def to_list(self):
        """
        convert Point to list
        :return: list
        """
        return [self.x, self.y]

    def to_array(self):
        """
        Convert Point to numpy.array
        :return: numpy.array
        """
        return np.array(self.to_list())
