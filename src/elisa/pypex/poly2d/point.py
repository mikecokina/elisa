from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from elisa.pypex.base.conf import ROUND_PRECISION

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from elisa.pypex.poly2d.polygon import Polygon

MIN_POLYGON_POINTS = 3


def _line_side(p1: ArrayLike, p2: ArrayLike, a: ArrayLike, b: ArrayLike) -> float:
    """Determine which side of a line two points lie on.

    Help determine whether both points of line p lie on the same side of line ab or not.

    :param p1: First point of line p
    :type p1: ArrayLike
    :param p2: Second point of line p
    :type p2: ArrayLike
    :param a: First point of line ab
    :type a: ArrayLike
    :param b: Second point of line ab
    :type b: ArrayLike
    :return: Cross product result indicating side orientation
    :rtype: float
    """
    p1, p2, a, b = np.array(p1), np.array(p2), np.array(a), np.array(b)
    cp1 = np.cross(b - a, p1 - a)
    cp2 = np.cross(b - a, p2 - a)
    return np.dot(cp1, cp2)


# /* Check whether p1 and p2 lie on the same side of line ab */
def same_side(p1: ArrayLike, p2: ArrayLike, a: ArrayLike, b: ArrayLike) -> bool:
    """Determine whether two points lie on the same side of a line.

    :param p1: First point of line p
    :type p1: ArrayLike
    :param p2: Second point of line p
    :type p2: ArrayLike
    :param a: First point of line ab
    :type a: ArrayLike
    :param b: Second point of line ab
    :type b: ArrayLike
    :return: True if points lie on the same side, False otherwise
    :rtype: bool
    """
    return _line_side(p1, p2, a, b) >= 0


def is_point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Test whether a point lies in a polygon.

    :param point: Point to test
    :type point: Point
    :param polygon: Polygon to test against
    :type polygon: Polygon
    :return: True if point is inside polygon, False otherwise
    :rtype: bool
    """
    if len(polygon) < MIN_POLYGON_POINTS:
        msg = "invalid polygon shape, expected at least 3 corners polygon"
        raise ValueError(msg)
    return polygon.mplpath.contains_point(point.to_array())


class _Point:
    """Internal point class for deduplication with tolerance."""

    def __init__(self, i: int, x: float, y: float) -> None:
        """Initialize an internal point.

        :param i: Index of the point
        :type i: int
        :param x: X coordinate
        :type x: float
        :param y: Y coordinate
        :type y: float
        """
        self.i = i
        self.x = x
        self.y = y

    def __key(self) -> tuple[float, float]:
        """Return the key for hashing.

        :return: Tuple of x and y coordinates
        :rtype: tuple[float, float]
        """
        return self.x, self.y

    def __hash__(self) -> int:
        """Return the hash of the point.

        :return: Hash value
        :rtype: int
        """
        return hash(self.__key())

    def __eq__(self, other: _Point | None) -> bool:
        """Check equality with another _Point.

        :param other: Another point
        :type other: _Point | None
        :return: True if equal, False otherwise
        :rtype: bool
        """
        if not isinstance(other, _Point):
            return False
        return (self.x == other.x) and (self.y == other.y)


class Point:
    """2D Point class."""

    def __init__(self, x: float, y: float) -> None:
        """Initialize a 2D point.

        :param x: X coordinate
        :type x: float
        :param y: Y coordinate
        :type y: float
        """
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """Return string representation of the point.

        :return: String representation
        :rtype: str
        """
        return f"Point [{self.x}, {self.y}]"

    def __repr__(self) -> str:
        """Return string representation of the point.

        :return: String representation
        :rtype: str
        """
        return f"Point [{self.x}, {self.y}]"

    def __hash__(self) -> int:
        """Return the hash of the point.

        :return: Hash value
        :rtype: int
        """
        return hash((self.x, self.y))

    def __eq__(self, other: Point | None) -> bool:
        """Check equality with another Point.

        :param other: Another point.
        :type other: Point | None
        :return: True if equal, False otherwise
        :rtype: bool
        """
        if not isinstance(other, Point):
            return False
        return (self.x == other.x) and (self.y == other.y)

    def __add__(self, other: Point) -> Point:
        """Add two points.

        :param other: Another point to add
        :type other: Point
        :return: New point with summed coordinates
        :rtype: Point
        """
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        """Subtract two points.

        :param other: Another point to subtract
        :type other: Point
        :return: New point with difference of coordinates
        :rtype: Point
        """
        return Point(self.x - other.x, self.y - other.y)

    @staticmethod
    def set(points: np.ndarray | list[Point], round_tol: int = ROUND_PRECISION) -> np.ndarray:
        """Deduplicate points with tolerance.

        Naive implementation of `set` like function in python.
        This method relies on tolerance. Points are the same up to supplied tolerance.

        :param points: Array or list of Point instances
        :type points: numpy.ndarray | list[Point]
        :param round_tol: Rounding precision tolerance
        :type round_tol: int
        :return: Array of unique Point instances
        :rtype: numpy.ndarray
        """
        points_array = np.asarray(points)
        _points = [_Point(i, round(cast("Point", point).x, round_tol), round(cast("Point", point).y, round_tol))
                   for i, point in enumerate(points_array)]
        indices = [_point.i for _point in set(_points)]
        return points_array[indices]

    def is_inside_polygon(self, polygon: Polygon) -> bool:
        """Test whether this point is inside a polygon.

        :param polygon: Polygon to test against
        :type polygon: Polygon
        :return: True if point is inside polygon, False otherwise
        :rtype: bool
        """
        return is_point_in_polygon(self, polygon)

    def to_list(self) -> list[float]:
        """Convert point to list.

        :return: List of x and y coordinates
        :rtype: list[float]
        """
        return [self.x, self.y]

    def to_array(self) -> np.ndarray:
        """Convert point to numpy array.

        :return: Array of x and y coordinates
        :rtype: numpy.ndarray
        """
        return np.array(self.to_list())
