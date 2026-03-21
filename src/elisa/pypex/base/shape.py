from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from elisa.pypex.poly2d.point import Point

if TYPE_CHECKING:
    from elisa.base.types import INT
    from elisa.pypex.base.types import Shape2DType


class Shape2D(metaclass=ABCMeta):
    """Convex 2D Shape."""

    def __init__(self, hull: Shape2DType, *, _validity: bool = True) -> None:
        """Initialize a convex 2D shape.

        :param hull: Iterable of points defining the convex hull
        :type hull: Point2DType
        :param _validity: Whether to check polygon validity
        :type _validity: bool
        """
        hull = self.normalize_hull(hull)
        if _validity:
            self.polygon_validity_check(hull=hull, _raise=True)
        self._hull = np.array(hull)
        self.xi, self.yi = 0, 1

    def __hash__(self) -> int:
        """Return the hash of the shape based on its hull.

        :return: Hash value
        :rtype: int
        """
        return hash(tuple(tuple(point) for point in self.hull))

    def __len__(self) -> int:
        """Return the number of points in the hull.

        :return: Number of points
        :rtype: int
        """
        return len(self.hull)

    def __eq__(self, other: Shape2D | None) -> bool:
        """Check equality with another Shape2D.

        Two shapes are equal if:
          - They are both Shape2D instances
          - They have the same number of vertices
          - Corresponding hull points are equal

        :param other: Another shape
        :type other: Shape2D | None
        :return: True if equal, False otherwise
        :rtype: bool
        """
        if not isinstance(other, Shape2D):
            return False

        if len(self) != len(other):
            return False

        return all(Point(*self.hull[i]) == Point(*other.hull[i]) for i in range(len(self)))

    def __str__(self) -> str:
        """Return string representation of the shape.

        :return: String representation
        :rtype: str
        """
        return "Poly ({}): [{}]".format(len(self.hull), ", ".join([str(v) for v in self.hull]))

    def __repr__(self) -> str:
        """Return string representation of the shape.

        :return: String representation
        :rtype: str
        """
        return "Poly ({}): [{}]".format(len(self.hull), ", ".join([str(v) for v in self.hull]))

    @staticmethod
    def normalize_hull(hull: Shape2DType) -> np.ndarray:
        """Convert definition of points to normal form (to numpy.ndarray).

        :param hull: Iterable of points
        :type hull: Point2DType
        :return: Numpy array of points
        :rtype: numpy.ndarray
        """
        return np.array([vertex.to_array() if isinstance(vertex, Point) else vertex for vertex in hull])

    @property
    def hull(self) -> np.ndarray:
        """Get the hull as a numpy array.

        :return: Hull points
        :rtype: numpy.ndarray
        """
        return self._hull

    @hull.setter
    def hull(self, hull: Shape2DType) -> None:
        """Set the hull from a list of points.

        :param hull: Iterable of points
        :type hull: Point2DType
        """
        self._hull = np.array(hull)

    @abstractmethod
    def intersects(self, shape: Shape2D, **kwargs) -> bool | tuple:
        """Check if this shape intersects with another shape.

        :param shape: Another shape
        :type shape: Shape2D
        :return: True/False or intersection details
        :rtype: bool | tuple
        """
        ...

    @abstractmethod
    def intersection(self, shape: Shape2D) -> Point | None:
        """Compute the intersection point with another shape.

        :param shape: Another shape
        :type shape: Shape2D
        :return: Intersection point or None
        :rtype: Point | None
        """
        ...

    def sort_clockwise(self, *, inplace: bool = False) -> np.ndarray:
        """Sort points of convex polygon in clockwise order.

        :param inplace: Whether to replace current polygon hull with new obtained values
        :type inplace: bool
        :return: Sorted hull
        :rtype: numpy.ndarray
        """
        center = np.sum(self.hull, axis=0) / self.hull.shape[0]
        x, y = self.hull.T[self.xi] - center[self.xi], self.hull.T[self.yi] - center[self.yi]
        atan2 = np.arctan2(y, x)
        arr1inds = atan2.argsort()[::-1][:len(atan2)]
        hull = self.hull[arr1inds[::-1]]
        if inplace:
            self.hull = hull
        return hull

    # noinspection PyPep8Naming
    def to_Points(self) -> np.ndarray:  # noqa: N802
        """Convert hull points to Point objects.

        :return: Array of Point objects
        :rtype: numpy.ndarray
        """
        return np.array([Point(*point) for point in self.hull])

    def to_array(self) -> np.ndarray:
        """Return the hull as a numpy array.

        :return: Hull as numpy array
        :rtype: numpy.ndarray
        """
        return np.array(self._hull)

    @staticmethod
    def validity_check(hull: Shape2DType, length: INT, *, _raise: bool = True) -> bool:
        """Check if the hull is a valid 2D polygon shape.

        :param hull: Iterable of points
        :type hull: Point2DType
        :param length: Required length
        :type length: INT
        :param _raise: Whether to raise an error if invalid
        :type _raise: bool
        :return: True if valid, False otherwise
        :rtype: bool
        """
        length_test = (len(hull) == length) if length in [3, 2] else (len(hull) > length)
        try:
            point_dim = 2
            if (length_test
                    & (isinstance(hull, (Iterable, np.ndarray)))
                    & np.all(np.array([len(v) == point_dim for v in hull]))):
                return True
        except TypeError:
            pass
        if _raise:
            msg = "invalid 2D polygon shape"
            raise ValueError(msg)
        return False

    @classmethod
    def polygon_validity_check(cls, hull: Shape2DType, *, _raise: bool = True) -> bool:
        """Check if the hull is a valid polygon.

        :param hull: Iterable of points
        :type hull: Point2DType
        :param _raise: Whether to raise an error if invalid
        :type _raise: bool
        :return: True if valid, False otherwise
        :rtype: bool
        """
        return cls.validity_check(hull, 1, _raise=_raise)

    @property
    def transpose(self) -> np.ndarray:
        """Transpose self._hull and return numpy.array.

        :return: Transposed hull
        :rtype: numpy.ndarray
        """
        return self._hull.T

    T = transpose
