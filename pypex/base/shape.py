import numpy as np

from abc import ABCMeta, abstractmethod
from collections import Iterable
from numpy import array
from pypex.poly2d.point import Point


class Shape2D(metaclass=ABCMeta):
    """
    Convex 2D Shape
    """

    def __init__(self, hull, _validity=True):
        hull = self.normalize_hull(hull)
        if _validity:
            self.polygon_validity_check(hull=hull, _raise=True)
        self._hull = np.array(hull)
        self.xi, self.yi = 0, 1

    def __len__(self):
        return len(self.hull)

    def __eq__(self, other):
        if len(other) != len(self):
            return False
        return all([Point(*self.hull[i]) == Point(*other.hull[i]) for i in range(len(self))])

    def __str__(self):
        return "Poly ({}): [{}]".format(len(self.hull), ", ".join([str(v) for v in self.hull]))

    def __repr__(self):
        return "Poly ({}): [{}]".format(len(self.hull), ", ".join([str(v) for v in self.hull]))

    @staticmethod
    def normalize_hull(hull):
        """
        Convert definition of points to  normal form (to numpy.array).

        :param hull: Iterable; iterable of points
        :return: numpy.array
        """
        return np.array([vertex.to_array() if isinstance(vertex, Point) else vertex for vertex in hull])

    @property
    def hull(self):
        return self._hull

    @hull.setter
    def hull(self, hull):
        self._hull = hull

    @abstractmethod
    def intersects(self, shape, **kwargs):
        pass

    @abstractmethod
    def intersection(self, shape):
        pass

    def sort_clockwise(self, inplace=False):
        """
        Sort points of convex polygon in clokwise order

        :param inplace: bool; replace current polygon hull with new obtained values
        :return: numpy.array
        """
        center = np.sum(self.hull, axis=0) / self.hull.shape[0]
        x, y = self.hull.T[self.xi] - center[self.xi], self.hull.T[self.yi] - center[self.yi]
        atan2 = np.arctan2(y, x)
        arr1inds = atan2.argsort()[::-1][:len(atan2)]
        hull = self.hull[arr1inds[::-1]]
        if inplace:
            self.hull = hull
        return hull

    def to_Points(self):
        return np.array([Point(*point) for point in self.hull])

    @staticmethod
    def validity_check(hull, length, _raise=True):
        length_test = (len(hull) == length) if length in [3, 2] else (len(hull) > length)
        try:
            if length_test & (isinstance(hull, (Iterable, array))) & np.all(np.array([len(v) == 2 for v in hull])):
                return True
        except TypeError:
            pass
        if _raise:
            raise ValueError("invalid 2D polygon shape")
        return False

    @classmethod
    def polygon_validity_check(cls, hull, _raise=True):
        return cls.validity_check(hull, 1, _raise)
