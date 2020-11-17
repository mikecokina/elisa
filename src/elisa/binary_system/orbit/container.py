import numpy as np

from collections.abc import Sequence
from matplotlib import pyplot as plt

from ... import umpy as up
from ... logger import getLogger
from ... utils import is_empty, polar_to_cartesian

logger = getLogger("orbit.container")


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
        return np.all(up.isnan(val))

    @classmethod
    def not_empty(cls, arr):
        """
        Return values where supplied array is not empty.

        :param arr: numpy.array;
        :return: numpy.array;
        """
        return arr[list(map(lambda x: not cls.is_empty(x), arr))]

    def sort(self, by='distance'):
        """
        Sort by given quantity.
        This method sorts bodies and mirrors based on quantity chosen on input.
        Sorting of mirrors is based on sorting of bodies.

        :param by: str;
        :return: self;
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

        x, y = polar_to_cartesian(arr1[:, 1], arr1[:, 2] - (up.pi / 2))
        plt.scatter(x, y, marker=marker)

        if not is_empty(arr2):
            x, y = polar_to_cartesian(arr2[:, 1], arr2[:, 2] - (up.pi / 2))
            plt.scatter(x, y, marker="x")

        plt.grid(True)
        plt.axes().set_aspect('equal')
        plt.show()

    def __iter__(self):
        for body, mirror in zip(self.body, self.mirror):
            yield body, mirror

    def __len__(self):
        return len(self.body)

    def __eq__(self, other):
        return np.all(self._body == other.body) & \
               np.all((self.mirror == other.mirror)[~np.all(up.isnan(other.mirror) & up.isnan(self.mirror), axis=1)])

    def __str__(self):
        return f"{self.__class__.__name__}\nbodies: {self.body}\nmirrors: {self._mirror}"

    def __repr__(self):
        return self.__str__()
