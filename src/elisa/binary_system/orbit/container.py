import numpy as np

from collections.abc import Sequence
from matplotlib import pyplot as plt

from ... import umpy as up
from ... logger import getLogger
from ... utils import is_empty, polar_to_cartesian

logger = getLogger("orbit.container")


class OrbitalSupplements(Sequence):
    """
    Structure designed to store orbital positions on (nearly) symmetrical orbital positions around apsidal line.
    The symmetrical counterparts are stored in the same positions within 'body' and 'mirror' position arrays.
    Corresponding `body` and `mirror` items share the same bindary system model which saves computational time.
    !!! BEWARE, THIS IS MUTABLE !!!

    :param body: numpy.array; N x 5 array containing row-wise orbital positions used as templates
                             (they will be evaluated exactly)
    :param mirror: numpy.array; N x 5 array containing row-wise orbital positions mirrored from corresponding
                               `body` orbital positions using symmetry along apsidal line.
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
        """
        Appending corresponding `body`/`mirror` couple to the end of their respective arrays.

        :param body: numpy.array; [index, ] + [items for items in elisa.const.Position]
        :param mirror: numpy.array; [index, ] + [items for items in elisa.const.Position]
        """
        self._body = np.vstack((self._body, body)) if not is_empty(self._body) else np.array([body])
        self._mirror = np.vstack((self._mirror, mirror)) if not is_empty(self._mirror) else np.array([mirror])

    @property
    def body(self):
        """Returns an array of body positions."""
        return self._body

    @property
    def mirror(self):
        """Returns an array of mirror positions."""
        return self._mirror

    @property
    def body_defined(self):
        """Returns True if `body` position array is not empty."""
        return self.not_empty(self.body)

    @property
    def mirror_defined(self):
        """Returns True if `mirror` position array is not empty."""
        return self.not_empty(self.mirror)

    @staticmethod
    def is_empty(val):
        """Return True if given orbital position is empty."""
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

        :param by: str; `index`, `distance` or `radius` sorting orbital positions in `body` an `mirror` according to
                         the index of `body` positions or by component distances.
        :return: self; sorted OrbitalSupplements object
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
        """
        Visualize orbital positions stored in `body` .
        """
        self._plot(self.body_defined)

    def plot_mirrors(self):
        """
        Visualize orbital positions stored in `mirror` .
        """
        self._plot(self.mirror_defined, markers=["x", "x"])

    def plot(self):
        """
        Visualize orbital positions stored in OrbitalSupplements.
        """
        self._plot(self.body_defined, self.mirror_defined)

    @classmethod
    def _plot(cls, arr1, arr2=None, markers=None):
        markers = ["o", "x"] if markers is None else markers
        x, y = polar_to_cartesian(arr1[:, 1], arr1[:, 2] - (up.pi / 2))
        plt.scatter(x, y, marker=markers[0])

        if not is_empty(arr2):
            x, y = polar_to_cartesian(arr2[:, 1], arr2[:, 2] - (up.pi / 2))
            plt.scatter(x, y, marker=markers[1])

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
