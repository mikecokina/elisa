from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from elisa import umpy as up
from elisa.base.types import FLOAT
from elisa.logger import getLogger
from elisa.types import Float, NumpyBool
from elisa.utils import is_empty, polar_to_cartesian

logger = getLogger("orbit.container")


class OrbitalSupplements(Sequence[tuple[NDArray[Float], NDArray[Float]]]):
    """Store nearly symmetrical orbital positions around the apsidal line.

    This structure is designed to store orbital positions on nearly
    symmetrical orbital positions around the apsidal line. The symmetrical
    counterparts are stored at the same indices within the ``body`` and
    ``mirror`` position arrays. Corresponding ``body`` and ``mirror`` items
    share the same binary-system model, which saves computational time.

    This object is mutable.

    :param body: ``N x M`` array containing row-wise orbital positions used as
        templates. These positions are evaluated exactly.
    :type body: NDArray | None
    :param mirror: ``N x M`` array containing row-wise orbital positions
        mirrored from corresponding ``body`` orbital positions using symmetry
        along the apsidal line.
    :type mirror: NDArray | None
    """

    def __init__(
        self,
        body: NDArray | None = None,
        mirror: NDArray | None = None,
    ) -> None:
        """Initialize the orbital supplements container.

        :param body: Initial body orbital positions.
        :type body: NDArray | None
        :param mirror: Initial mirrored orbital positions.
        :type mirror: NDArray | None
        :return: ``None``.
        :rtype: None
        """
        if body is None and mirror is None:
            self._body = np.array([], dtype=FLOAT)
            self._mirror = np.array([], dtype=FLOAT)
        else:
            self._body = np.asarray(body, dtype=FLOAT)
            self._mirror = np.asarray(mirror, dtype=FLOAT)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[NDArray[Float], NDArray[Float]]:
        """Return the body and mirror positions at ``index``.

        :param index: Row index.
        :type index: int
        :return: Body and mirror orbital positions at the selected index.
        :rtype: tuple[NDArray[numpy.float64], NDArray[numpy.float64]]
        """
        return self.body[index], self.mirror[index]

    def append(self, body: NDArray, mirror: NDArray) -> None:
        """Append a corresponding body and mirror pair.

        The appended rows are added to the end of their respective arrays.

        :param body: Orbital position row in the form
            ``[index, ...position fields...]``.
        :type body: NDArray
        :param mirror: Mirrored orbital position row in the form
            ``[index, ...position fields...]``.
        :type mirror: NDArray
        :return: ``None``.
        :rtype: None
        """
        body_row = np.asarray(body, dtype=FLOAT)
        mirror_row = np.asarray(mirror, dtype=FLOAT)

        self._body = (
            np.vstack((self._body, body_row)) if not is_empty(self._body) else np.array([body_row], dtype=FLOAT)
        )
        self._mirror = (
            np.vstack((self._mirror, mirror_row)) if not is_empty(self._mirror) else np.array([mirror_row], dtype=FLOAT)
        )

    @property
    def body(self) -> NDArray[Float]:
        """Return the array of body orbital positions.

        :return: Body orbital positions.
        :rtype: NDArray[numpy.float64]
        """
        return self._body

    @property
    def mirror(self) -> NDArray[Float]:
        """Return the array of mirrored orbital positions.

        :return: Mirrored orbital positions.
        :rtype: NDArray[numpy.float64]
        """
        return self._mirror

    @property
    def body_defined(self) -> NDArray[Float]:
        """Return non-empty body orbital positions.

        :return: Body orbital positions that are not empty.
        :rtype: NDArray[numpy.float64]
        """
        return self.not_empty(self.body)

    @property
    def mirror_defined(self) -> NDArray[Float]:
        """Return non-empty mirrored orbital positions.

        :return: Mirrored orbital positions that are not empty.
        :rtype: NDArray[numpy.float64]
        """
        return self.not_empty(self.mirror)

    @staticmethod
    def is_empty(val: NDArray) -> NumpyBool:
        """Return whether a single orbital position row is empty.

        A row is considered empty if all its values are ``NaN``.

        :param val: Orbital position row.
        :type val: NDArray
        :return: ``True`` if the row is empty, otherwise ``False``.
        :rtype: numpy.bool_
        """
        value_array = np.asarray(val, dtype=FLOAT)
        return np.all(up.isnan(value_array))

    @classmethod
    def not_empty(cls, arr: NDArray) -> NDArray[Float]:
        """Return only the rows in ``arr`` that are not empty.

        :param arr: Array of orbital positions.
        :type arr: NDArray
        :return: Orbital positions that are not empty.
        :rtype: NDArray[numpy.float64]
        """
        arr_array = np.asarray(arr, dtype=FLOAT)

        if arr_array.size == 0:
            return arr_array

        mask = np.array([not cls.is_empty(row) for row in arr_array], dtype=bool)
        return arr_array[mask]

    def sort(
        self,
        by: Literal["index", "distance", "radius"] = "distance",
    ) -> OrbitalSupplements:
        """Sort stored orbital positions by the selected quantity.

        This method sorts ``body`` and ``mirror`` based on the quantity chosen
        on input. Sorting of mirrors follows the sorting of bodies.

        ``by`` can be one of:

        - ``"index"`` - sort by the row index field
        - ``"distance"`` - sort by component distance
        - ``"radius"`` - sort by the same column as distance

        :param by: Sorting key.
        :type by: Literal["index", "distance", "radius"]
        :return: Sorted orbital supplements instance.
        :rtype: OrbitalSupplements
        :raises ValueError: If ``by`` has an invalid value.
        """
        if by == "index":
            sort_column = 0
        elif by in {"distance", "radius"}:
            sort_column = 1
        else:
            message = "Invalid value of `by`."
            raise ValueError(message)

        sort_index = np.argsort(self.body[:, sort_column])
        self._body = self.body[sort_index]
        self._mirror = self.mirror[sort_index]
        return self

    def size(self) -> int:
        """Return the number of stored orbital-position pairs.

        :return: Number of stored pairs.
        :rtype: int
        """
        return self.__len__()

    def to_orbital_position(self) -> None:
        """Convert stored rows to orbital-position objects.

        This method is currently not implemented.

        :return: ``None``.
        :rtype: None
        """
        message = "`to_orbital_position` is not implemented yet."
        raise NotImplementedError(message)

    def plot_bodies(self) -> None:
        """Visualize orbital positions stored in ``body``.

        :return: ``None``.
        :rtype: None
        """
        self._plot(self.body_defined)

    def plot_mirrors(self) -> None:
        """Visualize orbital positions stored in ``mirror``.

        :return: ``None``.
        :rtype: None
        """
        self._plot(self.mirror_defined, markers=("x", "x"))

    def plot(self) -> None:
        """Visualize orbital positions stored in the container.

        :return: ``None``.
        :rtype: None
        """
        self._plot(self.body_defined, self.mirror_defined)

    @classmethod
    def _plot(
        cls,
        arr1: NDArray,
        arr2: NDArray | None = None,
        markers: tuple[str, str] | None = None,
    ) -> None:
        """Plot one or two orbital-position arrays in polar projection.

        The orbital positions are converted from polar to Cartesian
        coordinates before plotting.

        :param arr1: Primary array of orbital positions to plot.
        :type arr1: NDArray
        :param arr2: Optional secondary array of orbital positions to plot.
        :type arr2: NDArray | None
        :param markers: Marker styles for the first and second arrays.
        :type markers: tuple[str, str] | None
        :return: ``None``.
        :rtype: None
        """
        marker_pair = ("o", "x") if markers is None else markers

        arr1_array = np.asarray(arr1, dtype=FLOAT)
        x_coord, y_coord = polar_to_cartesian(
            arr1_array[:, 1],
            arr1_array[:, 2] - (up.pi / 2),
        )
        plt.scatter(x_coord, y_coord, marker=marker_pair[0])

        if arr2 is not None and not is_empty(arr2):
            arr2_array = np.asarray(arr2, dtype=FLOAT)
            x_coord, y_coord = polar_to_cartesian(
                arr2_array[:, 1],
                arr2_array[:, 2] - (up.pi / 2),
            )
            plt.scatter(x_coord, y_coord, marker=marker_pair[1])

        plt.grid(visible=True)
        plt.gca().set_aspect("equal")
        plt.show()

    def __iter__(self) -> Iterator[tuple[NDArray[Float], NDArray[Float]]]:
        """Iterate over paired body and mirror rows.

        :return: Iterator over ``(body_row, mirror_row)`` pairs.
        :rtype: Iterator[tuple[NDArray[numpy.float64], NDArray[numpy.float64]]]
        """
        yield from zip(self.body, self.mirror, strict=True)

    def __len__(self) -> int:
        """Return the number of stored orbital-position pairs.

        :return: Number of stored pairs.
        :rtype: int
        """
        return len(self.body)

    def __eq__(self, other: object) -> bool:
        """Return whether two orbital supplement containers are equal.

        Equality compares body arrays directly. Mirror arrays are compared with
        ``NaN`` values treated as equal when they occur in the same rows.

        :param other: Object to compare with.
        :type other: object
        :return: ``True`` if both containers are equal, otherwise ``False``.
        :rtype: bool
        """
        if not isinstance(other, OrbitalSupplements):
            return False

        body_equal = np.all(self._body == other.body)

        mirror_nan_mask = np.all(
            up.isnan(other.mirror) & up.isnan(self.mirror),
            axis=1,
        )
        mirror_equal = np.all((self.mirror == other.mirror)[~mirror_nan_mask])

        return bool(body_equal and mirror_equal)

    def __str__(self) -> str:
        """Return the informal string representation.

        :return: String representation of the container.
        :rtype: str
        """
        return f"{self.__class__.__name__}\nbodies: {self.body}\nmirrors: {self._mirror}"

    def __repr__(self) -> str:
        """Return the developer-oriented string representation.

        :return: String representation of the container.
        :rtype: str
        """
        return self.__str__()

    def __hash__(self) -> int:
        """Return hash based on immutable snapshots of body and mirror arrays.

        :return: Hash value.
        :rtype: int
        """
        body_hash = hash(self._body.tobytes())
        mirror_hash = hash(self._mirror.tobytes())
        return hash((body_hash, mirror_hash))
