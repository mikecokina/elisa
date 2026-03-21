from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa.pypex.base import shape
from elisa.pypex.base.conf import ROUND_PRECISION
from elisa.pypex.poly2d.intersection import linter

if TYPE_CHECKING:
    from elisa.pypex.poly2d.point import Point


class Line(shape.Shape2D):
    __intersect__ = ("INTERSECT",)
    __overlapping__ = ("COLINEAR",)

    def __str__(self) -> str:
        return "Line: [{}]".format(", ".join([str(v) for v in self.hull]))

    def __repr__(self) -> str:
        return "Line: [{}]".format(", ".join([str(v) for v in self.hull]))

    def intersects(
            self,
            line: Line,
            *,
            _full: bool = False,
            touch_is_separated: bool = True,
            round_tol: int = ROUND_PRECISION,
    ) -> bool | tuple:
        """Figure out whether two line segments intersect.

        Touch handling policy:
          - touch_is_separated=True: touching counts as separated (strict intersection)
          - touch_is_separated=False: touching at one point counts as intersection (inclusive)

        Backward compatibility note:
          Previously this method used ``in_touch`` with default False (strict).
          The default behavior is preserved by using ``touch_is_separated=True`` as default.

        :param line: The other line to check intersection with.
        :param _full: If True, return full intersection tuple. Defaults to False.
        :param touch_is_separated: Touch policy, see above. Defaults to True.
        :param round_tol: Consider as same up to 'round_tol' decimal numbers. Defaults to ROUND_PRECISION.

        :returns: True if segments have a unique-point intersection (message "INTERSECT"),
                  or full intersection tuple if _full is True.
        :rtype: bool or tuple
        """
        intersection = linter.intersection(
            self.hull[0], self.hull[1], line.hull[0], line.hull[1],
            touch_is_separated=touch_is_separated,
            round_tol=round_tol,
        )
        if _full:
            return intersection

        # Keep original behavior: True only for unique-point intersection (not colinear overlap)
        return bool(intersection[1]) and (intersection[4] == "INTERSECT")

    def full_intersects(
            self,
            line: Line,
            *,
            touch_is_separated: bool = True,
            round_tol: int = ROUND_PRECISION,
    ) -> tuple:
        """Return full intersection tuple for two line segments.

        Touch handling policy:
          - touch_is_separated=True: touching counts as separated (strict)
          - touch_is_separated=False: touching counts as intersection (inclusive)

        :param line: The other line to check intersection with.
        :param touch_is_separated: Touch policy, see above. Defaults to True.
        :param round_tol: Consider as same up to 'round_tol' decimal numbers. Defaults to ROUND_PRECISION.

        :returns: Full intersection tuple as returned by linter.intersection.
        :rtype: tuple
        """
        return linter.intersection(
            self.hull[0], self.hull[1], line.hull[0], line.hull[1],
            touch_is_separated=touch_is_separated,
            round_tol=round_tol,
        )

    def intersection(
            self,
            line: Line,
            *,
            touch_is_separated: bool = True,
            round_tol: int = ROUND_PRECISION,
    ) -> Point | None:
        """Find unique intersection point of two line segments if it exists.

        Returns a Point only when there is a unique-point intersection ("INTERSECT").
        For colinear overlaps ("COLINEAR") there is no unique single point, so returns None.

        Touch handling policy:
          - touch_is_separated=True: touching counts as separated (strict)
          - touch_is_separated=False: touching counts as intersection (inclusive)

        :param line: The other line to check intersection with.
        :param touch_is_separated: Touch policy, see above. Defaults to True.
        :param round_tol: Consider as same up to 'round_tol' decimal numbers. Defaults to ROUND_PRECISION.

        :returns: Intersection point if exists, else None.
        :rtype: Point or None
        """
        intersection = self.full_intersects(line, touch_is_separated=touch_is_separated, round_tol=round_tol)
        intersect = bool(intersection[1]) and (intersection[4] == "INTERSECT")
        if not intersect:
            return None
        return intersection[2]

    def to_array(self) -> np.ndarray:
        """Get points of line in numpy array.

        :returns: Array of points for the line.
        :rtype: numpy.ndarray
        """
        return np.array([point.to_array() for point in self.to_Points()])

    def sort_clockwise(self, *args, **kwargs) -> np.ndarray:  # noqa: ARG002
        """Return hull points (no sorting for line).

        :returns: Array of hull points (unsorted for line).
        :rtype: numpy.ndarray
        """
        return self.hull

    def direction_vector(self) -> np.ndarray:
        """Get direction vector.

        :returns: Direction vector from first to second point.
        :rtype: numpy.ndarray
        """
        return self.hull[1] - self.hull[0]

    def parametrized(self) -> callable:
        """Return callable parametrization of given line as function.

        :returns: Function of t returning point on the line.
        :rtype: callable
        """

        def _parametrized(t: float) -> tuple[float, float]:
            v = self.direction_vector()
            x = float(self.hull[0][0] + (t * v[0]))
            y = float(self.hull[0][1] + (t * v[1]))
            return x, y

        return _parametrized

    def angle(self, other: shape.Shape2D, *, degrees: bool = False) -> float:
        """Return angle between vectors defined by self `Line` and other `Line`.

        :param other: The other line or shape.
        :param degrees: If True, return angle in degrees. Defaults to False.

        :returns: Angle between the two lines.
        :rtype: float
        """
        vector_self = self.hull[0] - self.hull[1]
        vector_other = other.hull[0] - other.hull[1]

        unit_vector_self = vector_self / np.linalg.norm(vector_self)
        unit_vector_other = vector_other / np.linalg.norm(vector_other)
        dot_product = np.dot(unit_vector_self, unit_vector_other)
        angle = np.arccos(dot_product)
        if degrees:
            angle = np.degrees(angle)
        return angle
