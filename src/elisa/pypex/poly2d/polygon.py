from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.path as mpltpath
import numpy as np

from elisa.pypex.base import shape
from elisa.pypex.base.conf import ROUND_PRECISION
from elisa.pypex.poly2d.intersection import linter, sat
from elisa.pypex.poly2d.line import Line
from elisa.pypex.poly2d.point import Point, is_point_in_polygon

if TYPE_CHECKING:
    from collections.abc import Iterator

    from elisa.pypex.base.types import Shape2DType


MIN_POLY_POINTS = 3


class Polygon(shape.Shape2D):
    """Convex Polygon."""

    def __init__(self, hull: Shape2DType, **kwargs) -> None:
        """Initialize a convex polygon.

        :param hull: Array or list of points defining the convex hull
        :type hull: Point2DType
        """
        super().__init__(hull=hull, **kwargs)
        self.sort_clockwise(inplace=True)
        self.mplpath = mpltpath.Path(self.hull)

    def edges(self, *, as_line: bool = False) -> Iterator[np.ndarray | Line]:
        """Iterate over edges in the polygon.

        :param as_line: If True, yield Line objects instead of numpy arrays
        :type as_line: bool
        :return: Iterator over edges as numpy arrays or Line objects
        :rtype: Iterator[numpy.ndarray | Line]
        """
        for i in range(-1, len(self.hull) - 1, 1):
            edge = np.array([self.hull[i], self.hull[i + 1]])
            if as_line:
                edge = Line(edge)
            yield edge

    def intersects(
            self,
            poly: Polygon,
            *,
            touch_is_separated: bool = True,
            round_tol: int = ROUND_PRECISION,
    ) -> bool:
        """Check whether two polygons intersect.

        Touch handling policy:
          - touch_is_separated=True: touching counts as separated (strict)
          - touch_is_separated=False: touching counts as intersection (inclusive)

        Backward compatibility note:
          Previously this method used ``in_touch`` with default False (strict).
          The default behavior is preserved by using ``touch_is_separated=True`` as default.

        :param poly: Another polygon.
        :type poly: Polygon
        :param touch_is_separated: Touch policy, see above. Defaults to True.
        :type touch_is_separated: bool
        :param round_tol: Precision for rounding
        :type round_tol: int
        :return: True if polygons intersect, False otherwise
        :rtype: bool
        """
        # sat.intersects returns True when objects intersect (not separated)
        return sat.intersects(self.hull, poly.hull, touch_is_separated=touch_is_separated, round_tol=round_tol)

    def intersection(self, poly: Polygon, *, round_tol: int = ROUND_PRECISION) -> Polygon | None:
        """Find intersection polygon created by clipping one polygon by another.

        :param poly: Another polygon.
        :type poly: Polygon
        :param round_tol: Precision for rounding
        :type round_tol: int
        :return: Intersection polygon or None
        :rtype: Polygon | None
        """
        # Vertices of each polygon that lie strictly inside the other
        in_poly1 = poly.hull[self.mplpath.contains_points(poly.hull)]
        in_poly2 = self.hull[poly.mplpath.contains_points(self.hull)]

        # Edge-edge intersection points (strict: touch_is_separated=False keeps endpoints)
        _, intersection_segment, intr_ptx, _, msg, _ = linter.intersections(
            self.hull, poly.hull, touch_is_separated=False,
        )
        edge_pts = intr_ptx[(msg == b"INTERSECT") & intersection_segment]

        # Collect all candidate vertices as raw float arrays — no Point wrappers needed.
        # normalize_hull() in Shape2D handles plain numpy rows via its `else vertex` branch.
        all_pts = np.concatenate((in_poly1, in_poly2, edge_pts), axis=0)
        if len(all_pts) < MIN_POLY_POINTS:
            return None

        # Deduplicate by rounding — vectorised replacement for Point.set.
        # np.unique sorts rows, which is fine: Polygon.__init__ re-sorts clockwise.
        all_pts = np.unique(np.round(all_pts, round_tol), axis=0)
        return Polygon(all_pts, _validity=False) if len(all_pts) >= MIN_POLY_POINTS else None

    def to_array(self) -> np.ndarray:
        """Return the hull as a numpy array.

        :return: Hull as numpy array
        :rtype: numpy.ndarray
        """
        return self.hull

    def contains_point(self, point: Point) -> bool:
        """Test whether a point lies in the polygon.

        :param point: Point to test
        :type point: Point
        :return: True if point is inside the polygon, False otherwise
        :rtype: bool
        """
        return is_point_in_polygon(point, self)

    contains_point_alias = contains_point

    def surface_area(self) -> float:
        """Compute surface area of the polygon.

        :return: Surface area of the polygon
        :rtype: float
        """
        lines = linter.polygon_hull_to_edges(self.hull)
        return 0.5 * np.abs(np.sum(lines[:, 0, 0] * lines[:, 1, 1] - lines[:, 1, 0] * lines[:, 0, 1]))

    def inpolygon(self) -> Polygon:
        """Find polygon whose points are defined as the center of each edge of the original polygon.

        :return: Polygon of edge centers
        :rtype: Polygon
        """
        _inpolygon = []
        for edge in self.edges(as_line=True):
            parametrized = edge.parametrized()
            _inpolygon.append(parametrized(0.5))
        return Polygon(np.array(_inpolygon))
