"""Implementation of Separating Axis Theorem (SAT) algorithm in 2D.

Detection of collision of two convex polygons.
If faces are just in touch, the function will return False unless in_touch=True.
Overlap is handled as no intersection at all.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa.pypex.base.conf import ROUND_PRECISION
from elisa.pypex.poly2d import projection

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.pypex.base.types import Shape2DType

MIN_POLYGON_VERTICES = 3
LINE_SEGMENT_VERTICES = 2


def separating_axis_theorem(
        poly1: NDArray,
        poly2: NDArray,
        *,
        touch_is_separated: bool = False,
        round_tol: int = ROUND_PRECISION,
) -> bool:
    """Test whether two convex polygons are separated using the Separating Axis Theorem (SAT).

    The Separating Axis Theorem states that two convex polygons are separated
    if and only if there exists at least one axis (derived from polygon edge normals)
    along which their projections do not overlap.

    Touch handling is controlled by ``touch_is_separated``:

        - If ``touch_is_separated`` is False:
            Polygons that touch at a boundary point or edge are considered
            NOT separated (i.e. touching counts as intersection).

        - If ``touch_is_separated`` is True:
            Polygons that only touch are considered separated.
            Only strict area overlap counts as intersection.

    :param poly1: First convex polygon as array of vertices
    :type poly1: NDArray
    :param poly2: Second convex polygon as array of vertices
    :type poly2: NDArray
    :param touch_is_separated:
        If True, touching polygons are treated as separated.
        If False, touching counts as intersection.
    :type touch_is_separated: bool
    :param round_tol: Decimal precision used to stabilize floating-point comparisons
    :type round_tol: int
    :return: True if polygons are separated according to the touch policy,
             False if they overlap (or touch when touching counts as intersection)
    :rtype: bool
    :raises ValueError: If either polygon has fewer than 3 vertices
    """
    # number of vertices in each polygon
    nv1, nv2 = len(poly1), len(poly2)

    if (nv1 < MIN_POLYGON_VERTICES) or (nv2 < MIN_POLYGON_VERTICES):
        msg = "invalid polygon shape, expected at least polygon with 3 corners"
        raise ValueError(msg)

    # Build a list of all edges from both polygons.
    # SAT for convex polygons needs testing axes derived from all polygon edges.
    edges = np.concatenate(
        (
            [[poly1[i], poly1[i + 1]] for i in range(-1, nv1 - 1)],
            [[poly2[i], poly2[i + 1]] for i in range(-1, nv2 - 1)],
        ),
        axis=0,
    )

    for edge in edges:
        # Tangent is the unit direction along the current edge.
        tangent = (edge[1] - edge[0]) / np.linalg.norm(edge[1] - edge[0])

        # Normal is perpendicular to the edge. This is the SAT axis we test.
        normal = np.array([-tangent[1], tangent[0]])

        # Convenience: we want to project onto the SAT axis (normal).
        # This code uses a helper that returns coordinates in a (tangent, normal) basis,
        # so we swap to make "tangent" be the axis we project onto.
        tangent, normal = normal, tangent

        # Project every vertex of each polygon onto the test axis (tangent).
        # After projection and basis conversion, the 1st component (x) is the scalar position
        # along the test axis. The 2nd component is ~0 (numerical noise) for pure projection.
        projection_poly1 = np.array([
            projection.cartesian_to_vectors_defined(
                tangent, normal, projection.projection(vertex, tangent),
            )
            for vertex in poly1
        ])
        projection_poly2 = np.array([
            projection.cartesian_to_vectors_defined(
                tangent, normal, projection.projection(vertex, tangent),
            )
            for vertex in poly2
        ])

        # Extract scalar coordinates along the test axis.
        projection_poly1_x = projection_poly1.T[0]
        projection_poly2_x = projection_poly2.T[0]

        # Each polygon's projection is an interval [min, max] on this axis.
        # We round to reduce floating point jitter near touching cases.
        interval1 = [
            round(projection_poly1_x.min(), round_tol),
            round(projection_poly1_x.max(), round_tol),
        ]
        interval2 = [
            round(projection_poly2_x.min(), round_tol),
            round(projection_poly2_x.max(), round_tol),
        ]

        # Sort intervals so interval[0] starts first.
        intervals = [interval1, interval2]
        intervals.sort(key=lambda x: x[0])

        # Separation on an axis means there is a gap between intervals:
        #   gap:     end1 < start2
        #   touch:   end1 == start2
        #
        # touch handling:
        # - touch_is_separated=False -> touching counts as NOT separated (intersection),
        #   so separation must be strict: end1 < start2
        # - touch_is_separated=True  -> touching counts as separated,
        #   so separation can be inclusive: end1 <= start2
        eval_method = np.less_equal if touch_is_separated else np.less

        if eval_method(intervals[0][1], intervals[1][0]):
            # Found a separating axis: polygons do not intersect (or only touch if policy says so).
            return True

    # No separating axis exists: polygons overlap (or touch counts as intersection).
    return False


def separating_axis_theorem_line_adapt(
        line1: NDArray,
        line2: NDArray,
        *,
        touch_is_separated: bool = False,
        round_tol: int = ROUND_PRECISION,
) -> bool:
    """Test whether two colinear line segments are separated.

     Using a 1D projection-based reduction of the Separating Axis Theorem (SAT).

    For colinear segments, the 2D problem reduces to checking overlap of
    their projections onto the common line direction:

        - If projected intervals strictly overlap → segments overlap.
        - If there is a gap between intervals → segments are separated.
        - If intervals meet at exactly one endpoint → segments are touching.

    Touch handling is controlled by ``touch_is_separated``:

        - If ``touch_is_separated`` is False:
            Touching segments are considered NOT separated
            (i.e. they count as intersecting).

        - If ``touch_is_separated`` is True:
            Touching segments are considered separated
            (i.e. only strict overlap counts as intersection).

    :param line1: First line segment as array of two vertices
    :type line1: NDArray
    :param line2: Second line segment as array of two vertices
    :type line2: NDArray
    :param touch_is_separated:
        If True, touching segments are treated as separated.
        If False, touching counts as intersection.
    :type touch_is_separated: bool
    :param round_tol: Decimal precision used to stabilize floating-point comparisons
    :type round_tol: int
    :return: True if segments are separated according to the touch policy,
             False if they overlap (or touch when touching counts as intersection)
    :rtype: bool
    """
    # Build a local coordinate system.
    tangent = (line1[1] - line1[0]) / np.linalg.norm(line1[1] - line1[0])
    normal = np.array([-tangent[1], tangent[0]])

    # Convert projected vectors into local coordinates. Project both segments onto the tangent axis.
    projection_line1 = np.array([projection.cartesian_to_vectors_defined(
        tangent, normal, projection.projection(vertex, tangent))
        for vertex in line1])
    projection_line2 = np.array([projection.cartesian_to_vectors_defined(
        tangent, normal, projection.projection(vertex, tangent))
        for vertex in line2])
    projection_x = [np.round(projection_line1.T[0], round_tol), np.round(projection_line2.T[0], round_tol)]

    # Sort intervals
    # After sorting:
    # - projection_x[0] is the interval that starts first
    # - projection_x[1] is the interval that starts later
    projection_x.sort(key=lambda x: x[0])

    # Does the first interval end before the second begins?
    # ...is end_of_first < start_of_second ?
    # If touch_is_separated=True: use <=
    # - touching counts as separated
    # If touch_is_separated=False: use <
    # - touching counts as NOT separated (so it counts as intersection)
    eval_method = np.less_equal if touch_is_separated else np.less
    return eval_method(projection_x[0][1], projection_x[1][0])


def intersects(
        poly1: NDArray | Shape2DType,
        poly2: NDArray | Shape2DType,
        *,
        touch_is_separated: bool = False,
        round_tol: int = ROUND_PRECISION,
) -> bool:
    """Determine whether two convex objects intersect using the Separating Axis Theorem (SAT).

    For convex polygons, all relevant separating axes are tested.
    For line segments, this correctly resolves overlap only when the segments
    are colinear. In that case, the problem reduces to a 1D interval overlap test.

    Touch handling is configurable:

    - If ``touch_is_separated`` is False:
        Touching objects (sharing exactly one boundary point) are considered
        intersecting.

    - If ``touch_is_separated`` is True:
        Touching objects are treated as separated, meaning only strict overlap
        counts as intersection.

    :param poly1: First convex polygon or line segment as array of vertices
    :type poly1: NDArray | Shape2DType
    :param poly2: Second convex polygon or line segment as array of vertices
    :type poly2: NDArray | Shape2DType
    :param touch_is_separated:
        If True, touching objects are considered separated.
        If False, touching counts as intersection.
    :type touch_is_separated: bool
    :param round_tol: Decimal precision used for numerical stability in comparisons
    :type round_tol: int
    :return: True if objects intersect (according to the touch policy), False if separated
    :rtype: bool
    """
    if (len(poly1) == LINE_SEGMENT_VERTICES) and (len(poly2) == LINE_SEGMENT_VERTICES):
        return not separating_axis_theorem_line_adapt(
            poly1,
            poly2,
            touch_is_separated=touch_is_separated,
            round_tol=round_tol,
        )
    return not separating_axis_theorem(poly1, poly2, touch_is_separated=touch_is_separated, round_tol=round_tol)
