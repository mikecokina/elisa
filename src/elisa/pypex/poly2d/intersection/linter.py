from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa.base.types import FLOAT, INT
from elisa.pypex.base.conf import PRECISION, ROUND_PRECISION
from elisa.pypex.poly2d.intersection import sat
from elisa.pypex.poly2d.point import Point
from elisa.pypex.utils import det_2d, multiple_determinants

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Return helpers:
# - Segment intersection flag is typically bool, but some legacy paths use np.nan
#   to represent "unknown".
SegmentIntersectionFlag = bool | float
# - Intersection point is a Point for unique intersection, otherwise np.nan.
IntersectionPoint = Point | float


def intersection(
        p1: NDArray,
        p2: NDArray,
        p3: NDArray,
        p4: NDArray,
        *,
        touch_is_separated: bool = False,
        tol: float = PRECISION,
        round_tol: int = ROUND_PRECISION,
) -> tuple[bool, SegmentIntersectionFlag, IntersectionPoint, float, str]:
    """Determine intersection.

    defs::

        x1, y1 = p1 + u * (p2 - p1) = p1 + u * dp1
        x2, y2 = a + v * (b - a) = a + v * dp2
        dp1 = p2 - p1 = (p2_x - p1_x, p2_y - p1_y)
        dp2 = pt4 - pt3 = (pt4_x - pt3_x, pt4_y - pt3_y)

    intersection::

        x1, y1 = x2, y2
        p1 + u * dp1 = a + v * dp2
        in coo
        p1_x + u * dp1_x = pt3_x + v * dp2_x
        p1_y + u * dp1_y = pt3_y + v * dp2_y

        variables::
            u, v

        solution::

            d = (dp1_x * dp2_y) - (dp1_y * dp2_x)
            u = (((p1_y - pt3_y) * dp2_x) - (dp2_y * (p1_x - pt3_x))) / d
            v = (((p1_y - pt3_y) * dp1_x) - (dp1_y * (p1_x - pt3_x))) / d

    Touch handling policy (applies to the segment-range check on ``u`` and ``v``):

    - ``touch_is_separated=False``:
      touching at endpoints counts as intersection (inclusive bounds, ``<=``).

    - ``touch_is_separated=True``:
      touching is treated as separated (strict bounds, ``<``).

    :param p1: First point of first segment.
    :param p2: Second point of first segment.
    :param p3: First point of second segment.
    :param p4: Second point of second segment.
    :param round_tol: Consider two numbers as same if match up to ``round_tol`` decimal numbers.
    :param tol: Consider number as zero if smaller than ``tol``.
    :param touch_is_separated: If True, touching counts as separated. If False, touching counts as intersection.
    :returns: Intersection tuple describing infinite lines and finite segments relationship.
    :rtype: tuple[bool, bool | float, Point | float, float, str]

    Return tuple layout:

    0: intersection_status::

          False: parallel (or coincident)
          True:  intersection of infinite lines (unique or colinear)

    1: segment intersection (finite segments)::

          False:     no intersection
          True:      intersection between defined points or overlap
          numpy.nan: unknown

    2: intersection Point (unique intersection point for non-parallel lines; np.nan for parallel/colinear)

    3: distance if parallel (separation of infinite lines; ~0 for colinear)

    4: string representation/description
    """
    p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
    # first line
    dp1 = p2 - p1
    # second line
    dp2 = p4 - p3
    # determinant
    matrix = np.array([dp1, dp2])
    d = det_2d(matrix)

    # test if d < 1e-10
    # testing on zero, but precission should cause p3 problem
    if np.abs(d) < tol:
        # test distance between lines
        # if general form is known (ax + by + c1 = 0 and ax + by + c2 = 0),
        # d = abs(c1 - c2) / sqrt(a**2 + b**2)
        # parametric equation in general:
        #   x, y = [p1_x, p1_y] + u * [T_x, T_y], where T is tangential vector defined as p2 - p1
        # N = (a, b) represent normal vector of line; `p3`, `p4` (method parametres) from general equation of line
        # N = [-Ty, Tx], can be obtained
        # general equation:
        #   -Ty * x + Tx * y + c = 0, then
        # c = Ty * p1_x - Tx * p1_y
        # finaly, general equation:
        #   -Ty * x + Tx * y + (Ty * p1_x - Tx * p1_y) = 0
        #
        #
        # a1, b1, c1 = -dp1_y, dp1_x, (dp1_y * pt1_x) - (dp1_x * pt1_y)
        # a2, b2, c2 = -dp2_y, dp2_x, (dp2_y * pt3_x) - (dp2_x * pt3_y)

        a1, b1, c1 = -dp1[1], dp1[0], det_2d(np.array([p1, dp1]))

        # second line has to be definable with same tangential and normal vector as first line
        # since ax + by + c = 0 and in our case [x, y] = p3 or p4 for second equation, then for c2
        # we have c2 = - (a1 * p3[0] + b1 * p3[1])
        c2 = - (a1 * p3[0] + b1 * p3[1])
        d = abs(c2 - c1) / (np.sqrt(a1 ** 2 + b1 ** 2))

        intersects, msg = (True, "COLINEAR") if abs(d) < tol else (False, "PARALLEL")
        int_in_segment = (
            False
            if msg == "PARALLEL"
            else sat.intersects(
                np.array([p1, p2]),
                np.array([p3, p4]),
                touch_is_separated=touch_is_separated,
                round_tol=round_tol,
            )
        )
        return intersects, int_in_segment, np.nan, float(d), msg

    # If not parallel, compute the unique intersection point
    # When d != 0 they solve for (u, v) using determinant formulas (Cramer's rule):
    # +0 because of negative zero (-0.0 is incorrect) formatting on output
    u = (det_2d([dp2, p1 - p3]) / d) + 0.0
    v = (det_2d([dp1, p1 - p3]) / d) + 0.0

    # Touch handling policy:
    # - touch_is_separated=False includes endpoints (0 <= u <= 1), so touching counts as intersection.
    # - touch_is_separated=True uses strict inequalities (0 < u < 1), so endpoint touches are excluded.
    eval_method = np.less if touch_is_separated else np.less_equal

    # Then compute the actual intersection coordinates on line1:
    int_x = p1[0] + (u * dp1[0])
    int_y = p1[1] + (u * dp1[1])

    # If the intersection is inside segment 1, then u must be between 0 and 1.
    # If it is inside segment 2, then v must be between 0 and 1.
    int_segment = (
            eval_method(0.0, u)
            and eval_method(u, 1.0)
            and eval_method(0.0, v)
            and eval_method(v, 1.0)
    )
    return True, bool(int_segment), Point(FLOAT(int_x), FLOAT(int_y)), np.nan, "INTERSECT"


def intersections(  # noqa: PLR0915
        poly1: NDArray[np.floating],
        poly2: NDArray[np.floating],
        *,
        touch_is_separated: bool = False,
        tol: float = PRECISION,
        round_tol: int = ROUND_PRECISION,
) -> tuple[
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.bytes_],
    NDArray[np.floating],
]:
    """Vectorised implementaion of lines intersection function.

    Compute intersections of all combination of supplied arrays of points which define convex polygon.

    Touch handling policy (applies to segment-segment intersection checks):
      - touch_is_separated=False: touching at endpoints counts as intersection (inclusive bounds, <=)
      - touch_is_separated=True:  touching counts as separated (strict bounds, <)

    :param poly1: Clockwise ordered numpy array of points.
    :param poly2: Clockwise ordered numpy array of points.
    :param touch_is_separated: If True, touching in one point is treated as separated.
    :param tol: Consider all numbers as zero when ``abs(number) < tol``.
    :param round_tol: Consider two numbers as same if match up to ``round_tol`` decimal numbers.
    :returns: Tuple of arrays with per-edge-pair results and index mapping.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    m1, _ = poly1.shape
    m2, _ = poly2.shape
    n1, n2 = 2, 2

    # mask to origin
    idx_mask = _index_map(m1, m2)

    intersection_status, intersection_segment = np.zeros(m1 * m2, dtype=bool), np.zeros(m1 * m2, dtype=bool)
    intr_ptx = np.full_like(np.empty((m1 * m2, 2), dtype=FLOAT), np.nan)
    distance = np.full_like(np.empty(m1 * m2, dtype=FLOAT), np.nan)

    msg = np.full(m1 * m2, b"", dtype="S9")

    poly1_edges = polygon_hull_to_edges(poly1)
    poly2_edges = polygon_hull_to_edges(poly2)

    dif_poly1 = poly1_edges[:, 1, :] - poly1_edges[:, 0, :]
    dif_poly2 = poly2_edges[:, 1, :] - poly2_edges[:, 0, :]

    # make all possible determinants matrix for all combination of lines (needs for equation solver 1/D)
    corr_dpoly1 = np.repeat(dif_poly1, m2, axis=0)
    corr_dpoly2 = np.tile(dif_poly2, (m1, 1))

    det_matrix = np.empty((m1 * m2, n1, n2))
    det_matrix[:, 0, :] = corr_dpoly1
    det_matrix[:, 1, :] = corr_dpoly2

    determinants = multiple_determinants(det_matrix)
    non_intersections = np.array(np.abs(determinants) < tol)

    if non_intersections.any():
        problem_poly1 = np.repeat(poly1, m2, axis=0)[non_intersections]
        problem_dif_poly1 = np.repeat(dif_poly1, m2, axis=0)[non_intersections]
        a1, b1 = -problem_dif_poly1[:, 1], problem_dif_poly1[:, 0]

        face_dface = np.empty((problem_poly1.shape[0], 2, problem_poly1.shape[1]), dtype=problem_poly1.dtype)
        face_dface[:, 0, :] = problem_poly1
        face_dface[:, 1, :] = problem_dif_poly1
        c1 = multiple_determinants(face_dface)
        problem_poly2_edges = np.tile(poly2_edges, (poly1.shape[0], 1, 1))[non_intersections]
        c2 = -(a1 * problem_poly2_edges[:, 1, 0] + b1 * problem_poly2_edges[:, 1, 1])
        dist = np.abs(c2 - c1) / (np.sqrt(np.power(a1, 2) + np.sqrt(np.power(b1, 2))))
        # fill output
        distance[non_intersections] = dist
        msg[non_intersections] = "PARALLEL"
        overlaps = non_intersections.copy()

        overlaps[non_intersections] = np.abs(dist) < tol
        intersection_status[overlaps] = True

        if np.any(overlaps):
            # assume that in real life, there will neglible amount of parallel lines with zero distance (overlap lines)
            # so we can use for loop without any significant loose of performance
            poly1_comb_overlap = np.repeat(poly1_edges, m2, axis=0)[overlaps]
            poly2_comb_overlap = np.tile(poly2_edges, (m1, 1, 1))[overlaps]

            intersection_segment[overlaps] = np.array(
                [
                    sat.intersects(a, b, touch_is_separated=touch_is_separated, round_tol=round_tol)
                    for a, b in zip(poly1_comb_overlap, poly2_comb_overlap, strict=True)
                ],
            )
            msg[overlaps] = "COLINEAR"

    ints = ~non_intersections
    ok_dif_poly1, ok_dif_poly2 = corr_dpoly1[ints], corr_dpoly2[ints]
    ok_poly1_edges, ok_poly2_edges = np.repeat(poly1_edges, m2, axis=0)[ints], np.tile(poly2_edges, (m1, 1, 1))[ints]

    p1_p3 = ok_poly1_edges[:, 0, :] - ok_poly2_edges[:, 0, :]

    dp2_p1_p3matrix = _dpx_p1_p3matrix(p1_p3, ok_dif_poly1, ok_dif_poly2)
    dp1_p1_p3matrix = _dpx_p1_p3matrix(p1_p3, ok_dif_poly1, ok_dif_poly1)

    d = determinants[ints]
    u = (multiple_determinants(dp2_p1_p3matrix) / d) + 0.0
    v = (multiple_determinants(dp1_p1_p3matrix) / d) + 0.0

    # Touch handling policy:
    # - touch_is_separated=False includes endpoints (0 <= u <= 1), so touching counts as intersection.
    # - touch_is_separated=True uses strict inequalities (0 < u < 1), so endpoint touches are excluded.
    eval_method = np.less if touch_is_separated else np.less_equal

    intersect_in = ok_poly1_edges[:, 0, :] + (u[:, np.newaxis] * ok_dif_poly1)

    u_in_range = np.logical_and(eval_method(0.0, u), eval_method(u, 1.0))
    v_in_range = np.logical_and(eval_method(0.0, v), eval_method(v, 1.0))
    segments_intersection_status = np.logical_and(u_in_range, v_in_range)

    # fill output
    intersection_status[ints] = True
    intersection_segment[ints] = segments_intersection_status
    msg[ints] = "INTERSECT"
    intr_ptx[ints] = intersect_in

    return intersection_status, intersection_segment, intr_ptx, distance, msg, idx_mask


def _index_map(m1: int, m2: int) -> NDArray[np.floating]:
    """Build index mapping for edge-pair combinations.

    The mapping describes which original edge indices correspond to each flattened
    (edge_i, edge_j) pair.

    The implementation keeps original dtypes as in the legacy code. Note that
    ``y`` and ``idx_map`` are created without an explicit integer dtype, so they
    default to float.

    :param m1: Number of points/edges in the first hull.
    :param m2: Number of points/edges in the second hull.
    :returns: Index map array of shape ``(m1 * m2, 4)``.
    :rtype: numpy.ndarray
    """
    x = np.empty((m1, 2), dtype=INT)
    x[:, 0] = np.arange(m1)
    x[:, 1] = np.roll(x[:, 0], axis=0, shift=-1)

    y = np.empty((m2, 2))
    y[:, 0] = np.arange(m2)
    y[:, 1] = np.roll(y[:, 0], axis=0, shift=-1)

    idx_map = np.empty((m1 * m2, 4))
    idx_map[:, :2] = np.repeat(x, m2, axis=0)
    idx_map[:, 2:] = np.tile(y, (m1, 1))
    return idx_map


def polygon_hull_to_edges(hull: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert hull vertices into an edge array.

    :param hull: Hull vertices of shape ``(n, 2)``.
    :returns: Edge array of shape ``(n, 2, 2)`` where each row is ``[v_i, v_{i+1}]``.
    :rtype: numpy.ndarray
    """
    edges = np.zeros((hull.shape[0], 2, 2))
    edges[:, 0, :] = hull
    edges[:, 1, :] = np.roll(hull, axis=0, shift=-1)
    return edges


def _dpx_p1_p3matrix(
        p1_p3: NDArray[np.floating],
        dpoly1: NDArray[np.floating],
        dpoly2: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Build helper matrix used for determinant-based parameter solving.

    :param p1_p3: Vector differences ``p1 - p3`` for each edge pair.
    :param dpoly1: Direction vectors for the first edge in each pair.
    :param dpoly2: Direction vectors for the second edge in each pair.
    :returns: Stacked matrix used in ``multiple_determinants``.
    :rtype: numpy.ndarray
    """
    dpx_p1_p3matrix = np.empty((dpoly1.shape[0], 2, p1_p3.shape[1]), dtype=p1_p3.dtype)
    dpx_p1_p3matrix[:, 0, :] = dpoly2
    dpx_p1_p3matrix[:, 1, :] = p1_p3
    return dpx_p1_p3matrix
