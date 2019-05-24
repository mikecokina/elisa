import numpy as np

from pypex.base.conf import PRECISION, ROUND_PRECISION
from pypex.poly2d.intersection import sat
from pypex.poly2d.point import Point
from pypex.utils import det_2d, multiple_determinants


def intersection(p1, p2, p3, p4, in_touch=False, tol=PRECISION, round_tol=ROUND_PRECISION):
    """
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

    :param p1: numpy.array; first point of first segment
    :param p2: numpy.array; second point of first segment
    :param p3: numpy.array; first point of second segment
    :param p4: numpy.array; second point of second segment
    :param round_tol: int; consider two numbers as same if match up to `round_tol` decimal numbers
    :param tol: float; consider number as zero if smaller than 'tol'
    :param in_touch: bool
    :return: tuple

        0: intersection_status::

              False: parallel
              True:  intersection

        1: segment intersection (if segments share common point/s)::

              False:     no intersection
              True:      intersection between defined points or overlap
              numpy.nan: uknown

        2: intersection Point

        3: distance if parallel
        
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

        intersects, msg = (True, "OVERLAP") if abs(d) < tol else (False, "PARALLEL")
        int_in_segment = False if msg in ["PARALLEL"] \
            else sat.intersects(np.array([p1, p2]), np.array([p3, p4]), in_touch, round_tol)
        return intersects, int_in_segment, np.nan, d, msg

    # +0 because of negative zero (-0.0 is incorrect) formatting on output
    u = (det_2d([dp2, p1 - p3]) / d) + 0.
    v = (det_2d([dp1, p1 - p3]) / d) + 0.

    eval_method = np.less_equal if in_touch else np.less
    int_x, int_y = p1[0] + (u * dp1[0]), p1[1] + (u * dp1[1])
    int_segment = True if np.logical_and(eval_method(0.0, u), eval_method(u, 1.0)) and \
        np.logical_and(eval_method(0.0, v), eval_method(v, 1.0)) else False
    return True, int_segment, Point(int_x, int_y), np.nan, "INTERSECT"


def intersections(poly1, poly2, in_touch=False, tol=PRECISION, round_tol=ROUND_PRECISION):
    """
    Vectorised implementaion of lines intersection function. Compute intersections of all combination of supplied
    arrays of points which define convex polygon

    :param poly1: numpy.array; clokwise ordered numpy array of points
    :param poly2: numpy.array; clokwise ordered numpy array of points
    :param in_touch: bool; consider touch in one point as intersection
    :param tol: consider all numbers as zero when abs(number) < tol
    :param round_tol: consider two numbers as same if match up to round_tol deciaml numbers
    :return: tuple;
    """
    m1, _ = poly1.shape
    m2, _ = poly2.shape
    n1, n2 = 2, 2

    # mask to origin
    idx_mask = _index_map(m1, m2)

    intersection_status, intersection_segment = np.zeros(m1 * m2, dtype=np.bool), np.zeros(m1 * m2, dtype=np.bool)
    intr_ptx = np.full_like(np.empty((m1 * m2, 2), dtype=np.float), np.nan)
    distance = np.full_like(np.empty(m1 * m2, dtype=np.float), np.nan)

    msg = np.chararray(m1 * m2, itemsize=9)

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
    non_intersections = np.abs(determinants) < tol

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
        msg[non_intersections] = np.str('PARALLEL')
        overlaps = non_intersections.copy()

        overlaps[non_intersections] = np.abs(dist) < tol
        intersection_status[overlaps] = True

        if np.any(overlaps):
            # assume that in real life, there will neglible amount of parallel lines with zero distance (overlap lines)
            # so we can use for loop without any significant loose of performance
            poly1_comb_overlap = np.repeat(poly1_edges, m2, axis=0)[overlaps]
            poly2_comb_overlap = np.tile(poly2_edges, (m1, 1, 1))[overlaps]

            intersection_segment[overlaps] = np.array([sat.intersects(a, b, in_touch=in_touch, round_tol=round_tol)
                                                       for a, b in zip(poly1_comb_overlap, poly2_comb_overlap)])
            msg[overlaps] = 'OVERLAP'

    ints = ~non_intersections
    ok_dif_poly1, ok_dif_poly2 = corr_dpoly1[ints], corr_dpoly2[ints]
    ok_poly1_edges, ok_poly2_edges = np.repeat(poly1_edges, m2, axis=0)[ints], np.tile(poly2_edges, (m1, 1, 1))[ints]

    p1_p3 = ok_poly1_edges[:, 0, :] - ok_poly2_edges[:, 0, :]

    dp2_p1_p3matrix = _dpx_p1_p3matrix(p1_p3, ok_dif_poly1, ok_dif_poly2)
    dp1_p1_p3matrix = _dpx_p1_p3matrix(p1_p3, ok_dif_poly1, ok_dif_poly1)

    d = determinants[ints]
    u = (multiple_determinants(dp2_p1_p3matrix) / d) + 0.0
    v = (multiple_determinants(dp1_p1_p3matrix) / d) + 0.0

    eval_method = np.less_equal if in_touch else np.less
    intersect_in = ok_poly1_edges[:, 0, :] + (u[:, np.newaxis] * ok_dif_poly1)

    u_in_range = np.logical_and(eval_method(0.0, u), eval_method(u, 1.0))
    v_in_range = np.logical_and(eval_method(0.0, v), eval_method(v, 1.0))
    segments_intersection_status = np.logical_and(u_in_range, v_in_range)

    # fill output
    intersection_status[ints] = True
    intersection_segment[ints] = segments_intersection_status
    msg[ints] = 'INTERSECT'
    intr_ptx[ints] = intersect_in

    return intersection_status, intersection_segment, intr_ptx, distance, msg, idx_mask


def _index_map(m1, m2):
    x = np.empty((m1, 2), dtype=int)
    x[:, 0] = np.arange(m1)
    x[:, 1] = np.roll(x[:, 0], axis=0, shift=-1)

    y = np.empty((m2, 2))
    y[:, 0] = np.arange(m2)
    y[:, 1] = np.roll(y[:, 0], axis=0, shift=-1)

    idx_map = np.empty((m1 * m2, 4))
    idx_map[:, :2] = np.repeat(x, m2, axis=0)
    idx_map[:, 2:] = np.tile(y, (m1, 1))
    return idx_map


def polygon_hull_to_edges(hull: np.array):
    edges = np.zeros((hull.shape[0], 2, 2))
    edges[:, 0, :] = hull
    edges[:, 1, :] = np.roll(hull, axis=0, shift=-1)
    return edges


def _dpx_p1_p3matrix(p1_p3, dpoly1, dpoly2):
    dpx_p1_p3matrix = np.empty((dpoly1.shape[0], 2, p1_p3.shape[1]), dtype=p1_p3.dtype)
    dpx_p1_p3matrix[:, 0, :] = dpoly2
    dpx_p1_p3matrix[:, 1, :] = p1_p3
    return dpx_p1_p3matrix
