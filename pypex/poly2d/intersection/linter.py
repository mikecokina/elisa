import numpy as np

from pypex.base.conf import PRECISION, ROUND_PRECISION
from pypex.poly2d.intersection import sat
from pypex.poly2d.point import Point
from pypex.utils import det_2d
from copy import copy


def intersection(p1, p2, p3, p4, in_touch=False, tol=ROUND_PRECISION):
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
    :param tol: int; consider two numbers as same if match up to `tol` decimal numbers
    :param in_touch: bool
    :return: tuple

        0: intersection_status::

              False: parallel
              True:  intersection

        1: segment intersection (if segments share common point/s)::

              False:     no intersection
              True:      intersection between defined points or overlap
              numpy.nan: unknown

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
    if np.abs(d) < PRECISION:
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

        int_segment, msg = (True, "OVERLAP") if d == 0 else (False, "PARALLEL")
        intersects = False if msg in ["PARALLEL"] \
            else sat.intersects(np.array([p1, p2]), np.array([p3, p4]), in_touch, tol)
        # fixme: different order in return!!!!
        return int_segment, intersects, np.nan, d, msg

    # +0 because of negative zero (-0.0 is incorrect) formatting on output
    u = (det_2d([dp2, p1 - p3]) / d) + 0.
    v = (det_2d([dp1, p1 - p3]) / d) + 0.

    eval_method = np.less_equal if in_touch else np.less
    int_x, int_y = p1[0] + (u * dp1[0]), p1[1] + (u * dp1[1])
    int_segment = True if np.logical_and(eval_method(0.0, u), eval_method(u, 1.0)) and \
        np.logical_and(eval_method(0.0, v), eval_method(v, 1.0)) else False
    return True, int_segment, Point(int_x, int_y), np.nan, "INTERSECT"


def polygons_intersection(face, polygon, in_touch=False, tol=PRECISION):
    """
    calculates whether intersection occurs between every combination of face and polygon edges
    :param in_touch:
    :param face:
    :param polygon:
    :param tol:
    :return:
    """
    m1, n1 = face.shape
    m2, n2 = polygon.shape

    face_edges = np.empty((m1, 2, 2))
    face_edges[:, 0, :] = face
    face_edges[:, 1, :] = np.roll(face, axis=0, shift=-1)

    polygon_edges = np.empty((m2, 2, 2))
    polygon_edges[:, 0, :] = polygon
    polygon_edges[:, 1, :] = np.roll(polygon, axis=0, shift=-1)

    dface = face_edges[:, 1, :] - face_edges[:, 0, :]
    dpolygon = polygon_edges[:, 1, :] - polygon_edges[:, 0, :]

    corr_dface = np.repeat(dface, m2, axis=0)
    corr_dpolygon = np.tile(dpolygon, (m1, 1))

    matrix = np.empty((m1 * m2, n1, n2))
    matrix[:, 0, :] = corr_dface
    matrix[:, 1, :] = corr_dpolygon

    determinants = multiple_determinants(matrix)

    intersection_status = np.empty(m1 * m2, dtype=np.bool)
    intersection_segment = np.empty(m1 * m2, dtype=np.bool)
    msg = np.chararray(m1 * m2, itemsize=9)
    intr_ptx = np.empty((m1 * m2, 2), dtype=np.float)
    distance = np.empty(m1 * m2, dtype=np.float)

    non_intersections = np.abs(determinants) < tol
    if non_intersections.any():
        problem_face = np.repeat(face, m2, axis=0)[non_intersections]
        problem_dface = np.repeat(dface, m2, axis=0)[non_intersections]
        a1, b1 = -problem_dface[:, 1], problem_dface[:, 0]

        face_dface = np.empty((problem_face.shape[0], 2, problem_face.shape[1]), dtype=dface.dtype)
        face_dface[:, 0, :] = problem_face
        face_dface[:, 1, :] = problem_dface
        c1 = multiple_determinants(face_dface)

        problem_polygon_edges = np.tile(polygon_edges, (face.shape[0], 1, 1))[non_intersections]
        c2 = -(a1 * problem_polygon_edges[:, 1, 0] + b1 * problem_polygon_edges[:, 1, 1])
        d = np.abs(c2 - c1) / (np.sqrt(np.power(a1, 2) + np.sqrt(np.power(b1, 2))))

        intersection_status[non_intersections] = False
        intersection_segment[non_intersections] = False
        intr_ptx[non_intersections] = np.NaN
        distance[non_intersections] = d
        msg[non_intersections] = np.str('PARALLEL')

        overlaps = copy(non_intersections)
        overlaps[non_intersections] = d < tol
        intersection_status[overlaps] = True
        # here should be the actual segment but nah...
        intersection_segment[overlaps] = True
        msg[overlaps] = 'OVERLAP'

    intersections = ~non_intersections
    ok_dface = np.repeat(dface, m2, axis=0)[intersections]
    ok_dpolygon = np.tile(dpolygon, (m1, 1))[intersections]
    ok_face_edges = np.repeat(face_edges, m2, axis=0)[intersections]
    ok_polygon_edges = np.tile(polygon_edges, (m1, 1, 1))[intersections]

    p1_p3 = ok_face_edges[:, 0, :] - ok_polygon_edges[:, 0, :]

    dp2_p1_p3matrix = np.empty((ok_dface.shape[0], 2, p1_p3.shape[1]), dtype=dface.dtype)
    dp2_p1_p3matrix[:, 0, :] = ok_dpolygon
    dp2_p1_p3matrix[:, 1, :] = p1_p3

    dp1_p1_p3matrix = np.empty((ok_dface.shape[0], 2, p1_p3.shape[1]), dtype=dface.dtype)
    dp1_p1_p3matrix[:, 0, :] = ok_dface
    dp1_p1_p3matrix[:, 1, :] = p1_p3

    d = determinants[intersections]
    u = (multiple_determinants(dp2_p1_p3matrix) / d) + 0.0
    v = (multiple_determinants(dp1_p1_p3matrix) / d) + 0.0

    eval_method = np.less_equal if in_touch else np.less
    intr = ok_face_edges[:, 0, :] + (u[:, np.newaxis] * ok_dface)

    u_in_range = np.logical_and(eval_method(0.0, u), eval_method(u, 1.0))
    v_in_range = np.logical_and(eval_method(0.0, v), eval_method(v, 1.0))

    true_intersections = np.logical_and(u_in_range, v_in_range)

    intersection_status[intersections] = True
    intersection_segment[intersections] = true_intersections
    intr_ptx[intersections] = np.NaN

    intr_candidates = intr_ptx[intersections]
    intr_candidates[true_intersections] = intr[true_intersections]
    intr_ptx[intersections] = intr_candidates

    distance[intersections] = np.NaN

    msg[intersections] = np.str('MISSED')
    intr_candidates = msg[intersections]
    intr_candidates[true_intersections] = 'INTERSECT'
    msg[intersections] = intr_candidates

    return intersection_status, intersection_segment, intr_ptx, distance, msg


def multiple_determinants(matrix):
    """
    calculates 2D determinant on every level of given 3D matrix

    :param matrix: np.array (Nx2x2), where i-th slice looks like:
                                    [[xi1, yi1],
                                     [xi2, yi2]]
    :return: np.array - N-dim vector where each element is 2D determinant of 2 2D vectors stored on given level in
                        `matrix'
    """
    return matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]
