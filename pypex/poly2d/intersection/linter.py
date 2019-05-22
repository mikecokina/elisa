import numpy as np

from pypex.base.conf import PRECISION, ROUND_PRECISION
from pypex.poly2d.intersection import sat
from pypex.poly2d.point import Point
from pypex.utils import det_2d


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
        return int_segment, intersects, np.nan, d, msg

    # +0 because of negative zero (-0.0 is incorrect) formatting on output
    u = (det_2d([dp2, p1 - p3]) / d) + 0.
    v = (det_2d([dp1, p1 - p3]) / d) + 0.

    eval_method = np.less_equal if in_touch else np.less
    int_x, int_y = p1[0] + (u * dp1[0]), p1[1] + (u * dp1[1])
    int_segment = True if np.logical_and(eval_method(0.0, u), eval_method(u, 1.0)) and \
        np.logical_and(eval_method(0.0, v), eval_method(v, 1.0)) else False
    return True, int_segment, Point(int_x, int_y), np.nan, "INTERSECT"
