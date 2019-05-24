"""
implementation of SAP algorithm in 2d
detection of collision of two faces
if will also return false if faces are just in touch

overlap is handled as no intersection at all
"""
import numpy as np

from pypex.poly2d import projection
from pypex.base.conf import ROUND_PRECISION


def separating_axis_theorem(poly1, poly2, in_touch=False, round_tol=ROUND_PRECISION):
    """

    :param round_tol: int
    :param in_touch: bool
    :param poly1: numpy.array;
    :param poly2: numpy.array
    :return: bool
    """
    # number of vertices in each polygon
    nv1, nv2 = len(poly1), len(poly2)

    if (nv1 < 3) or (nv2 < 3):
        raise ValueError("invalid polygon shape, supported at least polygon with 3 corners")

    edges = np.concatenate(([[poly1[i], poly1[i + 1]] for i in range(-1, nv1 - 1)],
                            [[poly2[i], poly2[i + 1]] for i in range(-1, nv2 - 1)]), axis=0)
    for edge in edges:
        # tangent vector of testing line
        tangent = (edge[1] - edge[0]) / np.linalg.norm(edge[1] - edge[0])
        # normal vector of testing line
        normal = -tangent[1], tangent[0]

        # we flip tangent and normal vector (all vertices of all polygons will be projected on line defined by normal
        # vector isntead of tangential); just convinience
        tangent, normal = normal, tangent

        # projection of each vertex on testing line (basically it is going to transform coo of each vertex
        # to coordinate system, where tangential vector represents new `x` axis)

        projection_poly1 = np.array([projection.cartesian_to_vectors_defined(
            tangent, normal, projection.projection(vertex, tangent)) for vertex in poly1])
        projection_poly2 = np.array([projection.cartesian_to_vectors_defined(
            tangent, normal, projection.projection(vertex, tangent)) for vertex in poly2])

        # new x coo (y should and has to be zero now)
        projection_poly1_x, projection_poly2_x = projection_poly1.T[0], projection_poly2.T[0]

        # maximal length projected of each face
        projection_edge1, projection_edge2 = [round(projection_poly1_x.min(), round_tol),
                                              round(projection_poly1_x.max(), round_tol)], \
                                             [round(projection_poly2_x.min(), round_tol),
                                              round(projection_poly2_x.max(), round_tol)]

        projection_edge = [projection_edge1, projection_edge2]
        projection_edge.sort(key=lambda x: x[0])

        # if intervals connected in point or separated return True
        eval_method = np.less if in_touch else np.less_equal
        if eval_method(projection_edge[0][1], projection_edge[1][0]):
            return True
    return False


def separating_axis_theorem_line_adapt(line1, line2, in_touch=False, round_tol=ROUND_PRECISION):
    """
    test overlap of segments that are situated on the same line

    :param round_tol: int
    :param in_touch: bool
    :param line1: numpy.array
    :param line2: numpy.array
    :return: bool
    """
    tangent = (line1[1] - line1[0]) / np.linalg.norm(line1[1] - line1[0])
    normal = -tangent[1], tangent[0]

    projection_line1 = np.array([projection.cartesian_to_vectors_defined(
        tangent, normal, projection.projection(vertex, tangent)) for vertex in line1])
    projection_line2 = np.array([projection.cartesian_to_vectors_defined(
        tangent, normal, projection.projection(vertex, tangent)) for vertex in line2])
    projection_x = [np.round(projection_line1.T[0], round_tol), np.round(projection_line2.T[0], round_tol)]
    projection_x.sort(key=lambda x: x[0])

    eval_method = np.less if in_touch else np.less_equal
    if eval_method(projection_x[0][1], projection_x[1][0]):
        return True
    return False


def intersects(poly1, poly2, in_touch=False, round_tol=ROUND_PRECISION):
    """
    Resolve whether two objects intersetcs.

    In case of two segments it will resolve correctly only segments situated on tme same line.
    It is suppose to resolve whether these two lines are separated or overlapped/touched in point.

    :param round_tol: int; consider two numbers as same if match up to `round_tol` decimal numbers
    :param in_touch: bool
    :param poly1: numpy.array; convex polygon
    :param poly2: numpy.array; convex polygon
    :return: bool
    """
    if (len(poly1) == 2) and (len(poly2) == 2):
        return not separating_axis_theorem_line_adapt(poly1, poly2, in_touch, round_tol)
    return not separating_axis_theorem(poly1, poly2, in_touch, round_tol)
