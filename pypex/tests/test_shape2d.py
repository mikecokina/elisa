import numpy as np
import unittest

from numpy.testing import assert_array_equal
from pypex.poly2d.polygon import Polygon
from pypex.poly2d.line import Line
from pypex.poly2d.point import Point
from pypex.poly2d.intersection.sat import intersects

parray = np.array([[0.5, 1.5], [0, 0], [1, 0], [0, 1], [1, 1]])


class Shape2DTestCase(unittest.TestCase):
    __parray__ = np.array([[0.5, 1.5], [0, 0], [1, 0], [0, 1], [1, 1]])

    def setUp(self):
        self.poly = Polygon(hull=self.__parray__)

    def test_sort_clockwise(self):
        expected = [[0, 0], [1, 0], [1, 1], [0.5, 1.5], [0, 1]]
        obtained = self.poly.sort_clockwise()
        assert_array_equal(expected, obtained)

    def test_sat_intersection_negative_case(self):
        # not intersection example:
        faces = [[[0.0, 0.5],
                  [1.0, 0.5],
                  [0.0, 1.0]],
                 [[0.0, 0.0],
                  [1.0, 0.4],
                  [0.0, -.7]]]
        self.assertFalse(intersects(faces[0], faces[1], in_touch=True))
        self.assertFalse(intersects(faces[0], faces[1], in_touch=False))

        # not intersection example:
        faces = [[[0.0, 0.5],
                  [1.0, 0.5],
                  [0.0, 1.0]],
                 [[0.0, 0.0],
                  [1.0, 0.3],
                  [0.0, 0.3]]]
        self.assertFalse(intersects(faces[0], faces[1], in_touch=True))
        self.assertFalse(intersects(faces[0], faces[1], in_touch=False))

    def test_sat_intersection_positive_case(self):
        faces = [[[0.0, 0.5],
                  [1.0, 0.5],
                  [0.0, 1.0]],
                 [[0.0, 2.0],
                  [1.0, 0.5],
                  [0.0, 0.5]]]
        self.assertTrue(intersects(faces[0], faces[1], in_touch=True))
        self.assertTrue(intersects(faces[0], faces[1], in_touch=False))

        faces = [[[0.0, 0.0],
                  [2.0, 0.0],
                  [0.0, 2.0]],
                 [[-.9, -.9],
                  [-3., 3.5],
                  [9.5, 1.0]]]
        self.assertTrue(intersects(faces[0], faces[1], in_touch=True))
        self.assertTrue(intersects(faces[0], faces[1], in_touch=False))

        faces = [[[0.0, 0.5],
                  [1.0, 0.5],
                  [1.0, 1.0],
                  [0.0, 1.0]],
                 [[0.5, 0.5],
                  [4.0, 0.5],
                  [4.0, 3.0],
                  [0.5, 3.0]]]
        self.assertTrue(intersects(faces[0], faces[1], in_touch=True))
        self.assertTrue(intersects(faces[0], faces[1], in_touch=False))

        faces = [[[0.0, 0.5],
                  [1.0, 0.5],
                  [1.0, 1.0],
                  [0.0, 1.0]],
                 [[0.5, 0.5],
                  [4.0, 0.5],
                  [4.0, 3.0]]]
        self.assertTrue(intersects(faces[0], faces[1], in_touch=True))
        self.assertTrue(intersects(faces[0], faces[1], in_touch=False))

    def test_sat_intersection_onedge_case(self):
        faces = [[[0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]],
                 [[-1., 1.0],
                  [0.0, 1.0],
                  [0.0, 0.0]]]
        self.assertFalse(intersects(faces[0], faces[1], in_touch=False))
        self.assertTrue(intersects(faces[0], faces[1], in_touch=True))

    def test_sat_intersection_touch_case(self):
        faces = [[[0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]],
                 [[0.0, 0.0],
                  [-.1, -.5],
                  [-.5, -.5]]]

        self.assertFalse(intersects(faces[0], faces[1], in_touch=False))
        self.assertTrue(intersects(faces[0], faces[1], in_touch=True))

    def test_line_sat_intersect(self):
        line1 = np.array([[-1., -1.], [0.3, 0.3]])
        line2 = np.array([[0.2, 0.2], [1.3, 1.3]])
        self.assertTrue(intersects(line1, line2, in_touch=True))
        self.assertTrue(intersects(line1, line2, in_touch=False))

        line1 = np.array([[-1., -1.], [0.3, 0.3]])
        line2 = np.array([[0.3, 0.3], [1.3, 1.3]])
        self.assertTrue(intersects(line1, line2, in_touch=True))
        self.assertFalse(intersects(line1, line2, in_touch=False))

        line1 = np.array([[1., 1.], [1., 2.]])
        line2 = np.array([[1., 2.], [1., 13.]])
        self.assertTrue(intersects(line1, line2, in_touch=True))
        self.assertFalse(intersects(line1, line2, in_touch=False))

        line1 = np.array([[1., 1.], [3.1, 1.]])
        line2 = np.array([[3.2, 1.], [5., 1.]])
        self.assertFalse(intersects(line1, line2, in_touch=True))
        self.assertFalse(intersects(line1, line2, in_touch=False))


class Line2DTestCase(unittest.TestCase):
    def test_intersects_negative(self):
        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[2, 0], [4, 0]]))
        self.assertFalse(line1.intersects(line2))

        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[1, 0], [4, 0]]))
        self.assertFalse(line1.intersects(line2, in_touch=True))
        self.assertFalse(line1.intersects(line2, in_touch=False))

        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[1, 0], [4, 1]]))
        self.assertFalse(line1.intersects(line2, in_touch=False))

    def test_intersects_positive(self):
        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[0.5, -0.5], [0.5, 1]]))
        self.assertTrue(line1.intersects(line2))

        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[1.0, 0.0], [0.5, 1.]]))
        self.assertTrue(line1.intersects(line2, in_touch=True))

    def test_intersection_positive(self):
        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[0.5, -0.5], [0.5, 1]]))
        obtained = line1.intersection(line2)
        expected = Point(0.5, 0)
        self.assertTrue(expected == obtained)

        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[1.0, 0.0], [0.5, 1.]]))
        obtained = line1.intersection(line2, in_touch=True)
        expected = Point(1, 0)
        self.assertTrue(expected == obtained)

    def test_intersection_negative(self):
        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[2, 0], [4, 0]]))
        obtained = line1.intersection(line2)
        self.assertIsNone(obtained)

        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[1, 0], [4, 0]]))
        obtained = line1.intersection(line2, in_touch=True)
        self.assertIsNone(obtained)
        obtained = line1.intersection(line2, in_touch=False)
        self.assertIsNone(obtained)

        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[1, 0], [4, 1]]))
        obtained = line1.intersection(line2, in_touch=False)
        self.assertIsNone(obtained)


class PolygonIntersectionObjectTestCase(unittest.TestCase):
    @staticmethod
    def test_polygons_intersection_positive_standard():
        faces = [[[0.0, 0.0],
                  [1.0, 0.0],
                  [0.5, 1.0]],
                 [[0.5, 0.2],
                  [0.0, -1.5],
                  [1.3, -1.3]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)

        expected = [[0.441, 0.], [0.607, 0.], [0.5, 0.2]]
        obtained = [[round(p[0], 3), round(p[1], 3)] for p in poly3.hull]
        assert_array_equal(expected, obtained)

        faces = [[[0.0, 0.0],
                  [1.0, 0.0],
                  [0.5, 1.0]],
                 [[0.5, 0.2],
                  [0.7, 0.3],
                  [0.6, -1.8]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)

        expected = [[0.51, 0.], [0.686, 0.], [0.7, 0.3], [0.5, 0.2]]
        obtained = [[round(p[0], 3), round(p[1], 3)] for p in poly3.hull]
        assert_array_equal(expected, obtained)

        faces = [[[0.0, 0.0],
                  [1.0, 0.0],
                  [0.5, 1.0]],
                 [[0.0, 0.5],
                  [0.9, 0.8],
                  [0.6, -1.8]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)

        expected = [[0.086, 0.171], [0.130, 0.], [0.808, 0.],
                    [0.844, 0.312], [0.643, 0.714], [0.3, 0.6]]
        obtained = [[round(p[0], 3), round(p[1], 3)] for p in poly3.hull]
        assert_array_equal(expected, obtained)

    def test_polygons_intersection_positive_negative(self):
        faces = [[[0.0, 0.0],
                  [1.0, 0.0],
                  [0.5, 1.0]],
                 [[0.5, -0.2],
                  [0.0, -1.5],
                  [1.3, -1.3]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)
        self.assertIsNone(poly3)

        faces = [[[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0],
                  [1.0, 0.0]],
                 [[1.0, 0.0],
                  [2.0, 0.0],
                  [2.0, 1.0],
                  [1.0, 1.0]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)
        self.assertIsNone(poly3)

    def test_polygons_intersection_positive_overlap(self):
        faces = [[[0.0, 0.0],
                  [1.0, 0.0],
                  [0.5, 1.0]],
                 [[0.5, 0.2],
                  [0.3, 0.5],
                  [0.6, 0.6]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)
        self.assertTrue(poly2 == poly3)

    def test_polygons_intersection_positive_patologic(self):
        # all equal
        faces = [[[0.0, 0.0],
                  [1.0, 0.0],
                  [0.5, 1.0]],
                 [[0.0, 0.0],
                  [1.0, 0.0],
                  [0.5, 1.0]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)
        self.assertTrue(poly1 == poly3)

        # some edges in overlap
        faces = [[[0.0, 0.0],
                  [1.0, 0.0],
                  [1.0, 1.0],
                  [0.0, 1.0]],

                 [[0.5, 0.0],
                  [1.5, 0.0],
                  [1.5, 2.0],
                  [0.5, 2.0]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)

        expected = [[0.5, 0.], [1., 0.], [1., 1.], [0.5, 1.]]
        obtained = [[round(p[0], 3), round(p[1], 3)] for p in poly3.hull]
        assert_array_equal(obtained, expected)


class PolygonSurfaceAreaObjectTestCase(unittest.TestCase):

    def test_surface_area(self):
        faces = [[[0.0, 0.0],
                  [1.0, 0.0],
                  [1.0, 1.0]],

                 [[0.0, 0.0],
                  [1.0, 0.0],
                  [1.0, 1.0],
                  [0.0, 1.0]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        area1 = poly1.surface_area()
        area2 = poly2.surface_area()

        self.assertEqual([0.5, 1.0], [area1, area2])
