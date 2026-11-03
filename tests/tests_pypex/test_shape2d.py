import numpy as np
from numpy.testing import assert_array_equal

from elisa.pypex.poly2d.intersection.sat import intersects
from elisa.pypex.poly2d.line import Line
from elisa.pypex.poly2d.point import Point
from elisa.pypex.poly2d.polygon import Polygon
from tests.utils import ElisaTestCase

parray = np.array([[0.5, 1.5], [0, 0], [1, 0], [0, 1], [1, 1]])


class Shape2DTestCase(ElisaTestCase):
    __parray__ = np.array([[0.5, 1.5], [0, 0], [1, 0], [0, 1], [1, 1]])

    def setUp(self) -> None:
        super().setUp()
        self.poly = Polygon(hull=self.__parray__)

    def test_sort_clockwise(self) -> None:
        expected = [[0, 0], [1, 0], [1, 1], [0.5, 1.5], [0, 1]]
        obtained = self.poly.sort_clockwise()
        assert_array_equal(expected, obtained)

    def test_sat_intersection_negative_case(self) -> None:
        # not intersection example:
        faces = [[[0.0, 0.5], [1.0, 0.5], [0.0, 1.0]], [[0.0, 0.0], [1.0, 0.4], [0.0, -0.7]]]
        assert not (intersects(faces[0], faces[1], touch_is_separated=False))
        assert not (intersects(faces[0], faces[1], touch_is_separated=True))

        # not intersection example:
        faces = [[[0.0, 0.5], [1.0, 0.5], [0.0, 1.0]], [[0.0, 0.0], [1.0, 0.3], [0.0, 0.3]]]
        assert not (intersects(faces[0], faces[1], touch_is_separated=False))
        assert not (intersects(faces[0], faces[1], touch_is_separated=True))

    def test_sat_intersection_positive_case(self) -> None:
        faces = [[[0.0, 0.5], [1.0, 0.5], [0.0, 1.0]], [[0.0, 2.0], [1.0, 0.5], [0.0, 0.5]]]
        assert intersects(faces[0], faces[1], touch_is_separated=False)
        assert intersects(faces[0], faces[1], touch_is_separated=True)

        faces = [[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], [[-0.9, -0.9], [-3.0, 3.5], [9.5, 1.0]]]
        assert intersects(faces[0], faces[1], touch_is_separated=False)
        assert intersects(faces[0], faces[1], touch_is_separated=True)

        faces = [[[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]], [[0.5, 0.5], [4.0, 0.5], [4.0, 3.0], [0.5, 3.0]]]
        assert intersects(faces[0], faces[1], touch_is_separated=False)
        assert intersects(faces[0], faces[1], touch_is_separated=True)

        faces = [[[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]], [[0.5, 0.5], [4.0, 0.5], [4.0, 3.0]]]
        assert intersects(faces[0], faces[1], touch_is_separated=False)
        assert intersects(faces[0], faces[1], touch_is_separated=True)

    def test_sat_intersection_onedge_case(self) -> None:
        faces = [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[-1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
        assert not (intersects(faces[0], faces[1], touch_is_separated=True))
        assert intersects(faces[0], faces[1], touch_is_separated=False)

    def test_sat_intersection_touch_case(self) -> None:
        faces = [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [-0.1, -0.5], [-0.5, -0.5]]]

        assert not (intersects(faces[0], faces[1], touch_is_separated=True))
        assert intersects(faces[0], faces[1], touch_is_separated=False)

    def test_line_sat_intersect(self) -> None:
        line1 = np.array([[-1.0, -1.0], [0.3, 0.3]])
        line2 = np.array([[0.2, 0.2], [1.3, 1.3]])
        assert intersects(line1, line2, touch_is_separated=False)
        assert intersects(line1, line2, touch_is_separated=True)

        line1 = np.array([[-1.0, -1.0], [0.3, 0.3]])
        line2 = np.array([[0.3, 0.3], [1.3, 1.3]])
        assert intersects(line1, line2, touch_is_separated=False)
        assert not (intersects(line1, line2, touch_is_separated=True))

        line1 = np.array([[1.0, 1.0], [1.0, 2.0]])
        line2 = np.array([[1.0, 2.0], [1.0, 13.0]])
        assert intersects(line1, line2, touch_is_separated=False)
        assert not (intersects(line1, line2, touch_is_separated=True))

        line1 = np.array([[1.0, 1.0], [3.1, 1.0]])
        line2 = np.array([[3.2, 1.0], [5.0, 1.0]])
        assert not (intersects(line1, line2, touch_is_separated=False))
        assert not (intersects(line1, line2, touch_is_separated=True))


class Line2DTestCase(ElisaTestCase):
    def test_intersects_negative(self) -> None:
        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[2, 0], [4, 0]]))
        assert not (line1.intersects(line2))

        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[1, 0], [4, 0]]))
        assert not (line1.intersects(line2, touch_is_separated=False))
        assert not (line1.intersects(line2, touch_is_separated=True))

        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[1, 0], [4, 1]]))
        assert not (line1.intersects(line2, touch_is_separated=True))

    def test_intersects_positive(self) -> None:
        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[0.5, -0.5], [0.5, 1]]))
        assert line1.intersects(line2)

        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[1.0, 0.0], [0.5, 1.0]]))
        assert line1.intersects(line2, touch_is_separated=False)

    def test_intersection_positive(self) -> None:
        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[0.5, -0.5], [0.5, 1]]))
        obtained = line1.intersection(line2)
        expected = Point(0.5, 0)
        assert expected == obtained

        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[1.0, 0.0], [0.5, 1.0]]))
        obtained = line1.intersection(line2, touch_is_separated=False)
        expected = Point(1, 0)
        assert expected == obtained

    def test_intersection_negative(self) -> None:
        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[2, 0], [4, 0]]))
        obtained = line1.intersection(line2)
        assert (obtained) is None

        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[1, 0], [4, 0]]))
        obtained = line1.intersection(line2, touch_is_separated=False)
        assert (obtained) is None
        obtained = line1.intersection(line2, touch_is_separated=True)
        assert (obtained) is None

        line1 = Line(np.array([[0, 0], [1, 0]]))
        line2 = Line(np.array([[1, 0], [4, 1]]))
        obtained = line1.intersection(line2, touch_is_separated=True)
        assert (obtained) is None


class PolygonIntersectionObjectTestCase(ElisaTestCase):
    @staticmethod
    def test_polygons_intersection_positive_standard() -> None:
        faces = [[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], [[0.5, 0.2], [0.0, -1.5], [1.3, -1.3]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)

        expected = [[0.441, 0.0], [0.607, 0.0], [0.5, 0.2]]
        obtained = [[round(p[0], 3), round(p[1], 3)] for p in poly3.hull]
        assert_array_equal(expected, obtained)

        faces = [[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], [[0.5, 0.2], [0.7, 0.3], [0.6, -1.8]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)

        expected = [[0.51, 0.0], [0.686, 0.0], [0.7, 0.3], [0.5, 0.2]]
        obtained = [[round(p[0], 3), round(p[1], 3)] for p in poly3.hull]
        assert_array_equal(expected, obtained)

        faces = [[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], [[0.0, 0.5], [0.9, 0.8], [0.6, -1.8]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)

        expected = [[0.086, 0.171], [0.130, 0.0], [0.808, 0.0], [0.844, 0.312], [0.643, 0.714], [0.3, 0.6]]
        obtained = [[round(p[0], 3), round(p[1], 3)] for p in poly3.hull]
        assert_array_equal(expected, obtained)

    def test_polygons_intersection_positive_negative(self) -> None:
        faces = [[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], [[0.5, -0.2], [0.0, -1.5], [1.3, -1.3]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)
        assert (poly3) is None

        faces = [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)
        assert (poly3) is None

    def test_polygons_intersection_positive_overlap(self) -> None:
        faces = [[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], [[0.5, 0.2], [0.3, 0.5], [0.6, 0.6]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)
        assert poly2 == poly3

    def test_polygons_intersection_positive_patologic(self) -> None:
        # all equal
        faces = [[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)
        assert poly1 == poly3

        # some edges in overlap
        faces = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [[0.5, 0.0], [1.5, 0.0], [1.5, 2.0], [0.5, 2.0]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        poly3 = poly1.intersection(poly2)

        expected = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
        obtained = [[round(p[0], 3), round(p[1], 3)] for p in poly3.hull]
        assert_array_equal(obtained, expected)


class PolygonSurfaceAreaObjectTestCase(ElisaTestCase):
    def test_surface_area(self) -> None:
        faces = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]

        poly1 = Polygon(faces[0])
        poly2 = Polygon(faces[1])
        area1 = poly1.surface_area()
        area2 = poly2.surface_area()

        assert ([area1, area2]) == ([0.5, 1.0])
