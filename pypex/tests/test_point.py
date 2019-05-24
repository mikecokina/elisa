import unittest

from pypex.poly2d.polygon import Polygon
from pypex.poly2d import point


def asser_points_equal(ps1, ps2):
    ps1 = sorted(ps1, key=lambda k: (k.x, k.y))
    ps2 = sorted(ps2, key=lambda k: (k.x, k.y))
    for p1, p2 in zip(ps1, ps2):
        assert p1, p2


class BaseTestCase(unittest.TestCase):
    def test_is_point_in_polygon_positive(self):
        poly = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        p = point.Point(x=0.5, y=0.5)
        obtained = point.is_point_in_polygon(p, poly)
        self.assertTrue(obtained)

    def test_is_point_in_polygon_negative(self):
        poly = Polygon([[-1, -1], [1, 0], [1, 1], [0.5, 1.9], [1, 1]])
        p = point.Point(x=-10, y=10)
        obtained = point.is_point_in_polygon(p, poly)
        self.assertFalse(obtained)

    def test_is_point_in_polygon_onedge(self):
        poly = Polygon([[0, 0], [1, 0], [1, 1]])
        p = point.Point(x=0.5, y=0)
        obtained = point.is_point_in_polygon(p, poly)
        self.assertFalse(obtained)


class PointTestCase(unittest.TestCase):
    def test_is_inside_polygon_positive(self):
        poly = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        p = point.Point(x=0.5, y=0.5)
        self.assertTrue(p.is_inside_polygon(poly))

    def test_set(self):
        points = [point.Point(0.1, 0.1), point.Point(0.1, 0.1)]
        points_set = point.Point.set(points)
        self.assertEqual(points_set, [point.Point(0.1, 0.1)])

        points = [point.Point(-2, -2), point.Point(-1, -1)]
        points_set = point.Point.set(points)
        asser_points_equal(points_set, points)

        # return Point(0.001, 0.0021) since 0.0022 will be rounded down to 0.002 and 0.0021 also to the 0.002
        points = [point.Point(0.001, 0.0021), point.Point(0.001, 0.0022)]
        points_set = point.Point.set(points, round_tol=3)
        asser_points_equal([point.Point(0.001, 0.0021)], points_set)

        # return all since 0.0025 will round up to 0.003
        points = [point.Point(0.001, 0.0021), point.Point(0.001, 0.0025)]
        points_set = point.Point.set(points, round_tol=3)
        asser_points_equal(points, points_set)
