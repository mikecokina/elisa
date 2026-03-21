import unittest
import numpy as np

from elisa.pypex.poly2d.polygon import Polygon
from elisa.pypex.poly2d import point


def asser_points_equal(ps1, ps2):
    ps1 = sorted(ps1, key=lambda k: (k.x, k.y))
    ps2 = sorted(ps2, key=lambda k: (k.x, k.y))
    for p1, p2 in zip(ps1, ps2):
        assert p1, p2


class LineSideTestCase(unittest.TestCase):
    """Test cases for _line_side function."""

    def test_line_side_same_side_positive(self):
        """Test points on same side of line."""
        p1 = np.array([0, 1])
        p2 = np.array([1, 1])
        a = np.array([0, 0])
        b = np.array([1, 0])
        result = point._line_side(p1, p2, a, b)
        self.assertGreater(result, 0)

    def test_line_side_opposite_sides(self):
        """Test points on opposite sides of line."""
        p1 = np.array([0, 1])
        p2 = np.array([0, -1])
        a = np.array([0, 0])
        b = np.array([1, 0])
        result = point._line_side(p1, p2, a, b)
        self.assertLess(result, 0)

    def test_line_side_with_lists(self):
        """Test _line_side accepts list inputs."""
        p1 = [0, 1]
        p2 = [1, 1]
        a = [0, 0]
        b = [1, 0]
        result = point._line_side(p1, p2, a, b)
        self.assertGreater(result, 0)


class SameSideTestCase(unittest.TestCase):
    """Test cases for same_side function."""

    def test_same_side_true(self):
        """Test points on the same side of a line."""
        p1 = np.array([0, 1])
        p2 = np.array([1, 1])
        a = np.array([0, 0])
        b = np.array([1, 0])
        self.assertTrue(point.same_side(p1, p2, a, b))

    def test_same_side_false(self):
        """Test points on opposite sides of a line."""
        p1 = np.array([0, 1])
        p2 = np.array([0, -1])
        a = np.array([0, 0])
        b = np.array([1, 0])
        self.assertFalse(point.same_side(p1, p2, a, b))

    def test_same_side_with_lists(self):
        """Test same_side accepts list inputs."""
        p1 = [0, 1]
        p2 = [1, 1]
        a = [0, 0]
        b = [1, 0]
        self.assertTrue(point.same_side(p1, p2, a, b))

    def test_same_side_collinear(self):
        """Test points on the line itself."""
        p1 = np.array([0.5, 0])
        p2 = np.array([0.75, 0])
        a = np.array([0, 0])
        b = np.array([1, 0])
        self.assertTrue(point.same_side(p1, p2, a, b))


class BaseTestCase(unittest.TestCase):
    """Test cases for is_point_in_polygon function."""

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

    def test_is_point_in_polygon_at_vertex(self):
        """Test point at polygon vertex."""
        poly = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        p = point.Point(x=0, y=0)
        obtained = point.is_point_in_polygon(p, poly)
        self.assertFalse(obtained)

    def test_is_point_in_polygon_invalid_polygon(self):
        """Test with invalid polygon (less than 3 points)."""
        poly = Polygon([[0, 0], [1, 0]])
        p = point.Point(x=0.5, y=0.5)
        with self.assertRaises(ValueError):
            point.is_point_in_polygon(p, poly)


class PointInitTestCase(unittest.TestCase):
    """Test cases for Point initialization."""

    def test_point_init(self):
        """Test Point initialization."""
        p = point.Point(1.5, 2.5)
        self.assertEqual(p.x, 1.5)
        self.assertEqual(p.y, 2.5)

    def test_point_init_integers(self):
        """Test Point initialization with integers."""
        p = point.Point(1, 2)
        self.assertEqual(p.x, 1)
        self.assertEqual(p.y, 2)

    def test_point_init_negative(self):
        """Test Point initialization with negative values."""
        p = point.Point(-1.5, -2.5)
        self.assertEqual(p.x, -1.5)
        self.assertEqual(p.y, -2.5)


class PointStringRepresentationTestCase(unittest.TestCase):
    """Test cases for Point string representations."""

    def test_point_str(self):
        """Test Point __str__ method."""
        p = point.Point(1.5, 2.5)
        self.assertEqual(str(p), "Point [1.5, 2.5]")

    def test_point_repr(self):
        """Test Point __repr__ method."""
        p = point.Point(1.5, 2.5)
        self.assertEqual(repr(p), "Point [1.5, 2.5]")

    def test_point_str_with_integers(self):
        """Test Point __str__ with integer coordinates."""
        p = point.Point(1, 2)
        self.assertEqual(str(p), "Point [1, 2]")

    def test_point_str_with_negative(self):
        """Test Point __str__ with negative coordinates."""
        p = point.Point(-1.5, -2.5)
        self.assertEqual(str(p), "Point [-1.5, -2.5]")


class PointEqualityTestCase(unittest.TestCase):
    """Test cases for Point equality and hashing."""

    def test_point_equality_true(self):
        """Test equality of two identical points."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(1.5, 2.5)
        self.assertEqual(p1, p2)

    def test_point_equality_false(self):
        """Test inequality of two different points."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(1.5, 2.6)
        self.assertNotEqual(p1, p2)

    def test_point_hash_same_points(self):
        """Test hash of identical points."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(1.5, 2.5)
        self.assertEqual(hash(p1), hash(p2))

    def test_point_hash_different_points(self):
        """Test hash of different points."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(1.5, 2.6)
        # Different points should (likely) have different hashes
        self.assertNotEqual(hash(p1), hash(p2))

    def test_point_hashable_in_set(self):
        """Test that Point can be used in sets."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(1.5, 2.5)
        p3 = point.Point(2.5, 3.5)
        point_set = {p1, p2, p3}
        self.assertEqual(len(point_set), 2)

    def test_point_hashable_in_dict(self):
        """Test that Point can be used as dict key."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(1.5, 2.5)
        point_dict = {p1: "first"}
        point_dict[p2] = "second"
        self.assertEqual(len(point_dict), 1)
        self.assertEqual(point_dict[p1], "second")


class PointArithmeticTestCase(unittest.TestCase):
    """Test cases for Point arithmetic operations."""

    def test_point_add(self):
        """Test Point addition."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(0.5, 0.5)
        result = p1 + p2
        self.assertEqual(result.x, 2.0)
        self.assertEqual(result.y, 3.0)
        self.assertIsInstance(result, point.Point)

    def test_point_add_negative(self):
        """Test Point addition with negative values."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(-0.5, -0.5)
        result = p1 + p2
        self.assertEqual(result.x, 1.0)
        self.assertEqual(result.y, 2.0)

    def test_point_add_integers(self):
        """Test Point addition with integer coordinates."""
        p1 = point.Point(1, 2)
        p2 = point.Point(3, 4)
        result = p1 + p2
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 6)

    def test_point_subtract(self):
        """Test Point subtraction."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(0.5, 0.5)
        result = p1 - p2
        self.assertEqual(result.x, 1.0)
        self.assertEqual(result.y, 2.0)
        self.assertIsInstance(result, point.Point)

    def test_point_subtract_negative(self):
        """Test Point subtraction with negative values."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(-0.5, -0.5)
        result = p1 - p2
        self.assertEqual(result.x, 2.0)
        self.assertEqual(result.y, 3.0)

    def test_point_subtract_to_negative(self):
        """Test Point subtraction resulting in negative values."""
        p1 = point.Point(0.5, 0.5)
        p2 = point.Point(1.5, 2.5)
        result = p1 - p2
        self.assertEqual(result.x, -1.0)
        self.assertEqual(result.y, -2.0)


class PointConversionTestCase(unittest.TestCase):
    """Test cases for Point conversion methods."""

    def test_point_to_list(self):
        """Test Point to_list method."""
        p = point.Point(1.5, 2.5)
        result = p.to_list()
        self.assertEqual(result, [1.5, 2.5])
        self.assertIsInstance(result, list)

    def test_point_to_list_integers(self):
        """Test Point to_list with integer coordinates."""
        p = point.Point(1, 2)
        result = p.to_list()
        self.assertEqual(result, [1, 2])

    def test_point_to_list_negative(self):
        """Test Point to_list with negative coordinates."""
        p = point.Point(-1.5, -2.5)
        result = p.to_list()
        self.assertEqual(result, [-1.5, -2.5])

    def test_point_to_array(self):
        """Test Point to_array method."""
        p = point.Point(1.5, 2.5)
        result = p.to_array()
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.5, 2.5]))

    def test_point_to_array_integers(self):
        """Test Point to_array with integer coordinates."""
        p = point.Point(1, 2)
        result = p.to_array()
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2]))

    def test_point_to_array_negative(self):
        """Test Point to_array with negative coordinates."""
        p = point.Point(-1.5, -2.5)
        result = p.to_array()
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([-1.5, -2.5]))


class PointTestCase(unittest.TestCase):
    """Test cases for Point set method and polygon membership."""

    def test_is_inside_polygon_positive(self):
        poly = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        p = point.Point(x=0.5, y=0.5)
        self.assertTrue(p.is_inside_polygon(poly))

    def test_is_inside_polygon_negative(self):
        """Test point is not inside polygon."""
        poly = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        p = point.Point(x=2.0, y=2.0)
        self.assertFalse(p.is_inside_polygon(poly))

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

    def test_set_no_duplicates(self):
        """Test set with multiple duplicates."""
        points = [point.Point(1.0, 1.0), point.Point(1.0, 1.0), point.Point(1.0, 1.0)]
        points_set = point.Point.set(points)
        self.assertEqual(len(points_set), 1)

    def test_set_all_unique(self):
        """Test set with all unique points."""
        points = [point.Point(1.0, 1.0), point.Point(2.0, 2.0), point.Point(3.0, 3.0)]
        points_set = point.Point.set(points)
        self.assertEqual(len(points_set), 3)

    def test_set_empty(self):
        """Test set with empty list."""
        points = []
        points_set = point.Point.set(points)
        self.assertEqual(len(points_set), 0)

    def test_set_single_point(self):
        """Test set with single point."""
        points = [point.Point(1.5, 2.5)]
        points_set = point.Point.set(points)
        self.assertEqual(len(points_set), 1)
        self.assertEqual(points_set[0], point.Point(1.5, 2.5))

    def test_set_with_numpy_array(self):
        """Test set with numpy array input."""
        points = np.array([point.Point(0.5, 0.5), point.Point(0.5, 0.5)], dtype=object)
        points_set = point.Point.set(points)
        self.assertEqual(len(points_set), 1)

