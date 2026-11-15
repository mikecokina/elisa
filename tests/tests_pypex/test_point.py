from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from elisa.pypex.poly2d import point
from elisa.pypex.poly2d.polygon import Polygon
from tests.utils import ElisaTestCase

if TYPE_CHECKING:
    from numpy.typing import NDArray


def asser_points_equal(
    ps1: list[point.Point] | NDArray[np.object_],
    ps2: list[point.Point] | NDArray[np.object_],
) -> None:
    ps1 = sorted(ps1, key=lambda k: (k.x, k.y))
    ps2 = sorted(ps2, key=lambda k: (k.x, k.y))
    for p1, p2 in zip(ps1, ps2, strict=True):
        assert p1, p2


class LineSideTestCase(ElisaTestCase):
    """Test cases for _line_side function."""

    def test_line_side_same_side_positive(self) -> None:
        """Test points on same side of line."""
        p1 = np.array([0, 1])
        p2 = np.array([1, 1])
        a = np.array([0, 0])
        b = np.array([1, 0])
        result = point._line_side(p1, p2, a, b)
        assert result > 0

    def test_line_side_opposite_sides(self) -> None:
        """Test points on opposite sides of line."""
        p1 = np.array([0, 1])
        p2 = np.array([0, -1])
        a = np.array([0, 0])
        b = np.array([1, 0])
        result = point._line_side(p1, p2, a, b)
        assert result < 0

    def test_line_side_with_lists(self) -> None:
        """Test _line_side accepts list inputs."""
        p1 = [0, 1]
        p2 = [1, 1]
        a = [0, 0]
        b = [1, 0]
        result = point._line_side(p1, p2, a, b)
        assert result > 0


class SameSideTestCase(ElisaTestCase):
    """Test cases for same_side function."""

    def test_same_side_true(self) -> None:
        """Test points on the same side of a line."""
        p1 = np.array([0, 1])
        p2 = np.array([1, 1])
        a = np.array([0, 0])
        b = np.array([1, 0])
        assert point.same_side(p1, p2, a, b)

    def test_same_side_false(self) -> None:
        """Test points on opposite sides of a line."""
        p1 = np.array([0, 1])
        p2 = np.array([0, -1])
        a = np.array([0, 0])
        b = np.array([1, 0])
        assert not point.same_side(p1, p2, a, b)

    def test_same_side_with_lists(self) -> None:
        """Test same_side accepts list inputs."""
        p1 = [0, 1]
        p2 = [1, 1]
        a = [0, 0]
        b = [1, 0]
        assert point.same_side(p1, p2, a, b)

    def test_same_side_collinear(self) -> None:
        """Test points on the line itself."""
        p1 = np.array([0.5, 0])
        p2 = np.array([0.75, 0])
        a = np.array([0, 0])
        b = np.array([1, 0])
        assert point.same_side(p1, p2, a, b)


class LineSideCompatibilityTestCase(ElisaTestCase):
    def test_line_side_magnitude_and_orientation(self) -> None:
        origin = np.array([-3.0, 2.0])
        for direction in ((4, 0), (0, 4), (3, 4)):
            ab = np.array(direction, dtype=float)
            normal = np.array([-ab[1], ab[0]])
            for side1, side2 in ((1, 2), (1, -2), (0, 2), (0, 0)):
                p1 = origin + 0.25 * ab + side1 * normal
                p2 = origin + 0.75 * ab + side2 * normal
                expected = side1 * side2 * np.dot(ab, ab) ** 2
                for reverse in (False, True):
                    a, b = (origin + ab, origin) if reverse else (origin, origin + ab)
                    for swap in (False, True):
                        first, second = (p2, p1) if swap else (p1, p2)
                        for as_list in (False, True):
                            with self.subTest(
                                direction=direction,
                                sides=(side1, side2),
                                reverse=reverse,
                                swap=swap,
                                as_list=as_list,
                            ):
                                args = (first, second, a, b)
                                if as_list:
                                    args = tuple(value.tolist() for value in args)
                                assert point._line_side(*args) == expected
                                assert point.same_side(*args) == (expected >= 0)


class BaseTestCase(ElisaTestCase):
    """Test cases for is_point_in_polygon function."""

    def test_is_point_in_polygon_positive(self) -> None:
        poly = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        p = point.Point(x=0.5, y=0.5)
        obtained = point.is_point_in_polygon(p, poly)
        assert obtained

    def test_is_point_in_polygon_negative(self) -> None:
        poly = Polygon([[-1, -1], [1, 0], [1, 1], [0.5, 1.9], [1, 1]])
        p = point.Point(x=-10, y=10)
        obtained = point.is_point_in_polygon(p, poly)
        assert not obtained

    def test_is_point_in_polygon_onedge(self) -> None:
        poly = Polygon([[0, 0], [1, 0], [1, 1]])
        p = point.Point(x=0.5, y=0)
        obtained = point.is_point_in_polygon(p, poly)
        assert not obtained

    def test_is_point_in_polygon_at_vertex(self) -> None:
        """Test point at polygon vertex."""
        poly = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        p = point.Point(x=0, y=0)
        obtained = point.is_point_in_polygon(p, poly)
        assert not obtained

    def test_is_point_in_polygon_invalid_polygon(self) -> None:
        """Test with invalid polygon (less than 3 points)."""
        poly = Polygon([[0, 0], [1, 0]])
        p = point.Point(x=0.5, y=0.5)
        with pytest.raises(ValueError, match="invalid polygon shape"):
            point.is_point_in_polygon(p, poly)


class PointInitTestCase(ElisaTestCase):
    """Test cases for Point initialization."""

    def test_point_init(self) -> None:
        """Test Point initialization."""
        p = point.Point(1.5, 2.5)
        assert p.x == 1.5
        assert p.y == 2.5

    def test_point_init_integers(self) -> None:
        """Test Point initialization with integers."""
        p = point.Point(1, 2)
        assert p.x == 1
        assert p.y == 2

    def test_point_init_negative(self) -> None:
        """Test Point initialization with negative values."""
        p = point.Point(-1.5, -2.5)
        assert p.x == -1.5
        assert p.y == -2.5


class PointStringRepresentationTestCase(ElisaTestCase):
    """Test cases for Point string representations."""

    def test_point_str(self) -> None:
        """Test Point __str__ method."""
        p = point.Point(1.5, 2.5)
        assert str(p) == "Point [1.5, 2.5]"

    def test_point_repr(self) -> None:
        """Test Point __repr__ method."""
        p = point.Point(1.5, 2.5)
        assert repr(p) == "Point [1.5, 2.5]"

    def test_point_str_with_integers(self) -> None:
        """Test Point __str__ with integer coordinates."""
        p = point.Point(1, 2)
        assert str(p) == "Point [1, 2]"

    def test_point_str_with_negative(self) -> None:
        """Test Point __str__ with negative coordinates."""
        p = point.Point(-1.5, -2.5)
        assert str(p) == "Point [-1.5, -2.5]"


class PointEqualityTestCase(ElisaTestCase):
    """Test cases for Point equality and hashing."""

    def test_point_equality_true(self) -> None:
        """Test equality of two identical points."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(1.5, 2.5)
        assert p1 == p2

    def test_point_equality_false(self) -> None:
        """Test inequality of two different points."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(1.5, 2.6)
        assert p1 != p2

    def test_point_hash_same_points(self) -> None:
        """Test hash of identical points."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(1.5, 2.5)
        assert hash(p1) == hash(p2)

    def test_point_hash_different_points(self) -> None:
        """Test hash of different points."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(1.5, 2.6)
        # Different points should (likely) have different hashes
        assert hash(p1) != hash(p2)

    def test_point_hashable_in_set(self) -> None:
        """Test that Point can be used in sets."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(1.5, 2.5)
        p3 = point.Point(2.5, 3.5)
        point_set = {p1, p2, p3}
        assert len(point_set) == 2

    def test_point_hashable_in_dict(self) -> None:
        """Test that Point can be used as dict key."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(1.5, 2.5)
        point_dict = {p1: "first"}
        point_dict[p2] = "second"
        assert len(point_dict) == 1
        assert point_dict[p1] == "second"


class PointArithmeticTestCase(ElisaTestCase):
    """Test cases for Point arithmetic operations."""

    def test_point_add(self) -> None:
        """Test Point addition."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(0.5, 0.5)
        result = p1 + p2
        assert result.x == 2.0
        assert result.y == 3.0
        assert isinstance(result, point.Point)

    def test_point_add_negative(self) -> None:
        """Test Point addition with negative values."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(-0.5, -0.5)
        result = p1 + p2
        assert result.x == 1.0
        assert result.y == 2.0

    def test_point_add_integers(self) -> None:
        """Test Point addition with integer coordinates."""
        p1 = point.Point(1, 2)
        p2 = point.Point(3, 4)
        result = p1 + p2
        assert result.x == 4
        assert result.y == 6

    def test_point_subtract(self) -> None:
        """Test Point subtraction."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(0.5, 0.5)
        result = p1 - p2
        assert result.x == 1.0
        assert result.y == 2.0
        assert isinstance(result, point.Point)

    def test_point_subtract_negative(self) -> None:
        """Test Point subtraction with negative values."""
        p1 = point.Point(1.5, 2.5)
        p2 = point.Point(-0.5, -0.5)
        result = p1 - p2
        assert result.x == 2.0
        assert result.y == 3.0

    def test_point_subtract_to_negative(self) -> None:
        """Test Point subtraction resulting in negative values."""
        p1 = point.Point(0.5, 0.5)
        p2 = point.Point(1.5, 2.5)
        result = p1 - p2
        assert result.x == -1.0
        assert result.y == -2.0


class PointConversionTestCase(ElisaTestCase):
    """Test cases for Point conversion methods."""

    def test_point_to_list(self) -> None:
        """Test Point to_list method."""
        p = point.Point(1.5, 2.5)
        result = p.to_list()
        assert result == [1.5, 2.5]
        assert isinstance(result, list)

    def test_point_to_list_integers(self) -> None:
        """Test Point to_list with integer coordinates."""
        p = point.Point(1, 2)
        result = p.to_list()
        assert result == [1, 2]

    def test_point_to_list_negative(self) -> None:
        """Test Point to_list with negative coordinates."""
        p = point.Point(-1.5, -2.5)
        result = p.to_list()
        assert result == [-1.5, -2.5]

    def test_point_to_array(self) -> None:
        """Test Point to_array method."""
        p = point.Point(1.5, 2.5)
        result = p.to_array()
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.5, 2.5]))

    def test_point_to_array_integers(self) -> None:
        """Test Point to_array with integer coordinates."""
        p = point.Point(1, 2)
        result = p.to_array()
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2]))

    def test_point_to_array_negative(self) -> None:
        """Test Point to_array with negative coordinates."""
        p = point.Point(-1.5, -2.5)
        result = p.to_array()
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([-1.5, -2.5]))


class PointTestCase(ElisaTestCase):
    """Test cases for Point set method and polygon membership."""

    def test_is_inside_polygon_positive(self) -> None:
        poly = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        p = point.Point(x=0.5, y=0.5)
        assert p.is_inside_polygon(poly)

    def test_is_inside_polygon_negative(self) -> None:
        """Test point is not inside polygon."""
        poly = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        p = point.Point(x=2.0, y=2.0)
        assert not p.is_inside_polygon(poly)

    def test_set(self) -> None:
        points = [point.Point(0.1, 0.1), point.Point(0.1, 0.1)]
        points_set = point.Point.set(points)
        assert points_set == [point.Point(0.1, 0.1)]

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

    def test_set_no_duplicates(self) -> None:
        """Test set with multiple duplicates."""
        points = [point.Point(1.0, 1.0), point.Point(1.0, 1.0), point.Point(1.0, 1.0)]
        points_set = point.Point.set(points)
        assert len(points_set) == 1

    def test_set_all_unique(self) -> None:
        """Test set with all unique points."""
        points = [point.Point(1.0, 1.0), point.Point(2.0, 2.0), point.Point(3.0, 3.0)]
        points_set = point.Point.set(points)
        assert len(points_set) == 3

    def test_set_empty(self) -> None:
        """Test set with empty list."""
        points = []
        points_set = point.Point.set(points)
        assert len(points_set) == 0

    def test_set_single_point(self) -> None:
        """Test set with single point."""
        points = [point.Point(1.5, 2.5)]
        points_set = point.Point.set(points)
        assert len(points_set) == 1
        assert points_set[0] == point.Point(1.5, 2.5)

    def test_set_with_numpy_array(self) -> None:
        """Test set with numpy array input."""
        points = np.array([point.Point(0.5, 0.5), point.Point(0.5, 0.5)], dtype=object)
        points_set = point.Point.set(points)
        assert len(points_set) == 1
