"""Unit tests for Separating Axis Theorem (SAT) intersection detection."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import numpy as np
import pytest

from elisa.pypex.base.conf import ROUND_PRECISION
from elisa.pypex.poly2d.intersection import linter, sat
from tests.utils import ElisaTestCase

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SeparatingAxisTheoremTestCase(ElisaTestCase):
    """Test cases for separating_axis_theorem() function."""

    def test_separated_squares(self) -> None:
        """Test two non-overlapping squares."""
        poly1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        poly2 = np.array([[2, 0], [3, 0], [3, 1], [2, 1]])
        assert sat.separating_axis_theorem(poly1, poly2, touch_is_separated=False)
        assert sat.separating_axis_theorem(poly1, poly2, touch_is_separated=True)

    def test_overlapping_squares(self) -> None:
        """Test two overlapping squares."""
        poly1 = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        poly2 = np.array([[1, 1], [3, 1], [3, 3], [1, 3]])
        assert not (sat.separating_axis_theorem(poly1, poly2, touch_is_separated=False))
        assert not (sat.separating_axis_theorem(poly1, poly2, touch_is_separated=True))

    def test_touching_squares_edge_to_edge(self) -> None:
        """Test two squares touching edge to edge."""
        poly1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        poly2 = np.array([[1, 0], [2, 0], [2, 1], [1, 1]])
        # touch_is_separated=False: touching counts as NOT separated
        assert not (sat.separating_axis_theorem(poly1, poly2, touch_is_separated=False))
        # touch_is_separated=True: touching counts as separated
        assert sat.separating_axis_theorem(poly1, poly2, touch_is_separated=True)

    def test_touching_squares_corner_to_corner(self) -> None:
        """Test two squares touching at a single corner point."""
        poly1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        poly2 = np.array([[1, 1], [2, 1], [2, 2], [1, 2]])
        # touch_is_separated=False: touching counts as NOT separated
        assert not (sat.separating_axis_theorem(poly1, poly2, touch_is_separated=False))
        # touch_is_separated=True: touching counts as separated
        assert sat.separating_axis_theorem(poly1, poly2, touch_is_separated=True)

    def test_nested_squares(self) -> None:
        """Test square completely inside another square."""
        poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])
        poly2 = np.array([[1, 1], [3, 1], [3, 3], [1, 3]])
        assert not (sat.separating_axis_theorem(poly1, poly2, touch_is_separated=False))
        assert not (sat.separating_axis_theorem(poly1, poly2, touch_is_separated=True))

    def test_invalid_polygon_too_few_vertices(self) -> None:
        """Test that error is raised for polygons with fewer than 3 vertices."""
        poly1 = np.array([[0, 0], [1, 1]])  # Only 2 vertices
        poly2 = np.array([[2, 0], [3, 0], [3, 1], [2, 1]])
        with pytest.raises(ValueError, match="invalid polygon shape"):
            sat.separating_axis_theorem(poly1, poly2)

    def test_triangles_separated(self) -> None:
        """Test two separated triangles."""
        poly1 = np.array([[0, 0], [1, 0], [0.5, 1]])
        poly2 = np.array([[3, 0], [4, 0], [3.5, 1]])
        assert sat.separating_axis_theorem(poly1, poly2, touch_is_separated=False)

    def test_triangles_overlapping(self) -> None:
        """Test two overlapping triangles."""
        poly1 = np.array([[0, 0], [2, 0], [1, 2]])
        poly2 = np.array([[1, 0], [3, 0], [2, 2]])
        assert not (sat.separating_axis_theorem(poly1, poly2, touch_is_separated=False))

    def test_complex_polygons_separated(self) -> None:
        """Test two separated complex (non-square) polygons."""
        poly1 = np.array([[0, 0], [2, 0], [2.5, 1], [2, 2], [0, 2]])
        poly2 = np.array([[4, 0], [6, 0], [6.5, 1], [6, 2], [4, 2]])
        assert sat.separating_axis_theorem(poly1, poly2, touch_is_separated=False)

    def test_complex_polygons_overlapping(self) -> None:
        """Test two overlapping complex polygons."""
        poly1 = np.array([[0, 0], [3, 0], [3, 2], [0, 2]])
        poly2 = np.array([[2, 0], [5, 0], [5, 2], [2, 2]])
        assert not (sat.separating_axis_theorem(poly1, poly2, touch_is_separated=False))


class SeparatingAxisTheoremLineAdaptTestCase(ElisaTestCase):
    """Test cases for separating_axis_theorem_line_adapt() function."""

    def test_overlapping_collinear_segments(self) -> None:
        """Test two overlapping collinear line segments."""
        line1 = np.array([[0, 0], [2, 0]])
        line2 = np.array([[1, 0], [3, 0]])
        # Overlapping: not separated
        assert not (sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=False))
        assert not (sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=True))

    def test_separated_collinear_segments(self) -> None:
        """Test two separated collinear line segments."""
        line1 = np.array([[0, 0], [1, 0]])
        line2 = np.array([[2, 0], [3, 0]])
        # Separated: should return True
        assert sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=False)
        assert sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=True)

    def test_touching_collinear_segments_in_touch_false(self) -> None:
        """Test two touching collinear line segments with in_touch=False."""
        line1 = np.array([[0, 0], [1, 0]])
        line2 = np.array([[1, 0], [2, 0]])
        # touch_is_separated=False: touching counts as NOT separated
        assert not (sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=False))

    def test_touching_collinear_segments_in_touch_true(self) -> None:
        """Test two touching collinear line segments with in_touch=True."""
        line1 = np.array([[0, 0], [1, 0]])
        line2 = np.array([[1, 0], [2, 0]])
        # touch_is_separated=True: touching counts as separated
        assert sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=True)

    def test_one_segment_inside_another(self) -> None:
        """Test one line segment completely inside another."""
        line1 = np.array([[0, 0], [4, 0]])
        line2 = np.array([[1, 0], [3, 0]])
        # One inside another: they overlap, not separated
        assert not (sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=False))
        assert not (sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=True))

    def test_vertical_segments(self) -> None:
        """Test two overlapping vertical line segments."""
        line1 = np.array([[0, 0], [0, 2]])
        line2 = np.array([[0, 1], [0, 3]])
        # Overlapping vertically: not separated
        assert not (sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=False))
        assert not (sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=True))

    def test_diagonal_segments(self) -> None:
        """Test two overlapping diagonal line segments."""
        line1 = np.array([[0, 0], [2, 2]])
        line2 = np.array([[1, 1], [3, 3]])
        # Overlapping diagonally: not separated
        assert not (sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=False))
        assert not (sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=True))

    def test_negative_coordinates_segments(self) -> None:
        """Test line segments with negative coordinates."""
        line1 = np.array([[-2, -2], [-1, -1]])
        line2 = np.array([[-1.5, -1.5], [0, 0]])
        # Overlapping with negative coords: not separated
        assert not (sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=False))
        assert not (sat.separating_axis_theorem_line_adapt(line1, line2, touch_is_separated=True))

    def test_with_custom_round_tol_segments(self) -> None:
        """Test line segments with custom rounding tolerance."""
        line1 = np.array([[0, 0], [1, 0]])
        line2 = np.array([[1.0001, 0], [2, 0]])
        # With high precision (round_tol=5), these are separated
        assert sat.separating_axis_theorem_line_adapt(line1, line2, round_tol=5)
        # With low precision (round_tol=3), these touch/overlap
        assert not (sat.separating_axis_theorem_line_adapt(line1, line2, round_tol=3))


class IntersectsTestCase(ElisaTestCase):
    """Test cases for intersects() function."""

    def test_two_overlapping_squares(self) -> None:
        """Test intersects() with two overlapping squares."""
        poly1 = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        poly2 = np.array([[1, 1], [3, 1], [3, 3], [1, 3]])
        assert sat.intersects(poly1, poly2, touch_is_separated=False)
        assert sat.intersects(poly1, poly2, touch_is_separated=True)

    def test_two_separated_squares(self) -> None:
        """Test intersects() with two separated squares."""
        poly1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        poly2 = np.array([[2, 0], [3, 0], [3, 1], [2, 1]])
        assert not (sat.intersects(poly1, poly2, touch_is_separated=False))
        assert not (sat.intersects(poly1, poly2, touch_is_separated=True))

    def test_two_touching_squares(self) -> None:
        """Test intersects() with two touching squares."""
        poly1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        poly2 = np.array([[1, 0], [2, 0], [2, 1], [1, 1]])
        # touch_is_separated=False: touching counts as intersecting
        assert sat.intersects(poly1, poly2, touch_is_separated=False)
        # touch_is_separated=True: touching counts as separated
        assert not (sat.intersects(poly1, poly2, touch_is_separated=True))

    def test_two_overlapping_lines(self) -> None:
        """Test intersects() with two overlapping line segments."""
        line1 = np.array([[0, 0], [2, 0]])
        line2 = np.array([[1, 0], [3, 0]])
        assert sat.intersects(line1, line2, touch_is_separated=False)

    def test_two_separated_lines(self) -> None:
        """Test intersects() with two separated line segments."""
        line1 = np.array([[0, 0], [1, 0]])
        line2 = np.array([[2, 0], [3, 0]])
        assert not (sat.intersects(line1, line2, touch_is_separated=False))

    def test_two_touching_lines_in_touch_false(self) -> None:
        """Test intersects() with two touching line segments (touch_is_separated=False)."""
        line1 = np.array([[0, 0], [1, 0]])
        line2 = np.array([[1, 0], [2, 0]])
        assert sat.intersects(line1, line2, touch_is_separated=False)

    def test_two_touching_lines_in_touch_true(self) -> None:
        """Test intersects() with two touching line segments (touch_is_separated=True)."""
        line1 = np.array([[0, 0], [1, 0]])
        line2 = np.array([[1, 0], [2, 0]])
        assert not (sat.intersects(line1, line2, touch_is_separated=True))

    def test_polygon_list_input(self) -> None:
        """Test intersects() with list input for polygons."""
        poly1 = [[0, 0], [2, 0], [2, 2], [0, 2]]
        poly2 = [[1, 1], [3, 1], [3, 3], [1, 3]]
        # Should work with list input (ArrayLike)
        assert sat.intersects(poly1, poly2, touch_is_separated=False)


class CollinearOrientationRegressionTestCase(ElisaTestCase):
    def assert_overlap_entry_points(
        self,
        line1: NDArray[np.float64],
        line2: NDArray[np.float64],
        expected: bool,
        touch_is_separated: bool,
        round_tol: int = ROUND_PRECISION,
    ) -> None:
        kwargs = {"touch_is_separated": touch_is_separated, "round_tol": round_tol}
        assert (sat.separating_axis_theorem_line_adapt(line1, line2, **kwargs)) == (not expected)
        assert (sat.intersects(line1, line2, **kwargs)) == (expected)
        scalar = linter.intersection(*line1, *line2, **kwargs)
        assert (scalar[:2]) == ((True, expected))
        assert np.isnan(scalar[2])
        assert (scalar[3:]) == ((0.0, "COLINEAR"))
        vector = linter.intersections(line1, line2, **kwargs)
        np.testing.assert_array_equal(vector[0], np.ones(4, dtype=bool))
        np.testing.assert_array_equal(vector[1], np.full(4, expected))
        np.testing.assert_array_equal(vector[2], np.full((4, 2), np.nan))
        np.testing.assert_array_equal(vector[3], np.zeros(4))
        np.testing.assert_array_equal(vector[4], np.full(4, b"COLINEAR"))

    def test_collinear_orientation_overlap_matrix(self) -> None:
        cases = [
            ("partial", (0, 2), (1, 3), True),
            ("contained", (0, 4), (1, 3), True),
            ("identical", (0, 2), (0, 2), True),
            ("disjoint", (0, 2), (3, 4), False),
            ("touch", (0, 2), (2, 4), None),
        ]
        for name, interval1, interval2, overlap in cases:
            for direction in ((1, 0), (0, 1), (1, -1)):
                origin = np.array([-2.0, 1.0])
                line1 = origin + np.array(interval1)[:, None] * direction
                line2 = origin + np.array(interval2)[:, None] * direction
                for reverse1 in (False, True):
                    for reverse2 in (False, True):
                        for swap in (False, True):
                            a = line1[::-1] if reverse1 else line1
                            b = line2[::-1] if reverse2 else line2
                            a, b = (b, a) if swap else (a, b)
                            for touch_is_separated in (False, True):
                                with self.subTest(
                                    case=name,
                                    direction=direction,
                                    reverse1=reverse1,
                                    reverse2=reverse2,
                                    swap=swap,
                                    touch_is_separated=touch_is_separated,
                                ):
                                    expected = not touch_is_separated if overlap is None else overlap
                                    self.assert_overlap_entry_points(a, b, expected, touch_is_separated)

    def test_collinear_orientation_round_tol(self) -> None:
        for reverse1 in (False, True):
            for reverse2 in (False, True):
                for swap in (False, True):
                    a = np.array([[0.0, 0.0], [1.0, 0.0]])
                    b = np.array([[1.0001, 0.0], [2.0, 0.0]])
                    a = a[::-1] if reverse1 else a
                    b = b[::-1] if reverse2 else b
                    a, b = (b, a) if swap else (a, b)
                    for round_tol in (3, 5):
                        for touch_is_separated in (False, True):
                            with self.subTest(
                                reverse1=reverse1,
                                reverse2=reverse2,
                                swap=swap,
                                round_tol=round_tol,
                                touch_is_separated=touch_is_separated,
                            ):
                                expected = round_tol == 3 and not touch_is_separated
                                self.assert_overlap_entry_points(
                                    a,
                                    b,
                                    expected,
                                    touch_is_separated,
                                    round_tol=round_tol,
                                )


if __name__ == "__main__":
    unittest.main()
