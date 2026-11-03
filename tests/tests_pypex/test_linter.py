from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.testing import assert_array_equal

from elisa.pypex.base.conf import PRECISION
from elisa.pypex.poly2d.intersection import linter
from elisa.pypex.poly2d.point import Point
from tests.utils import ElisaTestCase

if TYPE_CHECKING:
    from numpy.typing import NDArray


class LinterTestCase(ElisaTestCase):
    @staticmethod
    def intersection_equal(
        a: list[bool | Point | float | str]
        | list[bool | float | str]
        | tuple[bool, bool, Point, float, str]
        | tuple[bool, bool, float, float, str],
        b: list[bool | Point | float | str]
        | list[bool | float | str]
        | tuple[bool, bool, Point, float, str]
        | tuple[bool, bool, float, float, str],
    ) -> None:
        assert len(a) == len(b)
        for (
            x,
            y,
        ) in zip(a, b, strict=True):
            assert_array_equal([x], [y])

    def test_intersection_one_point_touch(self) -> None:
        line1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        line2 = np.array([[1.0, 0.0], [1.0, 0.3]])
        line3 = np.array([[0.5, 0.0], [0.3, 0.3]])

        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1], touch_is_separated=False))
        expected = [True, True, Point(1.0, 0.0), np.nan, "INTERSECT"]
        self.intersection_equal(obtained, expected)

        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1], touch_is_separated=True))
        expected = [True, False, Point(1.0, 0.0), np.nan, "INTERSECT"]
        self.intersection_equal(obtained, expected)

        obtained = list(linter.intersection(line1[0], line1[1], line3[0], line3[1], touch_is_separated=False))
        expected = [True, True, Point(0.5, 0.0), np.nan, "INTERSECT"]
        self.intersection_equal(obtained, expected)

        obtained = list(linter.intersection(line1[0], line1[1], line3[0], line3[1], touch_is_separated=True))
        expected = [True, False, Point(0.5, 0.0), np.nan, "INTERSECT"]
        self.intersection_equal(obtained, expected)

    def test_overlap_in_single_point(self) -> None:
        line1 = np.array([[1.2, 1.2], [1.5, 1.5]])
        line2 = np.array([[1.5, 1.5], [11.3, 11.3]])

        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1], touch_is_separated=False))
        # line, segment, intr_ptx, distance, msg
        expected = [True, True, np.nan, 0.0, "COLINEAR"]
        self.intersection_equal(obtained, expected)

        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1], touch_is_separated=True))
        expected = [True, False, np.nan, 0.0, "COLINEAR"]
        self.intersection_equal(obtained, expected)

    def test_intersection_intersect_common(self) -> None:
        line1 = np.array([[-1, 0], [1, 0]])
        line2 = np.array([[0, -1], [0, 1]])
        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1]))
        expected = (True, True, Point(0.0, 0.0), np.nan, "INTERSECT")
        self.intersection_equal(obtained, expected)

        line1 = np.array([[-13, 10], [10, -3]])
        line2 = np.array([[-5, -11], [15, 10]])
        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1]))
        obtained[2] = Point(round(obtained[2].x, 2), round(obtained[2].y, 2))
        expected = [True, True, Point(5.20, -0.29), np.nan, "INTERSECT"]
        self.intersection_equal(obtained, expected)

        # reversed order
        line1 = np.array([[-1, 0], [1, 0]])
        line2 = np.array([[0, 1], [0, -1]])
        obtained = linter.intersection(line1[0], line1[1], line2[0], line2[1])
        expected = (True, True, Point(0.0, 0.0), np.nan, "INTERSECT")
        self.intersection_equal(obtained, expected)

    def test_intersection_intersect_no_common(self) -> None:
        line1 = np.array([[-0.5, -0.5], [0.5, 0.25]])
        line2 = np.array([[0.5, 1.0], [1, 2]])
        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1]))
        obtained[2] = Point(round(obtained[2].x, 2), round(obtained[2].y, 2))
        expected = [True, False, Point(-0.1, -0.2), np.nan, "INTERSECT"]
        self.intersection_equal(obtained, expected)

    def test_intersection_parallel(self) -> None:
        line1 = np.array([[0, 1], [1, 1]])
        line2 = np.array([[-1, 0], [10, 0]])
        obtained = linter.intersection(line1[0], line1[1], line2[0], line2[1])
        expected = (False, False, np.nan, 1.0, "PARALLEL")
        self.intersection_equal(obtained, expected)

        line1 = np.array([[0, 1], [1, 1]])
        line2 = np.array([[-1, 0.3], [10, 0.3]])
        obtained = linter.intersection(line1[0], line1[1], line2[0], line2[1])
        expected = (False, False, np.nan, 0.7, "PARALLEL")
        self.intersection_equal(obtained, expected)

    def test_intersection_overlap(self) -> None:
        line1 = np.array([[-0.5, -0.5], [1.5, 1.5]])
        line2 = np.array([[-1, -1], [1, 1]])
        obtained = linter.intersection(line1[0], line1[1], line2[0], line2[1])
        expected = (True, True, np.nan, 0.0, "COLINEAR")
        self.intersection_equal(obtained, expected)

    def test_intersection_overlap_no_common(self) -> None:
        line1 = np.array([np.array([0, 1]), np.array([1, 1])])
        line2 = np.array([np.array([2, 1]), np.array([6, 1])])
        obtained = linter.intersection(line1[0], line1[1], line2[0], line2[1])
        expected = (True, False, np.nan, 0.0, "COLINEAR")
        self.intersection_equal(obtained, expected)


class LinterMultipleTestCase(ElisaTestCase):
    @staticmethod
    def test_linter_intersections() -> None:
        lines1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        lines2 = np.array([[0.2, -0.2], [0.6, 0.6], [1.2, 0.4]])

        _intersection_status, intersection_segment, _intr_ptx, _distance, _msg, _ = linter.intersections(
            lines1,
            lines2,
            touch_is_separated=False,
        )

        assert_array_equal([True, False, True, False, True, True, True, True, False], intersection_segment)

    @staticmethod
    def test_linter_intersections_negative() -> None:
        lines1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        lines2 = np.array([[-0.2, -0.2], [-0.75, -0.25], [-0.8, -0.8]])

        intersection_status, intersection_segment, _intr_ptx, _distance, _msg, _ = linter.intersections(
            lines1,
            lines2,
            touch_is_separated=False,
        )

        assert_array_equal([False, False, False, False, False, False, False, False, False], intersection_segment)
        assert_array_equal([True, True, True, True, True, True, True, True, True], intersection_status)


class ParallelDistanceRegressionTestCase(ElisaTestCase):
    def assert_parallel_results(
        self,
        line1: NDArray[np.float64],
        line2: NDArray[np.float64],
        expected_distance: float,
        tol: float = PRECISION,
        rtol: float = 1e-12,
        atol: float = 0.0,
    ) -> None:
        result = linter.intersections(line1, line2, tol=tol)
        assert isinstance(result, tuple)
        assert (len(result)) == (6)
        status, segment, points, distances, messages, indices = result
        collinear = expected_distance < tol
        assert_array_equal(status, np.full(4, collinear))
        assert_array_equal(segment, np.full(4, collinear))
        assert_array_equal(points, np.full((4, 2), np.nan))
        np.testing.assert_allclose(distances, expected_distance, rtol=rtol, atol=atol)
        assert_array_equal(messages, np.full(4, b"COLINEAR" if collinear else b"PARALLEL"))
        assert (status.dtype) == (np.dtype(bool))
        assert (segment.dtype) == (np.dtype(bool))
        assert (messages.dtype) == (np.dtype("S9"))
        assert_array_equal(indices, [[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0]])
        # Two-vertex hulls include both edge orientations. Check every mapped pair.
        for i, (a, b, c, d) in enumerate(indices.astype(int)):
            scalar = linter.intersection(line1[a], line1[b], line2[c], line2[d], tol=tol)
            assert isinstance(scalar, tuple)
            assert (len(scalar)) == (5)
            assert (scalar[:2]) == ((bool(status[i]), bool(segment[i])))
            assert np.isnan(scalar[2])
            np.testing.assert_allclose(scalar[3], expected_distance, rtol=rtol, atol=atol)
            np.testing.assert_allclose(distances[i], scalar[3], rtol=rtol, atol=atol)
            assert (scalar[4]) == (messages[i].decode())

    def test_parallel_distance_scales(self) -> None:
        for direction, offset in [((4, 0), (0, 3)), ((0, 4), (3, 0)), ((3, 4), (0, 5))]:
            for scale in (1e-3, 1.0, 1e6):
                with self.subTest(direction=direction, scale=scale):
                    origin = np.array([-7.0, 2.0]) * scale
                    line1 = np.array([origin, origin + np.array(direction) * scale])
                    line2 = line1 + np.array(offset) * scale
                    self.assert_parallel_results(line1, line2, 3 * scale)

    def test_parallel_distance_tolerance(self) -> None:
        # A geometric distance threshold, distinct from the determinant threshold:
        # these segments are exactly parallel for all nonzero separations below.
        for tol in (PRECISION, 1e-6):
            for direction, normal in [((4, 0), (0, 1)), ((0, 4), (1, 0)), ((3, 4), (-0.8, 0.6))]:
                for factor in (0.0, 0.5, 0.99, 1.01, 2.0):
                    with self.subTest(tol=tol, direction=direction, factor=factor):
                        line1 = np.array([(0, 0), direction], dtype=float)
                        line2 = line1 + np.array(normal) * (factor * tol)
                        self.assert_parallel_results(line1, line2, factor * tol, tol=tol, rtol=1e-5, atol=1e-15)
            # Exact boundary on axis-aligned lines: abs(distance) < tol is strict.
            for direction, normal in [((4, 0), (0, 1)), ((0, 4), (1, 0))]:
                with self.subTest(tol=tol, direction=direction, factor=1.0):
                    line1 = np.array([(0, 0), direction], dtype=float)
                    self.assert_parallel_results(line1, line1 + np.array(normal) * tol, tol, tol=tol)

    def test_nonparallel_endpoint_orientation_consistency(self) -> None:
        for second, interior, touches in [
            ([[1, -1], [1, 1]], True, False),
            ([[2, 0], [2, 1]], False, True),
            ([[1, 0], [1, 1]], False, True),
            ([[3, -1], [3, 1]], False, False),
        ]:
            for touch_is_separated in (False, True):
                with self.subTest(second=second, touch_is_separated=touch_is_separated):
                    line1 = np.array([[0, 0], [2, 0]], dtype=float)
                    line2 = np.array(second, dtype=float)
                    for a, b in ((line1, line2), (line2, line1)):
                        result = linter.intersections(a, b, touch_is_separated=touch_is_separated)
                        expected = interior or (touches and not touch_is_separated)
                        assert_array_equal(result[0], np.ones(4, dtype=bool))
                        assert_array_equal(result[1], np.full(4, expected))
                        assert_array_equal(result[3], np.full(4, np.nan))
                        assert_array_equal(result[4], np.full(4, b"INTERSECT"))
                        for i, (p, q, r, s) in enumerate(result[5].astype(int)):
                            scalar = linter.intersection(a[p], a[q], b[r], b[s], touch_is_separated=touch_is_separated)
                            assert (scalar[:2]) == ((True, expected))
                            np.testing.assert_allclose([scalar[2].x, scalar[2].y], result[2][i])
                            assert np.isnan(scalar[3])
                            assert (scalar[4]) == ("INTERSECT")
