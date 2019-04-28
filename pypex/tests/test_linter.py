import unittest
import numpy as np

from numpy.testing import assert_array_equal
from pypex.poly2d.intersection import linter
from pypex.poly2d.point import Point


class LinterTestCase(unittest.TestCase):
    @staticmethod
    def intersection_equal(a, b):
        for x, y, in zip(a, b):
            assert_array_equal([x], [y])

    def test_intersection_one_point_touch(self):
        line1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        line2 = np.array([[1.0, 0.0], [1.0, 0.3]])
        line3 = np.array([[0.5, 0.0], [0.3, 0.3]])
        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1], in_touch=True))
        expected = [True, True, Point(1.0, 0.0), np.nan, 'INTERSECT']
        self.intersection_equal(obtained, expected)

        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1], in_touch=False))
        expected = [True, False, Point(1.0, 0.0), np.nan, 'INTERSECT']
        self.intersection_equal(obtained, expected)

        obtained = list(linter.intersection(line1[0], line1[1], line3[0], line3[1], in_touch=True))
        expected = [True, True, Point(0.5, 0.0), np.nan, 'INTERSECT']
        self.intersection_equal(obtained, expected)

        obtained = list(linter.intersection(line1[0], line1[1], line3[0], line3[1], in_touch=False))
        expected = [True, False, Point(0.5, 0.0), np.nan, 'INTERSECT']
        self.intersection_equal(obtained, expected)

    def test_overlap_in_single_point(self):
        line1 = np.array([[1.2, 1.2], [1.5, 1.5]])
        line2 = np.array([[1.5, 1.5], [11.3, 11.3]])
        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1], in_touch=True))
        expected = [True, True, np.nan, 0.0, 'OVERLAP']
        self.intersection_equal(obtained, expected)

        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1], in_touch=False))
        expected = [True, False, np.nan, 0.0, 'OVERLAP']
        self.intersection_equal(obtained, expected)

    def test_intersection_intersect_common(self):
        line1 = np.array([[-1, 0], [1, 0]])
        line2 = np.array([[0, -1], [0, 1]])
        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1]))
        expected = (True, True, Point(0.0, 0.0), np.nan, 'INTERSECT')
        self.intersection_equal(obtained, expected)

        line1 = np.array([[-13, 10], [10, -3]])
        line2 = np.array([[-5, -11], [15, 10]])
        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1]))
        obtained[2] = Point(round(obtained[2].x, 2), round(obtained[2].y, 2))
        expected = [True, True, Point(5.20, -0.29), np.nan, 'INTERSECT']
        self.intersection_equal(obtained, expected)

        # reversed order
        line1 = np.array([[-1, 0], [1, 0]])
        line2 = np.array([[0, 1], [0, -1]])
        obtained = linter.intersection(line1[0], line1[1], line2[0], line2[1])
        expected = (True, True, Point(0.0, 0.0), np.nan, 'INTERSECT')
        self.intersection_equal(obtained, expected)

    def test_intersection_intersect_no_common(self):
        line1 = np.array([[-0.5, -0.5], [0.5, 0.25]])
        line2 = np.array([[0.5, 1.0], [1, 2]])
        obtained = list(linter.intersection(line1[0], line1[1], line2[0], line2[1]))
        obtained[2] = Point(round(obtained[2].x, 2), round(obtained[2].y, 2))
        expected = [True, False, Point(-0.1, -0.2), np.nan, 'INTERSECT']
        self.intersection_equal(obtained, expected)

    def test_intersection_parallel(self):
        line1 = np.array([[0, 1], [1, 1]])
        line2 = np.array([[-1, 0], [10, 0]])
        obtained = linter.intersection(line1[0], line1[1], line2[0], line2[1])
        expected = (False, False, np.nan, 1.0, 'PARALLEL')
        self.intersection_equal(obtained, expected)

        line1 = np.array([[0, 1], [1, 1]])
        line2 = np.array([[-1, 0.3], [10, 0.3]])
        obtained = linter.intersection(line1[0], line1[1], line2[0], line2[1])
        expected = (False, False, np.nan, 0.7, 'PARALLEL')
        self.intersection_equal(obtained, expected)

    def test_intersection_overlap(self):
        line1 = np.array([[-0.5, -0.5], [1.5, 1.5]])
        line2 = np.array([[-1, -1], [1, 1]])
        obtained = linter.intersection(line1[0], line1[1], line2[0], line2[1])
        expected = (True, True, np.nan, 0.0, 'OVERLAP')
        self.intersection_equal(obtained, expected)

    def test_intersection_overlap_no_common(self):
        line1 = np.array([np.array([0, 1]), np.array([1, 1])])
        line2 = np.array([np.array([2, 1]), np.array([6, 1])])
        obtained = linter.intersection(line1[0], line1[1], line2[0], line2[1])
        expected = (True, False, np.nan, 0.0, 'OVERLAP')
        self.intersection_equal(obtained, expected)
