import unittest
import numpy as np
from numpy.testing import assert_array_equal

from elisa.binary_system import geo
from pypex.poly2d import polygon


class SupportMethodsTestCse(unittest.TestCase):

    def test_darkside_filter(self):
        normals = np.array([[1, 1, 1], [0.3, 0.1, -5], [-2, -3, -4.1]])
        normals = np.array([a / b for a, b in zip(normals, np.linalg.norm(normals, axis=1))])
        los = [1, 0, 0]
        obtained = geo.darkside_filter(los, normals)
        expected = np.array([0, 1], dtype=int)
        self.assertTrue(np.all(obtained == expected))

    def _test_plane_projection(self, expected, plane):
        points = np.array([[1, 1, 1], [0.3, 0.1, -5], [-2, -3, -4.1]])
        obtained = geo.plane_projection(points, plane, keep_3d=False)
        self.assertTrue(np.all(obtained == expected))

    def test_plane_projection_xy(self):
        expeceted = np.array([[1, 1], [0.3, 0.1], [-2, -3]])
        self._test_plane_projection(expected=expeceted, plane="xy")

    def test_plane_projection_yz(self):
        expeceted = np.array([[1, 1], [0.1, -5], [-3, -4.1]])
        self._test_plane_projection(expected=expeceted, plane="yz")

    def test_plane_projection_zz(self):
        expeceted = np.array([[1, 1], [0.3, -5], [-2, -4.1]])
        self._test_plane_projection(expected=expeceted, plane="zx")

    def test_calculate_spot_longitudes(self):
        raise Exception("Unfinished unittest")

    def test_surface_area_coverage_not_partial(self):
        size = 5
        visible = [0, 2]
        coverage = [10, 20]

        obtained = geo.surface_area_coverage(size, visible, coverage)
        expected = np.array([10., 0., 20., 0., 0.])
        self.assertTrue(np.all(obtained == expected))

    def test_surface_area_coverage_partial(self):
        size = 5
        visible = [0, 2]
        partial = [1]
        coverage = [10, 20]
        partial_coverage = [30]

        obtained = geo.surface_area_coverage(size, visible, coverage, partial, partial_coverage)
        expected = np.array([10., 30., 20., 0., 0.])
        self.assertTrue(np.all(obtained == expected))

    def test_faces_to_pypex_poly(self):
        points = np.array([[1, 1], [0.3, 0.1], [-2, -3]])
        faces = np.array([points[[0, 1, 2]]])
        obtained = geo.faces_to_pypex_poly(faces)[0]
        expected = polygon.Polygon(points)
        self.assertTrue(obtained == expected)

    @staticmethod
    def test_pypex_poly_surface_area():
        points = np.array([[1, 1], [0.3, 0.1], [-2, -3]])
        polygons = [None, polygon.Polygon(points), None]
        obtained = np.round(geo.pypex_poly_surface_area(polygons), 5)
        expected = [0.0, 0.05, 0.0]
        assert_array_equal(obtained, expected)

    def test_hull_to_pypex_poly(self):
        hull = np.array([[0, 0], [0, 1], [1, 1]])
        obtained = geo.hull_to_pypex_poly(hull)
        self.assertTrue(isinstance(obtained, polygon.Polygon))


class SystemOrbitalPositionTestCase(unittest.TestCase):
    pass


class OrbitalSupplementsTestCase(unittest.TestCase):
    pass
