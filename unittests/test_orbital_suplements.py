import numpy as np

from numpy.testing import assert_array_equal
from unittests.utils import ElisaTestCase
from elisa.binary_system.orbit.container import OrbitalSupplements


class OrbitalSupplementsTestCase(ElisaTestCase):
    @staticmethod
    def test_init():
        bodies = [[1.0, 10], [2.0, 20], [3.0, 30]]
        mirrors = [[1.0, 10], [np.nan, np.nan], [3.0, 30]]

        supplements = OrbitalSupplements(bodies, mirrors)

        assert_array_equal(bodies, supplements.body)
        assert_array_equal(mirrors, supplements.mirror)

    @staticmethod
    def test_append():
        ad = ([0.0, 0.0], [5.0, 50.0])
        bodies = [[1.0, 10], [2.0, 20], [3.0, 30]]
        mirrors = [[1.0, 10], [np.nan, np.nan], [3.0, 30]]

        supplements = OrbitalSupplements(bodies, mirrors)
        supplements.append(*ad)

        bodies.append(ad[0])
        mirrors.append(ad[1])

        assert_array_equal(bodies, supplements.body)
        assert_array_equal(mirrors, supplements.mirror)

    def test_append_of_zero_value(self):
        supplements = OrbitalSupplements()
        supplements.append([1, 1], np.array([10, 10]))
        self.assertTrue(len(supplements.body.shape) == 2)

    def test_is_empty(self):
        arr = [[np.nan, np.nan], [np.nan, np.nan]]
        self.assertTrue(OrbitalSupplements.is_empty(arr))

        arr = [[6, 10], [np.nan, np.nan]]
        self.assertFalse(OrbitalSupplements.is_empty(arr))

    def test_not_empty(self):
        arr = np.array([[10, 10], [np.nan, np.nan], [20, 12], [np.nan, np.nan]])
        not_empty = OrbitalSupplements.not_empty(arr)
        self.assertTrue(np.all(np.array([[10, 10], [20, 12]]) == not_empty))

    def test_defined_only(self):
        arr = np.array([[10, 10], [np.nan, np.nan], [20, 12], [np.nan, np.nan]])
        supplements = OrbitalSupplements(arr, arr)
        not_empty = supplements.body_defined
        self.assertTrue(np.all(np.array([[10, 10], [20, 12]]) == not_empty))

        not_empty = supplements.mirror_defined
        self.assertTrue(np.all(np.array([[10, 10], [20, 12]]) == not_empty))

    def test_sort_by_index(self):
        bodies = [[2.0, 20], [1.0, 10], [4.0, 30], [3.0, 40]]
        mirrors = [[1.0, 10], [0.0, 0.0], [3.0, 30], [100.0, 5.0]]

        expected = OrbitalSupplements(
            [[1., 10.], [2., 20.], [3., 40.], [4., 30.]],
            [[0., 0.], [1., 10.], [100., 5.], [3., 30.]]
        )

        supplements = OrbitalSupplements(bodies, mirrors)
        obtained = supplements.sort(by="index")

        self.assertTrue(obtained == expected)

    def test_sort_by_distance(self):
        bodies = [[2.0, 20], [1.0, 10], [4.0, 30], [3.0, 40]]
        mirrors = [[1.0, 10], [0.0, 0.0], [3.0, 30], [100.0, 5.0]]

        expected = OrbitalSupplements(
            [[1., 10.], [2., 20.], [4., 30.], [3., 40.]],
            [[0., 0.], [1., 10.], [3., 30.], [100., 5.]]
        )

        supplements = OrbitalSupplements(bodies, mirrors)
        obtained = supplements.sort(by="distance")

        self.assertTrue(obtained == expected)
