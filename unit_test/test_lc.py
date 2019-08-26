import unittest

import numpy as np

from elisa.binary_system.lc import find_apsidally_corresponding_positions
from elisa.binary_system.geo import OrbitalSupplements


class SupportMethodsTestCase(unittest.TestCase):
    def _test_find_apsidally_corresponding_positions(self, arr1, arr2, expected, tol=1e-10):
        obtained = find_apsidally_corresponding_positions(arr1, arr2, tol)
        self.assertTrue(expected == obtained)

    # def test_find_apsidally_corresponding_positions_full_match(self):
    #     arr1 = np.array([1, 2, 3, 4, 5.0])
    #     arr2 = np.array([1, 3, 2, 4, 5.0])
    #     expected = OrbitalSupplements([1.0, 3.0, 2.0, 4.0, 5.0],
    #                                   [1.0, 3.0, 2.0, 4.0, 5.0])
    #     self._test_find_apsidally_corresponding_positions(arr1, arr2, expected)
    #
    # def test_find_apsidally_corresponding_positions_mixed_first_longer(self):
    #     arr1 = np.array([1, 2, 3, 4, 5.0, 6, 7])
    #     arr2 = np.array([1, 2, 3, 4, 5.5, 7])
    #     expected = OrbitalSupplements([1.0, 2.0, 3.0, 4.0, None, 7.0, 5.0, 6.0],
    #                                   [1.0, 2.0, 3.0, 4.0, 5.5, 7.0, None, None])
    #     self._test_find_apsidally_corresponding_positions(arr1, arr2, expected)
    #
    # def test_find_apsidally_corresponding_positions_mixed_second_longer(self):
    #     arr1 = np.array([1, 2, 3, 4, 5.5, 7])
    #     arr2 = np.array([1, 2, 3, 4, 5.0, 6, 7])
    #     expected = OrbitalSupplements([1.0, 2.0, 3.0, 4.0, None, None, 7.0, 5.5],
    #                                   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, None])
    #     self._test_find_apsidally_corresponding_positions(arr1, arr2, expected)
    #
    # def test_find_apsidally_corresponding_positions_mixed_under_tolerance(self):
    #     arr1 = np.array([1, 2, 3, 4])
    #     arr2 = np.array([1, 2, 2.02, 4])
    #     expected = OrbitalSupplements([1.0, 2.0, 2.0, 4.0, 3.0],
    #                                   [1.0, 2.0, 2.02, 4.0, None])
    #     self._test_find_apsidally_corresponding_positions(arr1, arr2, expected, tol=0.1)
    #
    # def test_find_apsidally_corresponding_positions_total_mixed(self):
    #     arr1 = np.array([1, 3, 5])
    #     arr2 = np.array([2, 4, 6])
    #     expected = OrbitalSupplements([None, None, None, 1, 3, 5],
    #                                   [2, 4, 6, None, None, None])
    #     self._test_find_apsidally_corresponding_positions(arr1, arr2, expected)
