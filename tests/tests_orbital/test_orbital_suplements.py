# The tests package selects Astropy constants before test-module imports.
import numpy as np
from numpy.testing import assert_array_equal

from elisa.binary_system.orbit.container import OrbitalSupplements
from tests import set_astropy_units
from tests.utils import ElisaTestCase

set_astropy_units()


class OrbitalSupplementsTestCase(ElisaTestCase):
    @staticmethod
    def test_init() -> None:
        bodies = [[1.0, 10], [2.0, 20], [3.0, 30]]
        mirrors = [[1.0, 10], [np.nan, np.nan], [3.0, 30]]

        supplements = OrbitalSupplements(bodies, mirrors)

        assert_array_equal(bodies, supplements.body)
        assert_array_equal(mirrors, supplements.mirror)

    @staticmethod
    def test_append() -> None:
        ad = ([0.0, 0.0], [5.0, 50.0])
        bodies = [[1.0, 10], [2.0, 20], [3.0, 30]]
        mirrors = [[1.0, 10], [np.nan, np.nan], [3.0, 30]]

        supplements = OrbitalSupplements(bodies, mirrors)
        supplements.append(*ad)

        bodies.append(ad[0])
        mirrors.append(ad[1])

        assert_array_equal(bodies, supplements.body)
        assert_array_equal(mirrors, supplements.mirror)

    def test_append_of_zero_value(self) -> None:
        supplements = OrbitalSupplements()
        supplements.append([1, 1], np.array([10, 10]))
        assert len(supplements.body.shape) == 2

    def test_is_empty(self) -> None:
        arr = [[np.nan, np.nan], [np.nan, np.nan]]
        assert OrbitalSupplements.is_empty(arr)

        arr = [[6, 10], [np.nan, np.nan]]
        assert not (OrbitalSupplements.is_empty(arr))

    def test_not_empty(self) -> None:
        arr = np.array([[10, 10], [np.nan, np.nan], [20, 12], [np.nan, np.nan]])
        not_empty = OrbitalSupplements.not_empty(arr)
        assert np.all(np.array([[10, 10], [20, 12]]) == not_empty)

    def test_defined_only(self) -> None:
        arr = np.array([[10, 10], [np.nan, np.nan], [20, 12], [np.nan, np.nan]])
        supplements = OrbitalSupplements(arr, arr)
        not_empty = supplements.body_defined
        assert np.all(np.array([[10, 10], [20, 12]]) == not_empty)

        not_empty = supplements.mirror_defined
        assert np.all(np.array([[10, 10], [20, 12]]) == not_empty)

    def test_sort_by_index(self) -> None:
        bodies = [[2.0, 20], [1.0, 10], [4.0, 30], [3.0, 40]]
        mirrors = [[1.0, 10], [0.0, 0.0], [3.0, 30], [100.0, 5.0]]

        expected = OrbitalSupplements(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 40.0], [4.0, 30.0]],
            [[0.0, 0.0], [1.0, 10.0], [100.0, 5.0], [3.0, 30.0]],
        )

        supplements = OrbitalSupplements(bodies, mirrors)
        obtained = supplements.sort(by="index")

        assert obtained == expected

    def test_sort_by_distance(self) -> None:
        bodies = [[2.0, 20], [1.0, 10], [4.0, 30], [3.0, 40]]
        mirrors = [[1.0, 10], [0.0, 0.0], [3.0, 30], [100.0, 5.0]]

        expected = OrbitalSupplements(
            [[1.0, 10.0], [2.0, 20.0], [4.0, 30.0], [3.0, 40.0]],
            [[0.0, 0.0], [1.0, 10.0], [3.0, 30.0], [100.0, 5.0]],
        )

        supplements = OrbitalSupplements(bodies, mirrors)
        obtained = supplements.sort(by="distance")

        assert obtained == expected


def test_sort_keys_accept_enum_and_preserve_string_behavior() -> None:
    from elisa.conf.enums import OrbitalSortKey

    body = np.array([[2.0, 0.5, 0.0, 0.0], [0.0, 1.5, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    mirror = body + 10.0
    for key in OrbitalSortKey:
        actual = OrbitalSupplements(body.copy(), mirror.copy()).sort(key)
        expected = OrbitalSupplements(body.copy(), mirror.copy()).sort(key.value)
        assert_array_equal(actual.body, expected.body)
        assert_array_equal(actual.mirror, expected.mirror)


def test_unknown_sort_key_is_a_contract_failure() -> None:
    import pytest

    from elisa.base.error import ElisaOperationInputError

    with pytest.raises(ElisaOperationInputError, match="Unsupported orbital sorting key"):
        OrbitalSupplements().sort("unknown")
