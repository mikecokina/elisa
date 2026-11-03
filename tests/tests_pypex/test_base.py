import numpy as np
from numpy.testing import assert_array_equal

from elisa.pypex import poly2d
from elisa.pypex.base import shape
from tests.utils import ElisaTestCase


class BaseTestCase(ElisaTestCase):
    def test_polygon_validity_check_valid(self) -> None:
        valid_hull = [[0, 0], [1, 0], [0.5, 1]]
        assert shape.Shape2D.polygon_validity_check(valid_hull, _raise=False)

    def test_polygon_validity_check_invalid(self) -> None:
        invalid_hull = [[0, 0], [1], [0.5, 1]]
        assert not (shape.Shape2D.polygon_validity_check(invalid_hull, _raise=False))
        invalid_hull = [[0.5, 1]]
        assert not (shape.Shape2D.polygon_validity_check(invalid_hull, _raise=False))
        invalid_hull = [0.5, 1]
        assert not (shape.Shape2D.polygon_validity_check(invalid_hull, _raise=False))

    @staticmethod
    def test_2d_projection() -> None:
        vector1 = np.array([4.0, 3.0])
        vector2 = np.array([5.0, 0.0])

        obtained = poly2d.projection.projection(vector1, vector2)
        expected = np.array([4.0, 0.0])
        assert_array_equal(obtained, expected)
