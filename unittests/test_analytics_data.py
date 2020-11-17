import os.path as op
import numpy as np

from numpy.testing import assert_array_equal
from elisa.analytics.dataset.utils import read_data_file
from unittests.utils import ElisaTestCase
from elisa.analytics import RVData, LCData
from elisa import units as u


class DataTestCase(ElisaTestCase):
    DATA = op.join(op.abspath(op.dirname(__file__)), "data", "dataset")


class DataReadTestCase(DataTestCase):
    def test_read_data_file(self):
        fpath = op.join(self.DATA, "dummy.csv")
        data = read_data_file(fpath, data_columns=(0, 1, 2)).T

        self.assertEqual(len(data), 3)
        assert_array_equal([0, 1, 2, 3, 4], data[0])
        assert_array_equal([1e-1] * 5, data[1])
        assert_array_equal([1e-3, 1e-3, 1e-3, 2e-3, 2e-3], data[2])


class LVDataTestCase(DataTestCase):
    def test_from_file(self):
        fpath = op.join(self.DATA, "dummy.csv")
        x_unit = u.dimensionless_unscaled
        y_unit = u.km/u.s
        rv_data = RVData.from_file(fpath, x_unit, y_unit=y_unit)

        self.assertEqual(rv_data.y_unit, u.m / u.s)
        self.assertEqual(rv_data.x_unit, u.dimensionless_unscaled)

        assert_array_equal(rv_data.x_data, np.arange(0, 5, 1))
        assert_array_equal(rv_data.y_data, [100] * 5)


class RVDataTestCase(DataTestCase):
    def test_from_file(self):
        fpath = op.join(self.DATA, "dummy.csv")
        x_unit = u.dimensionless_unscaled
        y_unit = u.dimensionless_unscaled
        lc_data = LCData.from_file(fpath, x_unit, y_unit=y_unit)

        self.assertEqual(lc_data.y_unit, u.dimensionless_unscaled)
        self.assertEqual(lc_data.x_unit, u.dimensionless_unscaled)

        assert_array_equal(lc_data.x_data, np.arange(0, 5, 1))
        assert_array_equal(lc_data.y_data, [0.1] * 5)

    def test_from_file_in_mag(self):
        fpath = op.join(self.DATA, "dummy.csv")
        x_unit = u.dimensionless_unscaled
        y_unit = u.mag
        lc_data = LCData.from_file(fpath, x_unit, y_unit=y_unit, reference_magnitude=1.0)

        assert_array_equal(lc_data.x_data, np.arange(0, 5, 1))
        assert_array_equal(np.round(lc_data.y_data, 2), [2.29] * 5)
