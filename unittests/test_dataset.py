import numpy as np

from numpy.testing import assert_array_equal
from unittests.utils import ElisaTestCase
from elisa.analytics.dataset.base import RVData, LCData
from elisa import units as u


class RVDatasetTestCase(ElisaTestCase):
    @staticmethod
    def test_unit_conversion():
        time = [0.2*86400, 0.5*86400]
        time_unit = u.s
        rv_data = [50.0, 60.0]
        rv_unit = u.km / u.s
        rv_err = [5.0, 6.0]

        rv_dataset = RVData(
            x_data=time,
            y_data=rv_data,
            x_unit=time_unit,
            y_unit=rv_unit,
            y_err=rv_err
        )

        expected_times = [0.2, 0.5]
        expected_rv = [50000.0, 60000.0]
        expected_rv_err = [5000.0, 6000.0]
        assert_array_equal(np.round(expected_times, 10), np.round(rv_dataset.x_data, 10))
        assert_array_equal(np.round(expected_rv, 10), np.round(rv_dataset.y_data, 10))
        assert_array_equal(np.round(expected_rv_err, 10), np.round(rv_dataset.y_err, 10))

    @staticmethod
    def test_phase_conversion():
        time = [0.2, 0.5]
        time_unit = None
        rv_data = [50.0, 60.0]
        rv_unit = u.km / u.s

        rv_dataset = RVData(
            x_data=time,
            y_data=rv_data,
            x_unit=time_unit,
            y_unit=rv_unit
        )

        expected_times = [0.2, 0.5]
        expected_rv = [50000.0, 60000.0]
        assert_array_equal(np.round(expected_times, 10), np.round(rv_dataset.x_data, 10))
        assert_array_equal(np.round(expected_rv, 10), np.round(rv_dataset.y_data, 10))


class LCDatasetTestCase(ElisaTestCase):
    @staticmethod
    def test_mag_to_flux_conversion():
        time = [0.2, 0.5]
        time_unit = u.dimensionless_unscaled
        lc_data = [10, 5]
        lc_unit = u.mag

        lc_dataset = LCData(
            x_data=time,
            y_data=lc_data,
            x_unit=time_unit,
            y_unit=lc_unit,
            reference_magnitude=10.0
        )

        expected_times = [0.2, 0.5]
        expected_lc = [1.0, 100.0]
        assert_array_equal(np.round(expected_times, 10), np.round(lc_dataset.x_data, 10))
        assert_array_equal(np.round(expected_lc, 10), np.round(lc_dataset.y_data, 10))

    @staticmethod
    def test_mag_to_flux_error_conversion():
        time = [0.2, 0.5]
        time_unit = u.dimensionless_unscaled
        lc_data = [10, 5]
        lc_unit = u.mag
        lc_err = [0.1, 0.1]

        lc_dataset = LCData(
            x_data=time,
            y_data=lc_data,
            x_unit=time_unit,
            y_unit=lc_unit,
            y_err=lc_err,
            reference_magnitude=10.0
        )

        expected_times = [0.2, 0.5]
        expected_lc_err = [0.09648, 0.09648]
        assert_array_equal(np.round(expected_times, 10), np.round(lc_dataset.x_data, 10))
        assert_array_equal(np.round(expected_lc_err, 5), np.round(lc_dataset.y_err, 5))
