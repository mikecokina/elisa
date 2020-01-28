import numpy as np
import astropy.units as u

from numpy.testing import assert_array_equal

from unittests.utils import ElisaTestCase

from elisa.analytics.dataset.base import RVdata


class RVdatasetTestCase(ElisaTestCase):
    @staticmethod
    def test_unit_conversion():
        time = [0.2*86400, 0.5*86400]
        time_unit = u.s
        rv_data = [50.0, 60.0]
        rv_unit = u.km / u.s

        rv_dataset = RVdata(
            x_data=time,
            y_data=rv_data,
            x_unit=time_unit,
            y_unit=rv_unit
        )

        expected_times = [0.2, 0.5]
        expected_rv = [50000.0, 60000.0]
        assert_array_equal(np.round(expected_times, 10), np.round(rv_dataset.x_data, 10))
        assert_array_equal(np.round(expected_rv, 10), np.round(rv_dataset.y_data))

    @staticmethod
    def test_phase_conversion():
        time = [0.2, 0.5]
        time_unit = None
        rv_data = [50.0, 60.0]
        rv_unit = u.km / u.s

        rv_dataset = RVdata(
            x_data=time,
            y_data=rv_data,
            x_unit=time_unit,
            y_unit=rv_unit
        )

        expected_times = [0.2, 0.5]
        expected_rv = [50000.0, 60000.0]
        assert_array_equal(np.round(expected_times, 10), np.round(rv_dataset.x_data, 10))
        assert_array_equal(np.round(expected_rv, 10), np.round(rv_dataset.y_data))
