import random
import sys
import unittest
import pandas as pd
import numpy as np

from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal
from elisa.conf import config
from os.path import join as pjoin
from os.path import dirname

from elisa.observer.observer import PassbandContainer


class TestPassbandContainer(unittest.TestCase):
    def setUp(self):
        self._data_path = pjoin(dirname(__file__), "data", "passband")
        self._bessel_v_df = pd.read_csv(pjoin(self._data_path, 'Generic.Bessell.V.csv'))
        self.pb = PassbandContainer(self._bessel_v_df, 'Generic.Bessell.V')

    def test_table_setter_bandwidth(self):
        miw, maw = min(self._bessel_v_df["wavelength"]), max(self._bessel_v_df["wavelength"])
        assert_array_equal([miw, maw], [self.pb.left_bandwidth, self.pb.right_bandwidth])

    def test_table_setter_akima(self):
        w = [450, 750.0, 746.5]
        expected = [0.000032, 0.000010, 0.000022]
        obtained = [np.round(self.pb.akima([_w])[0], 6) for _w in w]
        obtained_direct = np.round(self.pb.akima(w), 6)
        assert_array_equal(expected, obtained)
        assert_array_equal(expected, obtained_direct)

    def test_table_setter__table(self):
        assert_frame_equal(self._bessel_v_df, self.pb.table)

    def test_default_wave_to_si_mult(self):
        self.assertEqual(self.pb.wave_unit, "angstrom")

    def test_default_wave_unit(self):
        self.assertEqual(self.pb.wave_to_si_mult, 1e-10)

    def test_bolometric_bandwidth(self):
        df = pd.DataFrame(
            {config.PASSBAND_DATAFRAME_THROUGHPUT: [1.0, 1.0],
             config.PASSBAND_DATAFRAME_WAVE: [0.0, sys.float_info.max]})
        self.pb.passband = "bolometric"
        self.pb.table = df

        assert_array_equal([0.0, sys.float_info.max], [self.pb.left_bandwidth, self.pb.right_bandwidth])

    def test_bolometric_akima(self):
        df = pd.DataFrame(
            {config.PASSBAND_DATAFRAME_THROUGHPUT: [1.0, 1.0],
             config.PASSBAND_DATAFRAME_WAVE: [0.0, sys.float_info.max]})
        self.pb.passband = "bolometric"
        self.pb.table = df

        n, ma = 150, np.iinfo(np.intp).max
        expected = [1.0] * n
        obtained = [self.pb.akima([random.randrange(ma)])[0] for _ in range(n)]
        obtained_direct = self.pb.akima([random.randrange(ma) for _ in range(n)])
        assert_array_equal(expected, obtained)
        assert_array_equal(expected, obtained_direct)


class TestObserver(unittest.TestCase):
    pass
