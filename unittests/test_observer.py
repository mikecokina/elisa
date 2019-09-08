import random
import sys
import os

from os.path import dirname
from os.path import join as pjoin

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from elisa.binary_system.system import BinarySystem
from elisa.conf import config
from elisa.observer.observer import PassbandContainer, Observer
from unittests.utils import ElisaTestCase


class TestPassbandContainer(ElisaTestCase):
    def setUp(self):
        self._data_path = pjoin(dirname(os.path.abspath(__file__)), "data", "passband")
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


class TestObserver(ElisaTestCase):
    def setUp(self):
        self._data_path = pjoin(dirname(os.path.abspath(__file__)), "data", "passband")
        self._passband = 'Generic.Bessell.V'
        self._bessel_v_df = pd.read_csv(pjoin(self._data_path, f'{self._passband}.csv'))
        config.PASSBAND_TABLES = self._data_path

    def test_bolometric(self):
        r = random.randrange
        m = sys.float_info.max
        expected = [1.0, [1.0, 1.0, 1.0], np.array([1.0, 1.0, 1.0])]
        obtained = [

            Observer.bolometric(r(m)),
            Observer.bolometric([r(m) for _ in range(3)]),
            Observer.bolometric(np.array([r(m) for _ in range(3)]))
        ]

        for e, o in zip(expected, obtained):
            self.assertEqual(type(o), type(e))
            assert_array_equal(e, o)

    def test__systemc_cls(self):
        s = BinarySystemMock()
        o = Observer(self._passband, s)
        self.assertEqual(o._system_cls, BinarySystemMock)

    def test_get_passband_df(self):
        obtained = Observer.get_passband_df(self._passband)
        obtained[config.PASSBAND_DATAFRAME_WAVE] = obtained[config.PASSBAND_DATAFRAME_WAVE] / 10.0
        assert_frame_equal(obtained, self._bessel_v_df)

    def test_init_passband_has_table(self):
        s = BinarySystemMock()
        o = Observer(self._passband, s)
        expected = self._bessel_v_df
        obtained = o.passband[self._passband].table
        obtained[config.PASSBAND_DATAFRAME_WAVE] = obtained[config.PASSBAND_DATAFRAME_WAVE] / 10.0
        assert_frame_equal(expected, obtained)

    def test_setup_bandwidth(self):
        passbands = ['Generic.Bessell.V', 'SLOAN.SDSS.g']
        s = BinarySystemMock()
        o = Observer(passbands, s)
        expected = [3700.0, 7500.0]
        obtained = [o.left_bandwidth, o.right_bandwidth]
        assert_array_equal(expected, obtained)

    def test_setup_bandwidth_bolometric_in(self):
        passbands = ['Generic.Bessell.V', 'SLOAN.SDSS.g', 'bolometric']
        s = BinarySystemMock()
        o = Observer(passbands, s)
        expected = [0.0, sys.float_info.max]
        obtained = [o.left_bandwidth, o.right_bandwidth]
        assert_array_equal(expected, obtained)

    def test_init_passband_str_vs_list(self):
        s = BinarySystemMock()
        o_str = Observer(self._passband, s)
        o_lst = Observer([self._passband], s)
        self.assertEqual(len(o_str.passband), len(o_lst.passband))

    def test_BinarySystem_phase_interval_reduce_has_no_pulsation(self):
        s = BinarySystemMock(pp=False, sp=False)
        o = Observer(self._passband, s)
        o._system_cls = BinarySystem

        phases = np.array([-0.1, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 3.5])
        obtained_ph, obtained_ri = o.phase_interval_reduce(phases)

        expected_ph = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        expected_ri = [9, 0, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 5]

        assert_array_equal(np.round(expected_ph, 2), np.round(obtained_ph, 2))
        assert_array_equal(expected_ri, obtained_ri)

        expected_phases = np.array(phases) % 1
        obtained_phases = obtained_ph[obtained_ri]

        assert_array_equal(np.round(expected_phases, 2), np.round(obtained_phases, 2))

    def test_BinarySystem_phase_interval_reduce_has_pulsation_and_no_spots(self):
        s = BinarySystemMock(pp=False, sp=True)
        o = Observer(self._passband, s)
        o._system_cls = BinarySystem

        phases = np.array([-0.1, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 3.5])
        obtained_ph, obtained_ri = o.phase_interval_reduce(phases)

        expected_ph = phases
        expected_ri = np.arange(0, 14, 1)

        assert_array_equal(np.round(expected_ph, 2), np.round(obtained_ph, 2))
        assert_array_equal(expected_ri, obtained_ri)

        expected_phases = np.array(phases)
        obtained_phases = obtained_ph[obtained_ri]

        assert_array_equal(np.round(expected_phases, 2), np.round(obtained_phases, 2))


class BinarySystemMock(object):
    class Star(object):
        def __init__(self, p=False):
            self.p = p
            self._synchronicity = 1.0

        def has_pulsations(self):
            return self.p

        def has_spots(self):
            return False

        @property
        def synchronicity(self):
            return self._synchronicity

    def __init__(self, pp=False, sp=False):
        self.primary = self.Star(pp)
        self.secondary = self.Star(sp)
