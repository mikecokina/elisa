from __future__ import annotations

# The tests package selects Astropy constants before test-module imports.
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_equal
from pandas.testing import assert_frame_equal

from elisa import settings
from elisa import umpy as up
from elisa.base.error import ElisaOperationInputError
from elisa.binary_system.system import BinarySystem
from elisa.observer.observer import Observer
from elisa.observer.passband import PassbandContainer, bolometric
from elisa.observer.plot import Plot
from elisa.observer.utils import convert_to_magnitudes
from elisa.photometric_standards.standards_handlers import load_standard
from elisa.single_system.system import SingleSystem
from tests import set_astropy_units
from tests.utils import ElisaTestCase

set_astropy_units()


class TestPassbandContainer(ElisaTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._data_path = str(
            Path(__file__).resolve().parent.parent.joinpath("data", "passband"),
        )
        self._bessel_v_df = pd.read_csv(str(Path(self._data_path).joinpath("Generic.Bessell.V.csv")))
        self.pb = PassbandContainer(self._bessel_v_df, "Generic.Bessell.V")

    def test_table_setter_bandwidth(self) -> None:
        miw, maw = min(self._bessel_v_df["wavelength"]), max(self._bessel_v_df["wavelength"])
        assert_array_equal([miw, maw], [self.pb.left_bandwidth, self.pb.right_bandwidth])

    def test_table_setter_akima(self) -> None:
        w = [450, 750.0, 746.5]
        expected = [0.000032, 0.000010, 0.000022]
        obtained = [np.round(self.pb.akima([_w])[0], 6) for _w in w]
        obtained_direct = np.round(self.pb.akima(w), 6)
        assert_array_equal(expected, obtained)
        assert_array_equal(expected, obtained_direct)

    def test_table_setter__table(self) -> None:
        assert_frame_equal(self._bessel_v_df, self.pb.table)

    def test_default_wave_unit(self) -> None:
        assert self.pb.wave_unit == "angstrom"

    def test_default_wave_to_si_mult(self) -> None:
        assert self.pb.wave_to_si_mult == 1e-10

    def test_bolometric_bandwidth(self) -> None:
        df = pd.DataFrame(
            {
                settings.PASSBAND_DATAFRAME_THROUGHPUT: [1.0, 1.0],
                settings.PASSBAND_DATAFRAME_WAVE: [0.0, sys.float_info.max],
            },
        )
        self.pb.passband = "bolometric"
        self.pb.table = df

        assert_array_equal([0.0, sys.float_info.max], [self.pb.left_bandwidth, self.pb.right_bandwidth])

    def test_bolometric_akima(self) -> None:
        df = pd.DataFrame(
            {
                settings.PASSBAND_DATAFRAME_THROUGHPUT: [1.0, 1.0],
                settings.PASSBAND_DATAFRAME_WAVE: [0.0, sys.float_info.max],
            },
        )
        self.pb.passband = "bolometric"
        self.pb.table = df

        n, ma = 150, np.iinfo(np.intp).max
        expected = [1.0] * n
        obtained = [self.pb.akima([random.randrange(ma)])[0] for _ in range(n)]
        obtained_direct = self.pb.akima([random.randrange(ma) for _ in range(n)])
        assert_array_equal(expected, obtained)
        assert_array_equal(expected, obtained_direct)


class TestObserver(ElisaTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._data_path = str(
            Path(__file__).resolve().parent.parent.joinpath("data", "passband"),
        )
        self._passband = "Generic.Bessell.V"
        self._bessel_v_df = pd.read_csv(str(Path(self._data_path).joinpath(f"{self._passband}.csv")))
        settings.configure(PASSBAND_TABLES=self._data_path)

    def test_bolometric(self) -> None:
        r = random.randrange
        m = int(sys.float_info.max)
        expected = [1.0, [1.0, 1.0, 1.0], np.array([1.0, 1.0, 1.0])]
        obtained = [
            bolometric(r(m)),
            bolometric([r(m) for _ in range(3)]),
            bolometric(np.array([r(m) for _ in range(3)])),
        ]

        for e, o in zip(expected, obtained, strict=True):
            assert type(o) is type(e)
            assert_array_equal(e, o)

    def test__systemc_cls(self) -> None:
        s = BinarySystemMock()
        o = Observer(self._passband, s)
        assert o._system_cls == BinarySystemMock

    def test_get_passband_df(self) -> None:
        obtained = PassbandContainer.get_passband_df(self._passband)
        obtained[settings.PASSBAND_DATAFRAME_WAVE] = obtained[settings.PASSBAND_DATAFRAME_WAVE] / 10.0
        assert_frame_equal(obtained, self._bessel_v_df)

    def test_init_passband_has_table(self) -> None:
        s = BinarySystemMock()
        o = Observer(self._passband, s)
        expected = self._bessel_v_df
        obtained = o.passband[self._passband].table
        obtained[settings.PASSBAND_DATAFRAME_WAVE] = obtained[settings.PASSBAND_DATAFRAME_WAVE] / 10.0
        assert_frame_equal(expected, obtained)

    def test_setup_bandwidth(self) -> None:
        passbands = ["Generic.Bessell.V", "SLOAN.SDSS.g"]
        s = BinarySystemMock()
        o = Observer(passbands, s)
        expected = [3700.0, 7500.0]
        obtained = [o.left_bandwidth, o.right_bandwidth]
        assert_array_equal(expected, obtained)

    def test_setup_bandwidth_bolometric_in(self) -> None:
        passbands = ["Generic.Bessell.V", "SLOAN.SDSS.g", "bolometric"]
        s = BinarySystemMock()
        o = Observer(passbands, s)
        expected = [0.0, sys.float_info.max]
        obtained = [o.left_bandwidth, o.right_bandwidth]
        assert_array_equal(expected, obtained)

    def test_init_passband_str_vs_list(self) -> None:
        s = BinarySystemMock()
        o_str = Observer(self._passband, s)
        o_lst = Observer([self._passband], s)
        assert len(o_str.passband) == len(o_lst.passband)

    def test_BinarySystem_phase_interval_reduce_has_no_pulsation(self) -> None:
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

    def test_BinarySystem_phase_interval_reduce_has_pulsation_and_no_spots(self) -> None:
        s = BinarySystemMock(pp=False, sp=True)
        o = Observer(self._passband, s)
        o._system_cls = BinarySystem

        phases = np.array([-0.1, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 3.5])
        obtained_ph, obtained_ri = o.phase_interval_reduce(phases)

        expected_ph = phases
        expected_ri = up.arange(0, 14, 1)

        assert_array_equal(np.round(expected_ph, 2), np.round(obtained_ph, 2))
        assert_array_equal(expected_ri, obtained_ri)

        expected_phases = np.array(phases)
        obtained_phases = obtained_ph[obtained_ri]

        assert_array_equal(np.round(expected_phases, 2), np.round(obtained_phases, 2))

    def test_SingleSystem_phase_interval_on_clear_surface(self) -> None:
        s = SingleSystemMock(p=False, s=False)
        o = Observer(self._passband, s)
        o._system_cls = SingleSystem

        phases = np.array([-0.1, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 3.5])
        obtained_ph, obtained_ri = o.phase_interval_reduce(phases)

        expected_ph = [0.0]
        expected_ri = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert_array_equal(np.round(expected_ph, 2), np.round(obtained_ph, 2))
        assert_array_equal(expected_ri, obtained_ri)

        expected_phases = np.zeros(phases.shape)
        obtained_phases = obtained_ph[obtained_ri]

        assert_array_equal(np.round(expected_phases, 2), np.round(obtained_phases, 2))

    def test_SingleSystem_phase_interval_reduce_with_spots_no_pulsation(self) -> None:
        s = SingleSystemMock(p=False, s=True)
        o = Observer(self._passband, s)
        o._system_cls = SingleSystem

        phases = np.array([-0.1, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 3.5])
        obtained_ph, obtained_ri = o.phase_interval_reduce(phases)

        expected_ph = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        expected_ri = [9, 0, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 5]

        assert_array_equal(np.round(expected_ph, 2), np.round(obtained_ph, 2))
        assert_array_equal(expected_ri, obtained_ri)

        expected_phases = np.array(phases) % 1
        obtained_phases = obtained_ph[obtained_ri]

        assert_array_equal(np.round(expected_phases, 2), np.round(obtained_phases, 2))

    def test_BinarySystem_phase_interval_reduce_has_pulsation(self) -> None:
        s1 = SingleSystemMock(p=True, s=True)
        s2 = SingleSystemMock(p=True, s=False)
        o1 = Observer(self._passband, s1)
        o2 = Observer(self._passband, s2)
        o1._system_cls = SingleSystem
        o2._system_cls = SingleSystem

        phases = np.array([-0.1, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 3.5])
        obtained_ph1, obtained_ri1 = o1.phase_interval_reduce(phases)
        obtained_ph2, obtained_ri2 = o2.phase_interval_reduce(phases)

        expected_ph = phases
        expected_ri = up.arange(0, 14, 1)

        assert_array_equal(np.round(expected_ph, 2), np.round(obtained_ph1, 2))
        assert_array_equal(np.round(expected_ph, 2), np.round(obtained_ph2, 2))
        assert_array_equal(expected_ri, obtained_ri1)
        assert_array_equal(expected_ri, obtained_ri2)

        expected_phases = np.array(phases)
        obtained_phases1 = obtained_ph1[obtained_ri1]
        obtained_phases2 = obtained_ph2[obtained_ri2]

        assert_array_equal(np.round(expected_phases, 2), np.round(obtained_phases1, 2))
        assert_array_equal(np.round(expected_phases, 2), np.round(obtained_phases2, 2))

    def test_convert_to_magnitudes(self) -> None:
        for system in ["vega", "ab", "st"]:
            zero_points = load_standard(system)
            input_curves = {}
            for passband, zp in zero_points["fluxes"].items():
                input_curves[passband] = np.array([zp])

            mags = convert_to_magnitudes(input_curves, zero_points)
            for passband, zm in zero_points["reference_magnitudes"].items():
                assert_equal(mags[passband][0], zm)

    def test_system_cls_property(self) -> None:
        """Test Observer.system_cls property getter and setter."""
        s = BinarySystemMock()
        o = Observer(self._passband, s)
        assert o.system_cls == BinarySystemMock
        o.system_cls = SingleSystemMock
        assert o.system_cls == SingleSystemMock

    def test_flux_unit_property(self) -> None:
        """Test Observer.flux_unit property getter and setter."""
        s = BinarySystemMock()
        o = Observer(self._passband, s)
        assert o.flux_unit == o._flux_unit
        o.flux_unit = "W/m2"
        assert str(o.flux_unit) == "W / m2"

    def test_lc_invalid_flux_unit(self) -> None:
        """Test Observer.lc raises ValueError for invalid flux_unit."""
        s = BinarySystemMock()
        o = Observer(self._passband, s)
        with pytest.raises(ValueError):  # noqa: PT011 - assert type, identity, or reference parity without promising dependency wording
            # noinspection PyTypeChecker
            o.lc(flux_unit="invalid_unit")

    def test_plot_and_observe_attributes(self) -> None:
        """Test Observer.plot and Observer.observe attributes exist and are correct type."""
        s = BinarySystemMock()
        o = Observer(self._passband, s)
        assert o.plot is not None
        assert o.observe is not None
        assert isinstance(o.plot, Plot)
        assert hasattr(o.observe, "lc")
        assert hasattr(o.observe, "rv")

    def test_manage_time_series_phase_and_time(self) -> None:
        """Test Observer.manage_time_series for phase and time domain input and error cases."""
        s = BinarySystemMock()
        o = Observer(self._passband, s)
        # Phase domain
        phases = o.manage_time_series(from_phase=0.0, to_phase=1.0, phase_step=0.1)
        assert np.allclose(phases + o._system.phase_shift, np.arange(0.0, 1.0, 0.1))
        # Time domain
        times = np.arange(0.0, 1.0, 0.1)
        phases_time = o.manage_time_series(from_time=0.0, to_time=1.0, time_step=0.1, times=times)
        assert len(phases_time) == len(times)
        # Error: both phase and time domain
        with pytest.raises(ElisaOperationInputError):
            o.manage_time_series(
                from_phase=0.0,
                to_phase=1.0,
                phase_step=0.1,
                from_time=0.0,
                to_time=1.0,
                time_step=0.1,
            )
        # Error: missing arguments
        with pytest.raises(ElisaOperationInputError):
            o.manage_time_series()


class BinarySystemMock:
    class Star:
        """Observer test double with the interface required by this scenario."""

        def __init__(self, p: bool = False) -> None:
            self.p = p
            self._synchronicity = 1.0

        def has_pulsations(self) -> bool:
            return self.p

        @staticmethod
        def has_spots() -> bool:
            return False

        @property
        def synchronicity(self) -> float:
            return self._synchronicity

    def __init__(self, pp: bool = False, sp: bool = False) -> None:
        self.primary = self.Star(pp)
        self.secondary = self.Star(sp)
        self.phase_shift = 0.0
        self.period = 1.0
        self.t0 = 0.0


class SingleSystemMock:
    class Star:
        """Observer test double with the interface required by this scenario."""

        def __init__(self, p: bool = False, s: bool = False) -> None:
            self.p = p
            self.s = s

        def has_pulsations(self) -> bool:
            return self.p

        def has_spots(self) -> bool:
            return self.s

    def __init__(self, p: bool = False, s: bool = False) -> None:
        self.star = self.Star(p, s)
        self.phase_shift = 0.0
        self.period = 1.0
        self.t0 = 0.0
