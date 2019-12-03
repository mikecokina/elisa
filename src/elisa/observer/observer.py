import os
import sys
import numpy as np
import pandas as pd

from multiprocessing.pool import Pool
from scipy import interpolate

from elisa.binary_system.system import BinarySystem
from elisa.observer import mp
from elisa.observer.plot import Plot
from elisa.conf import config
from elisa.utils import is_empty
from elisa.logger import getLogger
from elisa.binary_system import utils as bsutils
from elisa import (
    units,
    umpy as up,
    utils
)

logger = getLogger('observer.observer')


class PassbandContainer(object):
    def __init__(self, table, passband):
        """
        Setup PassbandContainier object. It carres dependedncies of throughputs on wavelengths for given passband.

        :param table: pandads.DataFrame;
        :param passband: str;
        """
        self.left_bandwidth = np.nan
        self.right_bandwidth = np.nan
        self.akima = None
        self._table = pd.DataFrame({})
        self.wave_unit = "angstrom"
        self.passband = passband
        # in case this np.pi will stay here, there will be rendundant multiplication in intensity integration
        self.wave_to_si_mult = 1e-10

        setattr(self, 'table', table)

    @property
    def table(self):
        """
        Return pandas dataframe which represent pasband table as dependecy of throughput on wavelength.

        :return: pandas.DataFrame;
        """
        return self._table

    @table.setter
    def table(self, df):
        """
        Setter for passband table.
        It precompute left and right bandwidth for given table and also interpolation function placeholder.
        Akima1DInterpolator is used. If `bolometric` passband is used then interpolation function is like::

            lambda x: 1.0


        :param df: pandas.DataFrame;
        """
        self._table = df
        self.akima = Observer.bolometric if (self.passband.lower() in ['bolometric']) else \
            interpolate.Akima1DInterpolator(df[config.PASSBAND_DATAFRAME_WAVE],
                                            df[config.PASSBAND_DATAFRAME_THROUGHPUT])
        self.left_bandwidth = min(df[config.PASSBAND_DATAFRAME_WAVE])
        self.right_bandwidth = max(df[config.PASSBAND_DATAFRAME_WAVE])


class Observables(object):
    def __init__(self, observer):
        self.observer = observer

    def lc(self, from_phase=None, to_phase=None, phase_step=None, phases=None, normalize=False):
        return self.observer.lc(from_phase, to_phase, phase_step, phases, normalize)

    def rv(self, from_phase=None, to_phase=None, phase_step=None, phases=None, normalize=False):
        return self.observer.rv(from_phase, to_phase, phase_step, phases, normalize)


class Observer(object):
    def __init__(self, passband, system):
        """
        Initializer for observer class.

        :param passband: string; for valid filter name see config.py file
        :param system: system instance (BinarySystem or SingleSystem)
        """
        logger.info("initialising Observer instance")
        # specifying what kind of system is observed
        self._system = system
        self._system_cls = type(self._system)

        self.left_bandwidth = sys.float_info.max
        self.right_bandwidth = 0.0
        self.passband = dict()
        self.init_passband(passband)

        self._observables = ['lc', 'rv']
        self.phases = None
        self.times = None
        self.fluxes = None
        self.fluxes_unit = None
        self.radial_velocities = None
        self.rv_unit = None

        self.plot = Plot(self)
        self.observe = Observables(self)

    @staticmethod
    def bolometric(x):
        """
        Bolometric passband interpolation function in way of lambda x: 1.0

        :param x:
        :return: float or numpy.array; 1.0s in shape of x
        """
        if isinstance(x, (float, int)):
            return 1.0
        if isinstance(x, list):
            return [1.0] * len(x)
        if isinstance(x, np.ndarray):
            return np.array([1.0] * len(x))

    def init_passband(self, passband):
        """
        Passband initializing method for Observer instance.
        During initialialization `self.passband` Dict is fill in way::

            {`passband`: PassbandContainer()}

        and global left and right passband bandwidth is set.
        If there is several passband defined on different intervals, e.g. ([350, 650], [450, 750]) then global limits
        are total boarder values, in example case as [350, 750].

        :param passband: Union[str; Iterable[str]]
        """
        passband = [passband] if isinstance(passband, str) else passband
        for band in passband:
            if band in ['bolometric']:
                df = pd.DataFrame(
                    {config.PASSBAND_DATAFRAME_THROUGHPUT: [1.0, 1.0],
                     config.PASSBAND_DATAFRAME_WAVE: [0.0, sys.float_info.max]})
                right_bandwidth = sys.float_info.max
                left_bandwidth = 0.0
            else:
                df = self.get_passband_df(band)
                left_bandwidth = df[config.PASSBAND_DATAFRAME_WAVE].min()
                right_bandwidth = df[config.PASSBAND_DATAFRAME_WAVE].max()

            self.setup_bandwidth(left_bandwidth=left_bandwidth, right_bandwidth=right_bandwidth)
            self.passband[band] = PassbandContainer(table=df, passband=band)

    def setup_bandwidth(self, left_bandwidth, right_bandwidth):
        """
        Find whether supplied left and right bandwidth are in currently set boundaries and nothing has to be done
        or any is out of current bound and related has to be changed to higher (`right_bandwidth`)
        or lower (`left_bandwidth`).


        :param left_bandwidth: float;
        :param right_bandwidth: float;
        """
        if left_bandwidth < self.left_bandwidth:
            self.left_bandwidth = left_bandwidth
        if right_bandwidth > self.right_bandwidth:
            self.right_bandwidth = right_bandwidth

    @staticmethod
    def get_passband_df(passband):
        """
        Read content o passband table (csv file) based on passband name.

        :param passband: str;
        :return: pandas.DataFrame;
        """
        if passband not in config.PASSBANDS:
            raise ValueError('Invalid or unsupported passband function')
        file_path = os.path.join(config.PASSBAND_TABLES, str(passband) + '.csv')
        df = pd.read_csv(file_path)
        df[config.PASSBAND_DATAFRAME_WAVE] = df[config.PASSBAND_DATAFRAME_WAVE] * 10.0
        return df

    def lc(self, from_phase=None, to_phase=None, phase_step=None, phases=None, normalize=False):
        """
        Method for observation simulation. Based on input parmeters and supplied Ob server system on initialization
        will compute lightcurve.

        :param normalize: bool;
        :param from_phase: float;
        :param to_phase: float;
        :param phase_step: float;
        :param phases: Iterable float;
        :return: Dict;
        """

        if phases is None and (from_phase is None or to_phase is None or phase_step is None):
            raise ValueError("Missing arguments. Specify phases.")

        if is_empty(phases):
            phases = up.arange(start=from_phase, stop=to_phase, step=phase_step)

        phases = np.array(phases)

        # reduce phases to only unique ones from interval (0, 1) in general case without pulsations
        base_phases, base_phases_to_origin = self.phase_interval_reduce(phases)

        logger.info(f"observation is running")
        # calculates lines of sight for corresponding phases
        position_method = self._system.get_positions_method()

        lc_kwargs = dict(
            passband=self.passband,
            left_bandwidth=self.left_bandwidth,
            right_bandwidth=self.right_bandwidth,
            atlas="ck04",
            phases=base_phases,
            position_method=position_method
        )

        if config.NUMBER_OF_PROCESSES > 1 and self._system.is_eccentric():
            batch_size = int(np.ceil(len(base_phases) / config.NUMBER_OF_PROCESSES))
            phase_batches = utils.split_to_batches(batch_size=batch_size, array=base_phases)
            func = self._system.compute_lightcurve

            pool = Pool(processes=config.NUMBER_OF_PROCESSES)
            result = [pool.apply_async(mp.observe_lc_worker, (func, batch_idx, batch, lc_kwargs))
                      for batch_idx, batch in enumerate(phase_batches)]
            pool.close()
            pool.join()
            # this will return output in same order as was given on apply_async init
            result = [r.get() for r in result]
            curves = bsutils.renormalize_async_result(result)
        else:
            curves = self._system.compute_lightcurve(**lc_kwargs)

        # remap unique phases back to original phase interval
        for items in curves:
            curves[items] = np.array(curves[items])[base_phases_to_origin]

            # adding additional light
            correction = np.mean(curves[items]) * self._system.additional_light / (1.0 - self._system.additional_light)
            curves[items] += correction

        self.phases = phases
        if normalize:
            # TODO: here develop lc normalization method
            self.fluxes_unit = units.dimensionless_unscaled
        else:
            self.fluxes = curves
            self.fluxes_unit = units.W / units.m**2
        logger.info("observation finished")
        return phases, curves

    def rv(self, from_phase=None, to_phase=None, phase_step=None, phases=None, normalize=False):
        """
        Method for simulation of observation radial velocity curves.

        :param normalize: bool;
        :param from_phase: float;
        :param to_phase: float;
        :param phase_step: float;
        :param phases: Iterable float;
        :return: Tuple[numpy.array, numpy.array, numpy.array]; phases, primary rv, secondary rv
        """

        if phases is None and (from_phase is None or to_phase is None or phase_step is None):
            raise ValueError("Missing arguments. Specify phases.")

        if is_empty(phases):
            phases = up.arange(start=from_phase, stop=to_phase, step=phase_step)
        phases = np.array(phases)
        phases, primary_rv, secondary_rv = self._system.compute_rv(
            **dict(
                phases=phases,
                position_method=self._system.get_positions_method()
            )
        )

        self.rv_unit = units.m / units.s
        if normalize:
            self.rv_unit = units.dimensionless_unscaled
            _max = np.max([primary_rv, secondary_rv])
            primary_rv /= _max
            secondary_rv /= _max

        return phases, primary_rv, secondary_rv

    def phase_interval_reduce(self, phases):
        """
        Function reduces original phase interval to base interval (0, 1) in case of LC without pulsations.

        :param phases: ndarray; phases to reduce
        :return: Tuple; (base_phase: ndarray, reverse_indices: ndarray)

        ::

            base_phases:  ndarray of unique phases between (0, 1)
            reverse_indices: ndarray mask applicable to `base_phases` which will reconstruct original `phases`
        """
        if self._system_cls == BinarySystem or str(self._system_cls) == str(BinarySystem):
            # function shouldn't search for base phases if system has pulsations or is assynchronous with spots
            has_pulsation_test = self._system.primary.has_pulsations() | self._system.secondary.has_pulsations()

            test1 = (self._system.primary.synchronicity != 1.0) & self._system.primary.has_spots()
            test2 = (self._system.secondary.synchronicity != 1.0) & self._system.secondary.has_spots()
            assynchronous_spotty_test = test1 | test2

            if has_pulsation_test | assynchronous_spotty_test:
                return phases, up.arange(phases.shape[0])
            else:
                base_interval = np.round(phases % 1, 9)
                return np.unique(base_interval, return_inverse=True)
        else:
            raise NotImplemented("not implemented")
