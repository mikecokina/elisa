import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import interpolate

from elisa.conf import config
from elisa.engine.binary_system.system import BinarySystem
from elisa.engine.single_system.system import SingleSystem


config.set_up_logging()


class PassbandContainer(object):
    def __init__(self, table, passband):
        self.left_bandwidth = None
        self.right_bandwidth = None
        self.akima = None
        self._table = None
        self.wave_unit: str = "angstrom"
        self.passband: str = passband
        # in case this np.pi will stay here, there will be rendundant multiplication in intensity integration
        self.wave_to_si_mult: float = 1e-10

        setattr(self, 'table', table)

    @property
    def table(self):
        return self._table

    @table.setter
    def table(self, df: pd.DataFrame):
        self._table = df
        self.akima = Observer.bolometric if (self.passband.lower() in ['bolometric']) else \
            interpolate.Akima1DInterpolator(df[config.PASSBAND_DATAFRAME_WAVE],
                                            df[config.PASSBAND_DATAFRAME_THROUGHPUT])
        self.left_bandwidth = min(df[config.PASSBAND_DATAFRAME_WAVE])
        self.right_bandwidth = max(df[config.PASSBAND_DATAFRAME_WAVE])


class Observer(object):
    def __init__(self, passband: list or str, system: BinarySystem or SingleSystem):
        """
        initializer for observer class
        :param passband: string - for valid filter name see config.py file
        :param system:
        """
        self._logger = logging.getLogger(Observer.__name__)
        self._logger.info("initialising Observer instance")
        # specifying what kind of system is observed
        self._system = system
        self._system_cls = type(self._system)

        # self._system._suppress_logger = True

        self.left_bandwidth = sys.float_info.max
        self.right_bandwidth = 0.0
        self.passband = dict()
        self.init_passband(passband)

    @staticmethod
    def bolometric(*args, **kwargs):
        return 1.0

    def init_passband(self, passband):
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
        if left_bandwidth < self.left_bandwidth:
            self.left_bandwidth = left_bandwidth
        if right_bandwidth > self.right_bandwidth:
            self.right_bandwidth = right_bandwidth

    @staticmethod
    def get_passband_df(passband):
        logging.debug("obtaining passband response function: {}".format(passband))
        if passband not in config.PASSBANDS:
            raise ValueError('Invalid or unsupported passband function')
        file_path = os.path.join(config.PASSBAND_TABLES, str(passband) + '.csv')
        df = pd.read_csv(file_path)
        df[config.PASSBAND_DATAFRAME_WAVE] = df[config.PASSBAND_DATAFRAME_WAVE] * 10.0
        return df

    def observe(self, from_phase: float = None, to_phase: float = None, phase_step: float = None,
                phases: list or set = None):
        if not phases and (from_phase is None or to_phase is None or phase_step is None):
            raise ValueError("missing arguments")

        if phases is None:
            phases = np.arange(start=from_phase, stop=to_phase, step=phase_step)

        # reduce phases to only unique ones from interval (0, 1) in general case without pulsations
        base_phases, reverse_idx = self.base_phase_interval(phases)

        self._logger.info("observation start w/ following configuration {<add>}")
        # self._logger.warning("logger will be suppressed due multiprocessing incompatibility")
        """
        distance, azimut angle, true anomaly and phase
                           np.array((r1, az1, ni1, phs1),
                                    (r2, az2, ni2, phs2),
                                    ...
                                    (rN, azN, niN, phsN))
        """
        # calculates lines of sight for corresponding phases
        position_method = self._system.get_positions_method()
        args = position_method(phase=base_phases)

        curves = self._system.compute_lightcurve(
                     **dict(
                         positions=args,
                         passband=self.passband,
                         left_bandwidth=self.left_bandwidth,
                         right_bandwidth=self.right_bandwidth,
                         atlas="ck04",
                         phases=base_phases
                     )
                 )

        # pool = Pool(processes=config.NUMBER_OF_THREADS)
        # res = [pool.apply_async(mp.observe_worker,
        #                         (self._system.initial_kwargs, self._system_cls, _args)) for _args in args]
        # pool.close()
        # pool.join()
        # result_list = [np.array(r.get()) for r in res]
        #
        # print(result_list)
        # # r = np.array(sorted(result_list, key=lambda x: x[0])).T[1]
        # # return utils.spherical_to_cartesian(np.column_stack((r, phi, theta)))

        # remap unique phases back to original phase interval
        for items in curves:
            curves[items] = np.array(curves[items])[reverse_idx]
        self._logger.info("observation finished")
        return curves

    def base_phase_interval(self, phases):
        """
        function reduces original phase interval to base interval (0, 1) in case of LC without pulsations
        :param phases: np.array - phases to reduce
        :return: base_phases - np.array of unique phases between (0, 1)
                 reverse_indices - np.array - mask applicable to `base_phases` which will reconstruct original `phases`
        """
        if not self._system.primary.has_pulsations() and not self._system.primary.has_pulsations():
            base_interval = np.round(phases % 1, 9)
            return np.unique(base_interval, return_inverse=True)
        else:
            return phases, np.arange(phases.shape[0])


if __name__ == "__main__":
    o = Observer(passband=['Generic.Bessell.B', 'Generic.Bessell.V'], system=None)
    print(o.right_bandwidth, o.left_bandwidth)
    pass
