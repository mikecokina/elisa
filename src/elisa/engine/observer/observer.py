import logging
import os
import pandas as pd
import numpy as np

from multiprocessing.pool import Pool
from scipy import interpolate
from os.path import dirname
from elisa.conf import config
from elisa.engine.binary_system.system import BinarySystem
from elisa.engine.single_system.system import SingleSystem
from elisa.engine.observer import static, mp

config.set_up_logging()


class PassbandContainer(object):
    def __init__(self, table, akima, left_bandwidth, right_bandwidth):
        self.left_bandwidth = left_bandwidth
        self.right_bandwidth = right_bandwidth
        self.akima = akima
        self.table = table


class Observer(object):
    def __init__(self, passband: list or str, system: BinarySystem or SingleSystem):
        self._logger = logging.getLogger(Observer.__name__)
        self._logger.info("initialising Observer instance")
        # specifying what kind of system is observed
        self._system = system
        self._system_cls = type(self._system)

        # self._system._suppress_logger = True

        self.left_bandwidth = 0.0
        self.right_bandwidth = 1e6
        self.passband = dict()
        self.init_passband(passband)

    @staticmethod
    def bolometric(*args, **kwargs):
        return 1.0

    def init_passband(self, passband):
        passband = [passband] if isinstance(passband, str) else passband
        for band in passband:
            if band in ['bolometric']:
                df = pd.DataFrame({"throughput": [1.0, 1.0], "wavelength": [0.0, 1e6]})
                akima = self.bolometric
                right_bandwidth = 1e6
                left_bandwidth = 0.0
            else:
                df = self.get_passband_df(band)
                left_bandwidth = min(df["wavelength"])
                right_bandwidth = max(df["wavelength"])
                akima = interpolate.Akima1DInterpolator(df["wavelength"], df["throughput"])

            self.setup_bandwidth(left_bandwidth=left_bandwidth, right_bandwidth=right_bandwidth)
            self.passband[band] = PassbandContainer(
                table=df, akima=akima, left_bandwidth=right_bandwidth, right_bandwidth=left_bandwidth)

    def setup_bandwidth(self, left_bandwidth, right_bandwidth):
        if left_bandwidth > self.left_bandwidth:
            self.left_bandwidth = left_bandwidth
        if right_bandwidth < self.right_bandwidth:
            self.right_bandwidth = right_bandwidth

    @staticmethod
    def get_passband_df(passband):
        logging.debug("obtaining passband response function: {}".format(passband))
        if passband not in config.PASSBANDS:
            raise ValueError('Invalid or unsupported passband function')
        file_path = os.path.join(config.PASSBAND_TABLES, str(passband) + '.csv')
        return pd.read_csv(file_path)

    def observe(self, from_phase: float = None, to_phase: float = None, phase_step: float = None,
                phases: list or set = None):
        if not phases and (from_phase is None or to_phase is None or phase_step is None):
            raise ValueError("missing arguments")

        if phases is None:
            phases = np.linspace(start=from_phase, stop=to_phase, endpoint=True)

        self._logger.info("observation start w/ following configuration {<add>}")
        # self._logger.warning("logger will be suppressed due multiprocessing incompatibility")
        """
        distance, azimut angle, true anomaly and phase
                           np.array((r1, az1, ni1, phs1),
                                    (r2, az2, ni2, phs2),
                                    ...
                                    (rN, azN, niN, phsN))
        """
        orbital_motion = self._system.orbit.orbital_motion(phase=phases)
        args = mp.prepare_observe_args(orbital_motion)
        self._system.compute_lightcurve(
            **dict(
                orbital_motion=args,
                passband=self.passband,
                left_bandwidth=self.left_bandwidth,
                right_bandwidth=self.right_bandwidth
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

        self._logger.info("observation finished")


if __name__ == "__main__":
    o = Observer(passband=['Generic.Bessell.B', 'Generic.Bessell.V'], system=None)
    print(o.right_bandwidth, o.left_bandwidth)
    pass
