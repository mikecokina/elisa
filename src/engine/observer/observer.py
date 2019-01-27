import logging
import os
import pandas as pd
import numpy as np

from multiprocessing import Queue, Manager, Process
from os.path import dirname
from conf import config
from engine.binary_system.system import BinarySystem
from engine.observer import static

config.set_up_logging()


class Observer(object):
    def __init__(self, passband, system: BinarySystem):
        self._logger = logging.getLogger(Observer.__name__)
        self._logger.info("initialising Observer instance")
        self._passband = passband
        # specifying what system is observed
        self._system = system  # co je observe?

    @property
    def passband(self):
        return self._passband

    @passband.setter
    def passband(self, passband):
        self._passband = passband

    # def get_van_hamme_ld_table(self, metallicity):
    #     self._logger.debug("obtaining van hamme ld table")
    #     return get_van_hamme_ld_table(passband=self.passband, metallicity=metallicity)

    @staticmethod
    def get_passband_df(passband):
        logging.debug("obtaining passband response function")
        if passband not in config.PASSBANDS:
            raise ValueError('Invalid or unsupported passband function')
        file_path = os.path.join(dirname(dirname(dirname(__file__))), 'passband', str(passband) + '.csv')
        return pd.read_csv(file_path)

    def observe(self, from_phase: float = None, to_phase: float = None, phase_step: float = None,
                phases: list or set = None):
        if not phases and (from_phase is None or to_phase is None or phase_step is None):
            raise ValueError("missing arguments")

        if phases is None:
            phases = np.linspace(start=from_phase, stop=to_phase, endpoint=True)

        self._logger.info("observetaion start w/ following configuration {<add>}")
        """
        distance, azimut angle, true anomaly and phase
                           np.array((r1, az1, ni1, phs1),
                                    (r2, az2, ni2, phs2),
                                    ...
                                    (rN, azN, niN, phsN))
        """
        orbital_motion = self._system.orbit.orbital_motion(phase=phases)

        args_queue = Queue(maxsize=int(len(orbital_motion) + config.NUMBER_OF_THREADS))
        manager = Manager()

        result_list, error_list = manager.list(), manager.list
        jobs = list()

        writer_proc = Process(target=static.queue_writer,
                              args=(args_queue, orbital_motion, config.NUMBER_OF_THREADS))

        writer_proc.daemon = True
        writer_proc.start()
        jobs.append(writer_proc)

        try:
            for _ in range(config.NUMBER_OF_THREADS):
                p = Process(target=static.worker,
                            args=(args_queue, result_list, error_list, self._system.initial_kwargs))
                jobs.append(p)
                p.daemon = True
                p.start()
            for p in jobs:
                p.join()
        except KeyboardInterrupt:
            raise
        finally:
            pass

        if len(error_list) > 0:
            raise Exception("error occured: {}".format("\n".join(error_list)))

        # r = np.array(sorted(result_list, key=lambda x: x[0])).T[1]
        print(self._system.initial_kwargs)
        self._logger.info("observation finished")

    def apply_filter(self):
        pass

