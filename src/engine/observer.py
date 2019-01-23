import os
import numpy as np
import pandas as pd
import logging

from os.path import dirname
from conf import config
from engine import utils

config.set_up_logging()


class Observer(object):
    def __init__(self, passband, system):
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

    def observe(self, from_phase, to_phase):
        self._logger.info("observetaion start w/ following configuration {<add>}")

        # generate phase points
        # start multiprocess
        # start writer and write generator of binary systems
        # start worker, init yielded binary system
        # build all in system
        # compute lightcurve point
        print(self._system.initial_kwargs)
        self._logger.info("observation finished")

    def apply_filter(self):
        pass


if __name__ == '__main__':
    # todo: handle bolometric in way like lambda x: 1
    observer = Observer('Generic.Bessell.B', system=None)
