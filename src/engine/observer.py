import os
from os.path import dirname

import numpy as np
import pandas as pd

from conf import config


def limb_darkening_linear(gamma, xlin):
    return 1.0 - (xlin * (1. - abs(np.cos(gamma))))


def limb_darkening_logarithmic(gamma, xlog, ylog):
    return 1.0 - (xlog * (1.0 - abs(np.cos(gamma)))) - (ylog * abs(np.cos(gamma)) * np.log10(abs(np.cos(gamma))))


def limb_darkening_sqrt(gamma, xsqrt, ysqrt):
    return 1.0 - (xsqrt * (1.0 - abs(np.cos(gamma)))) - (ysqrt * (1.0 - np.sqrt(abs(np.cos(gamma)))))


def get_van_hamme_ld_table():
    pass


class Observer(object):
    def __init__(self, passband, system):
        self._passband = passband
        # specifying what system is observed
        self._system = system  # co je observe?

    @property
    def passband(self):
        return self._passband

    @passband.setter
    def passband(self, passband):
        self._passband = passband

    @staticmethod
    def get_passband_df(passband):
        if passband not in config.PASSBAND:
            raise ValueError('Invalid or unsupported passband function')
        file_path = os.path.join(dirname(dirname(dirname(__file__))), 'passband', str(passband) + '.csv')
        return pd.read_csv(file_path)

    def compute_lightcurve(self):
        pass

    def apply_filter(self):
        pass


if __name__ == '__main__':
    # todo: handle bolometric in way like lambda x: 1
    observer = Observer('Generic.Bessell.B', system=None)
    print(observer.get_passband_df(observer.passband).head())
