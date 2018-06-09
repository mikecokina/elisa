import os
import numpy as np
import pandas as pd
from engine.conf import PASSBAND
from os.path import dirname


class Observer(object):
    def __init__(self, passband):
        self._passband = passband

    @property
    def passband(self):
        return self._passband

    @passband.setter
    def passband(self, passband):
        self._passband = passband

    @staticmethod
    def get_passband_df(passband):
        if passband not in PASSBAND:
            raise ValueError('Invalid or unsupported passband function')
        file_path = os.path.join(dirname(dirname(__file__)), 'passband', str(passband) + '.csv')
        return pd.read_csv(file_path)

    @classmethod
    def limb_darkening_linear(cls, gamma, xlin):
        return 1.0 - (xlin * (1. - abs(np.cos(gamma))))

    @classmethod
    def limb_darkening_logarithmic(cls, gamma, xlog, ylog):
        return 1.0 - (xlog * (1.0 - abs(np.cos(gamma)))) - (ylog * abs(np.cos(gamma)) * np.log10(abs(np.cos(gamma))))

    @classmethod
    def limb_darkening_sqrt(cls, gamma, xsqrt, ysqrt):
        return 1.0 - (xsqrt * (1.0 - abs(np.cos(gamma)))) - (ysqrt * (1.0 - np.sqrt(abs(np.cos(gamma)))))


if __name__ == '__main__':
    observer = Observer('Generic.Bessell.B')
    print(observer.get_passband_df(observer.passband).head())
