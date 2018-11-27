import os
import numpy as np
import pandas as pd
from engine.conf import PASSBAND
from os.path import dirname


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
        if passband not in PASSBAND:
            raise ValueError('Invalid or unsupported passband function')
        file_path = os.path.join(dirname(dirname(__file__)), 'passband', str(passband) + '.csv')
        return pd.read_csv(file_path)

    def compute_lightcurve(self):
        pass

    def apply_filter(self):
        pass

if __name__ == '__main__':
    observer = Observer('Generic.Bessell.B')
    print(observer.get_passband_df(observer.passband).head())
