import os
import numpy as np
import pandas as pd

from os.path import dirname
from conf import config
from engine import utils


def limb_darkening_linear(gamma, xlin):
    return 1.0 - (xlin * (1. - abs(np.cos(gamma))))


def limb_darkening_logarithmic(gamma, xlog, ylog):
    return 1.0 - (xlog * (1.0 - abs(np.cos(gamma)))) - (ylog * abs(np.cos(gamma)) * np.log10(abs(np.cos(gamma))))


def limb_darkening_sqrt(gamma, xsqrt, ysqrt):
    return 1.0 - (xsqrt * (1.0 - abs(np.cos(gamma)))) - (ysqrt * (1.0 - np.sqrt(abs(np.cos(gamma)))))


def get_van_hamme_ld_table(metallicity):
    filename = "{model}.{passband}.{metallicity}.csv".format(
        model=config.LD_KEY_TO_FILE_PREFIX[config.LIMB_DARKENING_LAW],
        passband=config.PASSBAND,
        metallicity=utils.numeric_metallicity_to_string(metallicity)
    )
    path = os.path.join(config.VAN_HAMME_LD_TABLES, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError("there is no file like {}".format(path))
    return pd.read_csv(path)


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
        if passband not in config.PASSBANDS:
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
    print(get_van_hamme_ld_table(0.5).head())
