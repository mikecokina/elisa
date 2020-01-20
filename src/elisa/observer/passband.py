import numpy as np
import pandas as pd

from scipy import interpolate

from elisa.conf import config
from elisa.observer import utils as outils


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
        self.akima = outils.bolometric if (self.passband.lower() in ['bolometric']) else \
            interpolate.Akima1DInterpolator(df[config.PASSBAND_DATAFRAME_WAVE],
                                            df[config.PASSBAND_DATAFRAME_THROUGHPUT])
        self.left_bandwidth = min(df[config.PASSBAND_DATAFRAME_WAVE])
        self.right_bandwidth = max(df[config.PASSBAND_DATAFRAME_WAVE])

