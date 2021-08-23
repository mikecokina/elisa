import sys
import numpy as np
import pandas as pd

from scipy import interpolate
from .. import settings


def init_bolometric_passband():
    """
    initializing bolometric passband and its wavelength boundaries

    :return: Tuple;
    """
    df = pd.DataFrame(
        {
            settings.PASSBAND_DATAFRAME_THROUGHPUT: [1.0, 1.0],
            settings.PASSBAND_DATAFRAME_WAVE: [50.0, 2000000.0]
        }
    )
    right_bandwidth = sys.float_info.max
    left_bandwidth = 0.0
    bol_passband = PassbandContainer(table=df, passband='bolometric')

    return bol_passband, right_bandwidth, left_bandwidth


def init_rv_passband():
    """
    Initializing passband used to calculate radial velocities

    :return: Tuple
    """
    df = pd.DataFrame(
        {settings.PASSBAND_DATAFRAME_THROUGHPUT: [1.0, 1.0],
         settings.PASSBAND_DATAFRAME_WAVE: settings.RV_LAMBDA_INTERVAL})
    right_bandwidth = settings.RV_LAMBDA_INTERVAL[1]
    left_bandwidth = settings.RV_LAMBDA_INTERVAL[0]
    psmbnd = PassbandContainer(table=df, passband='rv_band')

    return psmbnd, right_bandwidth, left_bandwidth


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


class PassbandContainer(object):
    def __init__(self, table, passband):
        """
        Data container used for storing passband response curves. Fully initialized PassbandContainers contain following
        attributes:

            - left_bandwidth, right_bandwidth: left and right wavelength boundary of the passband
            - table: pandas.DataFrame; dataframe containing a `wavelength` column with corresponding `throughput` values
                                       defining a given passband
            - passband: name of the passband

        The response curve is stored in a pandas.DataFrame
        Setup PassbandContainier object. It carres dependedncies of throughputs on wavelengths for given passband.

        :param table: pandas.DataFrame;
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
        self.akima = bolometric if (self.passband.lower() in ['bolometric', 'rv_band']) else \
            interpolate.Akima1DInterpolator(df[settings.PASSBAND_DATAFRAME_WAVE],
                                            df[settings.PASSBAND_DATAFRAME_THROUGHPUT])
        self.left_bandwidth = min(df[settings.PASSBAND_DATAFRAME_WAVE])
        self.right_bandwidth = max(df[settings.PASSBAND_DATAFRAME_WAVE])
