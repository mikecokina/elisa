import sys
import numpy as np
import pandas as pd

from elisa.conf import config
from elisa.observer.passband import PassbandContainer


def init_bolometric_passband():
    """
    initializing bolometric passband and its wavelength boundaries

    :return: tuple;
    """
    df = pd.DataFrame(
        {config.PASSBAND_DATAFRAME_THROUGHPUT: [1.0, 1.0],
         config.PASSBAND_DATAFRAME_WAVE: [0.0, sys.float_info.max]})
    right_bandwidth = sys.float_info.max
    left_bandwidth = 0.0
    bol_passband = PassbandContainer(table=df, passband='bolometric')

    return bol_passband, right_bandwidth, left_bandwidth


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

