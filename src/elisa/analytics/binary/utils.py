import numpy as np


def normalize_lightcurve_to_max(lc):
    """
    Normalize light-curve dict to maximal value.
    Require light-curve in following shape::

        {
            <passband>: numpy.array(<flux>)
        }

    :param lc: Dict[str, numpy.array(float)];
    :return: Dict[str, numpy.array(float)];
    """
    _max = np.max(list(lc.values()))
    return {key: val/_max for key, val in lc.items()}


def normalize_rv_curve_to_max(rv):
    _max = np.max([rv[0], rv[1]])
    return rv[0]/_max, rv[1]/_max


def lightcurves_mean_error(lc):
    return np.mean(list(lc.values())) * 0.05
