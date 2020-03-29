import numpy as np


def normalize_lightcurve_to_max(lc):
    """
    Normalize light-curve dict to maximal value for each passband.
    Require light-curve in following shape::

        {
            <passband>: numpy.array(<flux>)
        }

    :param lc: Dict[str, numpy.array(float)];
    :return: Dict[str, numpy.array(float)];
    """
    return {key: np.array(val)/max(val) for key, val in lc.items()}


def normalize_rv_curve_to_max(rv):
    _max = np.max([rv['primary'], rv['secondary']])
    return {key: val/_max for key, val in rv.items()}


def lightcurves_mean_error(lc):
    return np.mean(list(lc.values())) * 0.05


def radialcurves_mean_error(rv):
    return np.mean(list(rv.values())) * 0.05
