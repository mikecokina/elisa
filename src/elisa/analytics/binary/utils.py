import numpy as np


def normalize_to_max(lc):
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
