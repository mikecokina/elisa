import numpy as np


def normalize_light_curve(lc, kind='global_maximum', top_fraction_to_average=0.1):
    """
    Function normalizes light by a method defined by `kind` argument.

    :param lc: Dict; dictionary containing curves, {filter: np.ndarray, ...}
    :param kind: str; specifies kind of normalization
    :**kind options**:
        * ** average ** * -- each curve is normalized to its average
        * ** global_average ** * -- curves are normalized to their global average
        * ** maximum ** * -- each curve is normalized to its own maximum
        * ** global_maximum -- curves are normalized to their global maximum
    :return: Dict;
    """
    valid_arguments = ['average', 'global_average', 'maximum', 'global_maximum']
    if kind == valid_arguments[0]:  # each curve is normalized to its average
        coeff = {key: np.mean(val) for key, val in lc.items()}
    elif kind == valid_arguments[1]:  # curves are normalized to their global average
        c = np.mean(list(lc.values()))
        coeff = {key: c for key, val in lc.items()}
    elif kind == valid_arguments[2]:  # each curve is normalized to its own average of the top fraction
        n = {key: int(top_fraction_to_average * len(val)) for key, val in lc.items()}
        coeff = {key: np.average(val[np.argsort(val)[-n[key]:]]) for key, val in lc.items()}
    elif kind == valid_arguments[3]:  # curves are normalized to their global maximum
        vals = np.array(list(lc.values())).flatten()
        n = int(top_fraction_to_average * len(vals) / len(lc))
        c = np.average(vals[np.argsort(vals)[-n:]])
        coeff = {key: c for key, val in lc.items()}
    else:
        raise ValueError(f'Argument `kind` = {kind} is not one of the valid arguments {valid_arguments}')

    return {key: np.array(val) / coeff[key] for key, val in lc.items()}


def normalize_rv_curve_to_max(rv):
    _max = np.max([rv['primary'], rv['secondary']])
    return {key: val/_max for key, val in rv.items()}


def lightcurves_mean_error(lc):
    return np.mean(lc) * 0.05


def radialcurves_mean_error(rv):
    return np.mean(rv) * 0.05
