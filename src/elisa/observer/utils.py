import numpy as np
from .. utils import is_empty


def normalize_light_curve(y_data, y_err=None, kind='global_maximum', top_fraction_to_average=0.1):
    """
    Function normalizes light by a method defined by `kind` argument.

    :param y_data: Dict; dictionary containing curves, {filter: np.ndarray, ...}
    :param y_err: Dict; dictionary containing errors, {filter: np.ndarray, ...}
    :param kind: str; specifies kind of normalization
    :**kind options**:
        * **average**- each curve is normalized to its average
        * **global_average** - curves are normalized to their global average
        * **maximum** - each curve is normalized to its own maximum
        * **global_maximum** - curves are normalized to their global maximum
    :param top_fraction_to_average: float; top portion of the dataset (in y-axis direction) used in the
                                           normalization process, from (0, 1) interval
    :return: Dict;
    """
    valid_arguments = ['average', 'global_average', 'maximum', 'global_maximum', 'minimum']
    if kind == valid_arguments[0]:  # each curve is normalized to its average
        coeff = {key: np.mean(val) for key, val in y_data.items()}
    elif kind == valid_arguments[1]:  # curves are normalized to their global average
        c = np.mean(list(y_data.values()))
        coeff = {key: c for key, val in y_data.items()}
    elif kind == valid_arguments[2]:  # each curve is normalized to its own average of the top fraction
        n = {key: int(top_fraction_to_average * len(val)) + 1 for key, val in y_data.items()}
        coeff = {key: np.average(val[np.argsort(val)[-n[key]:]]) for key, val in y_data.items()}
    elif kind == valid_arguments[3]:  # curves are normalized to their global maximum
        vals = np.concatenate(list(y_data.values()))
        n = int(top_fraction_to_average * len(vals) / len(y_data)) + 1
        c = np.average(vals[np.argsort(vals)[-n:]])
        coeff = {key: c for key, val in y_data.items()}
    elif kind == valid_arguments[4]:  # each curve is normalized to its own average of the top fraction
        n = {key: int(top_fraction_to_average * len(val)) for key, val in y_data.items()}
        coeff = {key: np.average(val[np.argsort(val)[:n[key]]]) for key, val in y_data.items()}
    else:
        raise ValueError(f'Argument `kind` = {kind} is not one of the valid arguments {valid_arguments}')

    y_data = {key: np.array(val) / coeff[key] for key, val in y_data.items()}
    y_err = {key: np.array(val) / coeff[key] if not is_empty(val) else None for key, val in y_err.items()} \
        if not is_empty(y_err) else None
    return y_data, y_err
