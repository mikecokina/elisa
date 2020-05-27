import numpy as np
from astropy import units as u

from elisa.binary_system import t_layer
from elisa.utils import is_empty


def convert_dict_to_json_format(dictionary):
    """
    Converts initial vector to JSON compatibile format.

    :param dictionary: Dict; vector of initial parameters

    ::

        {
            paramname: {
                            value: value,
                            min: ...
                       }, ...
        }

    :return: List; [{param: paramname, value: value, ...}, ...]
    """
    retval = list()
    for key, val in dictionary.items():
        val.update({'param': key})
        retval.append(val)
    return retval


def convert_json_to_dict_format(json):
    """
    Converts initial vector to JSON compatibile format.

    :param json: List; vector of initial parameters {paramname:{value: value, min: ...}, ...}
    :return: List; [{param: paramname, value: value, ...}, ...]
    """
    retval = dict()
    for item in json:
        param = item.pop('param')
        retval.update({param: item, })
    return retval


def unify_unit_string_representation(dictionary):
    """
    transform user units to unified format

    :param dictionary: dict; model parameter
    :return: dict; model parameter
    """
    for key, val in dictionary.items():
        if 'unit' in val.keys():
            val['unit'] = u.Unit(val['unit']) if isinstance(val['unit'], str) else val['unit']
            val['unit'] = val['unit'].to_string()

    return dictionary


def normalize_light_curve(y_data, y_err=None, kind='global_maximum', top_fraction_to_average=0.1):
    """
    Function normalizes light by a method defined by `kind` argument.

    :param y_data: Dict; dictionary containing curves, {filter: np.ndarray, ...}
    :param y_err: Dict; dictionary containing errors, {filter: np.ndarray, ...}
    :param kind: str; specifies kind of normalization
    :**kind options**:
        * ** average ** * -- each curve is normalized to its average
        * ** global_average ** * -- curves are normalized to their global average
        * ** maximum ** * -- each curve is normalized to its own maximum
        * ** global_maximum -- curves are normalized to their global maximum
    :param top_fraction_to_average: float;
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


def lightcurves_mean_error(lc, *args):
    return np.mean(lc) * 0.05


def radialcurves_mean_error(rv):
    return np.mean(rv) * 0.05


def is_time_dependent(labels):
    if 'system@period' in labels and 'system@primary_minimum_time' in labels:
        return True
    return False


def time_layer_resolver(x_data, pop=False, **kwargs):
    """
    If kwargs contain `period` and `primary_minimum_time`, then xs is expected to be JD time not phases.
    Then, xs has to be converted to phases.

    :param pop: bool; determine if kick system@primary_minimum_time or just read it
    :param x_data: Union[List, numpy.array];
    :param kwargs: Dict;
    :return: Tuple;
    """

    if is_time_dependent(list(kwargs.keys())):
        t0 = kwargs['system@primary_minimum_time']
        if pop:
            kwargs.pop('system@primary_minimum_time')
        period = kwargs['system@period']
        x_data = t_layer.jd_to_phase(t0, period, x_data)
    return x_data, kwargs
