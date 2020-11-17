import numpy as np

from ... import units as u
from ... binary_system import t_layer


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
