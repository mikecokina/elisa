from copy import copy
from astropy import units as u

from elisa.analytics.binary import params


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
        param = copy(item).pop('param')
        retval.update({param: item, })
    return retval


def transform_initial_values(x0):
    """
    Transforms initial vector to base units.

    :param x0: Dict; initial vector, {param_name: {value: value, unit: astropy.unit...}, ...}
    :return: Dict;
    """
    for key, val in x0.items():
        if 'unit' in val.keys():
            val['unit'] = u.Unit(val['unit']) if isinstance(val['unit'], str) else val['unit']
            if 'value' in val.keys():
                val['value'] = (val['value'] * val['unit']).to(params.PARAMS_UNITS_MAP[key]).value
            if 'min' in val.items():
                val['min'] = (val['min'] * val['unit']).to(params.PARAMS_UNITS_MAP[key]).value
            if 'max' in val.items():
                val['max'] = (val['max'] * val['unit']).to(params.PARAMS_UNITS_MAP[key]).value
            val['unit'] = params.PARAMS_UNITS_MAP[key]
    return x0
