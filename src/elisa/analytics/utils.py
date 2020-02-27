from copy import copy
from astropy import units as u

from elisa.analytics.binary import params


def convert_dict_to_json_format(dictionary):
    """
    Converts initial vector to JSON compatibile format

    :param dictionary: dict; vector of initial parameters {`paramname`:{`value`: value, `min`: ...}, ...}
    :return: list; [{`param`: paramname, `value`: value, ...}, ...]
    """
    retval = list()
    for key, val in dictionary.items():
        val.update({'param': key})
        retval.append(val)
    return retval


def convert_json_to_dict_format(json):
    """
    Converts initial vector to JSON compatibile format

    :param json: list; vector of initial parameters {`paramname`:{`value`: value, `min`: ...}, ...}
    :return: list; [{`param`: paramname, `value`: value, ...}, ...]
    """
    retval = dict()
    for item in json:
        param = copy(item).pop('param')
        retval.update({param: item, })
    return retval


def transform_initial_values(X0):
    """
    transforms initial vector to base units-

    :param X0: dict; initial vector, {`param_name: {`value`:value, `unit`: astropy.unit...}, ...}
    :return:
    """
    for key, val in X0.items():
        if 'unit' in val.keys():
            val['unit'] = u.Unit(val['unit']) if isinstance(val['unit'], str) else val['unit']
            if 'value' in val.keys():
                val['value'] = (val['value'] * val['unit']).to(params.PARAMS_UNITS_MAP[key]).value
            if 'min' in val.keys():
                val['min'] = (val['min'] * val['unit']).to(params.PARAMS_UNITS_MAP[key]).value
            if 'max' in val.keys():
                val['max'] = (val['max'] * val['unit']).to(params.PARAMS_UNITS_MAP[key]).value
            val['unit'] = params.PARAMS_UNITS_MAP[key]
    return X0


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

