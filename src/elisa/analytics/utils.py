from copy import copy
from astropy import units as u

from elisa.analytics.binary import params
from elisa.logger import getPersistentLogger


logger = getPersistentLogger('analytics.utils')


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
            if 'min' in val.keys():
                val['min'] = (val['min'] * val['unit']).to(params.PARAMS_UNITS_MAP[key]).value
            if 'max' in val.keys():
                val['max'] = (val['max'] * val['unit']).to(params.PARAMS_UNITS_MAP[key]).value
            val['unit'] = params.PARAMS_UNITS_MAP[key]
    return x0


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


def prep_constrained_params(x0):
    """
    Function treats constrained parameters by assingning None to `value`
    :param x0: dict;
    :return: dict;
    """
    for key, val in x0.items():
        if 'constraint' in val.keys():
            if 'value' in val.keys():
                logger.warning(f'Parameter `value` is meaningless and will not be used in case of contrained parameter '
                               '{key}.')
            val['value'] = None
    return x0
