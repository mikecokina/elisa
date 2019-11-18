from typing import List, Tuple, Dict

import numpy as np

from elisa.atm import atm_file_prefix_to_quantity_list
from elisa.base import error
from elisa.binary_system.system import BinarySystem
from elisa.conf import config
from elisa.observer.observer import Observer


# DO NOT CHANGE KEYS - NEVER EVER
PARAMS_KEY = {
    'inclination': 'inclination',
    'eccentricity': 'eccentricity',
    'aop ': 'argument_of_periastron',
    'gamma': 'gamma',
    'p__mass': 'p__mass',
    'p__t_eff': 'p__t_eff',
    'p__omega': 'p__surface_potential',
    'p__beta': 'p__gravity_darkening',
    'p__albedo': 'p__albedo',
    'p__mh': 'p__metallicity',
    's__mass': 's__mass',
    's__t_eff': 's__t_eff',
    's__omega': 's__surface_potential',
    's__beta': 's__gravity_darkening',
    's__albedo': 's__albedo',
    's__mh': 's__metallicity',
}

TEMPERATURES = atm_file_prefix_to_quantity_list("temperature", config.ATM_ATLAS)
METALLICITY = atm_file_prefix_to_quantity_list("metallicity", config.ATM_ATLAS)

NORMALIZATION_MAP = {
    'inclination': (0, 180),
    'eccentricity': (0, 1),
    'argument_of_periastron': (0, 360),
    'gamma': (0, 1e6),
    'p__mass': (0.5, 20),
    's__mass': (0.5, 20),
    'p__t_eff': (np.min(TEMPERATURES), np.max(TEMPERATURES)),
    's__t_eff': (np.min(TEMPERATURES), np.max(TEMPERATURES)),
    'p__metallicity': (np.min(METALLICITY), np.max(METALLICITY)),
    's__metallicity': (np.min(METALLICITY), np.max(METALLICITY)),
    'p__surface_potential': (2.0, 50.0),
    's__surface_potential': (2.0, 50.0),
    'p__albedo': (0, 1),
    's__albedo': (0, 1),
    'p__gravity_darkening': (0, 1),
    's__gravity_darkening': (0, 1)
}


def renormalize_value(val, _min, _max):
    """
    Renormalize value `val` to value from interval specific for given parameter defined my `_min` and `_max`.

    :param val: float;
    :param _min: float;
    :param _max: float;
    :return: float;
    """
    return (val * (_max - _min)) + _min


def normalize_value(val, _min, _max):
    """
    Normalize value `val` to value from interval (0, 1) based on `_min` and `_max`.

    :param val: float;
    :param _min: float;
    :param _max: float;
    :return: float;
    """
    return (val - _min) / (_max - _min)


def x0_vectorize(x0) -> Tuple:
    """
    Transform native JSON form of initial parameters to Tuple.
    JSON form::

        [
            {
                'value': 2.0,
                'param': 'p__mass',
                'fixed': False,
                'min': 1.0,
                'max': 3.0
            },
            {
                'value': 4000.0,
                'param': 'p__t_eff',
                'fixed': True,
                'min': 3500.0,
                'max': 4500.0
            },
            ...
        ]

    :param x0: List[Dict[str, Union[float, str, bool]]]; initial parmetres in JSON form
    :return: Tuple;
    """
    _x0 = [record['value'] for record in x0 if not record['fixed']]
    _kwords = [record['param'] for record in x0 if not record['fixed']]
    return _x0, _kwords


def x0_to_kwargs(x0):
    """
    Transform native JSON input form to `key, value` form::

        {
            key: value,
            ...
        }

    :param x0: List[Dict[str, Union[float, str, bool]]];
    :return: Dict[str, float];
    """
    return {record['param']: record['value'] for record in x0}


def x0_to_fixed_kwargs(x0):
    """
    Transform native JSON input form to `key, value` form, but select `fixed` parametres only::

        {
            key: value,
            ...
        }

    :param x0: List[Dict[str, Union[float, str, bool]]];
    :return: Dict[str, float];
    """
    return {record['param']: record['value'] for record in x0 if record['fixed']}


def update_normalization_map(update):
    """
    Update module normalization map with supplied dict.

    :param update: Dict;
    """
    NORMALIZATION_MAP.update(update)


def param_renormalizer(x, kwords):
    """
    Renormalize values from `x` to their native form.

    :param x: Iterable[float]; iterable of normalized parameter values
    :param kwords: Iterable[str]; related parmaeter names from `x`
    :return: List[float];
    """
    return [renormalize_value(_x, *get_param_boundaries(_kword)) for _x, _kword in zip(x, kwords)]


def param_normalizer(x: List, kwords: List) -> List:
    """
    Normalize values from `x` to value between (0, 1).

    :param x: Iterable[float]; iterable of values in their native form
    :param kwords: Iterable[str]; iterable str of names related to `x`
    :return: List[float];
    """
    return [normalize_value(_x, *get_param_boundaries(_kword)) for _x, _kword in zip(x, kwords)]


def get_param_boundaries(param):
    """
    Return normalization boundaries for given parmeter.

    :param param: str; name of parameter to get boundaries for
    :return: Tuple[float, float];
    """
    return NORMALIZATION_MAP[param]


def serialize_param_boundaries(x0):
    """
    Serialize boundaries of parameters if exists and parameter is not fixed.

    :param x0: List[Dict[str, Union[float, str, bool]]]; initial parmetres in JSON form
    :return: Dict[str, Tuple[float, float]]
    """
    return {record['param']: (record.get('min', NORMALIZATION_MAP[record['param']][0]),
                              record.get('max', NORMALIZATION_MAP[record['param']][1]))
            for record in x0 if not record['fixed']}


def fit_data_initializer(x0, passband=None):
    boundaries = serialize_param_boundaries(x0)
    update_normalization_map(boundaries)

    fixed = x0_to_fixed_kwargs(x0)
    x0_vectorized, kwords = x0_vectorize(x0)
    x0 = param_normalizer(x0_vectorized, kwords)

    observer = Observer(passband='bolometric' if passband is None else passband, system=None)
    observer._system_cls = BinarySystem

    return x0, kwords, fixed, observer


def initial_x0_validity_check(x0: List[Dict], morphology):
    hash_map = {val['param']: idx for idx, val in enumerate(x0)}
    param = 'surface_potential'
    is_oc = morphology in ['over-contact']
    are_same = x0[hash_map[f'p__{param}']]['value'] == x0[hash_map[f's__{param}']]['value']
    any_fixed = x0[hash_map[f'p__{param}']].get('fixed', False) or x0[hash_map[f's__{param}']].get('fixed', False)
    all_fixed = x0[hash_map[f'p__{param}']].get('fixed', False) and x0[hash_map[f's__{param}']].get('fixed', False)

    if is_oc and all_fixed and are_same:
        return x0
    if is_oc and all_fixed and not are_same:
        msg = 'different potential in over-contact morphology with all fixed (pontetial) value are not allowed'
        raise error.InitialParamsError(msg)
    if is_oc and any_fixed:
        msg = 'just one fixed potential in over-contact morphology is not allowed'
        raise error.InitialParamsError(msg)
    if is_oc:
        # if is overcontact, fix secondary pontetial for further steps
        x0[hash_map[f's__{param}']]['fixed'] = True
        _min, _max = x0[hash_map[f'p__{param}']]['min'], x0[hash_map[f'p__{param}']]['max']
        x0[hash_map[f's__{param}']]['min'] = _min
        x0[hash_map[f's__{param}']]['max'] = _max
        update_normalization_map({f's__{param}': (_min, _max)})
    return x0


def is_overcontact(morphology):
    return morphology in ['over-contact']


def adjust_constrained_potential(adjust_in, to_value=None):
    if to_value is not None:
        adjust_in[PARAMS_KEY['s__omega']] = to_value
    else:
        adjust_in[PARAMS_KEY['s__omega']] = adjust_in[PARAMS_KEY['p__omega']]
    return adjust_in


def adjust_result_constrained_potential(adjust_in, hash_map):
    value = adjust_in[hash_map[PARAMS_KEY['p__omega']]]["value"]
    adjust_in[hash_map[PARAMS_KEY['s__omega']]] = {
        "param": PARAMS_KEY['s__omega'],
        "value": value,
        "min": adjust_in[hash_map[PARAMS_KEY['p__omega']]].get("min", value),
        "max": adjust_in[hash_map[PARAMS_KEY['p__omega']]].get("max", value),
        "fixed": False
    }
    return adjust_in
