import numpy as np

from typing import List, Tuple, Dict
from ...atm import atm_file_prefix_to_quantity_list
from ...base import error
from ...binary_system.system import BinarySystem
from ...conf import config
from ...observer.observer import Observer


# DO NOT CHANGE KEYS - NEVER EVER
PARAMS_KEY_MAP = {
    'omega': 'argument_of_periastron',
    'i': 'inclination',
    'e': 'eccentricity',
    'gamma': 'gamma',
    'M1': 'p__mass',
    'T1': 'p__t_eff',
    'Omega1': 'p__surface_potential',
    'beta1': 'p__gravity_darkening',
    'A1': 'p__albedo',
    'MH1': 'p__metallicity',
    'F1': 'p__synchronicity',
    'M2': 's__mass',
    'T2': 's__t_eff',
    'Omega2': 's__surface_potential',
    'beta2': 's__gravity_darkening',
    'A2': 's__albedo',
    'MH2': 's__metallicity',
    'F2': 's__synchronicity',
}

PARAMS_KEY_TEX_MAP = {
    'argument_of_periastron': '$\\omega$',
    'inclination': '$i$',
    'eccentricity': '$e$',
    'gamma': '$\\gamma$',
    'p__mass': '$M_1$',
    'p__t_eff': '$T_1^{eff}$',
    'p__surface_potential': '$\\Omega_1$',
    'p__gravity_darkening': '$\\beta_1$',
    'p__albedo': '$A_1$',
    'p__metallicity': '$M/H_1$',
    'p__synchronicity': '$F_1$',
    's__mass': '$M_2$',
    's__t_eff': '$T_2^{eff}$',
    's__surface_potential': '$\\Omega_2$',
    's__gravity_darkening': '$\\beta_2$',
    's__albedo': '$A_2$',
    's__metallicity': '$M/H_2$',
    's__synchronicity': '$F_2$'
}


PARAMS_UNITS_MAP = {
    PARAMS_KEY_MAP['i']: 'degrees',
    PARAMS_KEY_MAP['e']: 'dimensionless',
    PARAMS_KEY_MAP['omega']: 'degrees',
    PARAMS_KEY_MAP['gamma']: 'm/s',
    PARAMS_KEY_MAP['M1']: 'solMass',
    PARAMS_KEY_MAP['M2']: 'solMass',
    PARAMS_KEY_MAP['T1']: 'K',
    PARAMS_KEY_MAP['T2']: 'K',
    PARAMS_KEY_MAP['MH1']: 'dimensionless',
    PARAMS_KEY_MAP['MH2']: 'dimensionless',
    PARAMS_KEY_MAP['Omega1']: 'dimensionless',
    PARAMS_KEY_MAP['Omega2']: 'dimensionless',
    PARAMS_KEY_MAP['A1']: 'dimensionless',
    PARAMS_KEY_MAP['A2']: 'dimensionless',
    PARAMS_KEY_MAP['beta1']: 'dimensionless',
    PARAMS_KEY_MAP['beta2']: 'dimensionless',
    PARAMS_KEY_MAP['F1']: 'dimensionless',
    PARAMS_KEY_MAP['F2']: 'dimensionless',
}


TEMPERATURES = atm_file_prefix_to_quantity_list("temperature", config.ATM_ATLAS)
METALLICITY = atm_file_prefix_to_quantity_list("metallicity", config.ATM_ATLAS)

NORMALIZATION_MAP = {
    PARAMS_KEY_MAP['i']: (0, 180),
    PARAMS_KEY_MAP['e']: (0, 1),
    PARAMS_KEY_MAP['omega']: (0, 360),
    PARAMS_KEY_MAP['gamma']: (0, 1e6),
    PARAMS_KEY_MAP['M1']: (0.5, 20),
    PARAMS_KEY_MAP['M2']: (0.5, 20),
    PARAMS_KEY_MAP['T1']: (np.min(TEMPERATURES), np.max(TEMPERATURES)),
    PARAMS_KEY_MAP['T2']: (np.min(TEMPERATURES), np.max(TEMPERATURES)),
    PARAMS_KEY_MAP['MH1']: (np.min(METALLICITY), np.max(METALLICITY)),
    PARAMS_KEY_MAP['MH2']: (np.min(METALLICITY), np.max(METALLICITY)),
    PARAMS_KEY_MAP['Omega1']: (2.0, 50.0),
    PARAMS_KEY_MAP['Omega2']: (2.0, 50.0),
    PARAMS_KEY_MAP['A1']: (0, 1),
    PARAMS_KEY_MAP['A2']: (0, 1),
    PARAMS_KEY_MAP['beta1']: (0, 1),
    PARAMS_KEY_MAP['beta2']: (0, 1),
    PARAMS_KEY_MAP['F1']: (0, 10),
    PARAMS_KEY_MAP['F2']: (0, 10),
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

    :param x: List[float]; iterable of normalized parameter values
    :param kwords: Iterable[str]; related parmaeter names from `x`
    :return: List[float];
    """
    return [renormalize_value(_x, *get_param_boundaries(_kword)) for _x, _kword in zip(x, kwords)]


def param_normalizer(x: List, kwords: List) -> List:
    """
    Normalize values from `x` to value between (0, 1).

    :param x: List[float]; iterable of values in their native form
    :param kwords: List[str]; iterable str of names related to `x`
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
    """
    Validate parameters.

    # Main idea of `initial_x0_validity_check` is to cut of initialization if over-contact system is expected,
    # but potentials are fixed both to different values or just one of them is fixed.
    # Valid input requires either both potentials fixed on same values or non-of them fixed.
    # When non of them are fixed, internaly is fixed secondary and its value is keep same as primary.

    :param x0: List[Dict];
    :param morphology: str;
    :return: List[Dict];
    """
    hash_map = {val['param']: idx for idx, val in enumerate(x0)}
    is_oc = morphology in ['over-contact']
    are_same = x0[hash_map[PARAMS_KEY_MAP['Omega1']]]['value'] == x0[hash_map[PARAMS_KEY_MAP['Omega2']]]['value']

    omega_1 = x0[hash_map[PARAMS_KEY_MAP['Omega1']]].get('fixed', False)
    omega_2 = x0[hash_map[PARAMS_KEY_MAP['Omega2']]].get('fixed', False)

    any_fixed = omega_1 | omega_2
    all_fixed = omega_1 & omega_2

    for x in x0:
        _min, _max = x.get('min', NORMALIZATION_MAP[x['param']][0]), x.get('max', NORMALIZATION_MAP[x['param']][1])
        if not (_min <= x['value'] <= _max):
            msg = f'Initial parametres are not fisible. Invalid bounds NOT: {_min} <= {x["param"]} <= {_max}'
            raise ValueError(msg)

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
        x0[hash_map[PARAMS_KEY_MAP['Omega2']]]['fixed'] = True
        _min, _max = x0[hash_map[PARAMS_KEY_MAP['Omega1']]]['min'], x0[hash_map[PARAMS_KEY_MAP['Omega1']]]['max']
        x0[hash_map[PARAMS_KEY_MAP['Omega2']]]['min'] = _min
        x0[hash_map[PARAMS_KEY_MAP['Omega2']]]['max'] = _max
        update_normalization_map({PARAMS_KEY_MAP['Omega2']: (_min, _max)})

    return x0


def is_overcontact(morphology):
    """
    Is string equal to `over-contact`?
    """
    return morphology in ['over-contact']


def adjust_constrained_potential(adjust_in, to_value=None):
    if to_value is not None:
        adjust_in[PARAMS_KEY_MAP['Omega2']] = to_value
    else:
        adjust_in[PARAMS_KEY_MAP['Omega2']] = adjust_in[PARAMS_KEY_MAP['Omega1']]
    return adjust_in


def adjust_result_constrained_potential(adjust_in, hash_map):
    """
    In constarained potentials (over-contact system), secondary potential is artificialy fixed and its values has to
    be changed to valid at the end of the fitting process.

    :param adjust_in: List[Dict]; result like Dict
    :param hash_map: Dict[str, int]; map of indices for parameters
    :return: List[Dict]; same shape as input
    """
    value = adjust_in[hash_map[PARAMS_KEY_MAP['Omega1']]]["value"]
    adjust_in[hash_map[PARAMS_KEY_MAP['Omega2']]] = {
        "param": PARAMS_KEY_MAP['Omega2'],
        "value": value,
        "min": adjust_in[hash_map[PARAMS_KEY_MAP['Omega1']]].get("min", value),
        "max": adjust_in[hash_map[PARAMS_KEY_MAP['Omega1']]].get("max", value),
    }
    return adjust_in


def extend_result_with_units(result):
    """
    Add unit information to `result` list.

    :param result: List[Dict];
    :return: List[Dict];
    """
    for res in result:
        key = res.get('param')
        if key in PARAMS_UNITS_MAP:
            res['unit'] = PARAMS_UNITS_MAP[key]
    return result
