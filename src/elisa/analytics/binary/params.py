import numpy as np

from astropy.time import Time
from typing import List, Tuple, Dict

from elisa import utils
from ...atm import atm_file_prefix_to_quantity_list
from ...base import error
from ...binary_system.system import BinarySystem
from ...conf import config
from ...observer.observer import Observer
from ...analytics.binary import bonds


# for parsing params inside spots or pulsation modes
PARAM_PARSER = '@'
USER_PARAM_PARSER = '.'

COMPOSITE_PARAMS = [
    'p__spots',
    's__spots',
    'p__pulsations',
    's__pulsations',
]

SPOT_PARAMS = [
    'p__spots',
    's__spots',
]

PULSATIONS_PARAMS = [
    'p__pulsations',
    's__pulsations',
]

# DO NOT CHANGE KEYS - NEVER EVER
PARAMS_KEY_MAP = {
    'omega': 'argument_of_periastron',
    'i': 'inclination',
    'e': 'eccentricity',
    'gamma': 'gamma',
    'q': 'mass_ratio',
    'a': 'semi_major_axis',
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
    'asini': 'asini',
    'P': 'period',
    'T0': 'primary_minimum_time',
    'l_add': 'additional_light',
    'phase_shift': 'phase_shift',
    'primary_spots': 'p__spots',
    'secondary_spots': 's__spots',
    'primary_pulsations': 'p__pulsations',
    'secondary_pulsations': 's__pulsations',
}

SPOTS_KEY_MAP = {
    'phi': 'longitude',
    'theta': 'latitude',
    'radius': 'angular_radius',
    't_factor': 'temperature_factor'
}

PULSATIONS_KEY_MAP = {
    'l': 'l',
    'm': 'm',
    'amplitude': 'amplitude',
    'frequency': 'frequency',
    'start_phase': 'start_phase',
    'mode_axis_phi': 'mode_axis_phi',
    'mode_axis_theta': 'mode_axis_theta',
}

PARAMS_KEY_TEX_MAP = {
    'argument_of_periastron': '$\\omega$',
    'inclination': '$i$',
    'eccentricity': '$e$',
    'gamma': '$\\gamma$',
    'mass_ratio': '$q$',
    'semi_major_axis': '$a$',
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
    's__synchronicity': '$F_2$',
    'asini': 'a$sin$(i)',
    'period': '$period$',
    'primary_minimum_time': '$T_0$',
    'additional_light': '$l_{add}$',
    'phase_shift': 'phase shift$',
}

SPOTS_KEY_TEX_MAP = {
    'longitude': '$\phi$',
    'latitude': '$\\theta$',
    'angular_radius': '$r$',
    'temperature_factor': '$T_{spot}/T_{eff}$'
}

PULSATIONS_KEY_TEX_MAP = {
    'l': '$\\ell$',
    'm': '$m$',
    'amplitude': '$A$',
    'frequency': '$f$',
    'start_phase': '$\\Phi_0$',
    'mode_axis_phi': '$\\phi_{mode}$',
    'mode_axis_theta': '$\\theta_{mode}$',
}

PARAMS_UNITS_MAP = {
    PARAMS_KEY_MAP['i']: 'degree',
    PARAMS_KEY_MAP['e']: '',
    PARAMS_KEY_MAP['omega']: 'degree',
    PARAMS_KEY_MAP['gamma']: 'm/s',
    PARAMS_KEY_MAP['M1']: 'solMass',
    PARAMS_KEY_MAP['M2']: 'solMass',
    PARAMS_KEY_MAP['T1']: 'K',
    PARAMS_KEY_MAP['T2']: 'K',
    PARAMS_KEY_MAP['MH1']: '',
    PARAMS_KEY_MAP['MH2']: '',
    PARAMS_KEY_MAP['Omega1']: '',
    PARAMS_KEY_MAP['Omega2']: '',
    PARAMS_KEY_MAP['A1']: '',
    PARAMS_KEY_MAP['A2']: '',
    PARAMS_KEY_MAP['beta1']: '',
    PARAMS_KEY_MAP['beta2']: '',
    PARAMS_KEY_MAP['F1']: '',
    PARAMS_KEY_MAP['F2']: '',
    PARAMS_KEY_MAP['q']: '',
    PARAMS_KEY_MAP['a']: 'solRad',
    PARAMS_KEY_MAP['asini']: 'solRad',
    PARAMS_KEY_MAP['P']: 'd',
    PARAMS_KEY_MAP['T0']: 'd',
    PARAMS_KEY_MAP['l_add']: '',
    PARAMS_KEY_MAP['phase_shift']: '',
    # SPOTS
    SPOTS_KEY_MAP['phi']: 'degree',
    SPOTS_KEY_MAP['theta']: 'degree',
    SPOTS_KEY_MAP['radius']: 'degree',
    SPOTS_KEY_MAP['t_factor']: '',
    # PULSATIONS
    PULSATIONS_KEY_MAP['l']: '',
    PULSATIONS_KEY_MAP['m']: '',
    PULSATIONS_KEY_MAP['amplitude']: 'm/s',
    PULSATIONS_KEY_MAP['frequency']: '1/d',
    PULSATIONS_KEY_MAP['start_phase']: 'degree',
    PULSATIONS_KEY_MAP['mode_axis_theta']: 'degree',
    PULSATIONS_KEY_MAP['mode_axis_phi']: 'degree',
}


TEMPERATURES = atm_file_prefix_to_quantity_list("temperature", config.ATM_ATLAS)
METALLICITY = atm_file_prefix_to_quantity_list("metallicity", config.ATM_ATLAS)

NORMALIZATION_MAP = {
    PARAMS_KEY_MAP['i']: (0, 180),
    PARAMS_KEY_MAP['e']: (0, 1),
    PARAMS_KEY_MAP['omega']: (0, 360),
    PARAMS_KEY_MAP['gamma']: (0, 1e6),
    PARAMS_KEY_MAP['M1']: (0.1, 50),
    PARAMS_KEY_MAP['M2']: (0.1, 50),
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
    PARAMS_KEY_MAP['q']: (0, 50),
    PARAMS_KEY_MAP['a']: (0, 100),
    PARAMS_KEY_MAP['asini']: (0, 100),
    PARAMS_KEY_MAP['P']: (0, 100),
    PARAMS_KEY_MAP['l_add']: (0, 1.0),
    PARAMS_KEY_MAP['phase_shift']: (-0.8, 0.8),
    PARAMS_KEY_MAP['T0']: (Time.now().jd - 365.0, Time.now().jd),
    # SPOTS
    SPOTS_KEY_MAP['phi']: (0, 360),
    SPOTS_KEY_MAP['theta']: (0, 180),
    SPOTS_KEY_MAP['radius']: (0, 90),
    SPOTS_KEY_MAP['t_factor']: (0.1, 3),
    # PULSATIONS
    PULSATIONS_KEY_MAP['l']: (0, 10),
    PULSATIONS_KEY_MAP['m']: (-10, 10),
    PULSATIONS_KEY_MAP['amplitude']: (0, 5000),
    PULSATIONS_KEY_MAP['frequency']: (0.01, 40),
    PULSATIONS_KEY_MAP['start_phase']: (0, 360),
    PULSATIONS_KEY_MAP['mode_axis_theta']: (0, 180),
    PULSATIONS_KEY_MAP['mode_axis_phi']: (0, 360),
}


def renormalize_value(val, _min, _max):
    """
    Renormalize value `val` to value from interval specific for given parameter defined my `_min` and `_max`. Inverse
    function to `normalize_value`.

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


def renormalize_flat_chain(flat_chain, labels, renorm=None):
    """
    Renormalize values in chain if renormalization Dict is supplied.
    """
    if renorm is not None:
        return np.array([[renormalize_value(val, renorm[key][0], renorm[key][1])
                          for key, val in zip(labels, sample)] for sample in flat_chain])


def x0_vectorize(x0):
    """
    Transform native dict form of initial parameters to Tuple.
    JSON form::

        {
            'p__mass': {
                'value': 2.0,
                'param': 'p__mass',
                'fixed': False,
                'min': 1.0,
                'max': 3.0
            },
            'p__t_eff': {
                'value': 4000.0,
                'param': 'p__t_eff',
                'fixed': True,
                'min': 3500.0,
                'max': 4500.0
            },
            ...
        }

    :param x0: Dict[Dict[str, Union[float, str, bool]]]; initial parmetres in JSON form
    :return: Tuple;
    """
    keys = [key for key, val in x0.items() if not val.get('fixed', False) and not val.get('constraint', False)
            and key not in COMPOSITE_PARAMS]
    values = [val['value'] for key, val in x0.items() if not val.get('fixed', False)
              and not val.get('constraint', False) and key not in COMPOSITE_PARAMS]

    composite_params = {key: val for key, val in x0.items() if key in COMPOSITE_PARAMS}
    for composite_name, composite_value in composite_params.items():
        for key, value in composite_value.items():
            keys += [PARAM_PARSER.join([composite_name, key, param_name]) for param_name, item in value.items()
                         if not item.get('fixed', False) and not item.get('constraint', False)]
            values += [item['value'] for item in value.values()
                       if not item.get('fixed', False) and not item.get('constraint', False)]
    return values, keys


def x0_to_kwargs(x0):
    """
    Transform native input form to `key, value` form::

        {
            key: value,
            ...
        }

    :param x0: Dict[str, Union[float, str, bool]];
    :return: Dict[str, float];
    """
    ret_dict = {key: value['value'] for key, value in x0.items() if key not in COMPOSITE_PARAMS}

    composite_params = {key: val for key, val in x0.items() if key in COMPOSITE_PARAMS}
    for composite_name, composite_value in composite_params.items():
        for key, value in composite_value.items():
            ret_dict.update({PARAM_PARSER.join([composite_name, key, param_name]): item['value']
                             for param_name, item in value.items()})
    return ret_dict


def x0_to_fixed_kwargs(x0):
    """
    Transform native dict input form to `key, value` form, but select `fixed` parametres only::

        {
            key: value,
            ...
        }

    :param x0: Dict[Dict[str, Union[float, str, bool]]];
    :return: Dict[str, float];
    """
    ret_dict = {key: value['value'] for key, value in x0.items() if value.get('fixed', False)
                and key not in COMPOSITE_PARAMS}

    composite_params = {key: val for key, val in x0.items() if key in COMPOSITE_PARAMS}
    for composite_name, composite_value in composite_params.items():
        for key, value in composite_value.items():
            ret_dict.update({PARAM_PARSER.join([composite_name, key, param_name]): item['value']
                             for param_name, item in value.items() if item.get('fixed', False)})
    return ret_dict


def x0_to_constrained_kwargs(x0):
    """
    Transform native dict input form to `key, value` form, but select `constraint` parameters only::

        {
            key: value,
            ...
        }

    :param x0: Dict[Dict[str, Union[float, str, bool]]];
    :return: Dict[str, float];
    """
    def _replace_parser(string, old_separator, new_separator):
        """
        replacing old separator (normally dot `.`) only when it is used as a parameter separator not in case of float
        numbers
        :param string: str; constraint
        :param old_separator: str; old separator (`.`)
        :param new_separator: str; new separator
        :return: str; updated constraint
        """
        test_string = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        components = string.split(old_separator)
        ret_val = components[0]
        for ii in range(1, len(components)):
            if ret_val[-1] in test_string and components[ii][0] in test_string:
                ret_val = old_separator.join([ret_val, components[ii]])
            else:
                ret_val = new_separator.join([ret_val, components[ii]])
        return ret_val

    ret_dict = {key: _replace_parser(value['constraint'], USER_PARAM_PARSER, PARAM_PARSER) for key, value in x0.items()
                if value.get('constraint', False)}

    composite_params = {key: val for key, val in x0.items() if key in COMPOSITE_PARAMS}
    for composite_name, composite_value in composite_params.items():
        for key, value in composite_value.items():
            ret_dict.update({PARAM_PARSER.join([composite_name, key, param_name]):
                                 _replace_parser(item['constraint'], USER_PARAM_PARSER, PARAM_PARSER)
                             for param_name, item in value.items() if item.get('constraint', False)})
    return ret_dict


def x0_to_variable_kwargs(x0):
    """
    Transform native dict input form to `key, value` form, but select variable parameters
    (not fixed or constrained)::

        {
            key: value,
            ...
        }

    :param x0: Dict[Dict[str, Union[float, str, bool]]];
    :return: Dict[str, float];
    """
    ret_dict = {key: value['value'] for key, value in x0.items()
                if not value.get('fixed', False) and not value.get('constraint', False) and key not in COMPOSITE_PARAMS}

    composite_params = {key: val for key, val in x0.items() if key in COMPOSITE_PARAMS}
    for composite_value in composite_params.values():
        for key, value in composite_value.items():
            ret_dict.update({PARAM_PARSER.join([key, param_name]): item['value'] for param_name, item in value.items()
                             if not value.get('fixed', False) and not value.get('constraint', False)})

    return ret_dict


def update_normalization_map(update):
    """
    Update module normalization map with supplied dict.

    :param update: Dict;
    """
    NORMALIZATION_MAP.update(update)


def param_renormalizer(xn, labels):
    """
    Renormalize values from `x` to their native form.

    :param xn: List[float]; iterable of normalized parameter values
    :param labels: Iterable[str]; related parmaeter names from `x`
    :return: List[float];
    """
    return [renormalize_value(_x, *get_param_boundaries(_kword)) for _x, _kword in zip(xn, labels)]


def param_normalizer(xn: List, labels: List) -> List:
    """
    Normalize values from `x` to value between (0, 1).

    :param xn: List[float]; iterable of values in their native form
    :param labels: List[str]; iterable str of names related to `x`
    :return: List[float];
    """
    return [normalize_value(_x, *get_param_boundaries(_kword)) for _x, _kword in zip(xn, labels)]


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

    :param x0: Dict[Dict[str, Union[float, str, bool]]]; initial parmetres in JSON form
    :return: Dict[str, Tuple[float, float]]
    """
    ret_dict = {key: (value.get('min', NORMALIZATION_MAP[key][0]),
                      value.get('max', NORMALIZATION_MAP[key][1]))
                for key, value in x0.items() if not value.get('fixed', False) and not value.get('constraint', False)
                and key not in COMPOSITE_PARAMS}

    composite_params = {key: val for key, val in x0.items() if key in COMPOSITE_PARAMS}
    for composite_name, composite_value in composite_params.items():
        for key, value in composite_value.items():
            ret_dict.update({PARAM_PARSER.join([composite_name, key, param_name]):
                                 (item.get('min', NORMALIZATION_MAP[param_name][0]),
                                  item.get('max', NORMALIZATION_MAP[param_name][1]))
                             for param_name, item in value.items()
                             if not item.get('fixed', False) and not item.get('constraint', False)})

    return ret_dict


def fit_data_initializer(x0, passband=None):
    boundaries = serialize_param_boundaries(x0)
    update_normalization_map(boundaries)

    fixed = x0_to_fixed_kwargs(x0)
    constraint = x0_to_constrained_kwargs(x0)
    x0_vectorized, labels = x0_vectorize(x0)
    x0 = param_normalizer(x0_vectorized, labels)

    observer = Observer(passband='bolometric' if passband is None else passband, system=None)
    observer._system_cls = BinarySystem

    return x0, labels, fixed, constraint, observer


def _check_param_borders(key, val):
    """
    Function checks that given parameter value is within borders.

    :param key: str; name of the parameter
    :param val: dict; parameter attributes
    :return: tuple, parameter borders
    """
    _min, _max = val.get('min', NORMALIZATION_MAP[key][0]), val.get('max', NORMALIZATION_MAP[key][1])
    if 'constraint' in val.keys():
        return _min, _max
    if not (_min <= val['value'] <= _max):
        raise error.InitialParamsError(f'Initial parameters in parameter `{key}` are not valid. Invalid bounds: '
                                       f'{_min} <= {val["value"]} <= {_max}')
    return _min, _max


def lc_initial_x0_validity_check(x0, morphology):
    """
    Validate parameters for light curve fitting.

    # Main idea of `lc_initial_x0_validity_check` is to cut of initialization if over-contact system is expected,
    # but potentials are fixed both to different values or just one of them is fixed.
    # Valid input requires either both potentials fixed on same values or non-of them fixed.
    # When non of them are fixed, internaly is fixed secondary and its value is keep same as primary.

    :param x0: Dict[Dict];
    :param morphology: str;
    :return: List[Dict];
    """
    # first valdiate constraints
    constraints_validator(x0)

    # invalidate fixed and constraints for same value
    for key, record in x0.items():
        if key not in COMPOSITE_PARAMS:
            if 'fixed' in record and 'constraint' in record:
                raise error.InitialParamsError(f'It is not allowed for `{key}` to contain `fixed` and `constraint` '
                                               f'parameter.')

        else:
            for composite_item_key, composite_item in record.items():
                for param_key, param_val in composite_item.items():
                    if 'fixed' in param_val and 'constraint' in param_val:
                        raise error.InitialParamsError(f'It is not allowed for `{param_key}` in `{composite_item_key}` '
                                                       f'to contain `fixed` and `constraint` parameter.')

    is_oc = is_overcontact(morphology)
    are_same = x0[PARAMS_KEY_MAP['Omega1']]['value'] == x0[PARAMS_KEY_MAP['Omega2']]['value']

    omega_1 = x0[PARAMS_KEY_MAP['Omega1']].get('fixed', False)
    omega_2 = x0[PARAMS_KEY_MAP['Omega2']].get('fixed', False)

    any_fixed = omega_1 | omega_2
    all_fixed = omega_1 & omega_2

    for key, val in x0.items():
        if key not in COMPOSITE_PARAMS:
            variable_test = 'fixed' in val.keys() and val['fixed'] is False
            _min, _max = _check_param_borders(key, val) if variable_test else None, None
        else:
            for composite_item_key, composite_item in record.items():
                for param_key, param_val in composite_item.items():
                    variable_test = 'fixed' in param_val.keys() and param_val['fixed'] is False
                    _min, _max = _check_param_borders(param_key, param_val) if variable_test else None, None

    if is_oc and all_fixed and are_same:
        return x0
    if is_oc and all_fixed and not are_same:
        msg = 'Different potential in over-contact morphology with all fixed (pontetial) value are not allowed.'
        raise error.InitialParamsError(msg)
    if is_oc and any_fixed:
        msg = 'Just one fixed potential in over-contact morphology is not allowed.'
        raise error.InitialParamsError(msg)
    if is_oc:
        # if is overcontact, add constraint for secondary pontetial
        _min, _max = x0[PARAMS_KEY_MAP['Omega1']]['min'], x0[PARAMS_KEY_MAP['Omega1']]['max']
        x0[PARAMS_KEY_MAP['Omega2']] = {
            "value": x0[PARAMS_KEY_MAP['Omega1']]['value'],
            "constraint": "{p__surface_potential}",
            "param": "s__surface_potential",
            "min": _min,
            "max": _max,
        }
        update_normalization_map({PARAMS_KEY_MAP['Omega2']: (_min, _max)})

    return x0


def rv_initial_x0_validity_check(x0: Dict):
    """
    Validate parameters for radial velocities curve fitting.

    :param x0: List[Dict];
    :return: List[Dict];
    """
    # validating constraints
    constraints_validator(x0)

    labels = x0.keys()
    has_t0, has_period = 'primary_minimum_time' in labels, 'period' in labels

    if has_t0:
        if not (has_t0 and has_period):
            raise error.ValidationError("Input requires both, period and primary minimum time.")
    else:
        if not has_period:
            raise error.ValidationError("Input requires at least period.")
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
        "value": value
    }
    return adjust_in


def extend_result_with_units(result):
    """
    Add unit information to `result` list.

    :param result: List[Dict];
    :return: List[Dict];
    """
    for key, value in result.items():
        if key in PARAMS_UNITS_MAP:
            value['unit'] = PARAMS_UNITS_MAP[key]
    return result


def constraints_validator(x0):
    """
    Validate constraints. Make sure there is no harmful code.
    Allowed methods used in constraints::

        'arcsin', 'arccos', 'arctan', 'log', 'sin', 'cos', 'tan', 'exp', 'degrees', 'radians'

    Allowed characters used in constraints::

        '(', ')', '+', '-', '*', '/', '.'

    :param x0: Dict[Dict]; initial values
    :raise: elisa.base.error.InitialParamsError;
    """
    x0 = x0.copy()
    allowed_methods = bonds.ALLOWED_CONSTRAINT_METHODS
    allowed_chars = bonds.ALLOWED_CONSTRAINT_CHARS

    x0c = x0_to_constrained_kwargs(x0)
    x0v = x0_to_variable_kwargs(x0)
    x0c = {key: utils.str_repalce(val, allowed_methods, [''] * len(allowed_methods)) for key, val in x0c.items()}

    if len(x0v) == 0:
        raise IndexError('There are no variable parameters to fit.')

    try:
        subst = {key: val.replace(USER_PARAM_PARSER, PARAM_PARSER).format(**x0v).replace(' ', '')
                 for key, val in x0c.items()}
    except KeyError:
        msg = f'It seems your constraint contain variable that cannot be resolved. ' \
              f'Make sure that linked constraint variable is not fixed or check for typos in variable name in ' \
              f'constraint expression.'
        raise error.InitialParamsError(msg)

    for key, val in subst.items():
        if not np.all(np.isin(list(val), allowed_chars + [PARAM_PARSER, ])):
            msg = f'Constraint {key} contain forbidden characters. Allowed: {allowed_chars}'
            raise error.InitialParamsError(msg)


def constraints_evaluator(floats, constraints):
    """
    Substitute variables in constraint with values and evaluate to number.

    :param floats: Dict[str, float]; non-fixed values (xn vector in dict form {label: xn_i})
    :param constraints: Dict[str, float]; values estimated as constraintes in form {label: constraint_string}
    :return: Dict[str, float]; evalauted constraints dict
    """
    allowed_methods = bonds.ALLOWED_CONSTRAINT_METHODS
    numpy_methods = [f'bonds.{method}' for method in bonds.TRANSFORM_TO_METHODS]
    constraints = constraints.copy()
    floats = {key.split(PARAM_PARSER, 1)[1] if PARAM_PARSER in key else key: val for key, val in floats.items()}

    numpy_callable = {key: utils.str_repalce(val, allowed_methods, numpy_methods) for key, val in constraints.items()}
    subst = {key: val.format(**floats) for key, val in numpy_callable.items()}
    try:
        evaluated = {key:  eval(val) for key, val in subst.items()}
    except Exception as e:
        raise error.InitialParamsError(f'Invalid syntax or value in constraint, {str(e)}.')
    return evaluated


def prepare_kwargs(xn, xn_lables, constraints, fixed):
    """
    This will prepare final kwargs for synthetic model evaluation.

    :return: Dict[str, float];
    """
    kwargs = dict(zip(xn_lables, xn))
    kwargs.update(constraints_evaluator(kwargs, constraints))
    kwargs.update(fixed)
    return kwargs


def mcmc_nwalkers_vs_ndim_validity_check(nwalkers, ndim):
    """
    Validate mcmc number of walkers and number of vector dimension.
    Has to be satisfied `nwalkers < ndim * 2`.

    :param nwalkers:
    :param ndim:
    :raise: RuntimeError; when condition `nwalkers < ndim * 2` is not satisfied
    """
    if nwalkers < ndim * 2:
        msg = f'Fit cannot be executed with fewer walkers ({nwalkers}) than twice the number of dimensions ({ndim})'
        raise RuntimeError(msg)


def xs_reducer(xs):
    """
    Convert phases `xs` to single list and inverse map related to given passband (in case of light curves)
    or component (in case of radial velocities).

    :param xs: Dict[str, numpy.array]; phases defined for each passband or ceomponent::

        {<passband>: <phases>} for light curves
        {<component>: <phases>} for radial velocities

    :return: Tuple; (numpy.array, Dict[str, List[int]]);
    """
    # this most likely cannot work corretly in python < 3.6, since dicts are not ordered
    x = np.hstack(list(xs.values())).flatten()
    y = np.arange(len(x)).tolist()
    reverse_dict = dict()
    for xs_key, phases in xs.items():
        reverse_dict[xs_key] = y[:len(phases)]
        del(y[:len(phases)])

    xs_reduced, inverse = np.unique(x, return_inverse=True)
    reverse = {band: inverse[indices] for band, indices in reverse_dict.items()}
    return xs_reduced, reverse


def is_time_dependent(labels):
    if 'period' in labels and 'primary_minimum_time' in labels:
        return True
    return False


def dict_to_user_format(dictionary):
    """
    function converts dictionary of results of parameter labels back to user format

    :param labels: list; list of labels
    :return: list; user formatted list of parameter labels
    """
    ret_dict = {}
    for param_name, param_val in dictionary.items():
        if PARAM_PARSER not in param_name:
            ret_dict[param_name] = param_val
            continue

        identificators = param_name.split(PARAM_PARSER)

        if identificators[0] not in ret_dict.keys():
            ret_dict[identificators[0]] = {}

        if identificators[1] not in ret_dict[identificators[0]].keys():
            ret_dict[identificators[0]][identificators[1]] = {}

        ret_dict[identificators[0]][identificators[1]][identificators[2]] = param_val

    return ret_dict
