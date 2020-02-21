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
    'phase_shift': 'phase_shift'
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


def x0_vectorize(x0) -> Tuple:
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
    keys = [key for key, val in x0.items() if not val.get('fixed', False) and not val.get('constraint', False)]
    values = [val['value'] for key, val in x0.items() if not val.get('fixed', False)
              and not val.get('constraint', False)]
    return values, keys


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
    Transform native dict input form to `key, value` form, but select `fixed` parametres only::

        {
            key: value,
            ...
        }

    :param x0: Dict[Dict[str, Union[float, str, bool]]];
    :return: Dict[str, float];
    """
    return {key: value['value'] for key, value in x0.items() if value.get('fixed', False)}


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
    return {key: value['constraint'] for key, value in x0.items() if value.get('constraint', False)}


def x0_to_variable_kwargs(x0):
    """
    Transform native dict input form to `key, value` form, but select `floats` parameters
    (as in not fixed or constrained)::

        {
            key: value,
            ...
        }

    :param x0: Dict[Dict[str, Union[float, str, bool]]];
    :return: Dict[str, float];
    """
    return {key: value['value'] for key, value in x0.items()
            if not value.get('fixed', False) and not value.get('constraint', False)}


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
    return {key: (value.get('min', NORMALIZATION_MAP[key][0]),
                  value.get('max', NORMALIZATION_MAP[key][1]))
            for key, value in x0.items() if not value.get('fixed', False) and not value.get('constraint', False)}


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
    for record in x0.values():
        if 'fixed' in record and 'constraint' in record:
            msg = f'It is not allowed for record {record} to contain fixed and constraint.'
            raise error.InitialParamsError(msg)

    is_oc = is_overcontact(morphology)
    are_same = x0[PARAMS_KEY_MAP['Omega1']]['value'] == x0[PARAMS_KEY_MAP['Omega2']]['value']

    omega_1 = x0[PARAMS_KEY_MAP['Omega1']].get('fixed', False)
    omega_2 = x0[PARAMS_KEY_MAP['Omega2']].get('fixed', False)

    any_fixed = omega_1 | omega_2
    all_fixed = omega_1 & omega_2

    for key, val in x0.items():
        _min, _max = val.get('min', NORMALIZATION_MAP[key][0]), val.get('max', NORMALIZATION_MAP[key][1])
        if not (_min <= val['value'] <= _max):
            msg = f'Initial parameters are not valid. Invalid bounds: {_min} <= {val["param"]} <= {_max}'
            raise error.InitialParamsError(msg)

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

    try:
        subst = {key: val.format(**x0v).replace(' ', '') for key, val in x0c.items()}
    except KeyError:
        msg = f'It seems your constraint contain variable that cannot be resolved. ' \
            f'Make sure that linked constraint variable is not fixed or check for typos.'
        raise error.InitialParamsError(msg)

    for key, val in subst.items():
        if not np.all(np.isin(list(val), allowed_chars)):
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
    floats, constraints = floats.copy(), constraints.copy()

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
