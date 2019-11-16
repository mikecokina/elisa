import numpy as np

from typing import List, Tuple
from elisa.atm import atm_file_prefix_to_quantity_list
from elisa.binary_system.system import BinarySystem
from elisa.conf import config
from elisa.observer.observer import Observer

ALL_PARAMS = ['inclination',
              'eccentricity',
              'argument_of_periastron'
              'gamma',
              'p__mass',
              'p__t_eff',
              'p__surface_potential',
              'p__gravity_darkening',
              'p__albedo',
              'p__metallicity',
              's__mass',
              's__t_eff',
              's__surface_potential',
              's__gravity_darkening',
              's__albedo',
              's__metallicity']

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
