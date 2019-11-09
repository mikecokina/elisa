import functools
import json
from copy import copy

import numpy as np

from typing import Tuple, List
from scipy.optimize import least_squares

from elisa import logger
from elisa.analytics.binary import model
from elisa.atm import atm_file_prefix_to_quantity_list
from elisa.conf import config

config.set_up_logging()
__logger__ = logger.getLogger('binary-fit')

ALL_PARAMS = ['inclination',
              'p__mass',
              'p__t_eff',
              'p__omega',
              'p__beta',
              'p__albedo',
              'p__mh',
              's__mass',
              's__t_eff',
              's__omega',
              's__beta',
              's__albedo',
              's__mh']

TEMPERATURES = atm_file_prefix_to_quantity_list("temperature", config.ATM_ATLAS)
METALLICITY = atm_file_prefix_to_quantity_list("metallicity", config.ATM_ATLAS)


NORMALIZE_MAP = {
    'inclination': (0, 180),
    'mass': (0.5, 20),
    't_eff': (np.min(TEMPERATURES), np.max(TEMPERATURES)),
    'mh': (np.min(METALLICITY), np.max(METALLICITY)),
    'omega': (2.0, 50.0),
    'albedo': (0, 1),
    'beta': (0, 1)
}


def _renormalize_value(val, _min, _max):
    """
    Renormalize value `val` to value from interval specific for given parameter defined my `_min` and `_max`.

    :param val: float;
    :param _min: float;
    :param _max: float;
    :return: float;
    """
    return (val * (_max - _min)) + _min


def _renormalize(x, kwords):
    """
    Renormalize values from `x` to their native form.

    :param x: Iterable[float]; iterable of normalized parameter values
    :param kwords: Iterable[str]; related parmaeter names from `x`
    :return: List[float];
    """
    return [_renormalize_value(_x, *_get_param_boundaries(_kword)) for _x, _kword in zip(x, kwords)]


def _normalize_value(val, _min, _max):
    """
    Normalize value `val` to value from interval (0, 1) based on `_min` and `_max`.

    :param val: float;
    :param _min: float;
    :param _max: float;
    :return: float;
    """
    return (val - _min) / (_max - _min)


def _normalize(x: List, kwords: List) -> List:
    """
    Normalize values from `x` to value between (0, 1).

    :param x: Iterable[float]; iterable of values in their native form
    :param kwords: Iterable[str]; iterable str of names related to `x`
    :return: List[float];
    """
    return [_normalize_value(_x, *_get_param_boundaries(_kword)) for _x, _kword in zip(x, kwords)]


def _get_param_boundaries(param):
    """
    Return normalization boundaries for given parmeter.

    :param param: str; name of parameter to get boundaries for
    :return: Tuple[float, float];
    """
    param = param[3:] if param not in ['inclination'] else param
    return NORMALIZE_MAP[param]


def _x0_vectorize(x0) -> Tuple:
    """
    Transform native JSON form of initial parameters to Tuple.
    JSON form::

        [
            {
                'value': 2.0,
                'param': 'p__mass',
                'fixed': False
            },
            {
                'value': 4000.0,
                'param': 'p__t_eff',
                'fixed': True
            },
            ...
        ]

    :param x0: List[Dict[str, Union[float, str, bool]]]; initial parmetres in JSON form
    :return: Tuple;
    """
    _x0 = [record['value'] for record in x0 if not record['fixed']]
    _kwords = [record['param'] for record in x0 if not record['fixed']]
    return _x0, _kwords


def _x0_to_kwargs(x0):
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


def _x0_to_fixed_kwargs(x0):
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


def _logger_decorator(suppress_logger=False):
    def do(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not suppress_logger:
                __logger__.info(f'current xn value: {json.dumps(kwargs, indent=4)}')
            return func(*args, **kwargs)
        return wrapper
    return do


def circular_sync_model_to_fit(x, *args):
    xs, ys, period, kwords, fixed, passband, discretization, suppress_logger = args
    x = _renormalize(x, kwords)
    kwargs = {k: v for k, v in zip(kwords, x)}
    kwargs.update(fixed)

    fn = model.circular_sync_synthetic
    synthetic = _logger_decorator(suppress_logger)(fn)(xs, period, passband, discretization, **kwargs)
    return synthetic - ys


def r_squared(*args, **x):
    """
    Compute R^2 (coefficient of determination).

    :param args: Tuple;
    :**args*::
        * **xs** * -- numpy.array; phases
        * **ys** * -- numpy.array; supplied fluxes (lets say fluxes from observation) normalized to max value
        * **period** * -- float;
        * **passband** * -- str;
    :param x: Dict;
    :** x options**: kwargs of current parameters to compute binary system
    :return: float;
    """

    xs, ys, period, passband, discretization, suppress_logger = args
    observed_mean = np.mean(ys)

    variability = np.sum(np.power(ys - observed_mean, 2))
    synthetic = model.circular_sync_synthetic(xs, period, passband, discretization, **x)
    residual = np.sum(np.power(ys - synthetic, 2))
    return 1.0 - (residual / variability)


class Fit(object):
    @staticmethod
    def circular_sync(xs, ys, period, x0, passband, discretization, xtol=1e-15, max_nfev=None, suppress_logger=False):
        initial_x0 = copy(x0)
        fixed = _x0_to_fixed_kwargs(x0)
        x0_vectorized, kwords = _x0_vectorize(x0)
        x0 = _normalize(x0_vectorized, kwords)
        bounds = (0.0, 1.0)

        args = (xs, ys, period, kwords, fixed, passband, discretization, suppress_logger)

        __logger__.info("fitting circular synchronous system...")
        result = least_squares(circular_sync_model_to_fit, x0, bounds=bounds, args=args, max_nfev=max_nfev, xtol=xtol)
        __logger__.info("fitting finished")

        result = _renormalize(result.x, kwords)
        result_dict = {k: v for k, v in zip(kwords, result)}
        result_dict.update(_x0_to_fixed_kwargs(initial_x0))
        return result_dict


fit = Fit()
