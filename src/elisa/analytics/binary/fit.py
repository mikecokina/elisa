import numpy as np

from typing import Tuple, List
from scipy.optimize import least_squares

from elisa.analytics.binary import model
from elisa.conf import config
from elisa.atm import atm_file_prefix_to_quantity_list


__ALL_PARAMS__ = ['inclination',
                  'p__mass',
                  'p__t_eff',
                  'p__omega',
                  's__mass',
                  's__t_eff',
                  's__omega']

__TEMPERATURES__ = atm_file_prefix_to_quantity_list("temperature", config.ATM_ATLAS)

__NORMALIZE_MAP__ = {
    'inclination': (0, 180),
    'mass': (0.5, 20),
    't_eff': (np.min(__TEMPERATURES__), np.max(__TEMPERATURES__)),
    'omega': (2.0, 50.0)
}


def _r_squared(*args, **x):
    # Coefficient of determination
    xs, ys, period, passband = args
    observed_mean = np.mean(ys)

    variability = np.sum(np.power(ys - observed_mean, 2))
    synthetic = model.synthetic(xs, period, passband, **x)
    residual = np.sum(np.power(ys - synthetic, 2))
    return 1.0 - (residual / variability)


def _renormalize_value(val, _min, _max):
    return (val * (_max - _min)) + _min


def _renormalize(x, kwords):
    return [_renormalize_value(_x, *_get_param_boundaries(_kword)) for _x, _kword in zip(x, kwords)]


def _normalize_value(val, _min, _max):
    return (val - _min) / (_max - _min)


def _normalize(x: List, kwords: List) -> List:
    return [_normalize_value(_x, *_get_param_boundaries(_kword)) for _x, _kword in zip(x, kwords)]


def _get_param_boundaries(param):
    param = param[3:] if param not in ['inclination'] else param
    return __NORMALIZE_MAP__[param]


def _x0_vectorize(x0) -> Tuple:
    _x0 = [record['value'] for record in x0]
    _kwords = [record['param'] for record in x0]
    return _x0, _kwords


def _input_x0_to_kwargs(x0):
    return {record['param']: record['value'] for record in x0}


def _model_to_fit(x, *args):
    xs, ys, period, kwords, passband = args
    x = _renormalize(x, kwords)
    kwargs = {k: v for k, v in zip(kwords, x)}
    synthetic = model.synthetic(xs, period, passband, **kwargs)
    return synthetic - ys


def fit(xs, ys, period, x0, passband, xtol=1e-15):
    inp = x0.copy()
    _x0, _kwords = _x0_vectorize(x0)
    x0 = _normalize(_x0, _kwords)
    bounds = (0.0, 1.0)
    result = least_squares(_model_to_fit, x0, bounds=bounds, args=(xs, ys, period, _kwords, passband), xtol=xtol)
    return result

    # import json
    # print(json.dumps({k: v for k, v in zip(_renormalize(x, _kwords), _kwords)}, indent=4))
    # print(_r_squared(*(xs, ys, period, passband), **_input_x0_to_kwargs(inp)))
