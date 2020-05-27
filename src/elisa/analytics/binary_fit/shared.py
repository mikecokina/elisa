from typing import Iterable, Dict

import numpy as np

from scipy import interpolate
from abc import ABCMeta, abstractmethod

from elisa.analytics.params import parameters
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.analytics.tools.utils import (
    normalize_light_curve,
    radialcurves_mean_error,
    lightcurves_mean_error,
    time_layer_resolver)
from elisa.binary_system.system import BinarySystem
from elisa.observer.observer import Observer
from elisa.utils import is_empty


class AbstractFit(metaclass=ABCMeta):
    MEAN_ERROR_FN = None

    __slots__ = ['fixed', 'constrained', 'fitable', 'normalized', 'observer', 'x_data', 'y_data',
                 'y_err', 'x_data_reduced', 'x_data_reducer', 'initial_vector', 'normalization', 'flat_result']

    def set_up(self, x0: BinaryInitialParameters, data: Dict, passband: Iterable = None, **kwargs):
        setattr(self, 'fixed', x0.get_fixed(jsonify=False))
        setattr(self, 'constrained', x0.get_constrained(jsonify=False))
        setattr(self, 'fitable', x0.get_fitable(jsonify=False))
        setattr(self, 'normalization', x0.get_normalization_map())

        observer = Observer(passband='bolometric' if passband is None else passband, system=None)
        observer._system_cls = BinarySystem
        setattr(self, 'observer', observer)

        setattr(self, 'x_data', {key: val.x_data for key, val in data.items()})
        setattr(self, 'y_data', {key: val.y_data for key, val in data.items()})

        err = {key: abs(self.__class__.MEAN_ERROR_FN(val)) if data[key].y_err is None else data[key].y_err
               for key, val in self.y_data.items()}
        setattr(self, 'y_err', err)

        x_data_reduced, x_data_reducer = parameters.xs_reducer({key: val.x_data for key, val in data.items()})
        setattr(self, 'x_data_reduced', x_data_reduced)
        setattr(self, 'x_data_reducer', x_data_reducer)

        setattr(self, 'initial_vector', [val.value for val in self.fitable.values()])
        setattr(self, 'flat_result', dict())

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @staticmethod
    def eval_constrained_results(result_dict: Dict[str, Dict], constraints: Dict[str, 'InitialParameter']):
        """
        Function adds constrained parameters into the resulting dictionary.

        :param constraints: Dict; contains constrained parameters
        :param result_dict: Dict; {'name': {'value': value, 'unit': unit, ...}
        :return: Dict; {'name': {'value': value, 'unit': unit, ...}
        """
        if is_empty(constraints):
            return result_dict

        res_val_dict = {key: val['value'] for key, val in result_dict.items()}
        constrained_values = parameters.constraints_evaluator(res_val_dict, constraints)
        result_dict.update(
            {
                key: {
                    'value': val,
                    'constraint': constraints[key].constraint,
                    'unit': constraints[key].to_dict()['unit']
                } for key, val in constrained_values.items()
            })
        return result_dict


class AbstractRVFit(AbstractFit):
    MEAN_ERROR_FN = radialcurves_mean_error

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def set_up(self, x0: BinaryInitialParameters, data: Dict, **kwargs):
        super().set_up(x0, data, passband=None, **kwargs)


class AbstractLCFit(AbstractFit):
    MEAN_ERROR_FN = lightcurves_mean_error

    __slots__ = ['fixed', 'constrained', 'fitable', 'normalized', 'observer', "discretization",
                 'interp_treshold', 'data', 'y_err', 'x_data_reduced', 'x_data_reducer',
                 'initial_vector', 'normalization', 'x_data', 'y_data']

    def set_up(self, x0: BinaryInitialParameters, data: Dict, passband: Iterable = None, **kwargs):
        super().set_up(x0, data, passband)
        setattr(self, 'discretization', kwargs.pop('discretization'))
        setattr(self, 'interp_treshold', kwargs.pop('interp_treshold'))
        self.normalize_data(kind='average')

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def normalize_data(self, kind='global_maximum', top_fraction_to_average=0.1):
        y_data, y_err = normalize_light_curve(self.y_data, self.y_err, kind, top_fraction_to_average)
        setattr(self, 'y_data', y_data)
        setattr(self, 'y_err', y_err)


def lc_r_squared(synthetic, *args, **x):
    """
    Compute R^2 (coefficient of determination).

    :param synthetic: callable; synthetic method
    :param args: Tuple;
    :**args**:
        * **x_data_reduced** * -- numpy.array; phases
        * **y_data** * -- numpy.array; supplied fluxes (lets say fluxes from observation) normalized to max value
        * **period** * -- float;
        * **passband** * -- Union[str, List[str]];
        * **discretization** * -- flaot;
        * **diff** * -- flaot; auxiliary parameter for interpolation defining spacing between evaluation phases if
                               interpolation is active
        * **interp_treshold** * - int; number of points
    :param x: Dict;
    :** x options**: kwargs of current parameters to compute binary system
    :return: float;
    """
    x_data_reduced, y_data, passband, discretization, x_data_reducer, diff, interp_treshold = args

    x_data_reduced, kwargs = time_layer_resolver(x_data_reduced, pop=False, **x)
    fit_xs = np.linspace(np.min(x_data_reduced) - diff, np.max(x_data_reduced) + diff, num=interp_treshold + 2) \
        if np.shape(x_data_reduced)[0] > interp_treshold else x_data_reduced

    observer = Observer(passband=passband, system=None)
    observer._system_cls = BinarySystem
    synthetic = synthetic(fit_xs, discretization, observer, **x)

    if np.shape(fit_xs) != np.shape(x_data_reduced):
        synthetic = {
            fltr: interpolate.interp1d(fit_xs, curve, kind='cubic')(x_data_reduced)
            for fltr, curve in synthetic.items()
        }
    synthetic = {band: synthetic[band][x_data_reducer[band]] for band in synthetic}
    synthetic, _ = normalize_light_curve(synthetic, kind='average')

    return r_squared(synthetic, y_data)


def rv_r_squared(synthetic, *args, **x):
    """
    Compute R^2 (coefficient of determination).

    :param synthetic: callable; synthetic method
    :param args: Tuple;
    :**args**:
        * **x_data_reduced** * -- numpy.array; phases
        * **y_data** * -- numpy.array; supplied fluxes (lets say fluxes from observation) normalized to max value
        * **period** * -- float;
        * **on_normalized** * -- bool;
    :param x: Dict;
    :** x options**: kwargs of current parameters to compute radial velocities curve
    :return: float;
    """
    x_data_reduced, y_data, x_data_reducer = args

    observer = Observer(passband='bolometric', system=None)
    observer._system_cls = BinarySystem
    synthetic = synthetic(x_data_reduced, observer, **x)
    synthetic = {comp: synthetic[comp][x_data_reducer[comp]] for comp in synthetic}

    return r_squared(synthetic, y_data)


def r_squared(synthetic, observed):
    variability = np.sum([np.sum(np.power(observed[item] - np.mean(observed[item]), 2)) for item in observed])
    residual = np.sum([np.sum(np.power(synthetic[item] - observed[item], 2)) for item in observed])

    return 1 - (residual / variability)
