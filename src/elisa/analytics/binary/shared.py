import numpy as np

from elisa.binary_system.system import BinarySystem
from elisa.conf.config import BINARY_COUNTERPARTS
from elisa.observer.observer import Observer
from elisa.analytics.binary import (
    utils as analutils,
    params,
    model
)


class AbstractCircularSyncLightCurve(object):
    """
    Params:

    * **_hash_map** * -- Dict[str, int];
    * **_xs** * -- numpy.array; phases
    * **_ys** * -- numpy.array; supplied fluxes (lets say fluxes from observation) normalized to max value
    * **_period** * -- float;
    * **_discretization** * -- flaot;
    * **_passband** * -- Iterable[str];
    * **_morphology** * -- str;
    * **_hash_map** * -- Dict;
    * **_observer** * -- elisa.observer.observer.Observer;
    * **_xtol** * -- float;
    * **_kwords** * -- Iterable[str];
    * **_fixed** * -- Dict;
    """
    def __init__(self):
        self._hash_map = dict()
        self._morphology = 'detached'
        self._discretization = np.nan
        self._passband = ''
        self._fixed = dict()
        self._kwords = list()
        self._observer = None
        self._period = np.nan

        self._xs = list()
        self._ys = dict()
        self._yerrs = np.nan
        self._xtol = np.nan

    def serialize_bubble(self, bubble):
        result = [{"param": key, "value": val, "fixed": True} for key, val in bubble.solution.items()]
        if params.is_overcontact(self._morphology):
            hash_map = {rec["param"]: idx for idx, rec in enumerate(result)}
            result = params.adjust_result_constrained_potential(result, hash_map)

        r_squared_args = self._xs, self._ys, self._period, self._passband, self._discretization, self._morphology
        r_squared_result = lc_r_squared(model.circular_sync_synthetic, *r_squared_args, **bubble.solution)
        result.append({"r_squared": r_squared_result})
        return result


def lc_r_squared(synthetic, *args, **x):
    """
    Compute R^2 (coefficient of determination).

    :param synthetic: callable; synthetic method
    :param args: Tuple;
    :**args*::
        * **xs** * -- numpy.array; phases
        * **ys** * -- numpy.array; supplied fluxes (lets say fluxes from observation) normalized to max value
        * **period** * -- float;
        * **passband** * -- Union[str, List[str]];
        * **discretization** * -- flaot;
    :param x: Dict;
    :** x options**: kwargs of current parameters to compute binary system
    :return: float;
    """
    xs, ys, period, passband, discretization, morphology = args
    observed_means = np.array([np.repeat(np.mean(ys[band]), len(xs)) for band in ys])
    variability = np.sum([np.sum(np.power(ys[band] - observed_means, 2)) for band in ys])

    observer = Observer(passband=passband, system=None)
    observer._system_cls = BinarySystem
    synthetic = synthetic(xs, period, discretization, morphology, observer, **x)

    synthetic = analutils.normalize_lightcurve_to_max(synthetic)
    residual = np.sum([np.sum(np.power(synthetic[band] - ys[band], 2)) for band in ys])
    return 1.0 - (residual / variability)


def rv_r_squared(synthetic, *args, **x):
    """
    Compute R^2 (coefficient of determination).
    """
    xs, ys, period, on_normalized = args
    observed_means = np.array([np.repeat(np.mean(ys[comp]), len(xs)) for comp in BINARY_COUNTERPARTS])
    variability = np.sum([np.sum(np.power(ys[comp] - observed_means, 2)) for comp in BINARY_COUNTERPARTS])

    observer = Observer(passband='bolometric', system=None)
    observer._system_cls = BinarySystem
    synthetic = synthetic(xs, period, observer, **x)
    if on_normalized:
        synthetic = analutils.normalize_rv_curve_to_max(synthetic)
    synthetic = {"primary": synthetic[0], "secondary": synthetic[1]}

    residual = np.sum([np.sum(np.power(synthetic[comp] - ys[comp], 2)) for comp in BINARY_COUNTERPARTS])
    return 1.0 - (residual / variability)
