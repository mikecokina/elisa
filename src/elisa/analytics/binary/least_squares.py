import functools
import numpy as np

from copy import copy
from scipy.optimize import least_squares

from elisa.binary_system.system import BinarySystem
from elisa.conf.config import BINARY_COUNTERPARTS
from elisa.observer.observer import Observer
from elisa.logger import getLogger
from elisa.analytics.binary import params
from elisa.analytics.binary import (
    utils as analutils,
    model
)

logger = getLogger('analytics.binary.fit')


def logger_decorator(suppress_logger=False):
    def do(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not suppress_logger:
                logger.info(f'current xn value: {kwargs}')
            return func(*args, **kwargs)
        return wrapper
    return do


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
    xs, ys, period, passband, discretization = args
    observed_means = np.array([np.repeat(np.mean(ys[band]), len(xs)) for band in ys])
    variability = np.sum([np.sum(np.power(ys[band] - observed_means, 2)) for band in ys])

    observer = Observer(passband=passband, system=None)
    observer._system_cls = BinarySystem
    synthetic = synthetic(xs, period, discretization, observer, **x)

    synthetic = analutils.normalize_lightcurve_to_max(synthetic)
    residual = np.sum([np.sum(np.power(synthetic[band] - ys[band], 2)) for band in ys])
    return 1.0 - (residual / variability)


def rv_r_squared(synthetic, *args, **x):
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


class CircularSyncLightCurve(object):
    @staticmethod
    def circular_sync_model_to_fit(x, *args):
        """
        Molde to find minimum.

        :param x: Iterable[float];
        :param args: Tuple;
         :**args*::
            * **xs** * -- numpy.array; phases
            * **ys** * -- numpy.array; supplied fluxes (lets say fluxes from observation) normalized to max value
            * **period** * -- float;
            * **discretization** * -- flaot;
            * **suppress_logger** * -- bool;
            * **passband** * -- Iterable[str];
            * **morphology** * -- str;
            * **observer** * -- elisa.observer.observer.Observer;
        :return: float;
        """
        xs, ys, period, kwords, fixed, discretization, suppress_logger, passband, morphology, observer = args
        x = params.param_renormalizer(x, kwords)
        kwargs = {k: v for k, v in zip(kwords, x)}
        kwargs.update(fixed)
        fn = model.circular_sync_synthetic
        synthetic = logger_decorator(suppress_logger)(fn)(xs, period, discretization, morphology, observer, **kwargs)
        synthetic = analutils.normalize_lightcurve_to_max(synthetic)
        return np.array([np.sum(np.power(synthetic[band] - ys[band], 2)) for band in synthetic])

    @staticmethod
    def fit(xs, ys, period, x0, passband, discretization, morphology='detached',
            xtol=1e-15, max_nfev=None, suppress_logger=False):

        initial_x0 = copy(x0)
        x0, kwords, fixed, observer = params.fit_data_initializer(x0, passband=passband)
        args = (xs, ys, period, kwords, fixed, discretization, suppress_logger, passband, morphology, observer)

        logger.info("fitting circular synchronous system...")
        func = CircularSyncLightCurve.circular_sync_model_to_fit
        result = least_squares(func, x0, bounds=(0, 1), args=args, max_nfev=max_nfev, xtol=xtol)
        logger.info("fitting finished")

        result = params.param_renormalizer(result.x, kwords)
        result_dict = {k: v for k, v in zip(kwords, result)}
        result_dict.update(params.x0_to_fixed_kwargs(initial_x0))

        r_squared_args = xs, ys, period, passband, discretization
        r_squared_result = lc_r_squared(model.circular_sync_synthetic, *r_squared_args, **result_dict)
        logger.info(f'r_squared: {r_squared_result}')

        return result_dict


class CentralRadialVelocity(object):
    @staticmethod
    def centarl_rv_model_to_fit(x, *args):
        xs, ys, period, kwords, fixed, suppress_logger, observer, on_normalized = args
        x = params.param_renormalizer(x, kwords)
        kwargs = {k: v for k, v in zip(kwords, x)}
        kwargs.update(fixed)
        fn = model.central_rv_synthetic
        synthetic = logger_decorator(suppress_logger)(fn)(xs, period, observer, **kwargs)
        if on_normalized:
            synthetic = analutils.normalize_rv_curve_to_max(synthetic)
        synthetic = {"primary": synthetic[0], "secondary": synthetic[1]}
        return np.array([np.sum(np.power(synthetic[comp] - ys[comp], 2)) for comp in BINARY_COUNTERPARTS])

    @staticmethod
    def fit(xs, ys, period, x0, xtol=1e-15, max_nfev=None, suppress_logger=False, on_normalized=False):
        initial_x0 = copy(x0)
        x0, kwords, fixed, observer = params.fit_data_initializer(x0)

        args = (xs, ys, period, kwords, fixed, suppress_logger, observer, on_normalized)
        logger.info("fitting radial velocity light curve...")
        func = CentralRadialVelocity.centarl_rv_model_to_fit
        result = least_squares(func, x0, bounds=(0, 1), args=args, max_nfev=max_nfev, xtol=xtol)
        logger.info("fitting finished")

        result = params.param_renormalizer(result.x, kwords)
        result_dict = {k: v for k, v in zip(kwords, result)}
        result_dict.update(params.x0_to_fixed_kwargs(initial_x0))

        r_squared_args = xs, ys, period, on_normalized
        r_squared_result = rv_r_squared(model.central_rv_synthetic, *r_squared_args, **result_dict)
        logger.info(f'r_squared: {r_squared_result}')

        return result_dict


circular_sync = CircularSyncLightCurve()
central_rv = CentralRadialVelocity()
