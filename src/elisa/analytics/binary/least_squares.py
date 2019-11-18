import functools
import numpy as np

from copy import copy
from scipy.optimize import least_squares

from elisa.base.error import HitSolutionBubble
from elisa.conf.config import BINARY_COUNTERPARTS
from elisa.logger import getPersistentLogger
from elisa.analytics.binary import params
from elisa.analytics.binary import (
    utils as analutils,
    model,
    shared
)

logger = getPersistentLogger('analytics.binary.fit')


def logger_decorator(suppress_logger=False):
    def do(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not suppress_logger:
                logger.info(f'current xn value: {kwargs}')
            return func(*args, **kwargs)
        return wrapper
    return do


class CircularSyncLightCurve(shared.AbstractCircularSyncLightCurve):
    def circular_sync_model_to_fit(self, xn):
        """
        Model to find minimum.

        :param xn: Iterable[float];
        :return: float;
        """
        xn = params.param_renormalizer(xn, self._kwords)
        kwargs = {k: v for k, v in zip(self._kwords, xn)}

        # if morphology is overcontact, secondary pontetial has to be same as primary
        if params.is_overcontact(self._morphology):
            to_value = xn[self._hash_map['p__surface_potential']]
            self._fixed = params.adjust_constrained_potential(self._fixed, to_value)

        kwargs.update(self._fixed)
        fn = model.circular_sync_synthetic
        args = self._xs, self._period, self._discretization, self._morphology, self._observer
        try:
            synthetic = logger_decorator()(fn)(*args, **kwargs)
            synthetic = analutils.normalize_lightcurve_to_max(synthetic)
        except Exception as e:
            logger.warning(f'mcmc hit invalid parameters, exception: {str(e)}')
            raise
        residua = np.array([np.sum(np.power(synthetic[band] - self._ys[band], 2)) for band in synthetic])

        if np.abs(residua) <= self._xtol:
            import sys
            sys.tracebacklimit = 0
            raise HitSolutionBubble(f"least_squares hit solution", solution=kwargs)

        return residua

    def fit(self, xs, ys, period, x0, passband, discretization, morphology='detached', xtol=1e-15, max_nfev=None):
        self._xs, self._ys = xs, ys
        self._xtol = xtol

        # Main idea of `initial_x0_validity_check` is to cut of initialization if over-contact system is expected,
        # but potentials are fixed both to different values or just one of them is fixed.
        # Valid input requires either both potentials fixed on same values or non-of them fixed.
        # When non of them are fixed, internaly is fixed secondary and its value is keep same as primary.
        x0 = params.initial_x0_validity_check(x0, morphology)
        initial_x0 = copy(x0)
        x0, kwords, fixed, observer = params.fit_data_initializer(x0, passband=passband)

        self._hash_map = {key: idx for idx, key in enumerate(kwords)}
        self._period = period
        self._morphology = morphology
        self._discretization = discretization
        self._passband = passband
        self._fixed = fixed
        self._kwords = kwords
        self._observer = observer

        logger.info("fitting circular synchronous system...")
        func = self.circular_sync_model_to_fit
        try:
            result = least_squares(func, x0, bounds=(0, 1), max_nfev=max_nfev, xtol=xtol)
        except HitSolutionBubble as bubble:
            return self.serialize_bubble(bubble)

        logger.info("fitting finished")

        result = params.param_renormalizer(result.x, kwords)
        result_dict = {k: v for k, v in zip(kwords, result)}
        result_dict.update(params.x0_to_fixed_kwargs(initial_x0))

        result = [{"param": key, "value": val} for key, val in result_dict.items()]

        if params.is_overcontact(morphology):
            hash_map = {rec["param"]: idx for idx, rec in enumerate(result)}
            result = params.adjust_result_constrained_potential(result, hash_map)

        r_squared_args = xs, ys, period, passband, discretization, morphology
        r_squared_result = shared.lc_r_squared(model.circular_sync_synthetic, *r_squared_args, **result_dict)
        result.append({"r_squared": r_squared_result})

        return result


class CentralRadialVelocity(object):
    @staticmethod
    def centarl_rv_model_to_fit(x, *args):
        xs, ys, period, kwords, fixed, observer, on_normalized = args
        x = params.param_renormalizer(x, kwords)
        kwargs = {k: v for k, v in zip(kwords, x)}
        kwargs.update(fixed)
        fn = model.central_rv_synthetic
        synthetic = logger_decorator()(fn)(xs, period, observer, **kwargs)
        if on_normalized:
            synthetic = analutils.normalize_rv_curve_to_max(synthetic)
        synthetic = {"primary": synthetic[0], "secondary": synthetic[1]}
        return np.array([np.sum(np.power(synthetic[comp] - ys[comp], 2)) for comp in BINARY_COUNTERPARTS])

    @staticmethod
    def fit(xs, ys, period, x0, xtol=1e-15, max_nfev=None, on_normalized=False):
        initial_x0 = copy(x0)
        x0, kwords, fixed, observer = params.fit_data_initializer(x0)

        args = (xs, ys, period, kwords, fixed, observer, on_normalized)
        logger.info("fitting radial velocity light curve...")
        func = CentralRadialVelocity.centarl_rv_model_to_fit
        result = least_squares(func, x0, bounds=(0, 1), args=args, max_nfev=max_nfev, xtol=xtol)
        logger.info("fitting finished")

        result = params.param_renormalizer(result.x, kwords)
        result_dict = {k: v for k, v in zip(kwords, result)}
        result_dict.update(params.x0_to_fixed_kwargs(initial_x0))

        r_squared_args = xs, ys, period, on_normalized
        r_squared_result = shared.rv_r_squared(model.central_rv_synthetic, *r_squared_args, **result_dict)

        result = [{"param": key, "value": val} for key, val in result_dict.items()]
        result.append({"r_squared": r_squared_result})

        return result_dict


circular_sync = CircularSyncLightCurve()
central_rv = CentralRadialVelocity()
