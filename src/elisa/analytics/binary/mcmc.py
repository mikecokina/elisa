from abc import ABC

import emcee
import numpy as np

from copy import copy

from elisa.logger import getPersistentLogger
from elisa.base.error import (
    ElisaError,
    SolutionBubbleException
)
from elisa.analytics.binary import (
    utils as analutils,
    params,
    models,
    shared
)

logger = getPersistentLogger('analytics.binary.mcmc')


class LightCurveFit(shared.AbstractLightCurveFit):
    def model_to_fit(self, *args, **kwargs):
        return self.likelihood(*args, **kwargs)

    def likelihood(self, xn):
        xn = params.param_renormalizer(xn, self._kwords)

        # if morphology is overcontact, secondary pontetial has to be same as primary
        if params.is_overcontact(self._morphology):
            self._fixed['s__surface_potential'] = xn[self._hash_map['p__surface_potential']]

        kwargs = {k: v for k, v in zip(self._kwords, xn)}
        kwargs.update(self._fixed)

        args = self._xs, self._period, self._discretization, self._morphology, self._observer
        synthetic = models.synthetic_binary(*args, **kwargs)
        synthetic = analutils.normalize_lightcurve_to_max(synthetic)
        lhood = -0.5 * np.sum(np.array([np.sum(np.power((synthetic[band] - self._ys[band]) / self._yerrs, 2))
                                        for band in synthetic]))

        if np.abs(lhood) <= self._xtol:
            import sys
            sys.tracebacklimit = 0
            raise SolutionBubbleException(f"mcmc hit solution", solution=kwargs)

        return lhood

    @staticmethod
    def ln_prior(xn):
        return np.all(np.bitwise_and(np.greater_equal(xn, 0.0), np.less_equal(xn, 1.0)))

    def ln_probability(self, xn):
        if not self.ln_prior(xn):
            return -np.inf

        try:
            likelihood = self.model_to_fit(xn)
        except SolutionBubbleException:
            raise
        except ElisaError as e:
            logger.warning(f'mcmc hit invalid parameters, exception: {str(e)}')
            return -10.0 * np.finfo(float).eps * np.sum(xn)

        return likelihood

    @staticmethod
    def resolve_mcmc_result(sampler, kwords, discard=False, thin=1):
        flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        result = list()
        for idx, key in enumerate(kwords):
            mcmc = np.percentile(flat_samples[:, idx], [16, 50, 84])
            val = params.param_renormalizer((mcmc[1],), (key,))[0]
            q = np.diff(params.param_renormalizer(mcmc, np.repeat(key, len(mcmc))))
            result.append({"param": key, "value": val, "min": val - q[0], "max": val + q[1], "fixed": False})
        return result

    def fit(self, xs, ys, period, x0, passband, discretization, nwalkers, nsteps,
            xtol=1e-6, p0=None, yerrs=None, nsteps_burn_in=10):

        yerrs = analutils.lightcurves_mean_error(ys) if yerrs is None else yerrs
        self._xs, self._ys, self._yerrs = xs, ys, yerrs
        self._xtol = xtol

        x0 = params.initial_x0_validity_check(x0, self._morphology)
        x0, kwords, fixed, observer = params.fit_data_initializer(x0, passband=passband)

        self._hash_map = {key: idx for idx, key in enumerate(kwords)}
        self._period = period
        self._morphology = self._morphology
        self._discretization = discretization
        self._passband = passband
        self._fixed = fixed
        self._kwords = kwords
        self._observer = observer

        ndim = len(x0)
        p0 = p0 if p0 is not None else np.random.uniform(0.0, 1.0, (nwalkers, ndim))
        # assign intial value
        p0[0] = x0
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=self.ln_probability)

        try:
            logger.info("running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, nsteps_burn_in if nsteps > nsteps_burn_in else nsteps)
            sampler.reset()

            logger.info("running production...")
            _, _, _ = sampler.run_mcmc(p0, nsteps)
        except SolutionBubbleException as bubble:
            result = self.serialize_bubble(bubble)
            return params.extend_result_with_units(result)

        result = self.resolve_mcmc_result(sampler, kwords)
        result = result + [{"param": key, "value": val} for key, val in fixed.items()]
        if params.is_overcontact(self._morphology):
            hash_map = {rec["param"]: idx for idx, rec in enumerate(result)}
            result = params.adjust_result_constrained_potential(result, hash_map)

        return params.extend_result_with_units(result)


class OvercontactLightCurveFit(LightCurveFit):
    def __init__(self):
        super().__init__()
        self._morphology = 'over-contact'


class DetachedLightCurveFit(LightCurveFit):
    def __init__(self):
        super().__init__()
        self._morphology = 'detached'


binary_detached = DetachedLightCurveFit()
binary_overcontact = OvercontactLightCurveFit()
