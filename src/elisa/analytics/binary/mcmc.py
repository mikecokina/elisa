import emcee
import numpy as np

from copy import copy
from elisa.logger import getPersistentLogger
from elisa.base.error import (
    ElisaError,
    HitSolutionBubble
)
from elisa.analytics.binary import (
    utils as analutils,
    params,
    model
)

logger = getPersistentLogger('analytics.binary.mcmc')


class CircularSyncLightCurve(object):
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

    def likelihood(self, xn):
        xn = params.param_renormalizer(xn, self._kwords)

        # if morphology is overcontact, secondary pontetial has to be same as primary
        if params.is_overcontact(self._morphology):
            self._fixed['s__surface_potential'] = xn[self._hash_map['p__surface_potential']]

        kwargs = {k: v for k, v in zip(self._kwords, xn)}
        kwargs.update(self._fixed)

        args = self._xs, self._period, self._discretization, self._morphology, self._observer
        synthetic = model.circular_sync_synthetic(*args, **kwargs)
        synthetic = analutils.normalize_lightcurve_to_max(synthetic)
        lhood = -0.5 * np.sum(np.array([np.sum(np.power((synthetic[band] - self._ys[band]) / self._yerrs, 2))
                                        for band in synthetic]))

        if np.abs(lhood) <= self._xtol:
            import sys
            sys.tracebacklimit = 0
            raise HitSolutionBubble(f"mcmc hit solution", solution=kwargs)

        return lhood

    @staticmethod
    def ln_prior(xn):
        return np.all(0 <= xn <= 1)

    def ln_probability(self, xn):
        if not self.ln_prior(xn):
            return -np.inf

        try:
            likelihood = self.likelihood(xn)
        except HitSolutionBubble:
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

    def eval_mcmc(self, xs, ys, period, x0, passband, discretization, nwalkers, nsteps,
                  niters=1, morphology="detached", xtol=1e-6, p0=None, yerrs=None, explore=10):

        yerrs = analutils.lightcurves_mean_error(ys) if yerrs is None else yerrs
        self._xs, self._ys, self._yerrs = xs, ys, yerrs
        self._xtol = xtol

        result = dict()
        for iter_num in range(niters):
            x0 = params.initial_x0_validity_check(x0, morphology)
            x0, kwords, fixed, observer = params.fit_data_initializer(x0, passband=passband)
            
            self._hash_map = {key: idx for idx, key in enumerate(kwords)}
            self._period = period
            self._morphology = morphology
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
                p0, _, _ = sampler.run_mcmc(p0, explore if nsteps > explore else nsteps)
                sampler.reset()

                logger.info("running production...")
                _, _, _ = sampler.run_mcmc(p0, nsteps)
            except HitSolutionBubble as e:
                result = [{"param": key, "value": val, "fixed": True} for key, val in e.solution.items()]
                if params.is_overcontact(morphology):
                    hash_map = {rec["param"]: idx for idx, rec in enumerate(result)}
                    result = params.adjust_result_constrained_potential(result, hash_map)
                return result

            result = self.resolve_mcmc_result(sampler, kwords)
            result = result + [{"param": key, "value": val, "fixed": True} for key, val in fixed.items()]
            if params.is_overcontact(morphology):
                hash_map = {rec["param"]: idx for idx, rec in enumerate(result)}
                result = params.adjust_result_constrained_potential(result, hash_map)

            logger.info(f'result after {iter_num}. iteration: {result}')
            x0 = copy(result)

        return result


circular_sync = CircularSyncLightCurve()
