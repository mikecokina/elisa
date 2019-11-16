import emcee
import numpy as np

from elisa.logger import getLogger
from elisa.analytics.binary import (
    utils as analutils,
    params,
    model
)

logger = getLogger('analytics.binary.mcmc')


class CircularSyncLightCurve(object):
    @classmethod
    def likelihood(cls, xn, *args):
        xs, ys, yerrs, period, kwords, fixed, discretization, observer = args
        xn = params.param_renormalizer(xn, kwords)
        kwargs = {k: v for k, v in zip(kwords, xn)}
        kwargs.update(fixed)
        synthetic = model.circular_sync_synthetic(xs, period, discretization, observer, **kwargs)
        synthetic = analutils.normalize_lightcurve_to_max(synthetic)
        return -0.5 * np.sum(np.array([np.sum(np.power((synthetic[band] - ys[band]) / yerrs, 2)) for band in synthetic]))

    @classmethod
    def ln_prior(cls, xn):
        in_bounds = [0 <= val <= 1 for val in xn]
        return 0.0 if np.all(in_bounds) else -np.inf

    @classmethod
    def ln_probability(cls, xn, *args):
        lp = cls.ln_prior(xn)

        if not np.isfinite(lp):
            return -np.inf
        try:
            likelihood = cls.likelihood(xn, *args)
        except Exception as e:
            logger.warning(f'mcmc hit invalid parameters, expcetion {str(e)}')
            return -np.inf

        return lp + likelihood

    @classmethod
    def eval_mcmc(cls, xs, ys, period, x0, passband, discretization,
                  nwalkers, nsteps, p0=None, yerrs=None, explore=100):

        x0, kwords, fixed, observer = params.fit_data_initializer(x0, passband=passband)
        yerrs = analutils.lightcurves_mean_error(ys) if yerrs is None else yerrs
        args = (xs, ys, yerrs, period, kwords, fixed, discretization, observer)

        ndim = len(x0)
        p0 = p0 if p0 is not None else np.random.uniform(0.0, 1.0, (nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=cls.ln_probability, args=args)

        logger.info("running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, explore)
        sampler.reset()

        logger.info("running production...")
        pos, prob, state = sampler.run_mcmc(p0, nsteps)

        result = cls.eval_mcmc_result(sampler, kwords)
        return result

    @staticmethod
    def eval_mcmc_result(sampler, kwords, discard=False, thin=1):
        flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        result = list()
        for idx, key in enumerate(kwords):
            mcmc = np.percentile(flat_samples[:, idx], [16, 50, 84])
            val = params.param_renormalizer((mcmc[1], ), (key, ))
            q = np.diff(params.param_renormalizer(mcmc, np.repeat(key, len(mcmc))))
            result.append({key: val, "error": {"min": -q[0], "max": q[1]}})
        return result


circular_sync = CircularSyncLightCurve()
