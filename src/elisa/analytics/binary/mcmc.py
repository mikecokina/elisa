import emcee
import numpy as np

from elisa.base import error
from elisa.logger import getPersistentLogger
from elisa.analytics.binary import (
    utils as analutils,
    params,
    model
)

logger = getPersistentLogger('analytics.binary.mcmc')


class CircularSyncLightCurve(object):
    @classmethod
    def likelihood(cls, xn, *args):
        xs, ys, yerrs, period, kwords, fixed, discretization, morphology, hash_map, observer = args
        xn = params.param_renormalizer(xn, kwords)

        # if morphology is overcontact, secondary pontetial has to be same as primary
        if morphology in ['over-contact']:
            fixed['s__surface_potential'] = xn[hash_map['p__surface_potential']]

        kwargs = {k: v for k, v in zip(kwords, xn)}
        kwargs.update(fixed)
        synthetic = model.circular_sync_synthetic(xs, period, discretization, morphology, observer, **kwargs)
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
        # fixme: this might filter important errors; it will require to change type of accross elisa and catch expected
        except Exception as e:
            logger.warning(f'mcmc hit invalid parameters, expcetion {str(e)}')
            return -np.inf

        return lp + likelihood

    @staticmethod
    def initial_x0_validity_check(x0, morphology):
        hash_map = {val['param']: idx for idx, val in enumerate(x0)}
        param = 'surface_potential'
        is_oc = morphology in ['over-contact']
        are_same = x0[hash_map[f'p__{param}']]['value'] == x0[hash_map[f's__{param}']]['value']
        any_fixed = x0[hash_map[f'p__{param}']].get('fixed', False) or x0[hash_map[f's__{param}']].get('fixed', False)
        all_fixed = x0[hash_map[f'p__{param}']].get('fixed', False) and x0[hash_map[f's__{param}']].get('fixed', False)

        if is_oc and all_fixed and not are_same:
            msg = 'different potential in over-contact morphology with all fixed (pontetial) value are not allowed'
            raise error.InitialParamsError(msg)
        if is_oc and any_fixed:
            msg = 'just one fixed potential in over-contact morphology is not allowed'
            raise error.InitialParamsError(msg)
        if is_oc:
            # if is overcontact, fix secondary pontetial for further steps
            x0[hash_map[f's__{param}']]['fixed'] = True
        return x0

    @classmethod
    def eval_mcmc(cls, xs, ys, period, x0, passband, discretization,
                  nwalkers, nsteps, morphology="detached", p0=None, yerrs=None, explore=100):

        x0 = cls.initial_x0_validity_check(x0, morphology)
        x0, kwords, fixed, observer = params.fit_data_initializer(x0, passband=passband)
        hash_map = {key: idx for idx, key in enumerate(kwords)}
        yerrs = analutils.lightcurves_mean_error(ys) if yerrs is None else yerrs
        args = (xs, ys, yerrs, period, kwords, fixed, discretization, morphology, hash_map, observer)

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
