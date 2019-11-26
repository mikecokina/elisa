import emcee
import numpy as np
import os
import os.path as op
import json

from multiprocessing import Pool
from typing import Iterable, Dict
from datetime import datetime

from ..binary.plot import Plot
from ...conf import config
from ...logger import getPersistentLogger
from ...base.error import (
    ElisaError,
    SolutionBubbleException
)
from ..binary import (
    utils as analutils,
    params,
    models,
    shared
)

logger = getPersistentLogger('analytics.binary.mcmc')


class McMcMixin(object):
    @staticmethod
    def ln_prior(xn):
        return np.all(np.bitwise_and(np.greater_equal(xn, 0.0), np.less_equal(xn, 1.0)))

    @staticmethod
    def resolve_mcmc_result(sampler, labels, discard=False, thin=1, quantiles=None):
        flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        quantiles = [16, 50, 84] if quantiles is None else quantiles
        result = list()
        for idx, key in enumerate(labels):
            mcmc = np.percentile(flat_samples[:, idx], quantiles)
            val = params.param_renormalizer((mcmc[1],), (key,))[0]
            q = np.diff(params.param_renormalizer(mcmc, np.repeat(key, len(mcmc))))
            result.append({"param": key, "value": val, "min": val - q[0], "max": val + q[1], "fixed": False})
        return result

    @staticmethod
    def _store_flat_chain(flat_chain: np.array, labels: Iterable[str], norm: Dict):
        """
        Store state of mcmc run.

        :param flat_chain: numpy.array; flatted array of parameters values in each mcmc step::

            [[param0_0, param1_0, ..., paramk_0],
             [param0_1, param1_1, ..., paramk_1]
             ...
             [param0_b, param1_n, ..., paramk_n]]

        :param labels: Union[List, numpy.array]; labels of parameters in order of params in `flat_chain`
        """
        now = datetime.now()
        fdir = now.strftime(config.DATE_MASK)
        fname = f'{now.strftime(config.DATETIME_MASK)}.json'
        fpath = op.join(config.HOME, fdir, fname)
        os.makedirs(op.join(config.HOME, fdir), exist_ok=True)
        data = {
            "flat_chain": flat_chain.tolist() if isinstance(flat_chain, np.ndarray) else flat_chain,
            "labels": labels,
            "normalization": norm
        }
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=4))

        return fname[:-5]

    @staticmethod
    def restore_flat_chain(fname):
        """
        Restore stored state from mcmc run.

        :param fname: str; base filename of stored state
        :return: Dict;
        """
        fdir = fname[:len(config.DATE_MASK) + 2]
        fname = f'{fname}.json'
        fpath = op.join(config.HOME, fdir, fname)
        with open(fpath, "r") as f:
            return json.loads(f.read())


class LightCurveFit(shared.AbstractLightCurveFit, McMcMixin):
    def __init__(self):
        self.plot = Plot()
        self.last_sampler = emcee.EnsembleSampler
        self.last_normalization = dict()
        self.last_fname = ''
        super(LightCurveFit, self).__init__()

    def model_to_fit(self, *args, **kwargs):
        return self.likelihood(*args, **kwargs)

    def likelihood(self, xn):
        xn = params.param_renormalizer(xn, self._labels)
        kwargs = params.prepare_kwargs(xn, self._labels, self._constraint, self._fixed)

        args = self._xs, self._period, self._discretization, self._morphology, self._observer, True
        synthetic = models.synthetic_binary(*args, **kwargs)
        synthetic = analutils.normalize_lightcurve_to_max(synthetic)

        lhood = -0.5 * np.sum(np.array([np.sum(np.power((synthetic[band] - self._ys[band]) / self._yerrs[band], 2))
                                        for band in synthetic]))

        if np.all(np.less_equal(np.abs(lhood), self._xtol)):
            import sys
            sys.tracebacklimit = 0
            raise SolutionBubbleException(f"mcmc hit solution", solution=kwargs)

        return lhood

    def ln_probability(self, xn):
        if not self.ln_prior(xn):
            return -np.inf

        try:
            likelihood = self.model_to_fit(xn)
        except SolutionBubbleException:
            raise
        except (ElisaError, ValueError) as e:
            logger.warning(f'mcmc hit invalid parameters, exception: {str(e)}')
            return -10.0 * np.finfo(float).eps * np.sum(xn)

        return likelihood

    def fit(self, xs, ys, period, x0, discretization, nwalkers, nsteps,
            xtol=1e-6, p0=None, yerrs=None, nsteps_burn_in=10, quantiles=None, discard=False):
        """
        Fit method using Markov Chain Monte Carlo.

        :param xs: Iterable[float];
        :param ys: Dict;
        :param period: float; sytem period
        :param x0: List[Dict]; initial state (metadata included)
        :param discretization: float; discretization of objects
        :param nwalkers: int; number of walkers
        :param nsteps: int; number of steps in mcmc eval
        :param xtol: float; tolerance of error to consider hitted solution as exact
        :param p0: numpy.array; inital priors for mcmc
        :param yerrs: Union[numpy.array, float]; errors for each point of observation
        :param nsteps_burn_in: int; numer of steps for mcmc to explore parameters
        :param quantiles: List[int];
        :param discard: Union[int, bool]; how many values of result discard when looking for solution
        :return: Dict; solution on supplied quantiles, default is [16, 50, 84]
        """

        def worker(_sampler, _p0):
            logger.info("running burn-in...")
            _p0, _, _ = sampler.run_mcmc(_p0, nsteps_burn_in if nsteps > nsteps_burn_in else nsteps)
            _sampler.reset()
            logger.info("running production...")
            _, _, _ = sampler.run_mcmc(_p0, nsteps)

        passband = list(ys.keys())
        yerrs = {band: analutils.lightcurves_mean_error(ys) for band in passband} if yerrs is None else yerrs
        # xs = xs if isinstance(xs, dict) else {band: xs for band in ys}
        self._xs, self._ys, self._yerrs = xs, ys, yerrs
        self._xtol = xtol

        x0 = params.initial_x0_validity_check(x0, self._morphology)
        x0, labels, fixed, constraint, observer = params.fit_data_initializer(x0, passband=passband)
        ndim = len(x0)

        if nwalkers < ndim * 2:
            msg = f'Fit cannot be executed with fewer walkers ({nwalkers}) than twice the number of dimensions ({ndim})'
            raise RuntimeError(msg)

        self._hash_map = {key: idx for idx, key in enumerate(labels)}
        self._period = period
        self._morphology = self._morphology
        self._discretization = discretization
        self._passband = passband
        self._fixed = fixed
        self._constraint = constraint
        self._labels = labels
        self._observer = observer

        p0 = p0 if p0 is not None else np.random.uniform(0.0, 1.0, (nwalkers, ndim))
        # assign intial value
        p0[0] = x0
        try:
            lnf = self.ln_probability
            if config.NUMBER_OF_MCMC_PROCESSES > 1:
                with Pool(processes=config.NUMBER_OF_MCMC_PROCESSES) as pool:
                    logger.info('starting parallel mcmc')
                    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=lnf, pool=pool)
                    worker(sampler, p0)
            else:
                logger.info('starting singlecore mcmc')
                sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=lnf)
                worker(sampler, p0)

        except SolutionBubbleException as bubble:
            result = self.serialize_bubble(bubble)
            return params.extend_result_with_units(result)

        result = self.resolve_mcmc_result(sampler, labels, discard=discard)
        result = result + [{"param": key, "value": val} for key, val in self._fixed.items()]

        self.last_sampler = sampler
        self.last_normalization = params.NORMALIZATION_MAP.copy()
        self.last_fname = self._store_flat_chain(sampler.get_chain(flat=True), labels, self.last_normalization)
        return params.extend_result_with_units(result)


class OvercontactLightCurveFit(LightCurveFit):
    """
    MCMC fitting implementation for light curves of over-contact binaries.
    It keeps eye on values of potentials - keep it same for primary and secondary component.
    """
    def __init__(self):
        super().__init__()
        self._morphology = 'over-contact'


class DetachedLightCurveFit(LightCurveFit):
    """
    MCMC fitting implementation for light curves of detached binaries.
    """
    def __init__(self):
        super().__init__()
        self._morphology = 'detached'


class CentralRadialVelocity(shared.AbstractCentralRadialVelocity, McMcMixin):
    def __init__(self):
        self.plot = Plot()
        self.last_sampler = emcee.EnsembleSampler
        self.last_normalization = dict()
        self.last_fname = ''
        super(CentralRadialVelocity, self).__init__()

    def likelihood(self, xn):
        pass

    def ln_probability(self, xn):
        pass

    def fit(self):
        pass


binary_detached = DetachedLightCurveFit()
binary_overcontact = OvercontactLightCurveFit()
central_rv = CentralRadialVelocity()
