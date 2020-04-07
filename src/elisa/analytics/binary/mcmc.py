import emcee
import numpy as np
import os
import os.path as op
import json

from abc import ABCMeta, abstractmethod
from multiprocessing import Pool
from typing import Iterable, Dict
from datetime import datetime

from ...conf.config import BINARY_COUNTERPARTS
from ..binary.plot import Plot
from ...conf import config
from ...logger import getPersistentLogger
from ...base.error import ElisaError
from ..binary import (
    utils as analutils,
    params,
    models
)
from ...analytics.binary.shared import (
    AbstractLightCurveDataMixin,
    AbstractCentralRadialVelocityDataMixin,
    AbstractFit
)

logger = getPersistentLogger('analytics.binary.mcmc')


class McMcMixin(object):
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

    @staticmethod
    def worker(sampler, p0, nsteps, nsteps_burn_in):
        logger.info("running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, nsteps_burn_in if nsteps > nsteps_burn_in else nsteps)
        sampler.reset()
        logger.info("running production...")
        _, _, _ = sampler.run_mcmc(p0, nsteps)


class McMcFit(AbstractFit, McMcMixin, metaclass=ABCMeta):
    def __init__(self):
        self.plot = Plot()
        self.last_sampler = emcee.EnsembleSampler
        self.last_normalization = dict()
        self.last_fname = ''
        self._last_known_lhood = np.finfo(float).max * np.finfo(float).eps

    @staticmethod
    def ln_prior(xn):
        return np.all(np.bitwise_and(np.greater_equal(xn, 0.0), np.less_equal(xn, 1.0)))

    @abstractmethod
    def likelihood(self, xn):
        pass

    def ln_probability(self, xn):
        if not self.ln_prior(xn):
            return -np.inf
        try:
            likelihood = self.likelihood(xn)
        except (ElisaError, ValueError) as e:
            logger.warning(f'mcmc hit invalid parameters, exception: {str(e)}')
            return self._last_known_lhood * 1e3
        return likelihood

    def _fit(self, x0, labels, nwalkers, ndim, nsteps, nsteps_burn_in, p0=None):

        p0 = p0 if p0 is not None else np.random.uniform(0.0, 1.0, (nwalkers, ndim))
        # assign intial value
        p0[0] = x0

        lnf = self.ln_probability
        if config.NUMBER_OF_MCMC_PROCESSES > 1:
            with Pool(processes=config.NUMBER_OF_MCMC_PROCESSES) as pool:
                logger.info('starting parallel mcmc')
                sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=lnf, pool=pool)
                self.worker(sampler, p0, nsteps, nsteps_burn_in)
        else:
            logger.info('starting singlecore mcmc')
            sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=lnf)
            self.worker(sampler, p0, nsteps, nsteps_burn_in)

        self.last_sampler = sampler
        self.last_normalization = params.NORMALIZATION_MAP.copy()
        self.last_fname = self._store_flat_chain(sampler.get_chain(flat=True), labels, self.last_normalization)

        return sampler


class LightCurveFit(McMcFit, AbstractLightCurveDataMixin):
    def likelihood(self, xn):
        """
        Liklehood function which defines goodnes of current `xn` vector to be fit of given model.
        Best is 0.0, worst is -inf.

        :param xn: Iterable[float]; vector of parameters we are looking for
        :return: float;
        """
        xn = params.param_renormalizer(xn, self.labels)
        kwargs = params.prepare_kwargs(xn, self.labels, self.constraint, self.fixed)

        args = self.xs, self.period, self.discretization, self.morphology, self.observer, True
        synthetic = models.synthetic_binary(*args, **kwargs)
        synthetic = analutils.normalize_lightcurve_to_max(synthetic)

        lhood = -0.5 * np.sum(np.array([np.sum(np.power((synthetic[band][self.xs_reverser[band]] - self.ys[band])
                                                        / self.yerrs[band], 2)) for band in synthetic]))
        self._last_known_lhood = lhood if lhood < self._last_known_lhood else self._last_known_lhood
        return lhood

    def fit(self, xs, ys, period, x0, discretization, nwalkers, nsteps,
            p0=None, yerrs=None, nsteps_burn_in=10, quantiles=None, discard=False):
        """
        Fit method using Markov Chain Monte Carlo.
        Once simulation is done, following valeus are stored and can be used for further evaluation::

            self.last_sampler: emcee.EnsembleSampler
            self.last_normalization: Dict; normalization map used during fitting
            self.last_fname: str; filename of last stored flatten emcee `sampler` with metadata

        Based on https://emcee.readthedocs.io/en/stable/.

        :param xs: Dict[str, Iterable[float]]; {<passband>: <phases>}
        :param ys: Dict[str, Iterable[float]]; {<passband>: <fluxes>};
        :param period: float; sytem period
        :param x0: List[Dict]; initial state (metadata included)
        :param discretization: float; discretization of objects
        :param nwalkers: int; number of walkers
        :param nsteps: int; number of steps in mcmc eval
        :param p0: numpy.array; inital priors for mcmc
        :param yerrs: Union[numpy.array, float]; errors for each point of observation
        :param nsteps_burn_in: int; numer of steps for mcmc to explore parameters
        :param quantiles: List[int];
        :param discard: Union[int, bool]; how many values of result discard when looking for solution
        :return: emcee.EnsembleSampler; sampler instance
        """

        self.passband = list(ys.keys())
        yerrs = {band: analutils.lightcurves_mean_error(ys) for band in self.passband} if yerrs is None else yerrs
        x0 = params.lc_initial_x0_validity_check(x0, self.morphology)
        x0, labels, fixed, constraint, observer = params.fit_data_initializer(x0, passband=self.passband)
        ndim = len(x0)
        params.mcmc_nwalkers_vs_ndim_validity_check(nwalkers, ndim)

        self.xs, self.xs_reverser = params.xs_reducer(xs)
        self.labels, self.observer, self.period = labels, observer, period
        self.fixed, self.constraint = fixed, constraint
        self.ys, self.yerrs = ys, yerrs
        self.discretization = discretization

        return self._fit(x0, self.labels, nwalkers, ndim, nsteps, nsteps_burn_in, p0)


class OvercontactLightCurveFit(LightCurveFit):
    """
    MCMC fitting implementation for light curves of over-contact binaries.
    It keeps eye on values of potentials - keep it same for primary and secondary component.
    """
    def __init__(self):
        super().__init__()
        self.morphology = 'over-contact'


class DetachedLightCurveFit(LightCurveFit):
    """
    MCMC fitting implementation for light curves of detached binaries.
    """
    def __init__(self):
        super().__init__()
        self.morphology = 'detached'


class CentralRadialVelocity(McMcFit, AbstractCentralRadialVelocityDataMixin):
    def likelihood(self, xn):
        """
        Liklehood function which defines goodnes of current `xn` vector to be fit of given model.
        Best is 0.0, worst is -inf.

        :param xn: Iterable[float]; vector of parameters we are looking for
        :return: float;
        """
        xn = params.param_renormalizer(xn, self.labels)
        kwargs = params.prepare_kwargs(xn, self.labels, self.constraint, self.fixed)

        args = self.xs, self.observer
        synthetic = models.central_rv_synthetic(*args, **kwargs)
        if self.on_normalized:
            synthetic = analutils.normalize_rv_curve_to_max(synthetic)

        lhood = -0.5 * np.sum(np.array([np.sum(np.power((synthetic[comp][self.xs_reverser[comp]] - self.ys[comp])
                                                        / self.yerrs[comp], 2)) for comp in BINARY_COUNTERPARTS]))
        self._last_known_lhood = lhood if lhood < self._last_known_lhood else self._last_known_lhood
        return lhood

    def fit(self, xs, ys, x0, nwalkers, nsteps, p0=None, yerrs=None, nsteps_burn_in=10):
        """
        Fit method using Markov Chain Monte Carlo.
        Once simulation is done, following valeus are stored and can be used for further evaluation::

            self.last_sampler: emcee.EnsembleSampler
            self.last_normalization: Dict; normalization map used during fitting
            self.last_fname: str; filename of last stored flatten emcee `sampler` with metadata

        Based on https://emcee.readthedocs.io/en/stable/.

        :param xs: Iterable[float];
        :param ys: Dict;
        :param x0: List[Dict]; initial state (metadata included)
        :param nwalkers: int; number of walkers
        :param nsteps: int; number of steps in mcmc eval
        :param p0: numpy.array; inital priors for mcmc
        :param yerrs: Union[numpy.array, float]; errors for each point of observation
        :param nsteps_burn_in: int; numer of steps for mcmc to explore parameters
        :return: emcee.EnsembleSampler; sampler instance
        """

        x0 = params.rv_initial_x0_validity_check(x0)
        yerrs = {c: analutils.radialcurves_mean_error(ys) for c in BINARY_COUNTERPARTS} if yerrs is None else yerrs
        x0, labels, fixed, constraint, observer = params.fit_data_initializer(x0)
        ndim = len(x0)
        params.mcmc_nwalkers_vs_ndim_validity_check(nwalkers, ndim)

        self.xs, self.xs_reverser = params.xs_reducer(xs)
        self.ys, self.yerrs = ys, yerrs
        self.labels, self.observer = labels, observer
        self.fixed, self.constraint = fixed, constraint

        return self._fit(x0, self.labels, nwalkers, ndim, nsteps, nsteps_burn_in, p0)


binary_detached = DetachedLightCurveFit()
binary_overcontact = OvercontactLightCurveFit()
central_rv = CentralRadialVelocity()
