import emcee
import numpy as np
import os
import os.path as op
import json
from scipy import interpolate

from abc import ABCMeta, abstractmethod
from multiprocessing import Pool
from typing import Iterable, Dict
from datetime import datetime

from elisa.graphic.mcmc_graphics import Plot
from elisa.conf import config
from elisa.logger import getPersistentLogger
from elisa.base.error import ElisaError
from elisa.analytics.binary import (
    utils as butils,
    params,
    models,
)
from elisa.analytics.binary.shared import (
    AbstractLightCurveDataMixin,
    AbstractCentralRadialVelocityDataMixin,
    AbstractFit,
    lc_r_squared,
    rv_r_squared
)


logger = getPersistentLogger('analytics.binary.mcmc')


class McMcMixin(object):
    @staticmethod
    def resolve_mcmc_result(flat_chain, labels, percentiles=None):
        """
        Function process flat chain (output from McMcFit._fit.get_chain(flat=True)) and produces dictionary with
        results.

        :param flat_chain: emcee.ensemble.EnsembleSampler.get_chain(flat=True);
        :param labels: List; list with names of variable fit parameters in correct order
                             (output of params.x0_to_variable_kwargs)
        :param percentiles: List; [percentile for left side error estimation, percentile of the centre,
                                  percentile for right side error estimation]
        :return: Dict;
        """
        # flat_chain = sampler.get_chain(discard=discard, thin=thin, flat=True)
        percentiles = [16, 50, 84] if percentiles is None else percentiles
        result = dict()
        for idx, key in enumerate(labels):
            mcmc = np.percentile(flat_chain[:, idx], percentiles)
            vals = params.param_renormalizer(mcmc, np.repeat(key, len(mcmc)))
            result[key] = {"value": vals[1], "min": min(vals), "max": max(vals), "fixed": False}

        return params.extend_result_with_units(result)

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
        logger.info(f'MCMC chain, variable`s labels and normalization constants were stored in: {fpath}')
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
    def worker(sampler, p0, nsteps, nsteps_burn_in, progress=False):
        logger.info("running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, nsteps_burn_in, progress=progress)
        sampler.reset()
        logger.info("running production...")
        _, _, _ = sampler.run_mcmc(p0, nsteps, progress=progress)


class McMcFit(AbstractFit, AbstractLightCurveDataMixin, McMcMixin, metaclass=ABCMeta):
    def __init__(self):
        self.plot = Plot()
        self.last_sampler = emcee.EnsembleSampler
        self.last_normalization = dict()
        self.last_fname = ''

        self.eval_counter = 0

    @staticmethod
    def ln_prior(xn):
        return np.all(np.bitwise_and(np.greater_equal(xn, 0.0), np.less_equal(xn, 1.0)))

    @abstractmethod
    def likelihood(self, xn):
        pass

    def lhood(self, synthetic):
        """
        Calculates likelihood function value for a synthetic model to be a correct model for given observational data.

        :param synthetic: Dict; {'dataset_name': numpy.array, }
        :return: float;
        """
        lh = - 0.5 * np.sum([np.power((self.ys[item] - synthetic[item][self.xs_reverser[item]]) / self.yerrs[item], 2)
                             for item, value in synthetic.items()])
        return lh

    def ln_probability(self, xn):
        if not self.ln_prior(xn):
            return -np.inf
        try:
            likelihood = self.likelihood(xn)
        except (ElisaError, ValueError) as e:
            logger.warning(f'mcmc hit invalid parameters, exception: {str(e)}')
            return -10.0 * np.finfo(float).eps * np.sum(xn)
        return likelihood

    @staticmethod
    def eval_constraints_after_mcmc(result_dict, constraints):
        """
        Function adds constrained parameters into the resulting dictionary.

        :param constraints: Dict; contains constrained parameters
        :param result_dict: Dict; {'name': {'value': value, 'unit': unit, ...}
        :return: Dict; {'name': {'value': value, 'unit': unit, ...}
        """
        res_val_dict = {key: val['value'] for key, val in result_dict.items()}
        constrained_values = params.constraints_evaluator(res_val_dict, constraints)
        result_dict.update({key: {'value': val} for key, val in constrained_values.items()})
        return result_dict

    def _fit(self, x0, labels, nwalkers, ndim, nsteps, nsteps_burn_in, p0=None, progress=False):

        p0 = p0 if p0 is not None else np.random.uniform(0.0, 1.0, (nwalkers, ndim))
        # assign intial value
        p0[0] = x0

        lnf = self.ln_probability
        if config.NUMBER_OF_MCMC_PROCESSES > 1:
            with Pool(processes=config.NUMBER_OF_MCMC_PROCESSES) as pool:
                logger.info('starting parallel mcmc')
                sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=lnf, pool=pool)
                self.worker(sampler, p0, nsteps, nsteps_burn_in, progress=progress)
        else:
            logger.info('starting singlecore mcmc')
            sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=lnf)
            self.worker(sampler, p0, nsteps, nsteps_burn_in, progress=progress)

        self.last_sampler = sampler
        self.last_normalization = params.NORMALIZATION_MAP.copy()
        self.last_fname = self._store_flat_chain(sampler.get_chain(flat=True), labels, self.last_normalization)

        return sampler


class LightCurveFit(McMcFit):
    def likelihood(self, xn):
        """
        Liklehood function which defines goodnes of current `xn` vector to be fit of given model.
        Best is 0.0, worst is -inf.

        :param xn: Iterable[float]; vector of parameters we are looking for
        :return: float;
        """
        xn = params.param_renormalizer(xn, self.labels)
        kwargs = params.prepare_kwargs(xn, self.labels, self.constraint, self.fixed)

        args = self.fit_xs, self.period, self.discretization, self.morphology, self.observer, True
        synthetic = models.synthetic_binary(*args, **kwargs)
        synthetic = butils.normalize_light_curve(synthetic, kind='average')

        if np.shape(self.fit_xs) != np.shape(self.xs):
            new_synthetic = dict()
            for fltr, curve in synthetic.items():
                f = interpolate.interp1d(self.fit_xs, curve, kind='cubic')
                new_synthetic[fltr] = f(self.xs)
            synthetic = new_synthetic

        lhood = self.lhood(synthetic)

        self.eval_counter += 1
        logger.debug(f'eval counter = {self.eval_counter}, likelihood = {lhood}')
        return lhood

    def fit(self, xs, ys, period, x0, discretization, nwalkers=None, nsteps=1000,
            p0=None, yerr=None, burn_in=None, quantiles=None, discard=False, interp_treshold=None, progress=False):
        """
        Fit method using Markov Chain Monte Carlo.
        Once simulation is done, following valeus are stored and can be used for further evaluation::

            self.last_sampler: emcee.EnsembleSampler
            self.last_normalization: Dict; normalization map used during fitting
            self.last_fname: str; filename of last stored flatten emcee `sampler` with metadata

        Based on https://emcee.readthedocs.io/en/stable/.

        :param progress: bool; visualize progress of the sampling
        :param xs: Dict[str, Iterable[float]]; {<passband>: <phases>}
        :param ys: Dict[str, Iterable[float]]; {<passband>: <fluxes>};
        :param period: float; system period
        :param x0: List[Dict]; initial state (metadata included)
        :param discretization: float; discretization of objects
        :param nwalkers: int; number of walkers
        :param nsteps: int; number of steps in mcmc eval
        :param p0: numpy.array; inital priors for mcmc
        :param yerr: Union[numpy.array, float]; errors for each point of observation
        :param burn_in: int; number of steps for mcmc to explore parameters
        :param quantiles: List[int];
        :param discard: Union[int, bool]; how many values of result discard when looking for solution
        :param interp_treshold: int; data binning treshold
        :return: emcee.EnsembleSampler; sampler instance
        """
        burn_in = int(nsteps / 10) if burn_in is None else burn_in

        self.passband = list(ys.keys())
        yerrs = {c: butils.lightcurves_mean_error(ys[c]) if yerr[c] is None else yerr[c] for c in xs.keys()}
        x0 = params.lc_initial_x0_validity_check(x0, self.morphology)
        x0_vector, labels, fixed, constraint, observer = params.fit_data_initializer(x0, passband=self.passband)
        ndim = len(x0_vector)
        nwalkers = 2 * len(labels) if nwalkers is None else nwalkers
        params.mcmc_nwalkers_vs_ndim_validity_check(nwalkers, ndim)

        self.xs, self.xs_reverser = params.xs_reducer(xs)
        self.labels, self.observer, self.period = labels, observer, period
        self.fixed, self.constraint = fixed, constraint
        self.ys, self.yerrs = ys, yerrs
        self.period = period
        self.discretization = discretization

        self.xs, x0 = models.time_layer_resolver(self.xs, x0)
        interp_treshold = config.MAX_CURVE_DATA_POINTS if interp_treshold is None else interp_treshold
        diff = 1.0 / interp_treshold
        self.fit_xs = np.linspace(np.min(self.xs) - diff, np.max(self.xs) + diff, num=interp_treshold + 2) \
            if np.shape(self.xs)[0] > interp_treshold else self.xs

        sampler = self._fit(x0_vector, self.labels, nwalkers, ndim, nsteps, burn_in, p0, progress=progress)

        # extracting fit results from MCMC sampler
        flat_chain = sampler.get_chain(flat=True)
        result_dict = McMcMixin.resolve_mcmc_result(flat_chain=flat_chain, labels=self.labels)
        result_dict.update({param: {'value': value} for param, value in self.fixed.items()})

        result_dict = self.eval_constraints_after_mcmc(result_dict, self.constraint)

        r_squared_args = self.xs, self.ys, self.period, self.passband, discretization, self.morphology, self.xs_reverser
        r_dict = {key: value['value'] for key, value in result_dict.items()}
        r_squared_result = lc_r_squared(models.synthetic_binary, *r_squared_args, **r_dict)
        result_dict["r_squared"] = {'value': r_squared_result}

        return params.extend_result_with_units(result_dict)


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
            synthetic = butils.normalize_rv_curve_to_max(synthetic)

        lhood = self.lhood(synthetic)

        self.eval_counter += 1
        logger.debug(f'eval counter = {self.eval_counter}, likehood = {lhood}')
        return lhood

    def fit(self, xs, ys, x0, nwalkers=None, nsteps=1000, p0=None, yerr=None, burn_in=None, progress=False):
        """
        Fit method using Markov Chain Monte Carlo.
        Once simulation is done, following valeus are stored and can be used for further evaluation::
        sampler = self._fit(x0_vector, self.labels, nwalkers, ndim, nsteps, nsteps_burn_in, p0)

        # extracting fit results from MCMC sampler::

            flat_chain = sampler.get_chain(flat=True)
            result_dict = McMcMixin.resolve_mcmc_result(flat_chain=flat_chain, labels=self.labels)
            result_dict.update({param: {value: value} for param, value in self.fixed.items()})
            result_dict.update(params.constraints_evaluator(result_dict, self.constraint))

            r_squared_args = self.xs, self.ys, False, self.xs_reverser
            r_dict = {key: value['value'] for key, value in result_dict.items()}
            r_squared_result = rv_r_squared(models.central_rv_synthetic, *r_squared_args, **r_dict)
            result_dict["r_squared"] = {value: r_squared_result}

            return params.extend_result_with_units(result_dict)
                self.last_sampler: emcee.EnsembleSampler
                self.last_normalization: Dict; normalization map used during fitting
                self.last_fname: str; filename of last stored flatten emcee `sampler` with metadata

        Based on https://emcee.readthedocs.io/en/stable/.

        :param progress: bool; visualize progress of the sampling
        :param xs: Iterable[float];
        :param ys: Dict;
        :param x0: List[Dict]; initial state (metadata included)
        :param nwalkers: int; number of walkers
        :param nsteps: int; number of steps in mcmc eval
        :param p0: numpy.array; inital priors for mcmc
        :param yerr: Union[numpy.array, float]; errors for each point of observation
        :param burn_in: int; numer of steps for mcmc to explore parameters
        :return: emcee.EnsembleSampler; sampler instancea
        """
        burn_in = int(nsteps / 10) if burn_in is None else burn_in

        x0 = params.rv_initial_x0_validity_check(x0)
        yerrs = {c: butils.radialcurves_mean_error(ys[c]) if yerr[c] is None else yerr[c]
                 for c in xs.keys()}
        x0_vector, labels, fixed, constrained, observer = params.fit_data_initializer(x0)
        ndim = len(x0_vector)
        nwalkers = 2*len(x0_vector) if nwalkers is None else nwalkers

        params.mcmc_nwalkers_vs_ndim_validity_check(nwalkers, ndim)

        self.xs, self.xs_reverser = params.xs_reducer(xs)
        self.ys, self.yerrs = ys, yerrs
        self.labels, self.observer = labels, observer
        self.fixed, self.constraint = fixed, constrained

        sampler = self._fit(x0_vector, self.labels, nwalkers, ndim, nsteps, burn_in, p0, progress=progress)

        # extracting fit results from MCMC sampler
        flat_chain = sampler.get_chain(flat=True)
        result_dict = McMcMixin.resolve_mcmc_result(flat_chain=flat_chain, labels=self.labels)
        result_dict.update({param: {'value': value} for param, value in self.fixed.items()})

        result_dict = self.eval_constraints_after_mcmc(result_dict, self.constraint)

        r_squared_args = self.xs, self.ys, False, self.xs_reverser
        r_dict = {key: value['value'] for key, value in result_dict.items()}
        r_squared_result = rv_r_squared(models.central_rv_synthetic, *r_squared_args, **r_dict)
        result_dict["r_squared"] = {'value': r_squared_result}

        return params.extend_result_with_units(result_dict)


binary_detached = DetachedLightCurveFit()
binary_overcontact = OvercontactLightCurveFit()
central_rv = CentralRadialVelocity()
