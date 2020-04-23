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
        percentiles = [16, 50, 84] if percentiles is None else percentiles
        result = dict()
        for idx, key in enumerate(labels):
            mcmc = np.percentile(flat_chain[:, idx], percentiles)
            vals = params.param_renormalizer(mcmc, np.repeat(key, len(mcmc)))
            result[key] = {"value": vals[1], "min": min(vals), "max": max(vals), "fixed": False}

        return params.extend_result_with_units(result)

    @staticmethod
    def _store_flat_chain(flat_chain: np.array, labels: Iterable[str], norm: Dict, filename=None):
        """
        Store state of mcmc run.

        :param flat_chain: numpy.array; flatted array of parameters values in each mcmc step::

            [[param0_0, param1_0, ..., paramk_0],
             [param0_1, param1_1, ..., paramk_1]
             ...
             [param0_b, param1_n, ..., paramk_n]]

        :param labels: Union[List, numpy.array]; labels of parameters in order of params in `flat_chain`
        """
        if filename is None:
            now = datetime.now()
            fdir = now.strftime(config.DATE_MASK)
            fname = f'{now.strftime(config.DATETIME_MASK)}.json'
            fpath = op.join(config.HOME, fdir, fname)
            os.makedirs(op.join(config.HOME, fdir), exist_ok=True)
        else:
            fpath = filename if filename.endswith('.json') else filename  + '.json'
        data = {
            "flat_chain": flat_chain.tolist() if isinstance(flat_chain, np.ndarray) else flat_chain,
            "labels": labels,
            "normalization": norm
        }
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=4))
        logger.info(f'MCMC chain, variable`s labels and normalization constants were stored in: {fpath}')
        return fpath[:-5]

    @staticmethod
    def restore_flat_chain(fname):
        """
        Restore stored state from mcmc run.

        :param fname: str; base filename of stored state
        :return: Dict;
        """
        fdir = fname[:len(config.DATE_MASK) + 2] if '/' not in fname else fname
        fname = f'{fname}.json'
        fpath = op.join(config.HOME, fdir, fname) if '/' not in fname else fname
        with open(fpath, "r") as f:
            return json.loads(f.read())

    @staticmethod
    def worker(sampler, p0, nsteps, nsteps_burn_in, progress=False):
        logger.info("running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, nsteps_burn_in, progress=progress) if nsteps_burn_in > 0 else p0, None, None
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
        self._last_known_lhood = -np.finfo(float).max * np.finfo(float).eps

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
        lh = - 0.5 * np.sum([np.sum(
            np.power((self.ys[item] - synthetic[item][self.xs_reverser[item]]) / self.yerrs[item], 2))
            for item, value in synthetic.items()])
        self._last_known_lhood = lh if lh < self._last_known_lhood else self._last_known_lhood
        return lh

    def ln_probability(self, xn):
        if not self.ln_prior(xn):
            return -np.inf
        try:
            likelihood = self.likelihood(xn)
        except (ElisaError, ValueError) as e:
            logger.warning(f'mcmc hit invalid parameters, exception: {str(e)}')
            return self._last_known_lhood * 1e3
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
        result_dict.update({key: {'value': val, 'constraint': constraints[key]}
                            for key, val in constrained_values.items()})
        return result_dict

    def _fit(self, x0, labels, nwalkers, ndim, nsteps, nsteps_burn_in, p0=None, progress=False):
        p0 = self.generate_initial_states(p0, nwalkers, ndim, x0_vector=x0)

        lnf = self.ln_probability
        logger.info('starting mcmc')
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

    def generate_initial_states(self, initial_state, nwalkers, ndim, x0_vector=None):
        """
        Function transforms user initial state to normalized format suitable for our mcmc chain, where all vales are in
        interval (0, 1).

        :param initial_state: numpy.ndarray; initial state matrix before normalization
        :param nwalkers: int;
        :param ndim: int;
        :param x0_vector: np.array; initial stat based on the firs value
        :return: initial_state: numpy.ndarray; initial state matrix after normalization
        """
        if initial_state is None:
            retval = np.random.uniform(0.0, 1.0, (nwalkers, ndim))
            retval[0] = x0_vector if x0_vector is not None else retval[0]
            return retval
        else:
            if initial_state.shape != (nwalkers, ndim):
                raise ValueError(f'your initial values for sampler do not have the correct shape ({nwalkers}, {ndim}). '
                                 f'Shape of your initial state matrix is {initial_state.shape}')
            initial_state[initial_state < 0] = 0.0
            initial_state[initial_state > 1] = 1.0
            return initial_state


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

        phases, kwargs = models.rvt_layer_resolver(self.xs, **kwargs)
        fit_xs = np.linspace(np.min(phases) - self.diff, np.max(phases) + self.diff, num=self.interp_treshold + 2) \
            if np.shape(phases)[0] > self.interp_treshold else phases

        args = fit_xs, self.discretization, self.morphology, self.observer, True
        synthetic = models.synthetic_binary(*args, **kwargs)
        synthetic = butils.normalize_light_curve(synthetic, kind='average')

        if np.shape(fit_xs) != np.shape(phases):
            new_synthetic = dict()
            for fltr, curve in synthetic.items():
                f = interpolate.interp1d(fit_xs, curve, kind='cubic')
                new_synthetic[fltr] = f(phases)
            synthetic = new_synthetic

        return self.lhood(synthetic)

    def fit(self, xs, ys, x0, discretization, nwalkers=None, nsteps=1000,
            initial_state=None, yerr=None, burn_in=None, percentiles=None, interp_treshold=None, progress=False):
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
        :param x0: List[Dict]; initial state (metadata included)
        :param discretization: float; discretization of objects
        :param interp_treshold: int; data binning treshold
        :param nwalkers: int; The number of walkers in the ensemble. Minimum is 2 * number of free parameters.
        :param nsteps: int; The number of steps to run.
        :param initial_state: numpy.array; initial priors for mcmc
        :param yerr: Union[numpy.array, float]; errors for each point of observation
        :param burn_in: int; number of steps for mcmc to explore parameters
        :param percentiles: List; [percentile for left side error estimation, percentile of the centre,
                                   percentile for right side error estimation]
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

        ys = butils.normalize_light_curve(ys, kind='average')
        self.xs, self.xs_reverser = params.xs_reducer(xs)
        self.labels, self.observer = labels, observer
        self.fixed, self.constraint = fixed, constraint
        self.ys, self.yerrs = ys, yerrs
        self.discretization = discretization

        # self.xs, x0 = models.time_layer_resolver(self.xs, x0)
        self.interp_treshold = config.MAX_CURVE_DATA_POINTS if interp_treshold is None else interp_treshold
        self.diff = 1.0 / self.interp_treshold

        sampler = self._fit(x0_vector, self.labels, nwalkers, ndim, nsteps, burn_in, initial_state, progress=progress)

        # extracting fit results from MCMC sampler
        flat_chain = sampler.get_chain(flat=True)
        result_dict = McMcMixin.resolve_mcmc_result(flat_chain=flat_chain, labels=self.labels, percentiles=percentiles)
        result_dict.update({lbl: {'value': val, 'fixed': True} for lbl, val in self.fixed.items()})

        result_dict = self.eval_constraints_after_mcmc(result_dict, self.constraint)

        r_squared_args = self.xs, self.ys, self.passband, discretization, self.morphology, \
                         self.xs_reverser, self.diff, self.interp_treshold
        r_dict = {key: value['value'] for key, value in result_dict.items()}
        r_squared_result = lc_r_squared(models.synthetic_binary, *r_squared_args, **r_dict)
        result_dict["r_squared"] = {'value': r_squared_result}

        result_dict = params.extend_result_with_units(result_dict)
        return params.dict_to_user_format(result_dict)


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

        return lhood

    def fit(self, xs, ys, x0, nwalkers=None, nsteps=1000, initial_state=None, yerr=None, burn_in=None, percentiles=None,
            progress=False):
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
        :param nwalkers: int; The number of walkers in the ensemble. Minimum is 2 * number of free parameters.
        :param nsteps: int; The number of steps to run.
        :param initial_state: numpy.array; initial priors for mcmc
        :param yerr: Union[numpy.array, float]; errors for each point of observation
        :param burn_in: int; numer of steps for mcmc to explore parameters
        :param percentiles: List[int]; [percentile for left side error estimation, percentile of the centre,
                                        percentile for right side error estimation]
        :return: dict; fit results
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

        sampler = self._fit(x0_vector, self.labels, nwalkers, ndim, nsteps, burn_in, initial_state, progress=progress)

        # extracting fit results from MCMC sampler
        flat_chain = sampler.get_chain(flat=True)
        result_dict = McMcMixin.resolve_mcmc_result(flat_chain=flat_chain, labels=self.labels, percentiles=percentiles)

        result_dict.update({lbl: {'value': val, 'fixed': True} for lbl, val in self.fixed.items()})

        result_dict = self.eval_constraints_after_mcmc(result_dict, self.constraint)

        r_squared_args = self.xs, self.ys, False, self.xs_reverser
        r_dict = {key: value['value'] for key, value in result_dict.items()}
        r_squared_result = rv_r_squared(models.central_rv_synthetic, *r_squared_args, **r_dict)
        result_dict[params.PARAM_PARSER.join(['system', 'r_squared'])] = {'value': r_squared_result}
        result_dict = params.extend_result_with_units(result_dict)
        return params.dict_to_user_format(result_dict)


binary_detached = DetachedLightCurveFit()
binary_overcontact = OvercontactLightCurveFit()
central_rv = CentralRadialVelocity()
