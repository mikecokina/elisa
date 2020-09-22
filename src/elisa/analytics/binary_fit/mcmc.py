from abc import ABCMeta, abstractmethod
from multiprocessing import Pool
from typing import Dict

import emcee
import numpy as np
from scipy import interpolate

from . shared import check_for_boundary_surface_potentials
from . mixins import MCMCMixin
from . shared import (
    lc_r_squared, rv_r_squared,
    AbstractLCFit, AbstractRVFit, AbstractFit
)

from .. import RVData, LCData
from .. models import lc as lc_model
from .. models import rv as rv_model
from .. params import parameters
from .. tools.utils import time_layer_resolver

from ... import const
from ... observer.utils import normalize_light_curve
from ... base.error import ElisaError
from ... import settings
from ... graphic.mcmc_graphics import Plot
from ... logger import getPersistentLogger
from ... binary_system.system import BinarySystem
from ... binary_system.curves.community import RadialVelocitySystem

logger = getPersistentLogger('analytics.binary_fit.mcmc')


class MCMCFit(AbstractFit, MCMCMixin, metaclass=ABCMeta):
    def __init__(self):
        self.plot = Plot()
        self.last_sampler = emcee.EnsembleSampler
        self.last_normalization = dict()
        self.flat_chain_path = ''
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
        lh = - 0.5 * np.sum(
            [np.sum(
                np.power((self.y_data[item] - synthetic[item][self.x_data_reducer[item]]) / self.y_err[item], 2)
                + np.log(2 * const.PI * np.power(self.y_err[item], 2))
            )
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

    def _fit(self, nwalkers, ndim, nsteps, nsteps_burn_in, p0=None, progress=False, save=False, fit_id=None):
        vector = parameters.vector_normalizer(self.initial_vector, self.fitable.keys(), self.normalization)
        p0 = self.generate_initial_states(p0, nwalkers, ndim, x0_vector=vector)
        lnf = self.ln_probability

        logger.info('starting mcmc')
        if settings.NUMBER_OF_MCMC_PROCESSES > 1:
            with Pool(processes=settings.NUMBER_OF_MCMC_PROCESSES) as pool:
                logger.info('starting parallel mcmc')
                sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=lnf, pool=pool)
                self.worker(sampler, p0, nsteps, nsteps_burn_in, progress=progress)
        else:
            logger.info('starting singlecore mcmc')
            sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=lnf)
            self.worker(sampler, p0, nsteps, nsteps_burn_in, progress=progress)

        self.last_sampler = sampler
        self.last_normalization = self.normalization

        if save:
            self.flat_chain_path = self.save_flat_chain(sampler.get_chain(flat=True), self.fitable,
                                                        self.normalization, fit_id=fit_id)

        return sampler

    @staticmethod
    def generate_initial_states(initial_state, nwalkers, ndim, x0_vector=None):
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
                raise ValueError(f'Your initial values for sampler do not satisfy required shape ({nwalkers}, {ndim}). '
                                 f'Shape of your initial state matrix is {initial_state.shape}')
            initial_state[initial_state < 0] = 0.0
            initial_state[initial_state > 1] = 1.0
            return initial_state


class LightCurveFit(MCMCFit, AbstractLCFit):
    MORPHOLOGY = None

    def likelihood(self, xn):
        """
        Liklehood function which defines goodnes of current `xn` vector to be fit of given model.
        Best is 0.0, worst is -inf.

        :param xn: Iterable[float]; vector of parameters we are looking for
        :return: float;
        """
        diff = 1.0 / self.interp_treshold
        xn = parameters.vector_renormalizer(xn, self.fitable.keys(), self.normalization)
        kwargs = parameters.prepare_properties_set(xn, self.fitable.keys(), self.constrained, self.fixed)
        phases, kwargs = time_layer_resolver(self.x_data_reduced, pop=False, **kwargs)

        fit_xs = np.linspace(np.min(phases) - diff, np.max(phases) + diff, num=self.interp_treshold + 2) \
            if np.shape(phases)[0] > self.interp_treshold else phases
        args = fit_xs, self.discretization, self.observer
        synthetic = lc_model.synthetic_binary(*args, **kwargs)
        synthetic, _ = normalize_light_curve(synthetic, kind='average')

        if np.shape(fit_xs) != np.shape(phases):
            synthetic = {
                band: interpolate.interp1d(fit_xs, curve, kind='cubic')(phases)
                for band, curve in synthetic.items()
            }
        return self.lhood(synthetic)

    def fit(self, data: Dict[str, LCData], x0: parameters.BinaryInitialParameters, discretization=5.0, nwalkers=None,
            nsteps=1000, initial_state=None, burn_in=None, percentiles=None, interp_treshold=None, progress=False,
            save=True, fit_id=None):
        """
        Fit method using Markov Chain Monte Carlo.
        Once simulation is done, following valeus are stored and can be used for further evaluation::

            self.last_sampler: emcee.EnsembleSampler
            self.last_normalization: Dict; normalization map used during fitting
            self.flat_chain_path: str; filename of last stored flatten emcee `sampler` with metadata

        Based on https://emcee.readthedocs.io/en/stable/.

        :param data: elisa.analytics.dataset.base.LCData;
        :param x0: List[Dict]; initial state (metadata included)
        :param discretization: float; discretization of objects
        :param interp_treshold: int; data binning treshold
        :param nwalkers: int; The number of walkers in the ensemble. Minimum is 2 * number of free parameters.
        :param nsteps: int; The number of steps to run.
        :param initial_state: numpy.array; initial priors for mcmc
        :param burn_in: int; number of steps for mcmc to explore parameters
        :param progress: bool; visualize progress of the sampling
        :param percentiles: List; [percentile for left side error estimation, percentile of the centre,
                                   percentile for right side error estimation]
        :param save: bool; wheterher stor chain or not
        :param fit_id: str; id which identifies fit file (if not specified, current dateime is used)
        :return: emcee.EnsembleSampler; sampler instance
        """
        burn_in = int(nsteps / 10) if burn_in is None else burn_in
        self.set_up(x0, data, passband=data.keys(), discretization=discretization, morphology=self.MORPHOLOGY,
                    interp_treshold=settings.MAX_CURVE_DATA_POINTS if interp_treshold is None else interp_treshold,
                    observer_system_cls=BinarySystem)

        ndim = len(self.initial_vector)
        nwalkers = 2 * len(self.initial_vector) if nwalkers is None else nwalkers
        self.mcmc_nwalkers_vs_ndim_validity_check(nwalkers, ndim)

        sampler = self._fit(nwalkers, ndim, nsteps, burn_in, initial_state, progress, save, fit_id)

        # extracting fit results from MCMC sampler
        flat_chain = sampler.get_chain(flat=True)
        result_dict = MCMCMixin.resolve_mcmc_result(flat_chain, self.fitable, self.normalization, percentiles)

        result_dict.update({lbl: {"value": val.value, 'fixed': True, 'unit': val.to_dict()['unit']}
                            for lbl, val in self.fixed.items()})
        result_dict = self.eval_constrained_results(result_dict, self.constrained)

        r_squared_args = (self.x_data_reduced, self.y_data, self.observer.passband, discretization,
                          self.x_data_reducer, 1.0 / self.interp_treshold, self.interp_treshold,
                          self.observer.system_cls)
        r_dict = {key: value['value'] for key, value in result_dict.items()}
        r_squared_result = lc_r_squared(lc_model.synthetic_binary, *r_squared_args, **r_dict)
        result_dict["r_squared"] = {'value': r_squared_result, "unit": None}

        result_dict = check_for_boundary_surface_potentials(result_dict)

        setattr(self, 'flat_result', result_dict)
        return parameters.serialize_result(result_dict)


class OvercontactLightCurveFit(LightCurveFit):
    """
    MCMC fitting implementation for light curves of over-contact binaries.
    It keeps eye on values of potentials - keep it same for primary and secondary component.
    """
    MORPHOLOGY = 'over-contact'


class DetachedLightCurveFit(LightCurveFit):
    """
    MCMC fitting implementation for light curves of detached binaries.
    """
    MORPHOLOGY = 'detached'


class CentralRadialVelocity(MCMCFit, AbstractRVFit):
    def likelihood(self, xn):
        """
        Liklehood function which defines goodnes of current `xn` vector to be fit of given model.
        Best is 0.0, worst is -inf.

        :param xn: Iterable[float]; vector of parameters we are looking for
        :return: float;
        """
        xn = parameters.vector_renormalizer(xn, self.fitable.keys(), self.normalization)
        kwargs = parameters.prepare_properties_set(xn, self.fitable.keys(), self.constrained, self.fixed)
        synthetic = rv_model.central_rv_synthetic(*(self.x_data_reduced, self.observer), **kwargs)
        lhood = self.lhood(synthetic)

        self.eval_counter += 1
        logger.debug(f'eval counter = {self.eval_counter}, likehood = {lhood}')
        return lhood

    def fit(self, data: Dict[str, RVData], x0: parameters.BinaryInitialParameters, nwalkers=None, nsteps=1000,
            initial_state=None, burn_in=None, percentiles=None, progress=False, save=True, fit_id=None):
        """
        Fit method using Markov Chain Monte Carlo.
        Once simulation is done, following valeus are stored and can be used for further evaluation::
        sampler = self._fit(x0_vector, self.labels, nwalkers, ndim, nsteps, nsteps_burn_in, p0)

        # extracting fit results from MCMC sampler::

            Based on https://emcee.readthedocs.io/en/stable/.

        :param save: bool; whether to store chain or not
        :param fit_id: str; id which identifies fit file (if not specified, current dateime is used)
        :param data: elisa.analytics.dataset.base.RVData;
        :param x0: elisa.analytics.params.parameters.BinaryInitialParameters;
        :param nwalkers: int; The number of walkers in the ensemble. Minimum is 2 * number of free parameters.
        :param nsteps: int; The number of steps to run.
        :param initial_state: numpy.array; initial priors for mcmc
        :param burn_in: int; numer of steps for mcmc to explore parameters
        :param percentiles: List[int]; [percentile for left side error estimation, percentile of the centre,
                                       percentile for right side error estimation]
        :param progress: bool; visualize progress of the sampling
        :return: Dict; fit results
        """
        burn_in = int(nsteps / 10) if burn_in is None else burn_in
        self.set_up(x0, data, observer_system_cls=RadialVelocitySystem)

        ndim = len(self.initial_vector)
        nwalkers = 2 * len(self.initial_vector) if nwalkers is None else nwalkers
        self.mcmc_nwalkers_vs_ndim_validity_check(nwalkers, ndim)

        sampler = self._fit(nwalkers, ndim, nsteps, burn_in, initial_state, progress, save, fit_id)

        # extracting fit results from MCMC sampler
        flat_chain = sampler.get_chain(flat=True)
        result_dict = MCMCMixin.resolve_mcmc_result(flat_chain, self.fitable, self.normalization, percentiles)

        result_dict.update({lbl: {
            'value': val.value,
            'fixed': True,
            'unit': val.to_dict()['unit']
        } for lbl, val in self.fixed.items()})
        result_dict = self.eval_constrained_results(result_dict, self.constrained)

        r_squared_args = self.x_data_reduced, self.y_data, self.x_data_reducer, self.observer.system_cls
        r_dict = {key: value['value'] for key, value in result_dict.items()}

        r_squared_result = rv_r_squared(rv_model.central_rv_synthetic, *r_squared_args, **r_dict)
        result_dict["r_squared"] = {'value': r_squared_result, "unit": None}

        setattr(self, 'flat_result', result_dict)
        return parameters.serialize_result(result_dict)
