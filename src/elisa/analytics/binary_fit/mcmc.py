from abc import ABCMeta, abstractmethod
from multiprocessing import Pool
from typing import Dict

import emcee
import numpy as np
from scipy import interpolate
from scipy.stats.distributions import norm

from . shared import check_for_boundary_surface_potentials
from . mixins import MCMCMixin
from . shared import (
    lc_r_squared, rv_r_squared,
    AbstractLCFit, AbstractRVFit, AbstractFit
)

from .. import RVData, LCData
from .. models import lc as lc_model
from .. models import rv as rv_model
from .. models import cost_fns
from .. params import parameters
from ..params.conf import NUISANCE_PARSER, PARAM_PARSER
from .. tools.utils import time_layer_resolver

from ... observer.utils import normalize_light_curve
from ... base.error import ElisaError
from ... import settings
from ... graphic.mcmc_graphics import Plot
from ... logger import getPersistentLogger
from ... binary_system.system import BinarySystem
from ... binary_system.curves.community import RadialVelocitySystem

logger = getPersistentLogger('analytics.binary_fit.mcmc')


class MCMCFit(AbstractFit, MCMCMixin, metaclass=ABCMeta):
    """
    General framework for MCMC sampling of binary systems.
    """
    def __init__(self):
        self.plot = Plot()
        self.last_sampler = emcee.EnsembleSampler
        self.last_normalization = dict()
        self.flat_chain_path = ''
        self.eval_counter = 0
        self._last_known_lhood = -np.finfo(float).max * np.finfo(float).eps
        self.sigmas = list()

    @staticmethod
    def ln_prior(xn, x0, sigmas):
        """
        Logarithmic value of prior (uniform, normal or combined).

        :param xn: numpy.array; current state of the sampler (normalized values of variable parameters)
        :param x0: numpy.array; mean (expected) values of normalized parameters with normal prior distribution
        :param sigmas: numpy.array; normalized standard deviations of normal prior distributions
        :return: numpy.array; sum of logarithms of prior distribution functions
        """
        retval = np.empty(sigmas.shape)

        nan_mask = np.isnan(sigmas)
        uni_prior = np.all(np.bitwise_and(np.greater_equal(xn[nan_mask], 0.0),
                                          np.less_equal(xn[nan_mask], 1.0))).astype(float)
        retval[nan_mask] = -np.inf if uni_prior == 0 else np.log(uni_prior)

        retval[~nan_mask] = np.log(norm().pdf(((xn[~nan_mask]-x0[~nan_mask])/sigmas[~nan_mask]))) \
            if 0.0 <= xn[~nan_mask].all() <= 1.0 else -np.inf
        return np.sum(retval)

    @abstractmethod
    def likelihood(self, xn):
        """
        Likelihood function depending on the type of the optimization.

        :param xn: numpy.array; current state of the sampler (normalized values of variable parameters)
        :return: float;
        """
        pass

    def likelihood_fn(self, synthetic, ln_f):
        """
        Calculates value of likelihood function for observational data being drawn from distribution around synthetic
        model.

        :param ln_f: float; marginalization parameters (currently supported single parameter for error penalization)
        :param synthetic: Dict; {'dataset_name': numpy.array, }
        :return: float; likelihood value
        """
        lh = cost_fns.likelihood_fn(self.y_data, self.y_err, synthetic, ln_f)
        self._last_known_lhood = lh if lh < self._last_known_lhood else self._last_known_lhood
        return lh

    def ln_probability(self, xn):
        """
        Resulting probability distribution made of likelihood and prior distribution.

        :param xn: numpy.array; current state of the sampler (normalized values of variable parameters)
        :return: float; likelihood
        """
        prior = self.ln_prior(xn, self.norm_init_vector, self.sigmas)
        if prior == -np.inf:
            return -np.inf
        try:
            likelihood = prior + self.likelihood(xn)
        except (ElisaError, ValueError) as e:
            if not settings.SUPPRESS_WARNINGS:
                logger.warning(f'mcmc hit invalid parameters, exception: {str(e)}')
            return self._last_known_lhood * 1e3
        return likelihood

    def normalized_sigma(self, vector):
        """
        Assigns normalized standard deviation for each variable parameter to attribute `sigma`. If sigma is not supplied for the
        parameter, np.nan is used instead.

        :param vector: List; normalized starting vector
        :return: None
        """
        sigmas = np.array([val.sigma if val.sigma is not None else np.nan for val in self.fitable.values()])
        perturbed = np.array(self.initial_vector) + sigmas
        perturbed_norm = parameters.vector_normalizer(perturbed, self.fitable.keys(), self.normalization)
        self.sigmas = np.array(perturbed_norm) - vector

    def _fit(self, nwalkers, ndim, nsteps, nsteps_burn_in, p0=None, progress=False, save=False, fit_id=None):
        """
        General MCMC sampling function for an inverse problem. Implementing sampler from the emcee package.

        :param nwalkers: int; The number of walkers in the ensemble. Minimum is 2 * number of free parameters.
        :param ndim: int; number of free variables
        :param nsteps: int; The number of steps to run.
        :param nsteps_burn_in: int; number of steps for mcmc to explore parameters
        :param p0: numpy.array; initial priors for mcmc
        :param progress: bool; display the progress bar of the sampling
        :param save: bool; if true, the MCMC flat chain will be stored
        :param fit_id: str; id or location (ending with .json) which identifies fit file (if not specified, current
                            datetime is used)
        :return: emcee.EnsembleSampler;
        """
        self.norm_init_vector = np.array(parameters.vector_normalizer(self.initial_vector, self.fitable.keys(),
                                                                      self.normalization))

        sigmas = np.array([val.sigma if val.sigma is not None else np.nan for val in self.fitable.values()])
        args = (np.array(self.initial_vector) + sigmas, self.fitable.keys(), self.normalization)
        perturbed_norm = parameters.vector_normalizer(*args)
        self.sigmas = np.array(perturbed_norm) - self.norm_init_vector

        p0 = self.generate_initial_states(p0, nwalkers, ndim, x0_vector=self.norm_init_vector)

        logger.info('starting mcmc')
        kwargs = dict(nwalkers=nwalkers, ndim=ndim, log_prob_fn=self.ln_probability)
        if settings.NUMBER_OF_MCMC_PROCESSES > 1:
            with Pool(processes=settings.NUMBER_OF_MCMC_PROCESSES) as pool:
                logger.info('starting parallel mcmc')
                sampler = emcee.EnsembleSampler(pool=pool, **kwargs)
                self.worker(sampler, p0, nsteps, nsteps_burn_in, save=save, fit_id=fit_id, fitable=self.fitable,
                            normalization=self.normalization, progress=progress)
        else:
            logger.info('starting singlecore mcmc')
            sampler = emcee.EnsembleSampler(**kwargs)
            self.worker(sampler, p0, nsteps, nsteps_burn_in, save=save, fit_id=fit_id, fitable=self.fitable,
                        normalization=self.normalization, progress=progress)

        self.last_sampler = sampler
        self.last_normalization = self.normalization

        if save:
            self.flat_chain_path = self.save_flat_chain(sampler.get_chain(flat=True), self.fitable,
                                                        self.normalization, fit_id=fit_id)

        return sampler

    @staticmethod
    def generate_initial_states(initial_state, nwalkers, ndim, x0_vector=None):
        """
        Function transforms user initial state to normalized format suitable for our MCMC chain, where all vales are in
        interval (0, 1).

        :param initial_state: numpy.ndarray; initial state matrix before normalization
        :param nwalkers: int; The number of walkers in the ensemble. Minimum is 2 * number of free parameters.
        :param ndim: int; number of free variables
        :param x0_vector: np.array; normalized vector of free parameters
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
    """
    LC fit class implementing an MCMC method.
    """
    MORPHOLOGY = None

    def likelihood(self, xn):
        """
        Likelihood function for given set of model variables.
        Best fit is 0.0, worst is -inf.

        :param xn: Iterable[float]; vector of optimized free parameters
        :return: float; likelihood
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
                band: interpolate.interp1d(fit_xs, curve, kind='cubic')(phases[self.x_data_reducer[band]])
                for band, curve in synthetic.items()
            }

        ln_f_key = f"{NUISANCE_PARSER}{PARAM_PARSER}ln_f"
        ln_f = parameters.prepare_nuisance_properties_set(xn, self.fitable, self.fixed)[ln_f_key]

        return self.likelihood_fn(synthetic, ln_f)

    def fit(self, data: Dict[str, LCData], x0: parameters.BinaryInitialParameters, discretization=5.0, nwalkers=None,
            nsteps=1000, initial_state=None, burn_in=None, percentiles=None, interp_treshold=None, progress=False,
            save=True, fit_id=None, samples="uniform"):
        """
        Fit method using Markov Chain Monte Carlo.
        Once simulation is done, following values are stored and can be used for further evaluation::

            self.last_sampler: emcee.EnsembleSampler
            self.last_normalization: Dict; normalization map used during fitting
            self.flat_chain_path: str; filename of last stored flatten emcee `sampler` with metadata

        Based on https://emcee.readthedocs.io/en/stable/.

        :param data: elisa.analytics.dataset.base.LCData;
        :param x0: List[Dict]; initial state (metadata included)
        :param discretization: float; discretization of objects
        :param interp_treshold: int; Above this total number of datapoints, light curve will be interpolated
                                     using model containing `interp_treshold` equidistant points per epoch
        :param nwalkers: int; The number of walkers in the ensemble. Minimum is 2 * number of free parameters.
        :param nsteps: int; The number of steps to run.
        :param initial_state: numpy.array; initial priors for MCMC
        :param burn_in: int; number of steps for MCMC to explore parameters
        :param progress: bool; display the progress bar of the sampling
        :param percentiles: List; [percentile for left side error estimation, percentile of the centre,
                                   percentile for right side error estimation]
        :param save: bool; whether to store the chain or not
        :param fit_id: str; id which identifies fit file (if not specified, current dateime is used)
        :param samples: Union[str, List]; `uniform`, `adaptive` or list with phases in (0, 1) interval
        :return: Dict; optimized model parameters in flattened form
        """
        burn_in = int(nsteps / 10) if burn_in is None else burn_in
        self.set_up(x0, data, passband=data.keys(), discretization=discretization, morphology=self.MORPHOLOGY,
                    interp_treshold=settings.MAX_CURVE_DATA_POINTS if interp_treshold is None else interp_treshold,
                    observer_system_cls=BinarySystem, samples=samples)

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

        result_dict = check_for_boundary_surface_potentials(result_dict, LightCurveFit.MORPHOLOGY)

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
    """
    RV fit class implementing an MCMC method using kinematic method.
    """
    def likelihood(self, xn):
        """
        Likelihood function for given set of model parameters.
        Best is 0.0, worst is -inf.

        :param xn: Iterable[float]; vector of parameters we are looking for
        :return: float;
        """
        xn = parameters.vector_renormalizer(xn, self.fitable.keys(), self.normalization)
        kwargs = parameters.prepare_properties_set(xn, self.fitable.keys(), self.constrained, self.fixed)
        synthetic = rv_model.central_rv_synthetic(*(self.x_data_reduced, self.observer), **kwargs)
        synthetic = {comp: rv[self.x_data_reducer[comp]] for comp, rv in synthetic.items()}

        ln_f_key = f"{NUISANCE_PARSER}{PARAM_PARSER}ln_f"
        ln_f = parameters.prepare_nuisance_properties_set(xn, self.fitable, self.fixed)[ln_f_key]
        lhood = self.likelihood_fn(synthetic, ln_f)

        self.eval_counter += 1
        logger.debug(f'eval counter = {self.eval_counter}, likehood = {lhood}')
        return lhood

    def fit(self, data: Dict[str, RVData], x0: parameters.BinaryInitialParameters, nwalkers=None, nsteps=1000,
            initial_state=None, burn_in=None, percentiles=None, progress=False, save=True, fit_id=None):
        """
        Fit method using Markov Chain Monte Carlo.
        Once simulation is done, following values are stored and can be used for further evaluation::
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
