from __future__ import annotations

from abc import ABCMeta, abstractmethod
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any

import emcee
import numpy as np
from scipy import interpolate
from scipy.stats.distributions import norm

from elisa import settings
from elisa.analytics.models import cost_fns
from elisa.analytics.models import lc as lc_model
from elisa.analytics.models import rv as rv_model
from elisa.analytics.params import parameters
from elisa.analytics.params.conf import NUISANCE_PARSER, PARAM_PARSER
from elisa.analytics.tools.utils import time_layer_resolver
from elisa.base.error import ElisaError
from elisa.base.types import FLOAT
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from elisa.graphic.mcmc_graphics import Plot
from elisa.logger import getPersistentLogger
from elisa.observer.utils import normalize_light_curve

from . import mixins
from .shared import (
    AbstractFit,
    AbstractLCFit,
    AbstractRVFit,
    check_for_boundary_surface_potentials,
    lc_r_squared,
    rv_r_squared,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.analytics.dataset.base import LCData, RVData
    from elisa.types import Float

logger = getPersistentLogger("analytics.binary_fit.mcmc")


class MCMCFit(AbstractFit, mixins.MCMCMixin, metaclass=ABCMeta):
    """General framework for MCMC sampling of binary systems."""

    # Attributes from parent class (set by set_up method)
    y_err: dict[str, Any]
    y_data: dict[str, Any]
    x_data_reducer: dict[str, NDArray]
    x_data_reduced: NDArray[Float]
    fitable: dict[str, Any]
    constrained: dict[str, Any]
    fixed: dict[str, Any]
    initial_vector: list[Any]
    normalization: dict[str, tuple[Any, ...]]
    flat_result: dict[str, Any]
    observer: Any
    atmosphere_model: None
    limb_darkening_coefficients: None
    interp_treshold: int
    discretization: float
    norm_init_vector: NDArray[Float]

    def __init__(self) -> None:
        self.plot = Plot()
        self.last_sampler = emcee.EnsembleSampler
        self.last_normalization: dict[str, Any] = {}
        self.flat_chain_path = ""
        self.eval_counter = 0
        self._last_known_lhood = -np.finfo(float).max * np.finfo(float).eps
        self.sigmas: NDArray[Float] | None = None

    @staticmethod
    def ln_prior(
        xn: NDArray[Float],
        x0: NDArray[Float],
        sigmas: NDArray[Float],
    ) -> Float:
        """Logarithmic value of prior (uniform, normal or combined).

        Computes the logarithm of the prior probability distribution for the sampler state.
        For parameters with NaN sigma values, a uniform prior is used (0 inside [0, 1], -inf outside).
        For parameters with defined sigma values, a normal prior is used with mean x0 and standard deviation sigma.

        :param xn: Current state of the sampler (normalized values of variable parameters in range [0, 1]).
        :type xn: NDArray[Float]
        :param x0: Mean (expected) values of normalized parameters with normal prior distribution.
        :type x0: NDArray[Float]
        :param sigmas: Normalized standard deviations of normal prior distributions (NaN indicates uniform prior).
        :type sigmas: NDArray[Float]
        :return: Sum of logarithms of prior distribution functions.
        :rtype: Float
        """
        retval = np.empty(sigmas.shape)

        nan_mask = np.isnan(sigmas)
        uni_prior = np.all(
            np.bitwise_and(
                np.greater_equal(xn[nan_mask], 0.0),
                np.less_equal(xn[nan_mask], 1.0),
            ),
        ).astype(float)
        retval[nan_mask] = -np.inf if uni_prior == 0 else np.log(uni_prior)

        retval[~nan_mask] = (
            np.log(norm().pdf((xn[~nan_mask] - x0[~nan_mask]) / sigmas[~nan_mask]))
            if np.logical_and(xn[~nan_mask] >= 0, xn[~nan_mask] <= 1.0).all()
            else -np.inf
        )
        return np.sum(retval)

    @abstractmethod
    def likelihood(self, xn: NDArray[Float]) -> Float:
        """Likelihood function depending on the type of the optimization.

        Abstract method to be implemented by subclasses.

        :param xn: Current state of the sampler (normalized values of variable parameters in range [0, 1]).
        :type xn: NDArray[Float]
        :return: Likelihood value.
        :rtype: Float
        """
        ...

    def likelihood_fn(self, synthetic: dict[str, Any], ln_f: Float) -> Float:
        """Calculate the value of likelihood function for observational data.

        Calculates value of likelihood function for observational data being drawn from distribution around synthetic
        model.

        :param synthetic: Dictionary mapping dataset names to numpy arrays containing synthetic model predictions.
        :type synthetic: dict[str, Any]
        :param ln_f: Marginalization parameters (currently supported single parameter for error penalization).
        :type ln_f: Float
        :return: Likelihood value.
        :rtype: Float
        """
        lh = cost_fns.likelihood_fn(self.y_data, self.y_err, synthetic, ln_f)
        self._last_known_lhood = min(self._last_known_lhood, lh)
        return lh

    def ln_probability(self, xn: NDArray[Float]) -> Float:
        """Calculate resulting probability distribution made of likelihood and prior distribution.

        Combines the prior and likelihood to compute the log probability for the given
        parameter state.

        :param xn: Current state of the sampler (normalized values of variable parameters in range [0, 1]).
        :type xn: NDArray[Float]
        :return: Log probability (likelihood).
        :rtype: Float
        """
        prior = self.ln_prior(xn, self.norm_init_vector, self.sigmas)
        if prior == -np.inf:
            return -np.inf
        try:
            likelihood = prior + self.likelihood(xn)
        except (ElisaError, ValueError) as e:
            if not settings.SUPPRESS_WARNINGS:
                error_msg = f"mcmc hit invalid parameters, exception: {e!s}"
                logger.warning(error_msg)
            return self._last_known_lhood * 1e3
        return likelihood

    def normalized_sigma(self, vector: NDArray[Float]) -> NDArray[Float]:
        """Assign normalized standard deviation for each variable parameter to attribute sigma.

        If sigma is not supplied for the parameter, np.nan is used instead.

        :param vector: Normalized starting vector.
        :type vector: NDArray[Float]
        :return: Normalized sigma values for each parameter.
        :rtype: NDArray[Float]
        """
        sigmas = np.array(
            [val.sigma if val.sigma is not None else np.nan for val in self.fitable.values()],
            dtype=FLOAT,
        )
        perturbed = np.array(self.initial_vector, dtype=FLOAT) + sigmas
        perturbed_norm = parameters.vector_normalizer(
            perturbed,
            self.fitable.keys(),
            self.normalization,
        )
        return np.array(perturbed_norm, dtype=FLOAT) - vector

    def _fit(
        self,
        nwalkers: int,
        ndim: int,
        nsteps: int,
        nsteps_burn_in: int,
        p0: NDArray[Float] | None = None,
        *,
        progress: bool = False,
        save: bool = False,
        fit_id: str | None = None,
    ) -> emcee.EnsembleSampler:
        """General MCMC sampling function for an inverse problem.

        Implements sampler from the emcee package. Handles both parallel and single-core
        execution based on the settings.NUMBER_OF_MCMC_PROCESSES configuration.

        :param nwalkers: The number of walkers in the ensemble. Minimum is 2 * number
            of free parameters.
        :type nwalkers: int
        :param ndim: Number of free variables.
        :type ndim: int
        :param nsteps: The number of steps to run.
        :type nsteps: int
        :param nsteps_burn_in: Number of steps for MCMC to explore parameters.
        :type nsteps_burn_in: int
        :param p0: Initial priors for MCMC. If None, random initial states are generated.
        :type p0: NDArray[Float] | None
        :param progress: Display the progress bar of the sampling.
        :type progress: bool
        :param save: If True, the MCMC flat chain will be stored to file.
        :type save: bool
        :param fit_id: ID or location (ending with .json) which identifies fit file.
            If not specified, current datetime is used.
        :type fit_id: str | None
        :return: MCMC ensemble sampler with results.
        :rtype: emcee.EnsembleSampler
        """
        self.norm_init_vector = np.array(
            parameters.vector_normalizer(
                self.initial_vector,
                self.fitable.keys(),
                self.normalization,
            ),
            dtype=FLOAT,
        )

        self.sigmas = self.normalized_sigma(self.norm_init_vector)

        p0 = self.generate_initial_states(p0, nwalkers, ndim, x0_vector=self.norm_init_vector)

        logger.info("starting mcmc")
        kwargs: dict[str, Any] = {
            "nwalkers": nwalkers,
            "ndim": ndim,
            "log_prob_fn": self.ln_probability,
        }
        if settings.NUMBER_OF_MCMC_PROCESSES > 1:
            with Pool(processes=settings.NUMBER_OF_MCMC_PROCESSES) as pool:
                logger.info("starting parallel mcmc")
                sampler = emcee.EnsembleSampler(pool=pool, **kwargs)  # type: ignore[arg-type]
                self.worker(
                    sampler,
                    p0,
                    nsteps,
                    nsteps_burn_in,
                    save=save,
                    fit_id=fit_id,
                    fitable=self.fitable,
                    normalization=self.normalization,
                    progress=progress,
                )
        else:
            logger.info("starting singlecore mcmc")
            sampler = emcee.EnsembleSampler(**kwargs)
            self.worker(
                sampler,
                p0,
                nsteps,
                nsteps_burn_in,
                save=save,
                fit_id=fit_id,
                fitable=self.fitable,
                normalization=self.normalization,
                progress=progress,
            )

        self.last_sampler = sampler
        self.last_normalization = self.normalization

        if save:
            self.flat_chain_path = self.save_flat_chain(
                sampler.get_chain(flat=True),
                self.fitable,
                self.normalization,
                fit_id=fit_id,
            )

        return sampler

    @staticmethod
    def generate_initial_states(
        initial_state: NDArray[Float] | None,
        nwalkers: int,
        ndim: int,
        x0_vector: NDArray[Float] | None = None,
    ) -> NDArray[Float]:
        """Transform user initial state to normalized format suitable for MCMC chain.

        Function transforms user initial state to normalized format suitable for MCMC chain,
        where all values are in the interval (0, 1).

        :param initial_state: Initial state matrix before normalization. If None, random
            initial states in (0, 1) are generated with the first walker set to x0_vector.
        :type initial_state: NDArray[Float] | None
        :param nwalkers: The number of walkers in the ensemble. Minimum is 2 * number
            of free parameters.
        :type nwalkers: int
        :param ndim: Number of free variables.
        :type ndim: int
        :param x0_vector: Normalized vector of free parameters for the first walker.
        :type x0_vector: NDArray[Float] | None
        :return: Initial state matrix after normalization, shape (nwalkers, ndim).
        :rtype: NDArray[Float]
        """
        if initial_state is None:
            rng = np.random.default_rng()
            retval = rng.uniform(0.0, 1.0, (nwalkers, ndim))
            retval[0] = x0_vector if x0_vector is not None else retval[0]
            return retval
        if initial_state.shape != (nwalkers, ndim):
            error_msg = (
                f"Your initial values for sampler do not satisfy required shape ({nwalkers}, {ndim}). "
                f"Shape of your initial state matrix is {initial_state.shape}"
            )
            raise ValueError(error_msg)
        initial_state[initial_state < 0] = 0.0
        initial_state[initial_state > 1] = 1.0
        return initial_state


class LightCurveFit(MCMCFit, AbstractLCFit):
    """LC fit class implementing an MCMC method.

    This class provides light curve fitting using Markov Chain Monte Carlo sampling.
    It handles synthetic light curve generation, likelihood calculation, and parameter
    optimization for binary systems.
    """

    MORPHOLOGY: str | None = None

    def likelihood(self, xn: NDArray[Float]) -> Float:
        """Likelihood function for given set of model variables.

        Calculates the likelihood value for the current model parameters by comparing
        synthetic and observed light curves. The best fit is 0.0, worst is -inf.

        :param xn: Vector of optimized free parameters (normalized to [0, 1] range).
        :type xn: NDArray[Float]
        :return: Likelihood value.
        :rtype: Float
        """
        diff = 1.0 / self.interp_treshold
        xn_list = parameters.vector_renormalizer(xn, self.fitable.keys(), self.normalization)
        xn_renorm = np.asarray(xn_list, dtype=FLOAT)
        kwargs: dict[str, Any] = parameters.prepare_properties_set(
            xn_renorm,
            self.fitable.keys(),
            self.constrained,
            self.fixed,
        )
        phases, kwargs = time_layer_resolver(self.x_data_reduced, pop=False, **kwargs)

        fit_xs: NDArray[Float] = (
            np.linspace(
                np.min(phases) - diff,
                np.max(phases) + diff,
                num=self.interp_treshold + 2,
            )
            if np.shape(phases)[0] > self.interp_treshold
            else phases
        )
        args: tuple[Any, ...] = (fit_xs, self.discretization, self.observer)
        synthetic: dict[str, NDArray[Float]] = lc_model.synthetic_binary(*args, **kwargs)
        synthetic, _ = normalize_light_curve(synthetic, kind="average")

        if np.shape(fit_xs) != np.shape(phases):
            synthetic = {
                band: interpolate.interp1d(fit_xs, curve, kind="cubic")(
                    phases[self.x_data_reducer[band]],
                )
                for band, curve in synthetic.items()
            }

        ln_f_key = f"{NUISANCE_PARSER}{PARAM_PARSER}ln_f"
        ln_f: Float = parameters.prepare_nuisance_properties_set(xn_renorm, self.fitable, self.fixed)[ln_f_key]

        return self.likelihood_fn(synthetic, ln_f)

    def fit(
        self,
        data: dict[str, LCData],
        x0: parameters.BinaryInitialParameters,
        *,
        discretization: Float = 5.0,
        nwalkers: int | None = None,
        nsteps: int = 1000,
        initial_state: NDArray[Float] | None = None,
        burn_in: int | None = None,
        percentiles: list | None = None,
        interp_treshold: int | None = None,
        samples: str | list = "uniform",
        progress: bool = False,
        save: bool = True,
        fit_id: str | None = None,
    ) -> dict[str, Any]:
        """Fit light curve using Markov Chain Monte Carlo sampling.

        Uses emcee's EnsembleSampler for parallel MCMC sampling. Once simulation is done,
        the following values are stored and can be used for further evaluation:

        .. code-block:: python

            self.last_sampler: emcee.EnsembleSampler
            self.last_normalization: dict; normalization map used during fitting
            self.flat_chain_path: str; filename of last stored flattened emcee sampler with metadata

        Based on https://emcee.readthedocs.io/en/stable/.

        :param data: Light curve dataset with observational data.
        :type data: dict[str, LCData]
        :param x0: Initial state of binary system parameters (metadata included).
        :type x0: parameters.BinaryInitialParameters
        :param discretization: Discretization of objects for surface generation.
            Default is 5.0.
        :type discretization: Float
        :param nwalkers: The number of walkers in the ensemble. If None, defaults to
            2 * number of free parameters. Minimum is 2 * number of free parameters.
        :type nwalkers: int | None
        :param nsteps: The number of steps to run in the MCMC chain. Default is 1000.
        :type nsteps: int
        :param initial_state: Initial priors for MCMC. If None, random initial states
            are generated with first walker set to x0.
        :type initial_state: NDArray[Float] | None
        :param burn_in: Number of initial steps for MCMC to explore parameter space
            before sampling. If None, defaults to nsteps / 10.
        :type burn_in: int | None
        :param percentiles: List with percentiles for error estimation:
            [percentile for left error, percentile of centre, percentile for right error].
            If None, uses default percentiles.
        :type percentiles: list | None
        :param interp_treshold: Above this total number of datapoints, light curve will be
            interpolated using model containing equidistant points. If None, uses
            settings.MAX_CURVE_DATAPOINTS.
        :type interp_treshold: int | None
        :param samples: Sampling method: 'uniform' (equidistant in phase), 'adaptive'
            (equidistant on curve), or list with phases in (0, 1) interval.
            Default is 'uniform'.
        :type samples: str | list
        :param progress: Display the progress bar of the sampling. Default is False.
        :type progress: bool
        :param save: Whether to store the chain to file. Default is True.
        :type save: bool
        :param fit_id: ID which identifies fit file (if not specified, current datetime is used).
        :type fit_id: str | None
        :return: Optimized model parameters in flattened form with statistics.
        :rtype: dict[str, Any]
        """
        burn_in = int(nsteps / 10) if burn_in is None else burn_in
        self.set_up(
            x0,
            data,
            passband=data.keys(),
            discretization=discretization,
            morphology=self.MORPHOLOGY,
            interp_treshold=(settings.MAX_CURVE_DATAPOINTS if interp_treshold is None else interp_treshold),
            observer_system_cls=BinarySystem,
            samples=samples,
        )

        ndim: int = len(self.initial_vector)
        nwalkers = 2 * len(self.initial_vector) if nwalkers is None else nwalkers
        self.mcmc_nwalkers_vs_ndim_validity_check(nwalkers, ndim)

        sampler: emcee.EnsembleSampler = self._fit(
            nwalkers,
            ndim,
            nsteps,
            burn_in,
            initial_state,
            progress=progress,
            save=save,
            fit_id=fit_id,
        )

        # Extract fit results from MCMC sampler
        flat_chain: NDArray[Float] = sampler.get_chain(flat=True)
        result_dict: dict[str, Any] = mixins.MCMCMixin.resolve_mcmc_result(
            flat_chain,
            self.fitable,
            self.normalization,
            percentiles,
        )

        result_dict.update(
            {
                lbl: {"value": val.value, "fixed": True, "unit": val.to_dict()["unit"]}
                for lbl, val in self.fixed.items()
            },
        )
        result_dict = self.eval_constrained_results(result_dict, self.constrained)

        r_squared_args: tuple[Any, ...] = (
            self.x_data_reduced,
            self.y_data,
            self.observer.passband,
            discretization,
            self.x_data_reducer,
            1.0 / self.interp_treshold,
            self.interp_treshold,
            self.observer.system_cls,
        )

        r_dict: dict[str, Any] = {key: value["value"] for key, value in result_dict.items()}
        r_dict = parameters.extend_json_with_atm_params(
            r_dict,
            atmosphere_model=self.atmosphere_model,
            limb_darkening_coefficients=self.limb_darkening_coefficients,
        )

        r_squared_result: Float = lc_r_squared(lc_model.synthetic_binary, *r_squared_args, **r_dict)
        result_dict["r_squared"] = {"value": r_squared_result, "unit": None}

        result_dict = check_for_boundary_surface_potentials(result_dict, self.MORPHOLOGY)

        self.flat_result = result_dict
        return parameters.serialize_result(result_dict)


class OvercontactLightCurveFit(LightCurveFit):
    """MCMC fitting implementation for light curves of over-contact binaries.

    This class keeps the values of potentials synchronized for primary and secondary
    components, which is required for over-contact binary systems.
    """

    MORPHOLOGY = "over-contact"


class DetachedLightCurveFit(LightCurveFit):
    """MCMC fitting implementation for light curves of detached binaries."""

    MORPHOLOGY = "detached"


class CentralRadialVelocity(MCMCFit, AbstractRVFit):
    """RV fit class implementing an MCMC method using kinematic method."""

    def likelihood(self, xn: NDArray[Float]) -> Float:
        """Likelihood function for given set of model parameters.

        Best is 0.0, worst is -inf.

        :param xn: Vector of parameters we are looking for.
        :type xn: NDArray[Float]
        :return: Likelihood value.
        :rtype: Float
        """
        xn_list = parameters.vector_renormalizer(xn, self.fitable.keys(), self.normalization)
        xn_renorm = np.asarray(xn_list, dtype=FLOAT)
        kwargs: dict[str, Any] = parameters.prepare_properties_set(
            xn_renorm,
            self.fitable.keys(),
            self.constrained,
            self.fixed,
        )
        synthetic: dict[str, NDArray[Float]] = rv_model.central_rv_synthetic(
            self.x_data_reduced,
            self.observer,
            **kwargs,
        )
        synthetic = {comp: synthetic[comp][self.x_data_reducer[comp]] for comp in synthetic}

        ln_f_key = f"{NUISANCE_PARSER}{PARAM_PARSER}ln_f"
        ln_f: Float = parameters.prepare_nuisance_properties_set(
            xn_renorm,
            self.fitable.keys(),
            self.fixed,
        )[ln_f_key]
        lhood = self.likelihood_fn(synthetic, ln_f)

        self.eval_counter += 1
        logger.debug("eval counter = %s, likehood = %s", self.eval_counter, lhood)
        return lhood

    def fit(
        self,
        data: dict[str, RVData],
        x0: parameters.BinaryInitialParameters,
        *,
        nwalkers: int | None = None,
        nsteps: int = 1000,
        initial_state: NDArray[Float] | None = None,
        burn_in: int | None = None,
        percentiles: list | None = None,
        progress: bool = False,
        save: bool = True,
        fit_id: str | None = None,
    ) -> dict[str, Any]:
        """Fit method using Markov Chain Monte Carlo.

        Once simulation is done, the following values are stored and can be used for further evaluation:

        .. code-block:: python

            self.last_sampler: emcee.EnsembleSampler
            self.last_normalization: dict; normalization map used during fitting
            self.flat_chain_path: str; filename of last stored flattened emcee sampler with metadata

        Based on https://emcee.readthedocs.io/en/stable/.

        :param data: Radial velocity dataset with observational data.
        :type data: dict[str, RVData]
        :param x0: Initial state of binary system parameters (metadata included).
        :type x0: parameters.BinaryInitialParameters
        :param nwalkers: The number of walkers in the ensemble. If None, defaults to
            2 * number of free parameters. Minimum is 2 * number of free parameters.
        :type nwalkers: int | None
        :param nsteps: The number of steps to run in the MCMC chain. Default is 1000.
        :type nsteps: int
        :param initial_state: Initial priors for MCMC. If None, random initial states
            are generated with first walker set to x0.
        :type initial_state: NDArray[Float] | None
        :param burn_in: Number of initial steps for MCMC to explore parameter space
            before sampling. If None, defaults to nsteps / 10.
        :type burn_in: int | None
        :param percentiles: List with percentiles for error estimation:
            [percentile for left error, percentile of centre, percentile for right error].
            If None, uses default percentiles.
        :type percentiles: list | None
        :param progress: Display the progress bar of the sampling. Default is False.
        :type progress: bool
        :param save: Whether to store the chain to file. Default is True.
        :type save: bool
        :param fit_id: ID which identifies fit file (if not specified, current datetime is used).
        :type fit_id: str | None
        :return: Optimized model parameters in flattened form with statistics.
        :rtype: dict[str, Any]
        """
        burn_in = int(nsteps / 10) if burn_in is None else burn_in
        self.set_up(x0, data, observer_system_cls=RadialVelocitySystem)

        ndim: int = len(self.initial_vector)
        nwalkers = 2 * len(self.initial_vector) if nwalkers is None else nwalkers
        self.mcmc_nwalkers_vs_ndim_validity_check(nwalkers, ndim)

        sampler: emcee.EnsembleSampler = self._fit(
            nwalkers,
            ndim,
            nsteps,
            burn_in,
            initial_state,
            progress=progress,
            save=save,
            fit_id=fit_id,
        )

        # Extract fit results from MCMC sampler
        flat_chain: NDArray[Float] = sampler.get_chain(flat=True)
        result_dict: dict[str, Any] = mixins.MCMCMixin.resolve_mcmc_result(
            flat_chain,
            self.fitable,
            self.normalization,
            percentiles,
        )

        result_dict.update(
            {
                lbl: {
                    "value": val.value,
                    "fixed": True,
                    "unit": val.to_dict()["unit"],
                }
                for lbl, val in self.fixed.items()
            },
        )
        result_dict = self.eval_constrained_results(result_dict, self.constrained)

        r_squared_args: tuple[Any, ...] = (
            self.x_data_reduced,
            self.y_data,
            self.x_data_reducer,
            self.observer.system_cls,
        )
        r_dict: dict[str, Any] = {
            key: value["value"] for key, value in result_dict.items()
        }

        r_squared_result: Float = rv_r_squared(rv_model.central_rv_synthetic, *r_squared_args, **r_dict)
        result_dict["r_squared"] = {"value": r_squared_result, "unit": None}

        self.flat_result = result_dict
        return parameters.serialize_result(result_dict)
