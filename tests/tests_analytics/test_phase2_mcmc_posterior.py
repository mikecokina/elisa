"""Phase 2 posterior, prior metadata and initialization regressions."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import emcee
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa.analytics.binary_fit.mcmc import CentralRadialVelocity, DetachedLightCurveFit, MCMCFit
from elisa.analytics.binary_fit.lc_prediction import LCSynthesisError, PredictionAlignmentError
from elisa.analytics.params import parameters as p
from elisa.base.error import ElisaError, MorphologyError, PhysicalStateError
from elisa.binary_system.system import BinarySystem
from elisa.binary_system.curves.community import RadialVelocitySystem
from tests.tests_analytics.test_lc_prediction_consistency import BASE, V
from tests.tests_analytics.test_phase1_uncertainties import lc_setup, make_lc
from tests.tests_analytics.test_lc_preparation_and_lsq import rv_problem


class ToyFit(MCMCFit):
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def likelihood(self, xn):
        if xn[0] < .25:
            raise MorphologyError('excluded geometry')
        return -float(np.sum(xn**2))


def toy():
    fit = ToyFit()
    fit.fitable = {'system@gamma': SimpleNamespace(sigma=None)}
    fit.fixed = {}
    fit.constrained = {}
    fit.normalization = {'system@gamma': (10., 20.)}
    fit.initial_vector = [15.]
    fit.prepare_priors()
    return fit


def test_history_independent_rejection_and_valid_acceptance():
    fit = toy()
    invalid = np.array([.1])
    assert fit.ln_probability(invalid) == -np.inf
    for value in (.5, 1., .25):
        assert fit.ln_probability(np.array([value])) == -value**2
        assert fit.ln_probability(invalid) == -np.inf
    assert not hasattr(fit, '_last_known_lhood')


@pytest.mark.parametrize('error', [ValueError('bug'), TypeError('bug'), ElisaError('unexpected'),
                                   RuntimeError('solver'), FloatingPointError('numerics'),
                                   PredictionAlignmentError('alignment')])
def test_unexpected_errors_escape(error):
    fit = toy()
    with patch.object(fit, 'likelihood', side_effect=error), pytest.raises(type(error), match=str(error)):
        fit.ln_probability(np.array([.5]))


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_unclassified_nonfinite_likelihood_raises(value):
    fit = toy()
    with patch.object(fit, 'likelihood', return_value=value), pytest.raises(FloatingPointError):
        fit.ln_probability(np.array([.5]))


def test_emcee_rejected_proposal_keeps_previous_valid_position():
    class InvalidMove(emcee.moves.Move):
        def propose(self, model, state):
            proposed = np.full_like(state.coords, .1)
            log_prob, blobs = model.compute_log_prob_fn(proposed)
            accepted = np.isfinite(log_prob)
            new = emcee.State(proposed, log_prob=log_prob, blobs=blobs)
            return self.update(state, new, accepted), accepted
    fit = toy()
    initial = np.array([[.4], [.6], [.8], [.9]])
    sampler = emcee.EnsembleSampler(4, 1, fit.ln_probability, moves=InvalidMove())
    sampler.run_mcmc(initial, 3)
    assert_array_equal(sampler.get_chain(), np.broadcast_to(initial, (3, 4, 1)))
    assert_array_equal(sampler.acceptance_fraction, np.zeros(4))


def test_automatic_initialization_uses_uniform_draws_and_preserves_x0():
    fit = toy()
    fit.norm_init_vector[:] = .1
    draws = np.array([[.9], [.2], [.4], [.6]])
    with patch('numpy.random.default_rng') as factory:
        factory.return_value.uniform.return_value = draws.copy()
        with patch.object(fit, 'ln_probability', side_effect=AssertionError('posterior evaluation')):
            with patch.object(fit, 'likelihood', side_effect=AssertionError('likelihood evaluation')):
                ensemble = fit.generate_initial_states(None, 4, 1, x0_vector=fit.norm_init_vector)
        factory.return_value.uniform.assert_called_once_with(0.0, 1.0, (4, 1))
    assert_array_equal(ensemble[:, 0], [.1, .2, .4, .6])
    assert_array_equal(fit.norm_init_vector, [.1])


def test_supplied_initialization_does_not_evaluate_posterior():
    fit = toy()
    states = np.array([[.1], [.7]])
    with patch.object(fit, 'ln_probability', side_effect=AssertionError('posterior evaluation')):
        with patch.object(fit, 'likelihood', side_effect=AssertionError('likelihood evaluation')):
            ensemble = fit.generate_initial_states(states, 2, 1)
    assert_array_equal(ensemble, states)
    assert not np.shares_memory(ensemble, states)


@pytest.mark.parametrize('states', [np.array([[-.1], [.7]]),
                                   np.array([[1.1], [.7]]), np.array([[np.nan], [.7]]),
                                   np.array([[np.inf], [.7]]), np.ones((2, 2))])
def test_supplied_initial_states_rejected_without_mutation(states):
    before = states.copy()
    states.flags.writeable = False
    with pytest.raises(ValueError, match='initial|walker'):
        toy().generate_initial_states(states, 2, 1)
    assert_array_equal(states, before)


def test_valid_supplied_states_are_copied_without_changes():
    states = np.array([[.25], [1.]])
    states.flags.writeable = False
    result = toy().generate_initial_states(states, 2, 1)
    assert_array_equal(result, states)
    assert not np.shares_memory(result, states)


@pytest.mark.parametrize('bounds', [(1., 1.), (2., 1.), (np.nan, 1.), (0., np.inf), (-np.inf, 1.),
                                   (-1e308, 1e308)])
def test_invalid_prior_bounds(bounds):
    fit = toy()
    fit.normalization['system@gamma'] = bounds
    with pytest.raises(ValueError, match='bounds|span'):
        fit.prepare_priors()


@pytest.mark.parametrize('sigma', [0., -1., np.nan, np.inf, -np.inf])
def test_invalid_gaussian_widths(sigma):
    fit = toy()
    fit.fitable['system@gamma'].sigma = sigma
    with pytest.raises(ValueError, match='Gaussian width'):
        fit.prepare_priors()


def test_normalized_width_avoids_cancellation_and_preserves_prior_definition():
    fit = toy()
    fit.initial_vector = [1e16]
    fit.normalization['system@gamma'] = (0., 2e16)
    fit.fitable['system@gamma'].sigma = .1
    fit.prepare_priors()
    assert_allclose(fit.sigmas, [5e-18], rtol=1e-15, atol=0)
    # Existing prior is log N(z;0,1), without an added -log(sigma).
    assert_allclose(fit.ln_prior(np.array([.5]), fit.norm_init_vector, fit.sigmas), -.5*np.log(2*np.pi))
    x = np.array([.3, .7])
    assert_allclose(fit.ln_prior(x, np.array([.5, .5]), np.array([.1, np.nan])),
                    -2.-.5*np.log(2*np.pi))
    for value in (-np.finfo(float).eps, 1.+np.finfo(float).eps, np.nan):
        assert fit.ln_prior(np.array([value]), np.array([.5]), np.array([.1])) == -np.inf


def test_metadata_checks_not_repeated_in_posterior():
    fit = toy()
    with patch.object(fit, 'prepare_priors', side_effect=AssertionError('hot path')):
        assert np.isfinite(fit.ln_probability(np.array([.5])))


def test_real_rv_prior_preparation_constraints_and_ensemble():
    cfg, data = rv_problem(constrained=True)
    fit = CentralRadialVelocity()
    fit.set_up(p.BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    # Constrained metadata and fixed parameters have no independent prior.
    fit.constrained['system@mass_ratio'].sigma = -1.
    fit.prepare_priors()
    assert len(fit.sigmas) == len(fit.fitable) == 2
    states = fit.generate_initial_states(None, 6, 2, x0_vector=fit.norm_init_vector)
    assert all(np.isfinite(fit.ln_probability(row)) for row in states)
    assert 'system@mass_ratio' not in fit.fitable
    assert fit.__class__.__mro__.index(MCMCFit) < fit.__class__.__mro__.index(type(fit).__bases__[1])


def test_lc_wrapped_programming_error_is_not_physical_rejection():
    fit = DetachedLightCurveFit()
    lc_setup(fit, {V: make_lc(errors=[.1]*3)})
    fit.prepare_priors()
    original = ValueError('bad input structure')
    wrapped = LCSynthesisError('wrapped')
    wrapped.__cause__ = original
    with patch.object(fit, 'predict_light_curves', side_effect=wrapped):
        with pytest.raises(ValueError, match='bad input structure'):
            fit.ln_probability(fit.norm_init_vector)


def test_real_roche_boundaries_through_lc_posterior():
    fit = DetachedLightCurveFit()
    lc_setup(fit, {V: make_lc(errors=[.1]*3)})
    fit.prepare_priors()
    # Exercise real BinarySystem construction and morphology; avoid surface synthesis.
    template = deepcopy(BASE)
    for component in ('primary', 'secondary'):
        template[component]['synchronicity'] = 1.
    def geometry(kwargs):
        config = deepcopy(template)
        for name, value in kwargs.items():
            group, key = name.split('@')
            if group in config:
                config[group][key] = value
        BinarySystem.from_json(config)
        return {V: np.ones(3)}
    reference = BinarySystem.from_json(template)
    _, inner, outer = reference.libration_potentials()
    assert inner > outer
    with patch.object(fit, 'predict_light_curves', side_effect=geometry):
        for potential in [inner + 1e-5, inner, inner - 1e-5, (inner+outer)/2]:
            fit.fixed['primary@surface_potential'].value = potential
            fit.fixed['secondary@surface_potential'].value = potential
            assert np.isfinite(fit.ln_probability(fit.norm_init_vector))
        fit.fixed['secondary@surface_potential'].value = inner + .1
        fit.fixed['primary@surface_potential'].value = inner - 1e-5
        assert fit.ln_probability(fit.norm_init_vector) == -np.inf
        for component in ('primary', 'secondary'):
            fit.fixed[f'{component}@surface_potential'].value = outer - 1e-5
        assert fit.ln_probability(fit.norm_init_vector) == -np.inf


@pytest.mark.parametrize('key', ['system@period', 'system@mass_ratio', 'primary@t_eff'])
def test_explicit_physical_scales(key):
    with pytest.raises(PhysicalStateError):
        MCMCFit.validate_physical_parameters({key: 0.})
    MCMCFit.validate_physical_parameters({key: np.nextafter(0., 1.)})
    with pytest.raises(FloatingPointError):
        MCMCFit.validate_physical_parameters({key: np.nan})


@pytest.mark.parametrize('sigma,bounds', [(np.nextafter(0., 1.), (0., 1e100)),
                                         (1e308, (0., 1e-100))])
def test_gaussian_width_must_be_representable_after_normalization(sigma, bounds):
    fit = toy()
    fit.fitable['system@gamma'].sigma = sigma
    fit.normalization['system@gamma'] = bounds
    fit.initial_vector = [bounds[1]/2]
    with pytest.raises(ValueError, match='normalized coordinates'):
        fit.prepare_priors()


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_initial_prior_center_rejected(value):
    fit = toy()
    fit.initial_vector = [value]
    with pytest.raises(ValueError, match='finite normalized'):
        fit.prepare_priors()


def test_current_constraints_are_physical_checks_and_fixed_scales_checked_once():
    cfg, data = rv_problem(constrained=True)
    fit = CentralRadialVelocity()
    fit.set_up(p.BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    fit.prepare_priors()
    fit.constrained['system@mass_ratio'].constraint = '-system@asini'
    assert fit.ln_probability(fit.norm_init_vector) == -np.inf
    fit.fixed['system@period'].value = 0.
    with pytest.raises(PhysicalStateError, match='period'):
        fit.prepare_priors()


def test_real_fit_prepares_once_and_starts_emcee_with_finite_states():
    from elisa import settings
    cfg, data = rv_problem(constrained=True)
    fit = CentralRadialVelocity()
    fit.set_up(p.BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    with patch.object(settings, 'NUMBER_OF_MCMC_PROCESSES', 1):
        with patch.object(fit, 'prepare_priors', wraps=fit.prepare_priors) as preparation:
            sampler = fit._fit(6, 2, 2, 0, save=False)
    assert preparation.call_count == 1
    assert sampler.get_chain().shape == (2, 6, 2)
    assert np.all(np.isfinite(sampler.get_log_prob()))


def test_new_physical_exception_keeps_legacy_error_mro():
    assert issubclass(MorphologyError, PhysicalStateError)
    assert issubclass(MorphologyError, ElisaError)
    assert not issubclass(MorphologyError, ValueError)
    for cls in (CentralRadialVelocity, DetachedLightCurveFit):
        fit = cls()
        assert isinstance(fit.sigmas, type(None))
        # Existing MCMCFit/MCMCMixin dictionaries own the sampler attributes.
        assert hasattr(fit, '__dict__')
        assert 'norm_init_vector' not in cls.__slots__


def test_lc_initialization_does_not_evaluate_physical_models():
    fit = DetachedLightCurveFit()
    lc_setup(fit, {V: make_lc(errors=[.1]*3)})
    fit.prepare_priors()
    states = np.array([[0.], [.25], [.75], [1.]])
    states.flags.writeable = False
    with patch.object(fit, 'ln_probability', side_effect=AssertionError('posterior evaluation')):
        with patch.object(fit, 'likelihood', side_effect=AssertionError('likelihood evaluation')):
            with patch.object(fit, 'predict_light_curves', side_effect=AssertionError('synthesis')):
                supplied = fit.generate_initial_states(states, 4, 1)
                automatic = fit.generate_initial_states(None, 4, 1, x0_vector=fit.norm_init_vector)
    assert_array_equal(supplied, states)
    assert_array_equal(automatic[0], fit.norm_init_vector)
    assert np.all((automatic >= 0.) & (automatic <= 1.))
