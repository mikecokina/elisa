"""Independent chain integrity regressions; no fitting or synthesis is performed."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import emcee
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa.analytics import DiscardUnit
from elisa.analytics.binary_fit import chain_tools as chains, io_tools, plot
from elisa.analytics.binary_fit.lc_fit import LCFitMCMC
from elisa.analytics.binary_fit.rv_fit import RVFitMCMC
from elisa.analytics.binary_fit.mixins import MCMCMixin
from elisa.analytics.tasks import AnalyticsTask

LABELS = ['primary@t_eff', 'secondary@t_eff']


def payload(chain=None):
    chain = np.arange(40., dtype=float).reshape(5, 4, 2) if chain is None else chain
    return {
        'flat_chain': chain.reshape(-1, chain.shape[-1]).tolist(),
        'chain_shape': list(chain.shape),
        'chain_order': 'step,walker,parameter',
        'fitable_parameters': LABELS[:chain.shape[-1]],
        'normalization': {key: (0., 1.) for key in LABELS[:chain.shape[-1]]},
        'fitable': {},
    }


@pytest.fixture(scope='module')
def correlated():
    # Independent AR(1) walkers, with equilibrium transient removed.
    rng = np.random.default_rng(401)
    noise = rng.normal(size=(17000, 8, 2))
    result = np.zeros_like(noise)
    for step in range(1, len(result)):
        result[step] = .95 * result[step - 1] + noise[step]
    return result[1000:]


def test_ensemble_time_axis_and_independent_acf(correlated):
    acf, tau = chains.ensemble_autocorrelation(correlated)
    assert acf.shape == (16000, 2)
    # Direct time-domain dot products, independent of emcee's FFT implementation.
    for param in range(2):
        centered = correlated[:, :, param] - correlated[:, :, param].mean(axis=0)
        denom = np.sum(centered**2, axis=0)
        for lag in (0, 1, 5, 20):
            end = len(centered) - lag
            expected = np.mean(np.sum(centered[:end] * centered[lag:], axis=0) / denom)
            assert_allclose(acf[lag, param], expected, atol=1e-14)
    # AR(1) theoretical integrated time: (1 + rho)/(1 - rho) = 39.
    assert_allclose(tau, 39., rtol=.20)
    assert_allclose(tau, emcee.autocorr.integrated_time(correlated))


def test_flattening_correlated_walkers_gives_wrong_answer(correlated):
    _, temporal = chains.ensemble_autocorrelation(correlated)
    incorrect = emcee.autocorr.integrated_time(correlated.reshape(-1, 2), has_walkers=False)
    assert np.all(temporal > 10 * incorrect)
    with pytest.raises(ValueError, match='steps, walkers, parameters'):
        chains.ensemble_autocorrelation(correlated.reshape(-1, 2))


@pytest.mark.parametrize('length', [1, 2, 8, 20])
def test_insufficient_chain_is_unavailable(length):
    # Smooth short trajectory has far fewer than 50 autocorrelation times.
    chain = np.broadcast_to(np.arange(length)[:, None, None], (length, 4, 2))
    _, tau = chains.ensemble_autocorrelation(chain)
    assert np.isnan(tau).all()


def test_one_stuck_walker_invalidates_only_its_parameter(correlated):
    chain = correlated.copy()
    chain[:, 2, 0] = 1.
    acf, tau = chains.ensemble_autocorrelation(chain)
    assert np.isnan(acf[:, 0]).all()
    assert np.isnan(tau[0])
    assert np.isfinite(tau[1])


def test_fully_constant_chain_has_no_runtime_warnings():
    with np.errstate(all='raise'):
        acf, tau = chains.ensemble_autocorrelation(np.ones((100, 4, 2)))
    assert np.isnan(acf).all()
    assert np.isnan(tau).all()


@pytest.mark.parametrize('value', [None, [], np.ones((8, 2)), np.full((8, 2, 1), np.nan)])
def test_unavailable_temporal_structure(value):
    with pytest.raises(ValueError, match='Temporal autocorrelation'):
        chains.ensemble_autocorrelation(value)


@pytest.mark.parametrize('value, expected', [
    (None, None), (DiscardUnit.STEPS, DiscardUnit.STEPS),
    (DiscardUnit.ROWS, DiscardUnit.ROWS), ('steps', DiscardUnit.STEPS), ('rows', DiscardUnit.ROWS),
])
def test_discard_unit_normalization(value, expected):
    assert chains.validate_discard(0, value) is expected


def test_structured_step_discard_before_flattening():
    original = np.arange(40.).reshape(5, 4, 2)
    for unit in (None, DiscardUnit.STEPS, 'steps'):
        flat, chain = chains.prepare_chain_samples(payload(original), 2, unit)
        assert_array_equal(chain, original[2:])
        assert_array_equal(flat, original[2:].reshape(-1, 2))


def test_legacy_row_discard_without_guessing_walkers():
    data = payload()
    data.pop('chain_shape')
    data.pop('chain_order')
    for unit in (None, DiscardUnit.ROWS, 'rows'):
        flat, chain = chains.prepare_chain_samples(data, 3, unit)
        assert chain is None
        assert_array_equal(flat, np.asarray(data['flat_chain'])[3:])
    with pytest.raises(ValueError, match='requires temporal chain metadata'):
        chains.prepare_chain_samples(data, 0, DiscardUnit.STEPS)


@pytest.mark.parametrize('discard,retains', [(0, True), (1, False), (4, True), (7, False), (16, True)])
def test_explicit_row_discard_preserves_only_complete_ensembles(discard, retains):
    original = np.arange(40.).reshape(5, 4, 2)
    flat, chain = chains.prepare_chain_samples(payload(original), discard, DiscardUnit.ROWS)
    assert_array_equal(flat, original.reshape(-1, 2)[discard:])
    assert (chain is not None) == retains
    if retains:
        assert_array_equal(chain, original[discard // 4:])


@pytest.mark.parametrize('discard,unit,exception', [
    (-1, None, ValueError), (True, None, TypeError), (1.5, None, TypeError),
    (np.bool_(True), None, TypeError), (0, 'walkers', ValueError),
    (5, None, ValueError), (6, 'steps', ValueError), (20, 'rows', ValueError), (100, 'rows', ValueError),
])
def test_invalid_or_excessive_discard(discard, unit, exception):
    with pytest.raises(exception):
        chains.prepare_chain_samples(payload(), discard, unit)


@pytest.mark.parametrize('metadata', [
    {'chain_shape': None}, {'chain_shape': [5, 4]}, {'chain_shape': [5, 4, True]},
    {'chain_shape': [5, 4., 2]}, {'chain_shape': [5, 0, 2]}, {'chain_shape': [-5, 4, 2]},
    {'chain_shape': [4, 4, 2]}, {'chain_order': 'walker,step,parameter'}, {'chain_order': None},
])
def test_invalid_temporal_metadata(metadata):
    data = payload()
    data.update(metadata)
    with pytest.raises(ValueError, match='temporal chain metadata'):
        chains.prepare_chain_samples(data)


@pytest.mark.parametrize('missing', ['chain_shape', 'chain_order'])
def test_partial_metadata_is_rejected(missing):
    data = payload()
    data.pop(missing)
    with pytest.raises(ValueError, match='temporal chain metadata'):
        chains.prepare_chain_samples(data)


@pytest.mark.parametrize('samples', [[], [1, 2], [[np.nan, 0]], [[np.inf, 0]], [[[1, 2]]]])
def test_invalid_sample_matrix(samples):
    data = payload()
    data['flat_chain'] = samples
    with pytest.raises(ValueError, match='finite nonempty matrix'):
        chains.prepare_chain_samples(data)


@pytest.mark.parametrize('labels', [[], ['a'], ['a', 'a'], ['a', 2]])
def test_invalid_labels(labels):
    data = payload()
    data['fitable_parameters'] = labels
    with pytest.raises(ValueError, match='labels'):
        chains.prepare_chain_samples(data)


def save(tmp_path, structured=True):
    original = np.arange(40.).reshape(5, 4, 2) / 40
    fitable = {label: SimpleNamespace(to_dict=lambda: {'fixed': False}) for label in LABELS}
    kwargs = {'chain_shape': original.shape} if structured else {}
    path = MCMCMixin.save_flat_chain(original.reshape(-1, 2), fitable,
                                    dict.fromkeys(LABELS, (0., 1.)), str(tmp_path / 'chain.json'), **kwargs)
    return original, path


@pytest.mark.parametrize('structured', [True, False])
def test_json_roundtrip_keeps_flat_public_format_and_order(tmp_path, structured):
    original, path = save(tmp_path, structured)
    data = MCMCMixin.load_flat_chain(path)
    assert_array_equal(data['flat_chain'], original.reshape(-1, 2))
    assert set(data) == {'flat_chain', 'fitable', 'normalization', 'fitable_parameters'} | (
        {'chain_shape', 'chain_order'} if structured else set())
    flat, chain = chains.prepare_chain_samples(data)
    assert_array_equal(flat, original.reshape(-1, 2))
    if structured:
        assert_array_equal(chain, original)
    else:
        assert chain is None


def test_invalid_export_does_not_overwrite_saved_file(tmp_path):
    _, path = save(tmp_path)
    before = Path(path).read_bytes()
    with pytest.raises(ValueError, match='metadata'):
        MCMCMixin.save_flat_chain(np.zeros((20, 2)), dict.fromkeys(LABELS, SimpleNamespace(to_dict=dict)),
                                 {}, path, chain_shape=(4, 4, 2))
    assert Path(path).read_bytes() == before


def test_existing_saved_json_fixture_loads_unchanged():
    fixture = Path(__file__).resolve().parents[2] / 'jupyter_tutorials/demo_data/assets/lc_mcmc_chain.json'
    original = json.loads(fixture.read_text())
    loaded = MCMCMixin.load_flat_chain(str(fixture))
    assert loaded == original
    flat, chain = chains.prepare_chain_samples(loaded, 7)
    assert chain is None
    assert_array_equal(flat, np.asarray(original['flat_chain'])[7:])


def test_csv_samples_remain_flat_without_external_temporal_metadata(tmp_path):
    # Core chain I/O is JSON; CSV arrays are imported by their caller, with labels.
    original = np.arange(40.).reshape(20, 2)
    path = tmp_path / 'legacy.csv'
    np.savetxt(path, original, delimiter=',')
    data = {'flat_chain': np.loadtxt(path, delimiter=','), 'fitable_parameters': LABELS}
    flat, chain = chains.prepare_chain_samples(data, 3)
    assert_array_equal(flat, original[3:])
    assert chain is None


@pytest.mark.parametrize('fit_cls', [LCFitMCMC, RVFitMCMC])
def test_public_load_initialization_and_temporal_state(tmp_path, fit_cls):
    fit = fit_cls('detached', None, None) if fit_cls is LCFitMCMC else fit_cls()
    assert fit.chain is None
    original, path = save(tmp_path)
    with patch.object(io_tools, 'update_solution'):
        result = fit.load_chain(path, discard=1)
        assert_array_equal(fit.chain, original[1:])
        assert_array_equal(result[0], original[1:].reshape(-1, 2))
        fit.load_chain(path, discard=1, discard_unit=DiscardUnit.ROWS)
        assert fit.chain is None
        # Reload legacy after structured must clear previous temporal data.
        fit.load_chain(path)
        _, legacy = save(tmp_path, structured=False)
        fit.load_chain(legacy, discard=1)
        assert fit.chain is None


def test_filtering_invalidates_temporal_diagnostics():
    original = np.arange(40.).reshape(5, 4, 2) / 40
    fit = SimpleNamespace(chain=original, flat_chain=original.reshape(-1, 2), variable_labels=LABELS,
                          normalization=dict.fromkeys(LABELS, (0., 1.)), flat_result=dict.fromkeys(LABELS, {}))
    with patch.object(io_tools, 'update_solution'):
        flat = io_tools.filter_chain(fit, **{LABELS[0]: (.2, .8)})
    expected = original.reshape(-1, 2)
    assert_array_equal(flat, expected[(expected[:, 0] > .2) & (expected[:, 0] < .8)])
    assert fit.chain is None
    with pytest.raises(ValueError, match='Temporal autocorrelation'):
        plot.autocorrelation(fit)


def test_plot_selection_uses_parameter_axis_without_flattening(correlated):
    fit = SimpleNamespace(chain=correlated, variable_labels=LABELS)
    with patch.object(plot.MCMCPlot, 'autocorr') as draw:
        plot.autocorrelation(fit, correlations_to_plot=[LABELS[1]])
    args = draw.call_args.kwargs
    assert args['correlations_to_plot'] == [LABELS[1]]
    assert args['autocorr_fns'].shape == (16000, 2)
    assert_allclose(args['autocorr_time'], emcee.autocorr.integrated_time(correlated))
    # Selecting/reordering columns retains their time and walker axes.
    with patch.object(plot.MCMCPlot, 'autocorr') as draw:
        plot.autocorrelation(fit, flat_chain=correlated[:, :, 1:], variable_labels=LABELS[1:])
    assert_allclose(draw.call_args.kwargs['autocorr_time'], args['autocorr_time'][1:])
    with pytest.raises(ValueError, match='parameter dimension'):
        plot.autocorrelation(fit, variable_labels=LABELS[:1])


def test_invalid_load_does_not_mutate_fit_state():
    fit = SimpleNamespace(chain='sentinel', flat_chain='sentinel')
    with patch.object(MCMCMixin, 'load_flat_chain', return_value=payload()):
        with pytest.raises(ValueError, match='no samples'):
            io_tools.load_chain(fit, 'unused', discard=5)
    assert fit.chain == fit.flat_chain == 'sentinel'


@pytest.mark.parametrize('fit_cls', [LCFitMCMC, RVFitMCMC])
def test_public_fit_extracts_ensemble_without_synthesis(fit_cls):
    fit = fit_cls('detached', None, None) if fit_cls is LCFitMCMC else fit_cls()
    original = np.arange(40.).reshape(5, 4, 2)
    sampler = Mock()
    sampler.get_chain.return_value = original
    fit.fit_method_instance = SimpleNamespace(
        fit=Mock(return_value={}), last_sampler=sampler, flat_result={}, flat_chain_path='',
        normalization={}, fitable=dict.fromkeys(LABELS))
    with patch.object(fit, 'fit_summary'):
        fit.fit(Mock(), {})
    sampler.get_chain.assert_called_once_with()
    assert_array_equal(fit.chain, original)
    assert_array_equal(fit.flat_chain, original.reshape(-1, 2))


@pytest.mark.parametrize('unit', [DiscardUnit.STEPS, DiscardUnit.ROWS, 'steps', 'rows'])
def test_task_forwards_explicit_discard_unit(unit):
    fit = Mock()
    task = SimpleNamespace(method='mcmc', MCMC_NAMES=['mcmc'], fit_cls=fit)
    assert AnalyticsTask.load_chain(task, 'chain.json', discard=3, discard_unit=unit) is task
    fit.load_chain.assert_called_once_with('chain.json', 3, None, discard_unit=unit)


def test_periodic_export_preserves_sampler_axes_without_diagnostics():
    original = np.arange(40.).reshape(5, 4, 2)
    sampler = Mock()
    sampler.get_chain.side_effect = lambda **kwargs: original.reshape(-1, 2) if kwargs.get('flat') else original
    sampler.sample.return_value = iter([None])
    with patch('elisa.analytics.binary_fit.mixins.time', side_effect=[0., 1e12, 1e12]), \
            patch.object(MCMCMixin, 'save_flat_chain') as save_chain, \
            patch.object(chains, 'ensemble_autocorrelation', side_effect=AssertionError('diagnostics during sampling')):
        MCMCMixin.worker(sampler, original[0], 1, 0, save=True, fitable={}, normalization={})
    kwargs = save_chain.call_args.kwargs
    assert kwargs['chain_shape'] == original.shape
    assert_array_equal(save_chain.call_args.args[0], original.reshape(-1, 2))
    sampler.reset.assert_called_once()


def test_worker_excludes_unstored_burn_in_from_production():
    # The actual sampler boundary resets after un-stored burn-in; no burn-in rows
    # are available to serialize or discard a second time automatically.
    sampler = Mock()
    initial = np.zeros((4, 2))
    burned = np.ones((4, 2))
    sampler.run_mcmc.side_effect = [(burned, None, None), (burned, None, None)]
    MCMCMixin.worker(sampler, initial, 5, 2)
    burn_call, production_call = sampler.run_mcmc.call_args_list
    assert_array_equal(burn_call.args[0], initial)
    assert burn_call.args[1] == 2
    assert burn_call.kwargs == {"progress": False, "store": False}
    assert_array_equal(production_call.args[0], burned)
    assert production_call.args[1] == 5
    assert production_call.kwargs == {"progress": False}
    sampler.reset.assert_called_once()
