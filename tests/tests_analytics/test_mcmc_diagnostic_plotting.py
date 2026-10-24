"""Plot diagnostics using synthetic chains and saved samples, without sampling."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa import units as u
from elisa.analytics.binary_fit import chain_tools, io_tools, plot
from elisa.analytics.binary_fit.mixins import MCMCMixin
from elisa.analytics.binary_fit.rv_fit import RVFitMCMC
from elisa.graphic.mcmc_graphics import Plot

LABELS = ['primary@t_eff', 'secondary@t_eff']


def fit_for(chain):
    return SimpleNamespace(
        chain=chain, flat_chain=chain.reshape(-1, chain.shape[-1]), variable_labels=LABELS,
        normalization={LABELS[0]: (1000., 2000.), LABELS[1]: (3000., 5000.)},
        flat_result={label: {'unit': 'K', 'value': 1500. if i == 0 else 4000.,
                             'confidence_interval': {'min': 1100., 'max': 4500.}}
                     for i, label in enumerate(LABELS)},
    )


@pytest.fixture(autouse=True)
def figures():
    with patch.object(plt, 'show'):
        yield
    plt.close('all')


def test_structured_walkers_units_summaries_and_no_mutation():
    chain = np.arange(30.).reshape(5, 3, 2) / 30
    before = chain.copy()
    fit = fit_for(chain)
    result = deepcopy(fit.flat_result)
    plot.traces(fit, truths=True, plot_units=dict.fromkeys(LABELS, u.K))
    axes = plt.gcf().axes
    assert len(axes) == 2
    assert axes[-1].get_xlabel() == 'Retained ensemble step'
    for p, ax in enumerate(axes):
        assert len(ax.lines) == 6  # Three walkers and three posterior summary lines.
        lower, upper = fit.normalization[LABELS[p]]
        for w, line in enumerate(ax.lines[:3]):
            assert_array_equal(line.get_xdata(), np.arange(5))
            assert_allclose(line.get_ydata(), lower + chain[:, w, p] * (upper - lower))
            assert line.get_linewidth() <= 1
            assert 0 < line.get_alpha() < 1
        assert 'K' in ax.get_ylabel()
        assert_allclose(ax.lines[3].get_ydata(), result[LABELS[p]]['value'])
    assert_array_equal(chain, before)
    assert_array_equal(fit.flat_chain, before.reshape(-1, 2))
    assert fit.flat_result == result


@pytest.mark.parametrize('indices', [[1, 0], [1]])
def test_custom_flat_source_column_mapping(indices):
    fit = fit_for(np.arange(30.).reshape(5, 3, 2) / 30)
    custom = fit.flat_chain[:, indices].copy()
    before = custom.copy()
    labels = [LABELS[i] for i in indices]
    plot.traces(fit, flat_chain=custom, variable_labels=labels, plot_units={})
    axes = plt.gcf().axes
    assert axes[-1].get_xlabel() == 'Flattened sample index'
    for p, ax in enumerate(axes):
        lower, upper = fit.normalization[labels[p]]
        assert_allclose(ax.collections[0].get_offsets()[:, 1], lower + custom[:, p] * (upper - lower))
        assert not ax.lines
    assert_array_equal(custom, before)


def test_internal_reordered_subset_maps_from_internal_source_axis():
    fit = fit_for(np.arange(30.).reshape(5, 3, 2) / 30)
    plot.traces(fit, variable_labels=LABELS[::-1], traces_to_plot=[LABELS[1]], plot_units={})
    assert len(plt.gcf().axes) == 1
    for w, line in enumerate(plt.gca().lines):
        assert_allclose(line.get_ydata(), 3000 + 2000 * fit.chain[:, w, 1])


@pytest.mark.parametrize('method, keyword', [(plot.traces, 'traces_to_plot'),
                                            (plot.autocorrelation, 'correlations_to_plot')])
@pytest.mark.parametrize('selection', [[], ['missing'], [LABELS[0], LABELS[0]], [LABELS[0], 'missing']])
def test_invalid_selections_create_no_figure(method, keyword, selection):
    fit = fit_for(np.ones((5, 3, 2)))
    with pytest.raises(ValueError, match='selection'):
        method(fit, **{keyword: selection})
    assert not plt.get_fignums()


@pytest.mark.parametrize('labels', [[], [LABELS[0]], [LABELS[0], LABELS[0]]])
@pytest.mark.parametrize('method', [plot.traces, plot.autocorrelation])
def test_bad_source_labels_create_no_figure(method, labels):
    fit = fit_for(np.ones((5, 3, 2)))
    custom = fit.flat_chain if method is plot.traces else fit.chain
    with pytest.raises(ValueError):
        method(fit, flat_chain=custom, variable_labels=labels)
    assert not plt.get_fignums()


@pytest.mark.parametrize('samples', [np.ones(5), np.ones((0, 2)), np.ones((5, 3, 2))])
def test_trace_custom_format_is_not_expanded(samples):
    with pytest.raises(ValueError):
        plot.traces(fit_for(np.ones((5, 3, 2))), flat_chain=samples)
    assert not plt.get_fignums()


def test_flat_fallback_and_acf_structured_requirement():
    fit = fit_for(np.ones((5, 3, 2)))
    fit.chain = None
    plot.traces(fit)
    assert plt.gca().get_xlabel() == 'Flattened sample index'
    with pytest.raises(ValueError, match='Temporal autocorrelation'):
        plot.autocorrelation(fit)


def test_acf_values_mapping_units_and_no_mutation():
    rng = np.random.default_rng(31)
    chain = rng.normal(size=(2048, 3, 2))
    before = chain.copy()
    fit = fit_for(chain)
    acf, tau = chain_tools.ensemble_autocorrelation(chain)
    plot.autocorrelation(fit, correlations_to_plot=[LABELS[1]])
    ax = plt.gca()
    assert len(plt.gcf().axes) == 1
    assert ax.get_xlabel() == 'Lag (ensemble steps)'
    assert_array_equal(ax.lines[0].get_xdata(), np.arange(len(chain)))
    assert_allclose(ax.lines[0].get_ydata(), acf[:, 1])
    assert f'{tau[1]:.2f} steps' in ax.get_legend().get_texts()[0].get_text()
    # Independent direct lag covariance confirms the plotted temporal ACF.
    centered = chain[:, :, 1] - chain[:, :, 1].mean(axis=0)
    expected = np.mean(np.sum(centered[:-1] * centered[1:], axis=0) / np.sum(centered**2, axis=0))
    assert_allclose(acf[1, 1], expected)
    assert_array_equal(chain, before)
    plt.close('all')
    plot.autocorrelation(fit, flat_chain=chain[:, :, ::-1], variable_labels=LABELS[::-1],
                         correlations_to_plot=[LABELS[1]])
    assert_allclose(plt.gca().lines[0].get_ydata(), acf[:, 1])


@pytest.mark.parametrize('constant', [False, True])
def test_unavailable_acf_annotations(constant):
    chain = np.broadcast_to(np.arange(8.)[:, None, None], (8, 3, 2)).copy()
    if constant:
        chain[:, 1, 0] = 1
    plot.autocorrelation(fit_for(chain))
    axes = plt.gcf().axes
    for ax in axes:
        legend = ax.get_legend().get_texts()[0].get_text()
        assert 'unavailable' in legend
        assert 'nan' not in legend
    if constant:
        assert 'constant walker' in axes[0].texts[0].get_text()
        assert np.isnan(axes[0].lines[0].get_ydata()).all()
        assert np.isfinite(axes[1].lines[0].get_ydata()).all()
    else:
        assert np.isfinite(axes[0].lines[0].get_ydata()).all()
        assert not axes[0].texts


@pytest.mark.parametrize('renderer, selection_key', [(Plot.paramtrace, 'traces_to_plot'),
                                                    (Plot.autocorr, 'correlations_to_plot')])
def test_backend_rejects_invalid_selection_before_figure(renderer, selection_key):
    with pytest.raises(ValueError, match='selection'):
        renderer(**{selection_key: [], 'variable_labels': LABELS, 'labels': LABELS}, **(
            {'flat_chain': np.ones((5, 2)), 'fit_params': {}, 'truths': False}
            if renderer is Plot.paramtrace else
            {'autocorr_fns': np.ones((5, 2)), 'autocorr_time': np.ones(2)}))
    assert not plt.get_fignums()


def test_structured_public_reload_preserves_plots(tmp_path):
    chain = np.random.default_rng(91).normal(size=(512, 3, 2))
    fit = fit_for(chain)
    path = MCMCMixin.save_flat_chain(
        fit.flat_chain, dict.fromkeys(LABELS, SimpleNamespace(to_dict=dict)),
        fit.normalization, str(tmp_path / 'chain.json'), chain_shape=chain.shape)
    loaded = RVFitMCMC()
    with patch.object(io_tools, 'update_solution'):
        loaded.load_chain(path)
    loaded.flat_result = deepcopy(fit.flat_result)
    for method in (plot.traces, plot.autocorrelation):
        method(fit)
        original = [[line.get_ydata().copy() for line in ax.lines] for ax in plt.gcf().axes]
        plt.close('all')
        method(loaded)
        for expected, ax in zip(original, plt.gcf().axes, strict=True):
            for ydata, line in zip(expected, ax.lines, strict=True):
                assert_allclose(line.get_ydata(), ydata)
        plt.close('all')
    assert_array_equal(loaded.chain, chain)
    assert_array_equal(loaded.flat_chain, fit.flat_chain)


def test_existing_saved_fixture_stays_flat_and_unmodified():
    path = Path(__file__).resolve().parents[2] / 'jupyter_tutorials/demo_data/assets/lc_mcmc_chain.json'
    data = json.loads(path.read_text())
    flat, chain = chain_tools.prepare_chain_samples(data, 7)
    before = flat.copy()
    fit = SimpleNamespace(chain=chain, flat_chain=flat, variable_labels=data['fitable_parameters'],
                          normalization=data['normalization'], flat_result={
                              label: {'unit': None} for label in data['fitable_parameters']})
    plot.traces(fit, traces_to_plot=[fit.variable_labels[-1]], plot_units={})
    assert plt.gca().get_xlabel() == 'Flattened sample index'
    label = fit.variable_labels[-1]
    lower, upper = fit.normalization[label]
    assert_allclose(plt.gca().collections[0].get_offsets()[:, 1], lower + flat[:, -1] * (upper - lower))
    assert_array_equal(flat, before)
    with pytest.raises(ValueError, match='Temporal autocorrelation'):
        plot.autocorrelation(fit)


def test_physical_unit_conversion_applies_to_samples_and_summary_only():
    chain = np.array([[[0.]], [[.5]], [[1.]]])
    label = 'system@gamma'
    summary = {'unit': 'm / s', 'value': 2000., 'confidence_interval': {'min': 1500., 'max': 2500.}}
    fit = SimpleNamespace(chain=chain, flat_chain=chain.reshape(-1, 1), variable_labels=[label],
                          normalization={label: (1000., 3000.)}, flat_result={label: deepcopy(summary)})
    plot.traces(fit, truths=True)  # Default display units are km/s.
    ax = plt.gca()
    assert_allclose(ax.lines[0].get_ydata(), [1., 2., 3.])
    assert_allclose([line.get_ydata()[0] for line in ax.lines[1:]], [2., 1.5, 2.5])
    assert 'km / s' in ax.get_ylabel()
    assert fit.flat_result[label] == summary
    assert_array_equal(chain[:, 0, 0], [0., .5, 1.])


def test_one_step_constant_panels_are_explained():
    plot.autocorrelation(fit_for(np.ones((1, 3, 2))))
    for ax in plt.gcf().axes:
        assert 'constant walker' in ax.texts[0].get_text()
        assert 'nan' not in ax.get_legend().get_texts()[0].get_text()
