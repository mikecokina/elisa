"""Focused regressions for the October 2026 UI reaudit."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import inspect

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from elisa.ui.shared import plotting
from elisa.ui.shared.logger import CircularBuffer, DualLoggingHandler
from elisa.ui.shared.utils import opt_int, result_temp_path
from elisa.ui.shared.validators import mcmc_integer, validated_float
from elisa.ui.tabs.lc_fitting import tab as lc_fit
from elisa.ui.tabs.lc_fitting.components import param_inputs
from elisa.ui.tabs.lc_modeling import tab as lc_tab
from elisa.ui.tabs.lc_modeling.components import spot_inputs
from elisa.ui.tabs.lc_modeling.logic import compute as lc_compute
from elisa.ui.tabs.rv_fitting import tab as rv_fit
from elisa.ui.tabs.rv_modeling.logic import compute as rv_compute


def test_lc_handler_forwards_both_spot_blocks(monkeypatch):
    calls = []
    monkeypatch.setattr(lc_compute, 'run_lc', lambda *args: (calls.append(args), pd.DataFrame(), None))
    primary = dict.fromkeys(spot_inputs.FIELD_ORDER, 0)
    secondary = dict(primary)
    primary.update(spot_0_enabled=True, spot_0_longitude=31, spot_0_angular_radius=12)
    secondary.update(spot_1_enabled=True, spot_1_longitude=122, spot_1_angular_radius=9)
    fn = lc_tab._make_handler(('mass',), ('mass',), ('period',), ('normalize',), ('enabled',), spot_inputs.FIELD_ORDER)
    fn(2, 1, 3, False, False, True, *primary.values(), *secondary.values())
    assert calls[0][:6] == ({'mass': 2}, {'mass': 1}, {'period': 3}, {'normalize': False}, {'enabled': False}, {'enabled': True})
    assert calls[0][6:] == (primary, secondary)
    assert spot_inputs.parse_spots(calls[0][6])[0]['longitude'] == 31
    assert spot_inputs.parse_spots(calls[0][7])[0]['longitude'] == 122


def test_every_passband_is_plotted():
    phases = np.array([0.1, 0.2])
    fluxes = {f'filter_{i}': np.array([i, i+1]) for i in range(12)}
    fig = plotting.render_lc_figure(phases, fluxes)
    try:
        assert [line.get_label() for line in fig.axes[0].lines] == list(fluxes)
        for line, flux in zip(fig.axes[0].lines, fluxes.values(), strict=True):
            np.testing.assert_array_equal(line.get_ydata(), flux)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('save', [lc_compute._save_lc_csv, rv_compute._save_rv_csv])
def test_csv_exports_preserve_each_result(save, monkeypatch, tmp_path):
    import elisa.ui.shared.utils as utils
    # Patch both implementations so this also reproduces the old timestamp collision.
    monkeypatch.setattr(utils.tempfile, 'gettempdir', lambda: str(tmp_path))
    first = Path(save(pd.DataFrame({'phase': [0.123456789], 'value': [1.23456789]})))
    second = Path(save(pd.DataFrame({'phase': [0.2], 'value': [2.0]})))
    assert first != second
    assert pd.read_csv(first)['value'][0] == 1.23456789
    assert pd.read_csv(second)['value'][0] == 2.0


def test_json_export_names_are_unique(monkeypatch, tmp_path):
    import elisa.ui.shared.utils as utils
    monkeypatch.setattr(utils.tempfile, 'gettempdir', lambda: str(tmp_path))
    with ThreadPoolExecutor(4) as pool:
        paths = list(pool.map(lambda _: result_temp_path('rv', 'mcmc'), range(100)))
    assert len(set(paths)) == 100
    assert all(p.parent == tmp_path and p.suffix == '.json' for p in paths)


@pytest.mark.parametrize('compute,run', [(lc_compute, 'run_lc'), (rv_compute, 'run_rv')])
@pytest.mark.parametrize('sampling', [
    {'phase_step': 0}, {'phase_step': -1}, {'from_phase': 1, 'to_phase': 0},
    {'from_phase': 1, 'to_phase': 1}, {'from_phase': float('nan')},
    {'to_phase': float('inf')}, {'phase_step': float('nan')},
    {'phase_step': 'bad'}, {'phase_step': 1e-300}, {'from_phase': 1e20, 'to_phase': 1e21, 'phase_step': 0.01},
])
def test_bad_sampling_rejected_before_model_construction(compute, run, sampling, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid sampling reached model construction')
    monkeypatch.setattr(compute, 'build_star', unexpected)
    with pytest.raises(ValueError):
        getattr(compute, run)({}, {}, {}, {'passband': ['bolometric'], **sampling})


@pytest.mark.parametrize('value', [1.5, '2.9', float('inf'), float('nan')])
def test_integer_inputs_are_not_truncated(value):
    with pytest.raises(ValueError):
        opt_int(value)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 'bad'])
def test_optional_float_rejects_invalid_supplied_value(value):
    with pytest.raises(ValueError):
        validated_float(value, name='albedo', lo=0, hi=1)


def test_mcmc_count_defaults_and_zero():
    assert opt_int(2**54 + 1) == 2**54 + 1
    assert mcmc_integer(0, name='burn_in', default=50, minimum=0) == 0
    assert mcmc_integer(None, name='burn_in', default=50, minimum=0) == 50
    assert mcmc_integer('', name='burn_in', default=50, minimum=0) == 50
    for value in [-1, 0.5, float('nan')]:
        with pytest.raises(ValueError):
            mcmc_integer(value, name='burn_in', default=50, minimum=0)


@pytest.fixture(scope='module')
def fit_apps():
    with gr.Blocks() as lc_app:
        lc_fit.build()
    with gr.Blocks() as rv_app:
        rv_fit.build()
    return lc_app, rv_app


def _named_callback(app, name):
    return next(fn for fn in app.fns.values() if fn.fn.__name__ == name or name in inspect.getclosurevars(fn.fn).nonlocals)


@pytest.mark.parametrize('which,name', [(0, '_unified_mcmc'), (1, '_mcmc_with_chain_handler')])
def test_actual_mcmc_callbacks_preserve_zero(fit_apps, which, name, monkeypatch):
    module = lc_fit if which == 0 else rv_fit
    callback = _named_callback(fit_apps[which], name)
    values = [c.value for c in callback.inputs]
    burn_index = next(i for i, c in enumerate(callback.inputs) if 'burn' in str(c.label).lower())
    values[burn_index] = 0
    if which == 0:
        monkeypatch.setattr(module, '_parse_lc_inputs_simple', lambda *a: ('Phases', [], 'detached', None))
    calls = []
    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {}, None, None, None, pd.DataFrame(), None
    monkeypatch.setattr(module.compute, 'run_mcmc', run)
    # File requirements in the RV handler are supplied with harmless stand-ins.
    if which == 1:
        for i, component in enumerate(callback.inputs):
            if isinstance(component, gr.File):
                values[i] = SimpleNamespace(name='/tmp/rv-data.dat')
    callback.fn(*values)
    assert calls
    assert (calls[0][1]['burn_in'] if which == 0 else calls[0][0][6]) == 0


def test_legacy_rv_mcmc_callback_preserves_zero(monkeypatch):
    calls = []
    def run(*args, **kwargs):
        calls.append(args)
        return {}, None, None, None, pd.DataFrame(), None
    monkeypatch.setattr(rv_fit.compute, 'run_mcmc', run)
    rv_fit._mcmc_handler(('primary_file',), (), ('burn_in',))(SimpleNamespace(name='data'), 0)
    assert calls[0][6] == 0


@pytest.mark.parametrize('approach', ['Community', 'Standard'])
@pytest.mark.parametrize('mode', ['free', 'fixed', 'constrained'])
def test_approach_callback_respects_modes(fit_apps, approach, mode):
    callback = _named_callback(fit_apps[0], '_on_approach_change')
    updates = callback.fn(approach, *[mode for _ in callback.inputs[1:]])
    for key, update in zip(param_inputs.APPROACH_TOGGLED_KEYS, updates):
        allowed = param_inputs.is_key_interactive_for_approach(key, approach.lower())
        expected = allowed and (mode == 'constrained' if key.endswith('_constraint') else mode == 'free' if key.endswith(('_min', '_max')) else True)
        assert update['interactive'] == expected


def test_initial_and_json_parameter_interactivity():
    with gr.Blocks():
        comps, _ = param_inputs.build(approach='community', defaults={'system_mass_ratio_fixed': True})
    assert comps['system_mass_ratio_constraint'].interactive is False
    assert comps['system_mass_ratio_min'].interactive is False
    for key in ('system_mass_ratio_value', 'system_mass_ratio_mode'):
        assert lc_fit._build_active_param_update(key, {}, 'community', 'Phases')['interactive'] is True
    for key in ('system_primary_minimum_time_value', 'system_primary_minimum_time_constraint'):
        assert lc_fit._build_active_param_update(key, {}, 'community', 'Phases')['interactive'] is False


def test_concurrent_capture_preserves_ownership_and_restores_show():
    original_show = plt.show
    first_entered, second_attempted, second_entered, release = Event(), Event(), Event(), Event()
    unrelated = plt.figure()
    def first():
        with plotting.capture_figure() as captured:
            figure = plt.figure()
            first_entered.set()
            assert release.wait(5)
            plt.show()
        assert captured == [figure]
        return figure
    def second():
        assert first_entered.wait(5)
        second_attempted.set()
        with plotting.capture_figure() as captured:
            second_entered.set()
            figure = plt.figure()
            plt.show()
        assert captured == [figure]
        return figure
    try:
        with ThreadPoolExecutor(2) as pool:
            a, b = pool.submit(first), pool.submit(second)
            assert second_attempted.wait(5)
            try:
                assert not second_entered.wait(0.1)
            finally:
                release.set()
            figures = [a.result(), b.result()]
        assert plt.show is original_show
        assert plt.fignum_exists(unrelated.number)
        for figure in figures:
            assert plotting.figure_to_pil(figure).size[0] > 0
    finally:
        plt.close(unrelated)


def test_capture_exception_cleans_only_owned_figures():
    original_show = plt.show
    existing = set(plt.get_fignums())
    with pytest.raises(RuntimeError):
        with plotting.capture_figure():
            plt.figure()
            raise RuntimeError('plot failed')
    assert plt.show is original_show
    assert set(plt.get_fignums()) == existing


def test_fits_run_concurrently_and_do_not_close_foreign_figures(monkeypatch):
    from elisa.ui.tabs.rv_fitting.logic import compute
    from contextlib import nullcontext
    barrier = Barrier(2)
    monkeypatch.setattr(compute, 'fit_logging', nullcontext)
    monkeypatch.setattr(compute, 'build_x0', lambda _: {})
    def fit(**kwargs):
        barrier.wait(timeout=5)
        return {}
    def model(**kwargs):
        return plt.figure()
    task = SimpleNamespace(fit=fit, plot=SimpleNamespace(model=model), fit_cls=SimpleNamespace(flat_result={}), save_result=lambda _: None)
    foreign = plt.figure()
    try:
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: compute.run_lsqrt('data', None, 'Phases', {}, task=task), range(2)))
        assert len(results) == 2
        assert plt.fignum_exists(foreign.number)
    finally:
        plt.close(foreign)


def test_previously_fixed_log_record_deduplication():
    import logging
    from io import StringIO
    buffer = CircularBuffer()
    handler = DualLoggingHandler(buffer, StringIO())
    record = logging.LogRecord('elisa.audit', logging.INFO, '', 1, 'single-message', (), None)
    handler.handle(record)
    handler.handle(record)
    assert buffer.getvalue().count('single-message') == 1


def test_overlapping_fit_logging_restores_profile_only_after_last_exit():
    from elisa import settings
    from elisa.ui.shared.logging_config import fit_logging
    first_entered, second_entered, first_exited = Event(), Event(), Event()
    def first():
        with fit_logging():
            first_entered.set()
            assert second_entered.wait(5)
        first_exited.set()
    def second():
        assert first_entered.wait(5)
        with fit_logging():
            second_entered.set()
            assert first_exited.wait(5)
            assert settings.LOG_CONFIG == 'fit'
    with settings.logging_context('default'):
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(first), pool.submit(second)]
            for future in futures:
                future.result()
        assert settings.LOG_CONFIG == 'default'
        with pytest.raises(RuntimeError):
            with fit_logging():
                with fit_logging():
                    raise RuntimeError('fit failed')
        assert settings.LOG_CONFIG == 'default'
