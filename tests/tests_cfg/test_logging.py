"""Tests for JSON-driven profiles and standard custom dictConfig behavior."""
from __future__ import annotations

import importlib.util
import io
import json
import logging
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from elisa.base.error import ElisaSettingsError

from elisa import settings
from elisa.conf import logconfig
from elisa.conf.enums import LoggingProfile


def load_ui_module(name):
    path = Path(__file__).parents[2] / 'src/elisa/ui/shared' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'logging_contract_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(name, message='logging contract'):
    return logging.LogRecord(name, logging.WARNING, __file__, 0, message, (), None)


def custom_file(tmp_path, level='ERROR'):
    path = tmp_path / 'custom.json'
    path.write_text(json.dumps({
        'version': 1, 'disable_existing_loggers': False,
        'handlers': {'null': {'class': 'logging.NullHandler'}},
        'root': {'level': level, 'handlers': ['null']},
        'loggers': {'logging.contract.custom': {'level': level, 'handlers': ['null'], 'propagate': False}},
    }))
    return path


def test_profiles_are_complete_and_repeated_switches_reuse_console():
    profiles = [json.loads((Path(logconfig.__file__).parent / 'logging_schemas' / f'{profile.value}.json').read_text())
                for profile in LoggingProfile]
    assert profiles[0]['loggers'].keys() == profiles[1]['loggers'].keys()
    observer = logging.getLogger('elisa.observer.observer')
    with settings.logging_context(LoggingProfile.DEFAULT.value):
        console = logconfig._console_handler
        for _ in range(3):
            for name, profile in zip((LoggingProfile.FIT.value, LoggingProfile.DEFAULT.value), reversed(profiles)):
                settings.configure_logging(name)
                assert console is logconfig._console_handler
                for logger_name, rule in profile['loggers'].items():
                    logger = logging.getLogger(logger_name)
                    assert logging.getLevelName(logger.level) == rule['level']
                    assert bool(logger.propagate) == bool(rule['propagate'])
                assert observer is logging.getLogger('elisa.observer.observer')
                assert observer.getEffectiveLevel() == (logging.WARNING if name == LoggingProfile.FIT.value else logging.DEBUG)


def test_contexts_restore_declared_policy_including_exceptions():
    with settings.logging_context(LoggingProfile.DEFAULT.value):
        with pytest.raises(RuntimeError):
            with settings.logging_context(LoggingProfile.FIT.value):
                with settings.logging_context(LoggingProfile.DEFAULT.value):
                    logging.getLogger('elisa.observer').setLevel(logging.CRITICAL)
                assert settings.LOG_CONFIG == LoggingProfile.FIT.value
                assert logging.getLogger('elisa.observer').level == logging.WARNING
                raise RuntimeError('fit failed')
        assert settings.LOG_CONFIG == LoggingProfile.DEFAULT.value
        assert logging.getLogger('elisa.observer').level == logging.NOTSET


def test_builtins_preserve_external_handlers_and_policies(tmp_path):
    root, host = logging.getLogger(), logging.getLogger('logging.contract.host')
    handler = logging.FileHandler(tmp_path / 'external.log')
    root.addHandler(handler)
    host.addHandler(handler)
    before = (host.level, host.disabled, host.propagate)
    host_filter = logging.Filter()
    root.addFilter(host_filter)
    stream = handler.stream
    try:
        with settings.logging_context(LoggingProfile.FIT.value):
            settings.configure_logging(LoggingProfile.DEFAULT.value)
            assert handler in root.handlers and handler in host.handlers
            assert host_filter in root.filters
            assert handler.stream is stream and not stream.closed
            host.handle(record(host.name))
            handler.flush()
            assert 'logging contract' in (tmp_path / 'external.log').read_text()
        assert (host.level, host.disabled, host.propagate) == before
    finally:
        root.removeFilter(host_filter)
        root.removeHandler(handler)
        host.removeHandler(handler)
        handler.close()


def test_custom_uses_standard_dictconfig(tmp_path):
    path = custom_file(tmp_path)
    with settings.logging_context(LoggingProfile.DEFAULT.value):
        with patch('logging.config.dictConfig') as configure:
            settings.configure_logging(str(path))
            configure.assert_called_once_with(json.loads(path.read_text()))


def test_custom_builtin_transitions_and_nested_contexts(tmp_path):
    path = custom_file(tmp_path)
    with settings.logging_context(LoggingProfile.DEFAULT.value):
        settings.configure_logging(str(path))
        assert logging.getLogger().level == logging.ERROR
        with settings.logging_context(LoggingProfile.FIT.value):
            assert logging.getLogger('scipy').level == logging.WARNING
            with settings.logging_context(str(path)):
                assert logging.getLogger().level == logging.ERROR
            assert settings.LOG_CONFIG == LoggingProfile.FIT.value
        assert settings.LOG_CONFIG == str(path)
        assert logging.getLogger().level == logging.ERROR
        settings.configure_logging(LoggingProfile.DEFAULT.value)
        assert logging.getLogger().handlers.count(logconfig._console_handler) == 1
        assert logging.getLogger().level == logging.DEBUG
        assert logging.getLogger('elisa.observer').level == logging.NOTSET


def test_custom_restoration_reloads_modified_file(tmp_path):
    path = custom_file(tmp_path)
    with settings.logging_context(LoggingProfile.DEFAULT.value):
        settings.configure_logging(str(path))
        with settings.logging_context(LoggingProfile.FIT.value):
            custom_file(tmp_path, level='CRITICAL')
        assert logging.getLogger().level == logging.CRITICAL


def test_custom_may_close_external_handlers(tmp_path):
    path = custom_file(tmp_path)
    handler = logging.FileHandler(tmp_path / 'external.log')
    stream = handler.stream
    logging.getLogger().addHandler(handler)
    try:
        with settings.logging_context(LoggingProfile.DEFAULT.value):
            settings.configure_logging(str(path))
            assert stream.closed  # This is intentionally standard dictConfig behavior.
            assert handler not in logging.getLogger().handlers
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_invalid_selectors_and_malformed_json_leave_selector_unchanged(tmp_path):
    broken = tmp_path / 'broken.json'
    broken.write_text('{broken')
    with settings.logging_context(LoggingProfile.DEFAULT.value):
        for profile in ('unknown-profile', str(tmp_path / 'missing.json'), str(broken)):
            with pytest.raises(ElisaSettingsError):
                settings.configure_logging(profile)
            assert settings.LOG_CONFIG == LoggingProfile.DEFAULT.value
        settings.configure(LOG_CONFIG=LoggingProfile.FIT.value)
        assert settings.LOG_CONFIG == LoggingProfile.FIT.value
        settings.configure(LOG_CONFIG=settings.DEFAULT_SETTINGS['LOG_CONFIG'])
        assert settings.LOG_CONFIG == LoggingProfile.DEFAULT.value


def test_ui_fitting_capture_and_progress_survive():
    ui, fitting = load_ui_module('logger'), load_ui_module('logging_config')
    capture, terminal = ui.CircularBuffer(), io.StringIO()
    handler = ui.DualLoggingHandler(capture, terminal, include_timestamp=False)
    root, elisa = logging.getLogger(), logging.getLogger('elisa')
    root.addHandler(handler)
    elisa.addHandler(handler)
    try:
        with settings.logging_context(LoggingProfile.DEFAULT.value):
            with redirect_stdout(ui.DualOutputBuffer(capture, terminal)):
                with pytest.raises(RuntimeError):
                    with fitting.fit_logging():
                        with fitting.fit_logging():
                            print('progress 50%')
                            logging.getLogger('elisa.analytics.binary_fit.least_squares').handle(record(
                                'elisa.analytics.binary_fit.least_squares', 'fit output'))
                            raise RuntimeError('fit failed')
                logging.getLogger('elisa.contract').handle(record('elisa.contract', 'after fit'))
        text = capture.getvalue()
        assert text.count('progress 50%') == 1
        assert text.count('fit output') == 1
        assert text.count('after fit') == 1
        assert handler in root.handlers and handler in elisa.handlers
    finally:
        root.removeHandler(handler)
        elisa.removeHandler(handler)
        handler.close()


def test_custom_deletion_during_context_reports_restoration_failure(tmp_path):
    path = custom_file(tmp_path)
    with settings.logging_context(LoggingProfile.DEFAULT.value):
        settings.configure_logging(str(path))
        with pytest.raises(ElisaSettingsError, match='missing JSON configuration'):
            with settings.logging_context(LoggingProfile.FIT.value):
                path.unlink()
        assert settings.LOG_CONFIG == LoggingProfile.FIT.value


def test_internal_module_is_renamed_without_a_shim():
    assert not Path(logconfig.__file__).with_name('logging.py').exists()


def test_production_loggers_use_module_names():
    from elisa import ld
    from elisa.atm import atm
    from elisa.analytics.binary_fit import least_squares
    from elisa.observer import observer
    for module in (atm, ld, least_squares, observer):
        assert module.logger.name == module.__name__


@pytest.mark.parametrize('fail_inside_capture', [False, True])
def test_console_created_during_ui_capture_returns_to_live_stdout(tmp_path, capsys, fail_inside_capture):
    import sys

    ui = load_ui_module('logger')
    captured = ui.CircularBuffer()
    with settings.logging_context(LoggingProfile.DEFAULT.value):
        # A custom configuration drops the old built-in console; the next one
        # is genuinely created inside capture, rather than reusing a safe stream.
        settings.configure_logging(str(custom_file(tmp_path)))
        try:
            with ui.TerminalCapture(captured):
                settings.configure_logging(LoggingProfile.FIT.value)
                console = logconfig._console_handler
                console.handle(record('elisa.contract', 'inside capture'))
                if fail_inside_capture:
                    raise RuntimeError('capture body failed')
        except RuntimeError:
            pass
        assert 'inside capture' in captured.getvalue()
        assert console.stream is sys.stdout
        captured.close()
        console.handle(record('elisa.contract', 'after capture'))
        assert 'after capture' in capsys.readouterr().out
