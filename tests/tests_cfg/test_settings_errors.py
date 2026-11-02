"""Settings errors preserve causes and never publish invalid candidate settings."""
from configparser import ConfigParser, InterpolationMissingOptionError, MissingSectionHeaderError
from json import JSONDecodeError
from pathlib import Path
from unittest.mock import patch

import pytest

from elisa import settings
from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.base.error import ElisaError, ElisaSettingsError, raise_elisa_settings_error
from elisa.conf.parsers import parse_tuple_interval


def test_settings_helper_preserves_cause_and_is_not_a_candidate_failure():
    cause = ValueError("invalid configuration")
    with pytest.raises(ElisaSettingsError, match="settings failed") as caught:
        raise_elisa_settings_error("settings failed", cause)
    assert isinstance(caught.value, ElisaError)
    assert caught.value.__cause__ is cause
    assert not isinstance(caught.value, SYNTHESIS_FAILURES)


@pytest.mark.parametrize(
    ("content", "cause_type"),
    [
        ("[computational]\nnumber_of_threads = not-an-int", ValueError),
        ("[general]\nsuppress_warnings = not-a-bool", ValueError),
        ("[general]\nhome = %(missing)s", InterpolationMissingOptionError),
    ],
)
def test_bad_ini_values_keep_the_original_cause_and_settings(content, cause_type):
    parser = ConfigParser()
    parser.read_string(content)
    before = settings.__dict__
    with pytest.raises(ElisaSettingsError) as caught:
        settings.update_config(parser)
    assert isinstance(caught.value.__cause__, cause_type)
    assert settings.__dict__ is before


def test_bad_ini_syntax_keeps_original_cause_and_settings(tmp_path):
    path = tmp_path / "invalid.ini"
    path.write_text("not an INI section\n")
    before = settings.__dict__
    with pytest.raises(ElisaSettingsError) as caught:
        settings.read_and_update_config(conf_path=str(path))
    assert isinstance(caught.value.__cause__, MissingSectionHeaderError)
    assert settings.__dict__ is before


def test_bad_logging_json_keeps_original_cause_and_settings(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{broken")
    before = settings.__dict__
    with pytest.raises(ElisaSettingsError) as caught:
        settings.configure(LOG_CONFIG=path)
    assert isinstance(caught.value.__cause__, JSONDecodeError)
    assert settings.__dict__ is before


def test_home_directory_failure_keeps_original_cause_and_settings(tmp_path):
    cause = PermissionError("directory is read only")
    before = settings.__dict__
    with patch.object(Path, "mkdir", side_effect=cause):
        with pytest.raises(ElisaSettingsError) as caught:
            settings.configure(HOME=tmp_path / "home")
    assert caught.value.__cause__ is cause
    assert settings.__dict__ is before


def test_override_body_errors_are_not_reclassified():
    original = settings.settings_serializer()
    cause = ValueError("failure in user computation")
    with pytest.raises(ValueError) as caught:
        with settings.override(NUMBER_OF_THREADS=2):
            raise cause
    assert caught.value is cause
    assert settings.settings_serializer() == original


def test_unknown_read_attribute_keeps_python_default_semantics():
    sentinel = object()
    assert getattr(settings, "_nonexistent_setting_for_check", sentinel) is sentinel


@pytest.mark.parametrize("raw", [123, [], {}])
def test_wrong_interval_input_type_is_a_settings_error(raw):
    with pytest.raises(ElisaSettingsError, match="invalid format"):
        parse_tuple_interval(raw, name="interval")
