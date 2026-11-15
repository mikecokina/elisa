from pathlib import Path
from unittest.mock import patch

import pytest

from elisa.managers.settings_manager import DefaultSettings, SettingsManager


@pytest.mark.parametrize("custom", [True, False])
def test_wizard_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom: bool) -> None:
    home = tmp_path / "default"
    monkeypatch.setattr(DefaultSettings, "HOME", str(home))
    destination = tmp_path / "custom" / "config.ini" if custom else home / "config.ini"
    answers = ["y", str(destination) if custom else "", *[str(tmp_path / name) for name in ("ld", "ck", "k")]]
    with patch("builtins.input", side_effect=answers):
        SettingsManager.run()
    assert f"home = {destination.parent}" in destination.read_text()
    assert (home / "config.ini").exists() is (not custom)


def test_wizard_refuses_existing(tmp_path: Path) -> None:
    destination = tmp_path / "config.ini"
    destination.write_text("original")
    with patch("builtins.input", side_effect=["y", str(destination)]), pytest.raises(FileExistsError):
        SettingsManager.run()
    assert destination.read_text() == "original"


def test_wizard_cancel() -> None:
    with patch("builtins.input", return_value="n"), pytest.raises(SystemExit):
        SettingsManager.run()


def test_wizard_write_failure(tmp_path: Path) -> None:
    destination = tmp_path / "config.ini"
    answers = ["y", str(destination), *[str(tmp_path / name) for name in ("ld", "ck", "k")]]
    with (
        patch("builtins.input", side_effect=answers),
        patch("configparser.ConfigParser.write", side_effect=OSError("write")),
        pytest.raises(OSError, match="write"),
    ):
        SettingsManager.run()
    assert not destination.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ck", "k", "ld"]


def test_wizard_write_failure_preserves_concurrent_config(tmp_path: Path) -> None:
    destination = tmp_path / "config.ini"
    answers = ["y", str(destination), *[str(tmp_path / name) for name in ("ld", "ck", "k")]]

    def fail_write(*_: object) -> None:
        destination.write_text("valid existing config")
        raise OSError("write")

    with (
        patch("builtins.input", side_effect=answers),
        patch("configparser.ConfigParser.write", side_effect=fail_write),
        pytest.raises(OSError, match="write"),
    ):
        SettingsManager.run()
    assert destination.read_text() == "valid existing config"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ck", "config.ini", "k", "ld"]
