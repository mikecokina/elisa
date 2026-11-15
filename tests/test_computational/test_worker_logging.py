"""Worker startup must not reopen parent-owned logging resources."""
from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("method", [m for m in ("fork", "spawn") if m in multiprocessing.get_all_start_methods()])
@pytest.mark.parametrize("remove_config", [False, True])
def test_worker_startup_preserves_custom_logging(tmp_path: Path, method: str, remove_config: bool) -> None:
    """Bound the missing-config regression so a respawning Pool cannot hang pytest."""
    script = tmp_path / "worker_logging.py"
    script.write_text("""
import json
import logging
import multiprocessing as mp
from pathlib import Path
import sys

from elisa import settings
from elisa.observer.mp_manager import initialize_worker


def read_state():
    handlers = logging.getLogger("worker-regression").handlers
    return (settings.REFLECTION_EFFECT_ITERATIONS, settings.LOG_CONFIG,
            any(isinstance(h, logging.FileHandler) for h in handlers))


if __name__ == "__main__":
    method, remove_config, directory = sys.argv[1:]
    directory = Path(directory)
    config = directory / "logging.json"
    log = directory / "run.log"
    config.write_text(json.dumps({
        "version": 1, "disable_existing_loggers": False,
        "handlers": {"file": {"class": "logging.FileHandler",
                              "filename": str(log), "mode": "w"}},
        "loggers": {"worker-regression": {"handlers": ["file"],
                                           "level": "INFO", "propagate": False}},
    }))
    settings.configure(LOG_CONFIG=str(config), REFLECTION_EFFECT_ITERATIONS=23)
    logger = logging.getLogger("worker-regression")
    logger.info("before pool")
    for handler in logger.handlers:
        handler.flush()
    snapshot = settings.settings_serializer()
    if remove_config == "True":
        config.unlink()
    with mp.get_context(method).Pool(1, initializer=initialize_worker,
                                     initargs=(snapshot,)) as pool:
        value, selector, has_file_handler = pool.apply_async(read_state).get(timeout=20)
    assert value == 23
    assert selector == str(config)
    if method == "fork":
        assert has_file_handler, "fork must retain inherited handlers"
    assert log.read_text() == "before pool\\n", "worker startup truncated the parent log"
    logger.info("after pool")
    for handler in logger.handlers:
        handler.flush()
    assert log.read_text() == "before pool\\nafter pool\\n"
""")
    config = tmp_path / "elisa.ini"
    config.write_text(f"[general]\nhome={tmp_path}\nsuppress_warnings=true\n")
    result = subprocess.run(  # noqa: S603 - Test-owned script, bounded child process.
        [sys.executable, str(script), method, str(remove_config), str(tmp_path)],
        env={**os.environ, "ELISA_CONFIG": str(config)}, capture_output=True, text=True, timeout=35, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_restoring_worker_snapshot_keeps_validation_and_derived_paths(tmp_path: Path) -> None:
    from elisa import settings
    from elisa.base.error import ElisaSettingsError

    previous = settings.settings_serializer()
    snapshot = {**previous, "HOME": str(tmp_path / "worker home"),
                "CK04_ATM_TABLES": str(tmp_path / "custom atlas"),
                "LOG_CONFIG": str(tmp_path / "already removed.json")}
    try:
        settings.restore_worker_settings(snapshot)
        assert settings.settings_serializer() == snapshot
        assert str(tmp_path / "custom atlas") in settings.ATLAS_TO_BASE_DIR.values()
        invalid = {**snapshot, "NUMBER_OF_PROCESSES": "invalid"}
        with pytest.raises(ElisaSettingsError):
            settings.restore_worker_settings(invalid)
        assert settings.settings_serializer() == snapshot
        with pytest.raises(ElisaSettingsError):
            settings.restore_worker_settings({"NOT_A_SETTING": 1})
        assert settings.settings_serializer() == snapshot
    finally:
        settings.restore_worker_settings(previous)
