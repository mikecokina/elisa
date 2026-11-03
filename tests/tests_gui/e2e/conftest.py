"""Opt-in browser tests using the selected installed scientific resources."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from elisa import settings

if TYPE_CHECKING:
    import collections.abc

    import playwright.sync_api

    import tests.tests_gui.e2e


INSTALLED_CONFIG = str(Path(settings.CONFIG_FILE).resolve())

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts/gui-fitting-e2e"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--run-fitting-e2e"):
        for item in items:
            if Path(__file__).parent in Path(item.path).parents:
                item.add_marker(pytest.mark.skip(reason="use --run-fitting-e2e explicitly"))


@pytest.fixture(scope="session")
def application() -> collections.abc.Iterator[tests.tests_gui.e2e.Application]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="elisa-browser-e2e-") as directory:
        work = Path(directory)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {
            **os.environ,
            "ELISA_CONFIG": INSTALLED_CONFIG,
            "E2E_WORK": str(work),
            "MPLBACKEND": "Agg",
            "GRADIO_ANALYTICS_ENABLED": "False",
            "TMPDIR": str(work),
            "PYTHONUNBUFFERED": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        log_path = ARTIFACTS / "application.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("server.py")), str(port)],
                cwd=work,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                url = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        pytest.fail(log_path.read_text())
                    try:
                        with urllib.request.urlopen(url + "/config", timeout=1) as response:  # noqa: S310 - fixed localhost HTTP URL constructed by this fixture
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(0.1)  # readiness polling, never a fit-completion delay
                else:
                    pytest.fail("Application did not become ready: " + log_path.read_text())
                yield {
                    "url": url,
                    "work": work,
                    "manifest": json.loads((work / "manifest.json").read_text()),
                    "log": log_path,
                }
            finally:
                import signal

                os.killpg(process.pid, signal.SIGTERM) if process.poll() is None else None
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                for name in ("backend.jsonl", "manifest.json", *[p.name for p in work.glob("*-chain.json")]):
                    if (work / name).exists():
                        (ARTIFACTS / name).write_bytes((work / name).read_bytes())


@pytest.fixture(scope="session")
def browser() -> collections.abc.Iterator[playwright.sync_api.Browser]:
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        yield browser
        browser.close()
