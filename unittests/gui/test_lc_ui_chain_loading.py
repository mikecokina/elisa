from __future__ import annotations

# ruff: noqa: S101
import json
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest

from elisa.ui.tabs.lc_fitting.logic import compute


class _DummyPlot:
    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def corner(self, *, truths: bool = True) -> None:
        del truths

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def traces(self, *, truths: bool = True) -> None:
        del truths


class _DummyTask:
    def __init__(self, data: dict, method: str) -> None:
        self.data = data
        self.method = method
        self.plot = _DummyPlot()
        self.fit_cls = SimpleNamespace(flat_result={})
        self.loaded_result_path: str | None = None
        self.loaded_chain_path: str | None = None

    def load_result(self, *, filename: str) -> None:
        self.loaded_result_path = filename

    # noinspection PyUnusedLocal
    def load_chain(self, filename: str, *, discard: int = 0) -> None:
        del discard
        self.loaded_chain_path = filename


def _write_json(path: object, payload: dict) -> None:
    path_obj = Path(str(path))
    path_obj.write_text(json.dumps(payload), encoding="utf-8")


def test_load_chain_raises_clear_error_when_flat_chain_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compute, "LCBinaryAnalyticsTask", _DummyTask)

    chain_path = tmp_path / "bad_chain.json"
    result_path = tmp_path / "result.json"
    _write_json(chain_path, {"system": {}})
    _write_json(result_path, {"system": {}})

    with pytest.raises(gr.Error, match="Invalid chain file: missing 'flat_chain' key"):
        compute.load_chain(str(chain_path), str(result_path))


def test_load_chain_raises_clear_error_when_result_missing_system(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compute, "LCBinaryAnalyticsTask", _DummyTask)

    chain_path = tmp_path / "chain.json"
    result_path = tmp_path / "bad_result.json"
    _write_json(chain_path, {"flat_chain": [[0.1, 0.2]]})
    _write_json(result_path, {"flat_chain": [[0.1, 0.2]]})

    with pytest.raises(gr.Error, match="Invalid result file: missing 'system' key"):
        compute.load_chain(str(chain_path), str(result_path))


def test_load_chain_swapped_files_are_handled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compute, "LCBinaryAnalyticsTask", _DummyTask)

    uploaded_chain_path = tmp_path / "uploaded_chain_slot.json"
    uploaded_result_path = tmp_path / "uploaded_result_slot.json"
    _write_json(uploaded_chain_path, {"system": {}})
    _write_json(uploaded_result_path, {"flat_chain": [[0.1, 0.2]]})

    task, *_ = compute.load_chain(str(uploaded_chain_path), str(uploaded_result_path))

    assert isinstance(task, _DummyTask)
    assert task.loaded_chain_path == str(uploaded_result_path)
    assert task.loaded_result_path == str(uploaded_chain_path)
