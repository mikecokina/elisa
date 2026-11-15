"""Operation boundaries contain failures and preserve validation messages."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Never

import gradio as gr
import pytest

from elisa.base.error import ElisaMaxIterationError, ElisaNumericError, ElisaParametersError, MorphologyError
from elisa.ui.tabs.lc_fitting.logic import compute as lc_fit
from elisa.ui.tabs.lc_modeling import tab as lc
from elisa.ui.tabs.rv_fitting.logic import compute as rv_fit
from elisa.ui.tabs.rv_modeling import tab as rv
from elisa.ui.tabs.system_visualization import tab as visualization


@pytest.fixture(params=[(lc, "run_lc", 6), (rv, "run_rv", 4), (visualization, "run_visualization", 6)])
def boundary(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Callable[[Exception], object]:
    module, operation, count = request.param

    def invoke(error: Exception) -> object:
        def fail(*_args: object) -> Never:
            raise error

        monkeypatch.setattr(module.compute, operation, fail)
        return module._make_handler(*((),) * count)()

    return invoke


@pytest.mark.parametrize("error_type", [ElisaNumericError, ElisaMaxIterationError])
def test_numerical_failure_has_actionable_message(
    boundary: Callable[[Exception], object],
    error_type: type[Exception],
) -> None:
    error = error_type("internal solver array details")
    with pytest.raises(gr.Error, match="Check the system parameters") as caught:
        boundary(error)
    assert caught.value.__cause__ is error
    assert "internal solver" not in str(caught.value)


@pytest.mark.parametrize("error_type", [ValueError, ElisaParametersError, MorphologyError])
def test_domain_validation_remains_visible(
    boundary: Callable[[Exception], object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(gr.Error, match="Invalid inclination"):
        boundary(error_type("Invalid inclination"))


@pytest.mark.parametrize("error_type", [TypeError, AttributeError, RuntimeError])
def test_programming_failure_is_displayed(boundary: Callable[[Exception], object], error_type: type[Exception]) -> None:
    error = error_type("implementation defect")
    with pytest.raises(gr.Error, match="implementation defect") as caught:
        boundary(error)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("payload", ["null", "42", '"system"', "[]"])
def test_result_loaders_reject_non_object_json(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(payload)
    for load in (lc_fit.detect_approach_from_json, lc_fit.load_params_from_json, rv_fit.load_params_from_json):
        with pytest.raises(ValueError, match=r"JSON object|missing 'system'"):
            load(str(path))


@pytest.mark.parametrize(
    ("compute", "task_name"),
    [(lc_fit, "LCBinaryAnalyticsTask"), (rv_fit, "RVBinaryAnalyticsTask")],
)
def test_chain_plot_programming_error_warns_and_returns_task(
    compute: ModuleType,
    task_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail(**_kwargs: object) -> Never:
        msg = "plot implementation defect"
        raise AttributeError(msg)

    task = SimpleNamespace(
        load_result=lambda **_kwargs: None,
        load_chain=lambda *_args, **_kwargs: None,
        plot=SimpleNamespace(corner=fail, traces=lambda **_kwargs: None),
        fit_cls=SimpleNamespace(flat_result={}),
    )
    monkeypatch.setattr(compute, task_name, lambda **_kwargs: task)
    result = tmp_path / "result.json"
    chain = tmp_path / "chain.json"
    result.write_text('{"system": {}}')
    chain.write_text('{"flat_chain": [[1, 2]]}')
    with pytest.warns(UserWarning, match="plot implementation defect"):
        loaded, corner, traces, _ = compute.load_chain(str(chain), str(result))
    assert loaded is task
    assert corner is None
    assert traces is None
