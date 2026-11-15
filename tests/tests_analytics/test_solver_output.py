"""Solver diagnostics are opt-in; ELISA's own fitting messages remain enabled."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest

from elisa.analytics.binary_fit import least_squares as fitting

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_lc_solver_default_is_quiet_and_explicit_verbose_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fit = SimpleNamespace(
        set_up=lambda *_args, **_kwargs: None,
        initial_vector=[0.8],
        fitable={"x": None},
        normalization={"x": (0.0, 1.0)},
        MORPHOLOGY="detached",
        model_to_fit=lambda vector: np.array([vector[0] - 0.25]),
    )
    results = []
    progress = Mock()
    monkeypatch.setattr(fitting, "logger", progress)

    class SolverFinishedError(Exception):
        pass

    def finish_after_solver(vector: NDArray[np.float64], *_args) -> None:
        # End before unrelated binary result serialization. The actual SciPy
        # solver and ELISA's start/finish reporting have both executed.
        results.append(vector.copy())
        raise SolverFinishedError

    monkeypatch.setattr(fitting.parameters, "vector_renormalizer", finish_after_solver)
    for options in ({}, {"verbose": 2}):
        with pytest.raises(SolverFinishedError):
            fitting.LightCurveFit.fit(fit, data={}, x0=None, **options)
        output = capsys.readouterr().out
        if options:
            assert "Iteration" in output
            assert "Total nfev" in output
        else:
            assert "Iteration" not in output
            assert "Total nfev" not in output
        progress.info.assert_any_call("fitting started...")
        progress.info.assert_any_call("fitting finished")
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_allclose(results[0], [0.25], atol=1e-7)
