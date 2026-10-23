"""Solver diagnostics are opt-in; ELISA's own fitting messages remain enabled."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from elisa.analytics.binary_fit import least_squares as fitting


def test_lc_solver_default_is_quiet_and_explicit_verbose_is_preserved(monkeypatch, capsys):
    fit = SimpleNamespace(
        set_up=lambda *args, **kwargs: None,
        initial_vector=[0.8], fitable={'x': None}, normalization={'x': (0., 1.)},
        MORPHOLOGY='detached', model_to_fit=lambda vector: np.array([vector[0] - 0.25]),
    )
    results = []
    progress = Mock()
    monkeypatch.setattr(fitting, 'logger', progress)

    class SolverFinished(Exception):
        pass

    def finish_after_solver(vector, *args):
        # End before unrelated binary result serialization. The actual SciPy
        # solver and ELISA's start/finish reporting have both executed.
        results.append(vector.copy())
        raise SolverFinished

    monkeypatch.setattr(fitting.parameters, 'vector_renormalizer', finish_after_solver)
    for options in ({}, {'verbose': 2}):
        with pytest.raises(SolverFinished):
            fitting.LightCurveFit.fit(fit, data={}, x0=None, **options)
        output = capsys.readouterr().out
        if options:
            assert 'Iteration' in output and 'Total nfev' in output
        else:
            assert 'Iteration' not in output and 'Total nfev' not in output
        progress.info.assert_any_call('fitting started...')
        progress.info.assert_any_call('fitting finished')
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_allclose(results[0], [0.25], atol=1e-7)
