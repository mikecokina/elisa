"""Numerical candidate failures remain distinct from input and structural errors."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from elisa import const
from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.analytics.models import cost_fns
from elisa.base.error import (
    ElisaAnalyticsDataError,
    ElisaNumericError,
    ElisaMaxIterationError,
    ElisaPredictionAlignmentError,
    MaxIterationError,
    raise_elisa_numeric_error,
    raise_elisa_max_iteration_error,
)
from elisa.binary_system import radius as binary_radius
from elisa.binary_system.system import BinarySystem
from elisa.opt.newton import newton
from elisa.single_system import radius as single_radius
from elisa.single_system.system import SingleSystem


@pytest.mark.parametrize(
    ("helper", "category", "parent"),
    [
        (raise_elisa_numeric_error, ElisaNumericError, FloatingPointError),
        (raise_elisa_max_iteration_error, ElisaMaxIterationError, MaxIterationError),
    ],
)
def test_numerical_helpers_preserve_cause_and_penalty_category(helper, category, parent):
    cause = ArithmeticError("original solver diagnostic")
    with pytest.raises(parent, match="failed candidate") as caught:
        helper("failed candidate", cause)
    assert type(caught.value) is category
    assert caught.value.__cause__ is cause
    assert isinstance(caught.value, SYNTHESIS_FAILURES)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_prediction_is_numerical_failure(value):
    observed = {"V": np.zeros(2)}
    errors = {"V": np.ones(2)}
    with pytest.raises(ElisaNumericError):
        cost_fns.likelihood_fn(observed, errors, {"V": np.array([0.0, value])}, 0.0)


@pytest.mark.parametrize(
    ("observed", "errors"),
    [
        ({}, {}),
        ({"V": np.zeros(2)}, {}),
        ({"V": np.zeros((2, 1))}, {"V": np.ones(2)}),
        ({"V": np.array([0.0, np.nan])}, {"V": np.ones(2)}),
        ({"V": np.zeros(2)}, {"V": np.array([1.0, np.inf])}),
        ({"V": np.zeros(2)}, {"V": np.array([1.0, 0.0])}),
        ({"V": np.zeros(2)}, {"V": np.array([1.0, -1.0])}),
    ],
)
def test_invalid_observations_are_data_errors_not_candidate_failures(observed, errors):
    with pytest.raises(ElisaAnalyticsDataError) as caught:
        cost_fns.prepare_observations(observed, errors)
    assert not isinstance(caught.value, SYNTHESIS_FAILURES)


def test_wrong_prediction_shape_remains_alignment_failure():
    prepared = cost_fns.prepare_observations({"V": np.zeros(2)}, {"V": np.ones(2)})
    with pytest.raises(ElisaPredictionAlignmentError) as caught:
        cost_fns.validate_predictions(prepared, {"V": np.zeros((2, 1))})
    assert not isinstance(caught.value, SYNTHESIS_FAILURES)


@pytest.mark.parametrize("system", ["binary", "single"])
@pytest.mark.parametrize("status", [2, 3, 4, 5])
def test_radius_solver_cannot_return_unconverged_finite_root(system, status):
    module, args = radius_call(system)
    with patch.object(module, "fsolve", return_value=(np.array([0.5]), {}, status, "no convergence")):
        with pytest.raises(ElisaMaxIterationError):
            module.calculate_radius(*args)


@pytest.mark.parametrize("system", ["binary", "single"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.5])
def test_converged_radius_solver_must_return_usable_root(system, value):
    module, args = radius_call(system)
    with patch.object(module, "fsolve", return_value=(np.array([value]), {}, 1, "converged")):
        with pytest.raises(ElisaNumericError):
            module.calculate_radius(*args)


@pytest.mark.parametrize("system", ["binary", "single"])
def test_radius_solver_preserves_valid_result_and_unexpected_exceptions(system):
    module, args = radius_call(system)
    with patch.object(module, "fsolve", return_value=(np.array([0.5]), {}, 1, "converged")):
        assert module.calculate_radius(*args) == 0.5
    cause = TypeError("broken solver implementation")
    with patch.object(module, "fsolve", side_effect=cause), pytest.raises(TypeError) as caught:
        module.calculate_radius(*args)
    assert caught.value is cause


def radius_call(system):
    if system == "binary":
        return binary_radius, (1.0, 1.0, 5.0, "primary", 1.0, 0.0, 0.0)
    return single_radius, (1.0, 0.0, -const.G, 0.0)


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize(
    ("converged", "root", "category"),
    [(False, 0.5, ElisaMaxIterationError), (True, np.nan, ElisaNumericError)],
)
def test_binary_critical_solver_distinguishes_convergence_and_numerical_failure(component, converged, root, category):
    with patch("elisa.binary_system.system.optimize.newton", return_value=(root, SimpleNamespace(converged=converged))):
        with pytest.raises(category):
            BinarySystem.critical_potential_static(component, 1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("converged", "root", "category"),
    [(False, 0.5, ElisaMaxIterationError), (True, np.nan, ElisaNumericError)],
)
def test_single_critical_solver_distinguishes_convergence_and_numerical_failure(converged, root, category):
    system = SimpleNamespace(star=SimpleNamespace(mass=1.0, surface_potential=-1.0), angular_velocity=1.0)
    with patch("elisa.single_system.system.optimize.newton", return_value=(root, SimpleNamespace(converged=converged))):
        with pytest.raises(category):
            SingleSystem.calculate_critical_potential(system)


def test_newton_exhaustion_uses_elisa_convergence_category():
    with pytest.raises(ElisaMaxIterationError, match="iteration limit"):
        newton(lambda x: x, 1.0, lambda x: 1.0, maxiter=0)


@pytest.mark.parametrize("category", [TypeError, ValueError, RuntimeError])
def test_unexpected_lagrange_solver_failure_propagates(category):
    cause = category("broken Lagrange implementation")
    with patch("elisa.binary_system.system.fsolve", side_effect=cause), pytest.raises(category) as caught:
        BinarySystem._lagrangian_points_static(1.0, 1.0)
    assert caught.value is cause


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_standalone_residuals_reject_nonfinite_predictions(value):
    with pytest.raises(ElisaNumericError):
        cost_fns.standardized_residuals({"V": np.zeros(2)}, {"V": 1.0}, {"V": np.array([0.0, value])})


@pytest.mark.parametrize("error", [0.0, -1.0, np.nan, np.inf])
def test_standalone_residuals_reject_invalid_uncertainty_values(error):
    with pytest.raises(ElisaAnalyticsDataError):
        cost_fns.standardized_residuals({"V": np.zeros(2)}, {"V": error}, {"V": np.zeros(2)})


def test_invalid_observation_conversion_preserves_cause():
    with pytest.raises(ElisaAnalyticsDataError) as caught:
        cost_fns.prepare_observations({"V": ["not a number"]}, {"V": [1.0]})
    assert isinstance(caught.value.__cause__, ValueError)
