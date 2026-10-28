"""Shared physical scales at LC/RV setup and proposal rejection boundaries."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

from elisa import const
from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.binary_fit.shared import AbstractFit, AbstractLCFit
from elisa.analytics.models import rv
from elisa.analytics.params import parameters as p
from elisa.base.error import PhysicalStateError
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from tests.tests_analytics.fitting_helpers import V, lc_parameters, make_lc, normalized, rv_problem


@pytest.fixture(params=[
    least_squares.DetachedLightCurveFit,
    mcmc.DetachedLightCurveFit,
    least_squares.CentralRadialVelocity,
    mcmc.CentralRadialVelocity,
])
def fitting_problem(request: pytest.FixtureRequest) -> tuple[AbstractFit, dict[str, Any], dict[str, Any]]:
    fit = request.param()
    if isinstance(fit, AbstractLCFit):
        config, _ = lc_parameters()
        data = {V: make_lc(errors=np.full(3, 0.1))}
    else:
        config, data = rv_problem()
    return fit, config, data


def prepare_fit(fit: AbstractFit, config: dict[str, Any], data: dict[str, Any]) -> None:
    initial = p.BinaryInitialParameters(**config)
    if isinstance(fit, AbstractLCFit):
        fit.set_up(
            initial, data, passband=data.keys(), discretization=10.0,
            interp_treshold=100, samples="uniform", observer_system_cls=BinarySystem,
        )
    else:
        fit.set_up(initial, data, observer_system_cls=RadialVelocitySystem)
    if isinstance(fit, mcmc.MCMCFit):
        fit.prepare_priors()


def evaluate(fit: AbstractFit, vector: NDArray[np.float64]) -> float | NDArray[np.float64]:
    if isinstance(fit, mcmc.MCMCFit):
        return float(fit.ln_probability(vector))
    if isinstance(fit, least_squares.LightCurveFit):
        return fit.model_to_fit(vector)
    assert isinstance(fit, least_squares.CentralRadialVelocity)
    return fit.central_rv_model_to_fit(vector)


@pytest.fixture
def synthesis_spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Mock, Mock]]:
    # Track real RV synthesis; LC predictions avoid mesh/table work.
    with patch.object(rv, "central_rv_synthetic", wraps=rv.central_rv_synthetic) as rv_spy:
        lc_spy = Mock(return_value={V: np.ones(3)})
        monkeypatch.setattr(AbstractLCFit, "predict_light_curves", lc_spy)
        yield rv_spy, lc_spy


def assert_evaluation(fit: AbstractFit, vector: NDArray[np.float64], *, valid: bool) -> None:
    actual = evaluate(fit, vector)
    if isinstance(fit, mcmc.MCMCFit):
        if valid:
            assert np.isfinite(actual)
        else:
            assert actual == -np.inf
    else:
        count = sum(fit.num_of_points.values())
        penalty = np.full(count, np.sqrt(const.MAX_USABLE_FLOAT / count))
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (count,)
        assert np.all(np.isfinite(actual))
        if valid:
            assert not np.array_equal(actual, penalty)
        else:
            assert_array_equal(actual, penalty)
            assert_allclose(np.dot(actual, actual), const.MAX_USABLE_FLOAT, rtol=2e-15)


@pytest.mark.parametrize("period", [-3.0, 0.0, np.nan, np.inf, -np.inf])
def test_invalid_fixed_period_aborts_before_optimizer(fitting_problem, period: float) -> None:
    fit, config, data = fitting_problem
    config["system"]["period"] = {"value": period, "fixed": True}
    initial = p.BinaryInitialParameters(**config)
    error_type = PhysicalStateError if np.isfinite(period) else FloatingPointError
    solver = (
        patch.object(fit, "_fit") if isinstance(fit, mcmc.MCMCFit)
        else patch.object(least_squares, "least_squares")
    )
    with solver as optimizer, pytest.raises(error_type, match="period"):
        fit.fit(data=data, x0=initial)
    optimizer.assert_not_called()


def test_free_period_candidates_reject_and_recover(fitting_problem, synthesis_spy) -> None:
    fit, config, data = fitting_problem
    config["system"]["period"] = {"value": 3.0, "fixed": False, "min": -4.0, "max": 4.0}
    prepare_fit(fit, config, data)
    vector = normalized(fit)
    index = list(fit.fitable).index("system@period")
    synthesis = synthesis_spy[1 if isinstance(fit, AbstractLCFit) else 0]
    for period in (3.0, -3.0, 0.0, 2.0):
        vector[index] = p.normalize_value(period, *fit.normalization["system@period"])
        synthesis.reset_mock()
        assert_evaluation(fit, vector, valid=period > 0)
        assert synthesis.call_count == int(period > 0)


def test_constrained_period_candidates_reject_and_recover(fitting_problem, synthesis_spy) -> None:
    fit, config, data = fitting_problem
    is_lc = isinstance(fit, AbstractLCFit)
    dependency, centre = ("inclination", 85.0) if is_lc else ("asini", 9.0)
    config["system"]["period"] = {"constraint": f"system@{dependency} - {centre}"}
    prepare_fit(fit, config, data)
    vector = normalized(fit)
    key = f"system@{dependency}"
    index = list(fit.fitable).index(key)
    synthesis = synthesis_spy[1 if is_lc else 0]
    for period in (0.5, -0.5, 0.0, 1.0):
        vector[index] = p.normalize_value(centre + period, *fit.normalization[key])
        synthesis.reset_mock()
        assert_evaluation(fit, vector, valid=period > 0)
        assert synthesis.call_count == int(period > 0)
        if period > 0:
            supplied = synthesis.call_args.args[0] if is_lc else synthesis.call_args.kwargs
            assert supplied["system@period"] == period


@pytest.mark.parametrize("expression", ["1e309", "-1e309", "log(-1)"])
def test_nonfinite_constrained_period_propagates(fitting_problem, synthesis_spy, expression: str) -> None:
    fit, config, data = fitting_problem
    config["system"]["period"] = {"constraint": "3.0"}
    prepare_fit(fit, config, data)
    fit.constrained["system@period"].constraint = expression
    synthesis = synthesis_spy[1 if isinstance(fit, AbstractLCFit) else 0]
    with np.errstate(invalid="ignore"), pytest.raises(FloatingPointError, match="Nonfinite physical parameter"):
        evaluate(fit, normalized(fit))
    synthesis.assert_not_called()


def test_unexpected_constraint_arithmetic_propagates(fitting_problem, synthesis_spy) -> None:
    fit, config, data = fitting_problem
    config["system"]["period"] = {"constraint": "3.0"}
    prepare_fit(fit, config, data)
    fit.constrained["system@period"].constraint = "1/0"
    with pytest.raises(ZeroDivisionError):
        evaluate(fit, normalized(fit))
    for synthesis in synthesis_spy:
        synthesis.assert_not_called()


@pytest.mark.parametrize("key", [
    "system@period", "system@mass_ratio", "system@semi_major_axis", "system@asini",
    "primary@mass", "secondary@mass", "primary@t_eff", "secondary@t_eff",
])
def test_shared_positive_scale_boundaries(key: str) -> None:
    for value in (-1.0, 0.0):
        with pytest.raises(PhysicalStateError):
            AbstractFit.validate_physical_parameters({key: value})
    AbstractFit.validate_physical_parameters({key: np.nextafter(0.0, 1.0)})
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(FloatingPointError) as caught:
            AbstractFit.validate_physical_parameters({key: value})
        assert type(caught.value) is FloatingPointError


def test_shared_eccentricity_and_unrelated_parameters_keep_existing_rules() -> None:
    for value in (0.0, np.nextafter(1.0, 0.0)):
        AbstractFit.validate_physical_parameters({"system@eccentricity": value})
    for value in (-np.finfo(float).eps, 1.0):
        with pytest.raises(PhysicalStateError):
            AbstractFit.validate_physical_parameters({"system@eccentricity": value})
    AbstractFit.validate_physical_parameters({
        "primary@surface_potential": -1.0, "primary@synchronicity": 0.0, "system@gamma": np.nan,
    })
