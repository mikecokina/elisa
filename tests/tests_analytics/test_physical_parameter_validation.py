"""Candidate models own physical validation; fitters do not rescan their inputs."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa import const
from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.binary_fit.shared import AbstractLCFit
from elisa.analytics.models import lc, rv
from elisa.analytics.params import parameters as p
from elisa.base.error import ElisaInitialParametersError, ElisaParametersError
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from elisa.binary_system.transform import RadialVelocityObserverProperties
from tests.tests_analytics.fitting_helpers import V, lc_parameters, make_lc, normalized, rv_problem


@pytest.fixture(params=[
    least_squares.DetachedLightCurveFit, mcmc.DetachedLightCurveFit,
    least_squares.CentralRadialVelocity, mcmc.CentralRadialVelocity,
])
def fitting_problem(request, monkeypatch):
    fit = request.param()
    if isinstance(fit, AbstractLCFit):
        config, _ = lc_parameters()
        data = {V: make_lc(errors=np.full(3, 0.1))}

        def synthesis(phases, discretization, observer, **kwargs):
            # Keep real serialization, community conversion, constructors and
            # property validation; only skip the surface/table curve calculation.
            lc.prepare_binary(discretization=discretization, **kwargs)
            return {V: np.ones(len(phases))}

        spy = Mock(side_effect=synthesis)
        monkeypatch.setattr(lc, "synthetic_binary", spy)
        original_predict = AbstractLCFit.predict_light_curves

        def predict(self, model_parameters, **kwargs):
            kwargs.setdefault("synthesis", spy)
            return original_predict(self, model_parameters, **kwargs)

        monkeypatch.setattr(AbstractLCFit, "predict_light_curves", predict)
    else:
        config, data = rv_problem()
        spy = Mock(wraps=rv.central_rv_synthetic)
        monkeypatch.setattr(rv, "central_rv_synthetic", spy)
    return fit, config, data, spy


def prepare_fit(fit, config, data):
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


def evaluate(fit, vector):
    if isinstance(fit, mcmc.MCMCFit):
        return fit.ln_probability(vector)
    if isinstance(fit, least_squares.LightCurveFit):
        return fit.model_to_fit(vector)
    return fit.central_rv_model_to_fit(vector)


def assert_evaluation(fit, vector, *, valid):
    actual = evaluate(fit, vector)
    if isinstance(fit, mcmc.MCMCFit):
        assert np.isfinite(actual) if valid else actual == -np.inf
    else:
        count = sum(fit.num_of_points.values())
        penalty = np.full(count, np.sqrt(const.MAX_USABLE_FLOAT / count))
        assert actual.shape == (count,)
        assert np.all(np.isfinite(actual))
        if valid:
            assert not np.array_equal(actual, penalty)
        else:
            assert_array_equal(actual, penalty)
            assert_allclose(np.dot(actual, actual), const.MAX_USABLE_FLOAT, rtol=2e-15)


@pytest.mark.parametrize("period", [-3.0, 0.0, np.nan, np.inf, -np.inf])
def test_invalid_fixed_period_is_rejected_in_the_initial_definition(fitting_problem, period):
    _, config, _, synthesis = fitting_problem
    config["system"]["period"] = {"value": period, "fixed": True}
    with pytest.raises(ElisaInitialParametersError, match="period"):
        p.BinaryInitialParameters(**config)
    synthesis.assert_not_called()


def test_free_period_candidates_reject_and_recover_at_the_model_boundary(fitting_problem):
    fit, config, data, synthesis = fitting_problem
    config["system"]["period"] = {"value": 3.0, "fixed": False, "min": 1.0, "max": 4.0}
    prepare_fit(fit, config, data)
    # Widen the internal normalization only to exercise invalid optimizer values.
    fit.normalization["system@period"] = (-4.0, 4.0)
    if isinstance(fit, mcmc.MCMCFit):
        fit.prepare_priors()
    vector = normalized(fit)
    index = list(fit.fitable).index("system@period")
    for period in (3.0, -3.0, 0.0, 2.0):
        vector[index] = p.normalize_value(period, *fit.normalization["system@period"])
        synthesis.reset_mock()
        assert_evaluation(fit, vector, valid=period > 0)
        # Synthesis enters the real model constructor; there is no fitter precheck.
        synthesis.assert_called_once()


@pytest.mark.parametrize("expression", ["-1", "0", "1e309", "-1e309", "log(-1)", "3"])
def test_constrained_period_is_validated_by_the_candidate_model(fitting_problem, expression):
    fit, config, data, synthesis = fitting_problem
    config["system"]["period"] = {"constraint": expression}
    with np.errstate(invalid="ignore"):
        prepare_fit(fit, config, data)
    with np.errstate(invalid="ignore"):
        assert_evaluation(fit, normalized(fit), valid=expression == "3")
    synthesis.assert_called_once()


def test_unexpected_constraint_arithmetic_keeps_the_fitter_failure_policy(fitting_problem):
    fit, config, data, synthesis = fitting_problem
    config["system"]["period"] = {"constraint": "3"}
    prepare_fit(fit, config, data)
    fit.constrained["system@period"].constraint = "1/0"
    if isinstance(fit, mcmc.MCMCFit):
        assert fit.ln_probability(normalized(fit)) == -np.inf
    else:
        with pytest.raises(ZeroDivisionError):
            evaluate(fit, normalized(fit))
    synthesis.assert_not_called()


@pytest.mark.parametrize("fit_class", [least_squares.CentralRadialVelocity, mcmc.CentralRadialVelocity])
@pytest.mark.parametrize("constrained", [False, True])
def test_asini_is_validated_once_per_rv_candidate(fit_class, constrained):
    config, data = rv_problem()
    if constrained:
        config["system"]["asini"] = {"constraint": "system@gamma / 1000"}
    fit = fit_class()
    prepare_fit(fit, config, data)
    with patch.object(RadialVelocityObserverProperties, "asini", wraps=RadialVelocityObserverProperties.asini) as validator:
        assert_evaluation(fit, normalized(fit), valid=True)
    validator.assert_called_once()


@pytest.mark.parametrize("fit_class", [least_squares.CentralRadialVelocity, mcmc.CentralRadialVelocity])
@pytest.mark.parametrize("parameter,expression", [
    ("asini", "-1"), ("asini", "1e309"),
    ("mass_ratio", "-1"), ("mass_ratio", "1e309"),
    ("eccentricity", "1"), ("eccentricity", "-1e309"),
])
def test_invalid_community_constraints_are_rejected_without_a_fitter_scan(fit_class, parameter, expression):
    config, data = rv_problem()
    config["system"][parameter] = {"constraint": expression}
    fit = fit_class()
    prepare_fit(fit, config, data)
    assert_evaluation(fit, normalized(fit), valid=False)
    # Direct synthesis still exposes the model's own exception category.
    physical = p.prepare_properties_set(fit.initial_vector, fit.fitable, fit.constrained, fit.fixed)
    with pytest.raises(ElisaParametersError, match=parameter):
        rv.central_rv_synthetic(fit.x_data_reduced, fit.observer, **physical)


@pytest.mark.parametrize("fitting_problem", [
    least_squares.DetachedLightCurveFit, mcmc.DetachedLightCurveFit,
], indirect=True)
@pytest.mark.parametrize("parameter,expression", [
    ("t_eff", "-1"), ("t_eff", "1e309"), ("t_eff", "log(-1)"),
    ("surface_potential", "-1"), ("surface_potential", "1e309"),
])
def test_invalid_component_constraints_are_rejected_by_star_properties(fitting_problem, parameter, expression):
    fit, config, data, synthesis = fitting_problem
    config["secondary"][parameter] = {"constraint": expression}
    with np.errstate(invalid="ignore"):
        prepare_fit(fit, config, data)
        assert_evaluation(fit, normalized(fit), valid=False)
    synthesis.assert_called_once()
