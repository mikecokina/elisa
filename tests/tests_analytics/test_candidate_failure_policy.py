"""Callback mechanics are isolated; RV numeric/state regressions use real synthesis."""

from types import SimpleNamespace

import numpy as np
import pytest

from elisa import const, settings, units as u
from elisa.analytics import RVData
from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.binary_fit.failure_policy import CandidateFailureDiagnostics, SYNTHESIS_FAILURES
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.base import error
from elisa.binary_system.curves.community import RadialVelocitySystem
from tests.tests_analytics.fitting_helpers import normalized, rv_problem


def callbacks(monkeypatch, exc):
    """Inject only at callback boundaries to verify catch mechanics, not model reachability."""

    def fail(*args, **kwargs):
        raise exc

    fit = SimpleNamespace(
        fitable={},
        normalization={},
        prepare_synthetic=fail,
        num_of_points={"primary": 3},
        ln_prior=lambda *args: 0.0,
        norm_init_vector=np.array([]),
        sigmas=np.array([]),
        likelihood=fail,
        _candidate_failure_diagnostics=CandidateFailureDiagnostics(),
    )
    monkeypatch.setattr(least_squares.parameters, "vector_renormalizer", fail)
    return fit, (
        least_squares.LightCurveFit.model_to_fit,
        least_squares.CentralRadialVelocity.central_rv_model_to_fit,
        mcmc.MCMCFit.ln_probability,
    )


def assert_penalty(result, index):
    if index == 2:
        assert result == -np.inf
    else:
        assert result.shape == (3,)
        assert np.isfinite(result).all()
        assert np.dot(result, result) == pytest.approx(const.MAX_USABLE_FLOAT)


@pytest.mark.parametrize("kind", SYNTHESIS_FAILURES)
def test_recognized_failures_rejected_by_all_callbacks(monkeypatch, kind):
    fit, calls = callbacks(monkeypatch, kind("candidate"))
    for enabled in (False, True):
        with settings.override(PENALIZE_ALL_EXCEPTIONS=enabled, SUPPRESS_WARNINGS=True):
            for index, call in enumerate(calls):
                assert_penalty(call(fit, np.array([])), index)


@pytest.mark.parametrize(
    "kind",
    [
        error.ElisaAnalyticsTaskError,
        error.ElisaAnalyticsDataError,
        error.ElisaInitialParametersError,
        error.ElisaOperationInputError,
        error.ElisaSettingsError,
        error.ElisaPassbandError,
        error.ElisaInvalidKwargsError,
        error.ElisaMissingKwargsError,
        error.ElisaPredictionAlignmentError,
        error.ElisaUnexpectedError,
        error.ElisaMonteCarloError,
        error.ElisaGraphicsError,
        error.ElisaNotImplementedError,
        error.ElisaNumericSolverError,
        ValueError,
    ],
)
def test_other_exceptions_follow_override(monkeypatch, kind):
    exc = kind("original")
    fit, calls = callbacks(monkeypatch, exc)
    with settings.override(PENALIZE_ALL_EXCEPTIONS=False):
        for call in calls:
            with pytest.raises(kind) as caught:
                call(fit, np.array([]))
            assert caught.value is exc
    with settings.override(PENALIZE_ALL_EXCEPTIONS=True, SUPPRESS_WARNINGS=True):
        for index, call in enumerate(calls):
            assert_penalty(call(fit, np.array([])), index)
        assert fit._candidate_failure_diagnostics.last["message"] == "original"
    # The same callback instance returns to strict propagation after override exits.
    with settings.override(PENALIZE_ALL_EXCEPTIONS=False):
        for call in calls:
            with pytest.raises(kind):
                call(fit, np.array([]))


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_process_control_never_rejected(monkeypatch, kind):
    fit, calls = callbacks(monkeypatch, kind())
    for enabled in (False, True):
        with settings.override(PENALIZE_ALL_EXCEPTIONS=enabled):
            for call in calls:
                with pytest.raises(kind):
                    call(fit, np.array([]))


def test_lookup_translation_and_emergency_original_diagnostic(monkeypatch):
    marker = error.ElisaLookupError("lookup context")
    fit, calls = callbacks(monkeypatch, marker)
    with settings.override(PENALIZE_ALL_EXCEPTIONS=False):
        for call in calls:
            with pytest.raises(error.ElisaUnexpectedError) as caught:
                call(fit, np.array([]))
            assert caught.value.__cause__ is marker
    with settings.override(PENALIZE_ALL_EXCEPTIONS=True, SUPPRESS_WARNINGS=True):
        for index, call in enumerate(calls):
            assert_penalty(call(fit, np.array([])), index)
        assert "ElisaLookupError" in fit._candidate_failure_diagnostics.last["traceback"]


@pytest.mark.parametrize("uncertainty", [1e-310, 1e-160])
def test_real_rv_residual_overflow_is_penalized(uncertainty):
    cfg, original = rv_problem()
    data = {
        k: RVData(
            x_data=d.x_data,
            y_data=d.y_data,
            y_err=np.full(d.y_data.shape, uncertainty),
            x_unit=u.dimensionless_unscaled,
            y_unit=u.m / u.s,
        )
        for k, d in original.items()
    }
    for enabled in (False, True):
        with settings.override(PENALIZE_ALL_EXCEPTIONS=enabled, SUPPRESS_WARNINGS=True, RV_METHOD="kinematic"):
            fit = least_squares.CentralRadialVelocity()
            fit.set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
            result = fit.central_rv_model_to_fit(normalized(fit))
            assert result.shape == (27,)
            assert np.isfinite(result).all()
            assert np.dot(result, result) == pytest.approx(const.MAX_USABLE_FLOAT)


def test_real_rv_valid_invalid_valid_and_setup():
    cfg, data = rv_problem()
    cfg["system"]["mass_ratio"] = {"constraint": "1 / (system@asini - 10)**2"}
    cfg["nuisance"] = {"ln_f": {"value": -2.0, "fixed": True}}
    for enabled in (False, True):
        with settings.override(PENALIZE_ALL_EXCEPTIONS=enabled, SUPPRESS_WARNINGS=True, RV_METHOD="kinematic"):
            for cls in (least_squares.CentralRadialVelocity, mcmc.CentralRadialVelocity):
                fit = cls()
                fit.set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
                x = normalized(fit)
                bad = x.copy()
                bad[list(fit.fitable).index("system@asini")] = 0.5
                if isinstance(fit, mcmc.MCMCFit):
                    fit.prepare_priors()
                    call = fit.ln_probability
                else:
                    call = fit.central_rv_model_to_fit
                before = call(x)
                assert np.isfinite(before).all()
                invalid = call(bad)
                if isinstance(fit, mcmc.MCMCFit):
                    assert invalid == -np.inf
                else:
                    assert np.dot(invalid, invalid) == pytest.approx(const.MAX_USABLE_FLOAT)
                np.testing.assert_array_equal(call(x), before)
                with pytest.raises(error.ElisaAnalyticsTaskError):
                    cls().set_up(
                        BinaryInitialParameters(**cfg),
                        {"unknown": data["primary"]},
                        observer_system_cls=RadialVelocitySystem,
                    )


def test_setting_ini_default_validation_and_restore():
    from configparser import ConfigParser
    from elisa.conf.settings import Settings
    from elisa.managers.settings_manager import DefaultSettings

    assert DefaultSettings.PENALIZE_ALL_EXCEPTIONS is False
    original = settings.PENALIZE_ALL_EXCEPTIONS
    with settings.override(PENALIZE_ALL_EXCEPTIONS=False):
        parser = ConfigParser()
        parser.read_string("[computational]\npenalize_all_exceptions = yes\n")
        Settings().update_config(parser)
        assert settings.PENALIZE_ALL_EXCEPTIONS is True
        parser.read_string("[computational]\npenalize_all_exceptions = no\n")
        Settings().update_config(parser)
        assert settings.PENALIZE_ALL_EXCEPTIONS is False
        with pytest.raises(error.ElisaSettingsError):
            settings.configure(PENALIZE_ALL_EXCEPTIONS="yes")
    assert settings.PENALIZE_ALL_EXCEPTIONS is original


@pytest.mark.parametrize("enabled", [False, True])
def test_lsq_penalty_warning_remains_concise(monkeypatch, enabled):
    from unittest.mock import Mock

    warning = Mock()
    monkeypatch.setattr(least_squares.logger, "warning", warning)
    fit, calls = callbacks(monkeypatch, error.ElisaNumericError("unusable candidate"))
    with settings.override(PENALIZE_ALL_EXCEPTIONS=enabled, SUPPRESS_WARNINGS=False):
        for call in calls[:2]:
            call(fit, np.array([]))
    assert warning.call_count == 2
    for call in warning.call_args_list:
        assert call.args == ("lsqrt hit invalid parameters, exception: unusable candidate",)
        assert call.kwargs == {}


def test_public_residual_cost_accepts_representable_large_norm():
    from elisa.analytics.models.cost_fns import standardized_residuals

    # One large residual requires the scaled fallback but its squared norm fits.
    observed = np.array([1e154, 0.0, 0.0, 0.0])
    result = standardized_residuals(
        {"primary": observed},
        {"primary": np.ones(4)},
        {"primary": np.zeros(4)},
    )
    np.testing.assert_array_equal(result, observed)
    assert np.dot(result, result) == pytest.approx(1e308)
