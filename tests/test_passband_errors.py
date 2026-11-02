"""Fixed passband failures propagate instead of rejecting fitting candidates."""
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from elisa import Observer, settings
from elisa.analytics.binary_fit import least_squares, shared
from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.base.error import ElisaError, ElisaPassbandError, raise_elisa_passband_error
from elisa.observer.passband import PassbandContainer, init_bolometric_passband, init_rv_passband
from tests.tests_analytics.fitting_helpers import V, lc_setup, make_lc, normalized
from tests.tests_analytics.test_mcmc_posterior import make_quadratic_posterior


def response(wave=None, throughput=None):
    return pd.DataFrame({
        settings.PASSBAND_DATAFRAME_WAVE: [400.0, 500.0, 600.0] if wave is None else wave,
        settings.PASSBAND_DATAFRAME_THROUGHPUT: [0.0, 1.0, 0.0] if throughput is None else throughput,
    })


def test_passband_helper_preserves_cause_without_penalty_semantics():
    cause = FileNotFoundError("missing filter")
    with pytest.raises(ElisaPassbandError, match="bad passband") as caught:
        raise_elisa_passband_error("bad passband", cause)
    assert caught.value.__cause__ is cause
    assert isinstance(caught.value, ElisaError)
    assert not isinstance(caught.value, SYNTHESIS_FAILURES)
    assert type(pickle.loads(pickle.dumps(caught.value))) is ElisaPassbandError


@pytest.mark.parametrize("name", ["unknown", "", 123, None])
def test_unsupported_passband_uses_passband_category(name):
    with pytest.raises(ElisaPassbandError):
        PassbandContainer.get_passband_df(name)


@pytest.mark.parametrize("cause", [FileNotFoundError("missing"), PermissionError("denied"), pd.errors.ParserError("bad CSV")])
def test_unreadable_passband_source_preserves_cause(cause):
    with patch("elisa.observer.passband.pd.read_csv", side_effect=cause):
        with pytest.raises(ElisaPassbandError, match="Cannot read passband table") as caught:
            PassbandContainer.get_passband_df(V)
    assert caught.value.__cause__ is cause


@pytest.mark.parametrize("content", ["", "wrong,columns\n1,2", "wavelength,throughput\nabc,1\n500,0"])
def test_corrupt_csv_source_uses_passband_category(tmp_path, content):
    (tmp_path / f"{V}.csv").write_text(content)
    with settings.override(PASSBAND_TABLES=tmp_path):
        with pytest.raises(ElisaPassbandError):
            PassbandContainer.from_name(V)


@pytest.mark.parametrize(
    "frame",
    [
        response().drop(columns=["throughput"]),
        response(wave=[400.0, np.nan, 600.0]),
        response(wave=[400.0, 500.0, np.inf]),
        response(wave=[-1.0, 500.0, 600.0]),
        response(wave=[400.0, 400.0, 600.0]),
        response(wave=[600.0, 500.0, 400.0]),
        response(wave=[400.0], throughput=[1.0]),
        response(wave=[], throughput=[]),
        response(throughput=[0.0, np.nan, 0.0]),
        response(throughput=[0.0, np.inf, 0.0]),
        response(throughput=[0.0, 0.0, 0.0]),
    ],
)
def test_invalid_response_does_not_replace_a_working_passband(frame):
    table = response()
    band = PassbandContainer(table, "custom")
    interpolator = band.akima
    with pytest.raises(ElisaPassbandError):
        band.table = frame
    assert band.table is table
    assert band.akima is interpolator
    assert band.get_bandwidth() == (400.0, 600.0)


@pytest.mark.parametrize("interval", [(6000, 5000), (5000, 5000), (-1, 5000), (np.nan, 5000), (0, np.inf), (1,), (1, 2, 3)])
def test_invalid_fixed_rv_bounds_are_passband_errors(interval):
    with pytest.raises(ElisaPassbandError):
        init_rv_passband(interval)


def test_valid_passbands_preserve_units_interpolation_and_special_filters(tmp_path):
    table = response()
    table.to_csv(tmp_path / f"{V}.csv", index=False)
    with settings.override(PASSBAND_TABLES=tmp_path):
        band = PassbandContainer.from_name(V)
        np.testing.assert_array_equal(band.table["wavelength"], table["wavelength"] * 10.0)
        assert band.get_bandwidth() == (4000.0, 6000.0)
        np.testing.assert_allclose(band.akima([4000, 5000, 6000]), [0, 1, 0])
    bolometric, right, left = init_bolometric_passband()
    assert left == 0.0 and np.isfinite(right)
    np.testing.assert_array_equal(bolometric.akima(np.array([0.0, right])), [1, 1])
    rv, right, left = init_rv_passband((5000, 6000))
    assert (left, right) == rv.get_bandwidth() == (5000, 6000)


@pytest.mark.parametrize("bounds", [(5, 4), (4, 4), (np.nan, 4), (0, np.inf)])
def test_observer_rejects_invalid_bandwidth_before_updating(bounds):
    observer = Observer()
    original = (observer.left_bandwidth, observer.right_bandwidth)
    with pytest.raises(ElisaPassbandError):
        observer.setup_bandwidth(*bounds)
    assert (observer.left_bandwidth, observer.right_bandwidth) == original


@pytest.mark.parametrize("stage", ["ln_prior", "likelihood"])
def test_mcmc_passband_failure_propagates_without_rejection_diagnostics(stage):
    fit = make_quadratic_posterior()
    failure = ElisaPassbandError("fixed passband is invalid")
    before = fit._candidate_failure_diagnostics.total
    with patch.object(fit, stage, side_effect=failure), pytest.raises(ElisaPassbandError) as caught:
        fit.ln_probability(np.array([0.5]))
    assert caught.value is failure
    assert fit._candidate_failure_diagnostics.total == before


def test_lsq_and_adaptive_sampling_do_not_reclassify_passband_failures():
    fit = least_squares.DetachedLightCurveFit()
    lc_setup(fit, {V: make_lc(errors=[0.1, 0.1, 0.1])})
    failure = ElisaPassbandError("invalid fixed passband")
    with patch.object(fit, "predict_light_curves", side_effect=failure), pytest.raises(ElisaPassbandError) as caught:
        fit.model_to_fit(normalized(fit))
    assert caught.value is failure
    with (
        patch("elisa.analytics.models.lc.synthetic_binary", side_effect=failure),
        pytest.raises(ElisaPassbandError) as caught,
    ):
        fit.adaptive_sampling()
    assert caught.value is failure


def test_unknown_fitting_filter_is_a_passband_input_error():
    with pytest.raises(ElisaPassbandError):
        shared._fit_passbands(["unknown"], {V: make_lc()})


def test_signed_finite_response_samples_remain_supported():
    table = response(throughput=[-0.001, 1.0, -0.001])
    band = PassbandContainer(table, "custom")
    np.testing.assert_allclose(band.akima(table["wavelength"]), table["throughput"], rtol=1e-12, atol=1e-15)
