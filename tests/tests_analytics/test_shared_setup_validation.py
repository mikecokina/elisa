"""Explicit fitting setup failures keep their data or task configuration category."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from elisa import settings
from elisa import units as u
from elisa.analytics import LCData
from elisa.analytics.binary_fit import least_squares, shared
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.base.error import ElisaAnalyticsDataError, ElisaAnalyticsTaskError
from elisa.binary_system.curves.community import RadialVelocitySystem
from tests.tests_analytics.fitting_helpers import V, lc_setup, make_lc, rv_problem


def test_requested_passbands_must_match_observation_identifiers():
    data = {V: make_lc(errors=[0.1] * 3)}
    with pytest.raises(ElisaAnalyticsTaskError, match="passbands"):
        shared._fit_passbands(["Generic.Bessell.B"], data)
    assert shared._fit_passbands([V], data) == [V]
    assert shared._fit_passbands(None, data) == "bolometric"


def test_absent_observations_are_a_data_error():
    with pytest.raises(ElisaAnalyticsDataError, match="at least one"):
        shared._observation_coordinate_unit({})


@pytest.mark.parametrize("identifier", ["", None, 1])
def test_invalid_observation_identifiers_are_task_errors(identifier):
    with pytest.raises(ElisaAnalyticsTaskError, match="identifiers"):
        shared._observation_coordinate_unit({identifier: make_lc(errors=[0.1] * 3)})


def test_coordinate_units_are_checked_across_datasets():
    phases = make_lc(errors=[0.1] * 3)
    times = LCData(x_data=[2450000.0, 2450001.0], y_data=[1.0, 2.0], y_err=[0.1, 0.1],
                   x_unit=u.d, y_unit=u.dimensionless_unscaled)
    with pytest.raises(ElisaAnalyticsDataError, match="BJD or photometric phases"):
        shared._observation_coordinate_unit({"a": phases, "b": times})
    assert shared._observation_coordinate_unit({"a": phases}) == u.dimensionless_unscaled


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_normalization_fraction_is_task_configuration(fraction):
    with pytest.raises(ElisaAnalyticsTaskError, match="fraction"):
        shared._validate_lc_normalization({V: np.ones(3)}, "maximum", fraction)


@pytest.mark.parametrize("flux", [0.0, -1.0, np.nan, np.inf])
def test_unusable_mean_flux_is_a_data_error(flux):
    with np.errstate(all="ignore"), pytest.raises(ElisaAnalyticsDataError, match="mean flux"):
        shared._validate_lc_normalization({V: np.full(2, flux)}, "average", 1.0)


@pytest.mark.parametrize("flux,error", [
    (np.nan, 0.1), (np.inf, 0.1), (1.0, np.nan), (1.0, np.inf), (1.0, 0.0), (1.0, -0.1),
])
def test_normalized_values_and_uncertainties_use_the_data_category(flux, error):
    with pytest.raises(ElisaAnalyticsDataError, match="normalization"):
        shared._validated_normalized_errors({V: np.array([flux])}, {V: np.array([error])})


def test_valid_normalized_errors_are_preserved():
    errors = np.array([0.1, 0.2])
    result = shared._validated_normalized_errors({V: np.array([1.0, 2.0])}, {V: errors})
    assert result[V] is errors


@pytest.mark.parametrize("scale", [0.0, np.nan, np.inf])
def test_unusable_estimated_uncertainties_use_the_data_category(scale):
    fit = least_squares.DetachedLightCurveFit()
    with patch.object(type(fit), "MEAN_ERROR_FN", return_value=scale):
        with pytest.raises(ElisaAnalyticsDataError, match="Cannot estimate positive y_err"):
            fit._prepare_observation_errors({V: make_lc()})


def test_unexpected_estimation_failures_keep_their_original_exception():
    fit = least_squares.DetachedLightCurveFit()
    failure = RuntimeError("unexpected estimation failure")
    with patch.object(type(fit), "MEAN_ERROR_FN", side_effect=failure):
        with pytest.raises(RuntimeError) as caught:
            fit._prepare_observation_errors({V: make_lc()})
    assert caught.value is failure


@pytest.mark.parametrize("option", ["atmosphere_model", "limb_darkening_coefficients"])
@pytest.mark.parametrize("value", [[], "invalid", {"tertiary": {}}])
def test_configuration_format_and_components_use_the_task_category(option, value):
    fit = least_squares.DetachedLightCurveFit()
    fit.atmosphere_model = fit.limb_darkening_coefficients = None
    setattr(fit, option, value)
    with pytest.raises(ElisaAnalyticsTaskError):
        fit._validate_atmosphere_parameters()


@pytest.mark.parametrize("use_epoch", [False, True])
def test_time_and_initial_parameter_incompatibility_is_task_configuration(use_epoch):
    system = {"period": {"value": 1.0, "min": 0.5, "max": 2.0}}
    if use_epoch:
        system["primary_minimum_time"] = {"value": 2450000.0, "fixed": True}
        data = make_lc(errors=[0.1] * 3)
    else:
        data = LCData(x_data=[2450000.0, 2450001.0], y_data=[1.0, 2.0], y_err=[0.1, 0.1],
                      x_unit=u.d, y_unit=u.dimensionless_unscaled)
    with pytest.raises(ElisaAnalyticsTaskError, match="primary_minimum_time"):
        shared.AbstractFit._check_data_param_consistency(BinaryInitialParameters(system=system), {V: data})


def test_community_rv_rejects_an_incompatible_method_during_setup():
    config, data = rv_problem()
    fit = least_squares.CentralRadialVelocity()
    with patch.object(settings, "RV_METHOD", "radiometric"):
        with pytest.raises(ElisaAnalyticsTaskError, match="kinematic RV"):
            fit.set_up(BinaryInitialParameters(**config), data, observer_system_cls=RadialVelocitySystem)


@pytest.mark.parametrize("component", ["tertiary", "", 1])
def test_rv_setup_rejects_unknown_component_identifiers(component):
    config, data = rv_problem()
    with pytest.raises(ElisaAnalyticsTaskError, match="keyed by primary or secondary"):
        least_squares.CentralRadialVelocity().set_up(
            BinaryInitialParameters(**config), {component: data["primary"]}, observer_system_cls=RadialVelocitySystem,
        )


@pytest.mark.parametrize("samples", ["unknown", None, {}])
def test_invalid_sampling_selection_retains_its_conversion_cause(samples):
    fit = least_squares.DetachedLightCurveFit()
    lc_setup(fit, {V: make_lc(errors=[0.1] * 3)})
    fit.interp_treshold = 2
    with pytest.raises(ElisaAnalyticsTaskError, match="SamplingMethod.UNIFORM") as caught:
        fit.generate_sample_phases(samples)
    assert isinstance(caught.value.__cause__, (ValueError, TypeError))


def test_sampling_does_not_reclassify_unexpected_errors():
    fit = least_squares.DetachedLightCurveFit()
    lc_setup(fit, {V: make_lc(errors=[0.1] * 3)})
    fit.interp_treshold = 2
    failure = RuntimeError("unexpected sampling failure")
    with patch.object(shared, "SamplingMethod", Mock(side_effect=failure)):
        with pytest.raises(RuntimeError) as caught:
            fit.generate_sample_phases("uniform")
    assert caught.value is failure
