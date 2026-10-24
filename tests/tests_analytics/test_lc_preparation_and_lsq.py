"""Prepared LC mappings and residual-vector LS tests using real LC/RV synthesis."""
from copy import deepcopy
from contextlib import contextmanager
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import OptimizeResult, least_squares as solve

from elisa import const, units as u
from elisa.analytics import LCData, RVData, SamplingMethod
from elisa.analytics.binary_fit import lc_prediction, least_squares, mcmc
from elisa.analytics.binary_fit.shared import lc_r_squared
from elisa.analytics.models import cost_fns, rv
from elisa.analytics.params import parameters as p
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from elisa.observer.observer import Observer
from tests.tests_analytics import test_lc_prediction_consistency as scientific


@pytest.fixture
def science():
    fixture = scientific.LCScientificConsistencyTestCase()
    try:
        fixture.setUp()
        yield fixture
    finally:
        fixture.doCleanups()


@contextmanager
def validation_calls():
    """Count actual static validation calls without replacing validation."""
    calls = []
    previous = sys.getprofile()
    def profile(frame, event, result):
        if event == 'call' and frame.f_code is lc_prediction.validate_observations.__code__:
            calls.append(frame.f_locals['xs'])
    try:
        sys.setprofile(profile)
        yield calls
    finally:
        sys.setprofile(previous)


def normalized(fit):
    return np.asarray(p.vector_normalizer(fit.initial_vector, fit.fitable.keys(), fit.normalization))


def physical(fit):
    return p.prepare_properties_set(fit.initial_vector, fit.fitable.keys(), fit.constrained, fit.fixed)


def lc_problem(science):
    """Noiseless two-band LC with two free values and a temperature constraint."""
    cfg, flat = science.physical()
    cfg['system']['inclination']['value'] = 83.
    cfg['primary']['t_eff']['value'] = 6300.
    data = {}
    for band, xs in {scientific.V: np.linspace(0., .98, 31),
                     scientific.B: np.array([.5, .02, .12, .25, .47, 0., .6, .8, .02])}.items():
        flux = science.synth(xs, [band], flat)[band]
        data[band] = LCData(x_data=xs, y_data=flux, y_err=np.full(len(xs), .002),
                            x_unit=u.dimensionless_unscaled, y_unit=u.dimensionless_unscaled)
    return cfg, data


def rv_problem(constrained=False, outlier=False):
    """Independent real kinematic RV observations with different component grids."""
    truth = dict(eccentricity=.15, argument_of_periastron=25., period=3.,
                 mass_ratio=.7, asini=10., gamma=12000.)
    flat = {f'system@{key}': value for key, value in truth.items()}
    cfg = {'system': {key: dict(value=value, fixed=True) for key, value in truth.items()}}
    cfg['system']['gamma'] = dict(value=9000., fixed=False, min=0., max=25000.)
    if not outlier:
        cfg['system']['asini'] = dict(value=9., fixed=False, min=8., max=12.)
        cfg['system']['mass_ratio'] = ({'constraint': 'system@asini * 0.07'} if constrained else
                                      dict(value=.8, fixed=False, min=.5, max=1.))
    data = {}
    for component, xs in {'secondary': np.array([.7, 0., .12, .3, .5, .12]),
                           'primary': np.linspace(0., .98, 21)}.items():
        obs = Observer(passband='bolometric', system=None)
        obs.system_cls = RadialVelocitySystem
        curve = rv.central_rv_synthetic(xs, obs, **flat)[component]
        if outlier and component == 'primary':
            curve = curve.copy()
            curve[5] += 20000.
        data[component] = RVData(x_data=xs, y_data=curve, y_err=np.full(len(xs), 100.),
                                 x_unit=u.dimensionless_unscaled, y_unit=u.m/u.s)
    return cfg, data


def capture_fit(callback, *args, **kwargs):
    """Capture actual SciPy diagnostics while running ELISA's full fit/report path."""
    results = []
    previous = sys.getprofile()
    def profile(frame, event, result):
        if event == 'return' and isinstance(result, OptimizeResult) and 'jac' in result and 'cost' in result:
            results.append(result)
    try:
        sys.setprofile(profile)
        report = callback(*args, **kwargs)
    finally:
        sys.setprofile(previous)
    assert results
    return report, results[-1]


def test_static_validation_once_per_setup_and_reused_by_evaluations(science):
    with validation_calls() as calls:
        fits = science.prepare({scientific.V: scientific.X, scientific.B: [.25]})
        assert len(calls) == 2
        for fit in fits:
            mappings = fit._validated_reducers
            arrays = tuple(mappings.values())
            obs = fit.observer
            for _ in range(3):
                if isinstance(fit, mcmc.LightCurveFit):
                    fit.likelihood(normalized(fit))
                else:
                    fit.model_to_fit(normalized(fit))
                fit.prediction_r_squared(physical(fit))
            assert fit.observer is obs
            assert fit._validated_reducers is mappings
            assert all(a is b for a, b in zip(arrays, mappings.values()))
        assert len(calls) == 2
        # Standalone public API still validates on every call.
        fit = fits[0]
        score = lc_r_squared(scientific.lc.synthetic_binary, fit.x_data_reduced, fit.y_data,
                             fit.observer.passband, fit.discretization, fit.x_data_reducer,
                             .01, fit.interp_treshold, BinarySystem, **physical(fit))
        assert_allclose(score, fit.prediction_r_squared(physical(fit)), atol=2e-12)
        assert len(calls) == 3


def test_second_setup_replaces_and_validates_data(science):
    fits = science.prepare({scientific.V: scientific.X})
    cfg, data = lc_problem(science)
    with validation_calls() as calls:
        for fit in fits:
            old = fit._validated_reducers
            fit.set_up(p.BinaryInitialParameters(**deepcopy(cfg)), data, passband=data.keys(),
                       discretization=10., interp_treshold=100, observer_system_cls=BinarySystem,
                       samples='uniform')
            assert fit._validated_reducers is not old
            assert set(fit._validated_reducers) == set(data)
            predictions = fit.predict_light_curves(physical(fit))
            assert {b: len(y) for b, y in predictions.items()} == {b: len(d.y_data) for b, d in data.items()}
        assert len(calls) == 2
        data[scientific.V].y_err = np.ones(2)
        for fit in fits:
            with pytest.raises(lc_prediction.PredictionAlignmentError, match='uncertainty length'):
                fit.set_up(p.BinaryInitialParameters(**deepcopy(cfg)), data, passband=data.keys(),
                           discretization=10., interp_treshold=100, observer_system_cls=BinarySystem,
                           samples='uniform')
        assert len(calls) == 4


def test_normalization_and_explicit_manual_revalidation(science):
    fit = science.prepare({scientific.V: scientific.X})[0]
    with validation_calls() as calls:
        fit.normalize_data(kind='average')
        assert len(calls) == 1
        before = fit.predict_light_curves(physical(fit))
        # Corresponding edits remain usable after the explicit revalidation step.
        fit.x_data[scientific.V] = fit.x_data[scientific.V][::-1]
        fit.y_data[scientific.V] = fit.y_data[scientific.V][::-1]
        fit.y_err[scientific.V] = fit.y_err[scientific.V][::-1]
        fit.x_data_reducer[scientific.V] = fit.x_data_reducer[scientific.V][::-1]
        fit.validate_observations()
        assert len(calls) == 2
        after = fit.predict_light_curves(physical(fit))
        assert_allclose(after[scientific.V], before[scientific.V][::-1], atol=2e-12)
        assert len(calls) == 2


@pytest.mark.parametrize('defect', ['short', 'column', 'passband'])
def test_dynamic_malformed_real_synthesis_output_is_rejected(science, defect):
    fit = science.prepare({scientific.V: scientific.X})[0]
    def malformed(*args, **kwargs):
        raw = scientific.lc.synthetic_binary(*args, **kwargs)
        if defect == 'passband':
            return {'other': raw[scientific.V]}
        raw[scientific.V] = raw[scientific.V][:-1] if defect == 'short' else raw[scientific.V][:, None]
        return raw
    with validation_calls() as calls, pytest.raises(lc_prediction.PredictionAlignmentError):
        fit.predict_light_curves(physical(fit), synthesis=malformed)
    assert not calls


def test_lc_parameter_recovery_result_and_residual_jacobian(science):
    cfg, data = lc_problem(science)
    fit = least_squares.DetachedLightCurveFit()
    fit.atmosphere_model = fit.limb_darkening_coefficients = None
    report, result = capture_fit(fit.fit, data, p.BinaryInitialParameters(**cfg),
                                 discretization=10., interp_treshold=100,
                                 ftol=1e-10, gtol=1e-9, max_nfev=80)
    assert result.success
    assert result.jac.shape == (40, 2)
    assert np.linalg.matrix_rank(result.jac) == 2
    assert_allclose(result.cost, .5*np.dot(result.fun, result.fun), atol=1e-12)
    assert_allclose(report['system']['inclination']['value'], 85., atol=.03)
    assert_allclose(report['primary']['t_eff']['value'], 6500., atol=5.)
    assert_allclose(report['secondary']['t_eff']['value'],
                    report['primary']['t_eff']['value']*0.7692307692307693, atol=1e-10)
    assert report['system']['period']['value'] == 3.
    assert report['system']['period']['fixed']
    assert report['secondary']['t_eff']['constraint'] == 'primary@t_eff * 0.7692307692307693'
    assert_allclose(report['r_squared']['value'], 1., atol=1e-8)
    assert_allclose(fit.model_to_fit(result.x), result.fun, atol=2e-12)


@pytest.mark.parametrize('constrained', [False, True])
def test_rv_residuals_parameter_recovery_and_results(constrained):
    cfg, data = rv_problem(constrained)
    fit = least_squares.CentralRadialVelocity()
    fit.set_up(p.BinaryInitialParameters(**deepcopy(cfg)), data, observer_system_cls=RadialVelocitySystem)
    predictions = fit.prepare_synthetic(normalized(fit))
    expected = np.concatenate([(d.y_data-predictions[b][fit.x_data_reducer[b]])/d.y_err
                               for b, d in data.items()])
    assert_allclose(fit.central_rv_model_to_fit(normalized(fit)), expected, atol=1e-12)
    mapped = {b: predictions[b][fit.x_data_reducer[b]] for b in data}
    assert_allclose(np.dot(expected, expected), cost_fns.wssr(fit.y_data, fit.y_err, mapped), rtol=2e-15)
    report, result = capture_fit(fit.fit, data, p.BinaryInitialParameters(**cfg),
                                 ftol=1e-12, gtol=1e-10)
    assert result.success
    assert result.jac.shape == (27, 2 if constrained else 3)
    gamma_column = list(fit.fitable).index('system@gamma')
    assert_allclose(result.jac[:, gamma_column], -250., atol=2e-5)
    assert_allclose(result.cost, .5*np.dot(result.fun, result.fun), atol=1e-12)
    for key, value in dict(asini=10., mass_ratio=.7, gamma=12000.).items():
        assert_allclose(report['system'][key]['value'], value, atol=1e-7)
    assert report['system']['period']['fixed']
    if constrained:
        assert report['system']['mass_ratio']['constraint'] == 'system@asini * 0.07'
    assert_allclose(report['r_squared']['value'], 1., atol=1e-12)
    assert_allclose(fit.central_rv_model_to_fit(result.x), result.fun, atol=1e-10)


def test_robust_rv_loss_acts_on_each_standardized_observation():
    cfg, data = rv_problem(outlier=True)
    records = {}
    for loss in ('linear', 'soft_l1'):
        fit = least_squares.CentralRadialVelocity()
        report, result = capture_fit(fit.fit, data, p.BinaryInitialParameters(**deepcopy(cfg)), loss=loss)
        expected = (.5*np.dot(result.fun, result.fun) if loss == 'linear' else
                    np.sum(np.sqrt(1+result.fun**2)-1))
        assert_allclose(result.cost, expected, rtol=2e-12)
        records[loss] = report['system']['gamma']['value']
    assert_allclose(records['linear'], 12000.+20000./27, atol=.01)
    assert abs(records['soft_l1']-12000.) < 10.


def test_lc_invalid_penalty_has_fixed_length_and_finite_linear_cost(science):
    fit = science.prepare({scientific.V: scientific.X, scientific.B: [.25]})[0]
    xn = normalized(fit)
    fit.fixed['system@eccentricity'].value = 1.1
    residuals = fit.model_to_fit(xn)
    assert residuals.shape == (9,)
    assert np.all(np.isfinite(residuals))
    assert_allclose(np.dot(residuals, residuals), const.MAX_USABLE_FLOAT, rtol=2e-15)
    result = solve(fit.model_to_fit, xn, bounds=(0, 1), max_nfev=2)
    assert np.isfinite(result.cost)
    assert_allclose(result.cost, .5*const.MAX_USABLE_FLOAT, rtol=2e-15)
    # A flat invalid region can still satisfy gtol, as the old penalty intended.
    assert result.jac.shape == (9, len(xn))


def test_residual_order_scalar_uncertainty_and_broadcasting_guard():
    observed = {'b': np.array([3., 5.]), 'a': np.array([7.])}
    model = {'a': np.array([5.]), 'b': np.array([1., 1.])}
    errors = {'a': np.array([.5]), 'b': np.array(2.)}
    assert_array_equal(cost_fns.standardized_residuals(observed, errors, model), [1., 2., 4.])
    with pytest.raises(ValueError, match='prediction shape'):
        cost_fns.standardized_residuals(observed, errors, {**model, 'a': np.ones(3)})


def test_rv_invalid_physical_synthesis_still_raises():
    cfg, data = rv_problem()
    fit = least_squares.CentralRadialVelocity()
    fit.set_up(p.BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    fit.fixed['system@eccentricity'].value = 1.1
    with pytest.raises(ValueError):
        fit.central_rv_model_to_fit(normalized(fit))


@pytest.mark.parametrize('strategy', list(SamplingMethod))
def test_sampling_enum_matches_legacy_strings_for_both_fitters(science, strategy):
    enum_fits = science.prepare({scientific.V: scientific.X}, 4, samples=strategy)
    legacy_fits = science.prepare({scientific.V: scientific.X}, 4, samples=strategy.value)
    for enum_fit, legacy_fit in zip(enum_fits, legacy_fits):
        assert_array_equal(enum_fit.fit_xs, legacy_fit.fit_xs)
        enum_prediction = enum_fit.predict_light_curves(physical(enum_fit))
        legacy_prediction = legacy_fit.predict_light_curves(physical(legacy_fit))
        assert_array_equal(enum_prediction[scientific.V], legacy_prediction[scientific.V])


def test_sampling_enum_defaults_serialization_and_invalid_strategy(science):
    import inspect
    import json
    import pickle

    assert inspect.signature(least_squares.LightCurveFit.fit).parameters['samples'].default is SamplingMethod.UNIFORM
    assert inspect.signature(mcmc.LightCurveFit.fit).parameters['samples'].default is SamplingMethod.UNIFORM
    assert json.dumps(SamplingMethod.ADAPTIVE) == '"adaptive"'
    assert str(SamplingMethod.UNIFORM) == 'uniform'
    assert pickle.loads(pickle.dumps(SamplingMethod.ADAPTIVE)) is SamplingMethod.ADAPTIVE
    with pytest.raises(ValueError, match='SamplingMethod.UNIFORM'):
        science.prepare({scientific.V: scientific.X}, 4, samples='unknown')
