"""Phase 1 scientific expectations and fitter preparation compatibility."""
from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa import units as u
from elisa.analytics import LCData, RVData
from elisa.analytics.binary_fit import least_squares, mcmc, shared
from elisa.analytics.dataset.utils import convert_flux_error
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.analytics.tools.utils import radialcurves_mean_error
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from tests.tests_analytics.test_lc_prediction_consistency import BASE, V
from tests.tests_analytics.test_lc_preparation_and_lsq import rv_problem, normalized

LC_CLASSES = (least_squares.DetachedLightCurveFit, mcmc.DetachedLightCurveFit)
RV_CLASSES = (least_squares.CentralRadialVelocity, mcmc.CentralRadialVelocity)


def lc_setup(fit, data):
    cfg = {group: {key: dict(value=value, fixed=True) for key, value in values.items()}
           for group, values in deepcopy(BASE).items()}
    cfg['system']['inclination'] = dict(value=85., fixed=False, min=80., max=90.)
    fit.set_up(BinaryInitialParameters(**cfg), data, passband=data.keys(), discretization=10.,
               interp_treshold=100, samples='uniform', observer_system_cls=BinarySystem)


def make_lc(values=(2., 4., 3.), errors=None, **kwargs):
    return LCData(x_data=np.arange(len(values)) / 4., y_data=values, y_err=errors,
                  x_unit=u.dimensionless_unscaled, y_unit=kwargs.pop('y_unit', u.dimensionless_unscaled), **kwargs)


@pytest.mark.parametrize('unit,factor', [(u.mag, 1.), (u.mmag, 1000.)])
def test_pointwise_magnitude_uncertainties(unit, factor):
    magnitudes = np.array([9., 10., 11.])
    sigma_mag = np.array([.01, .02, .03])
    data = make_lc(factor * magnitudes, factor * sigma_mag, y_unit=unit, reference_magnitude=10.)
    expected_flux = 10. ** ((10. - magnitudes) / 2.5)
    assert_allclose(data.y_data, expected_flux, rtol=1e-14)
    assert_allclose(data.y_err, np.log(10.) / 2.5 * expected_flux * sigma_mag, rtol=1e-14)
    missing = make_lc(factor * magnitudes, y_unit=unit, reference_magnitude=10.)
    assert missing.y_err is None
    for cls in LC_CLASSES:
        fit = cls()
        lc_setup(fit, {V: missing})
        assert_allclose(fit.y_err[V], .05)


def test_magnitude_zero_point_alone_cannot_determine_flux_errors():
    with pytest.raises(ValueError, match='pointwise'):
        convert_flux_error([.01], u.mag, zero_point=10.)


@pytest.mark.parametrize('cls', LC_CLASSES)
@pytest.mark.parametrize('errors', [None, [.02, .04, .03]])
def test_lc_normalization_and_repeated_setup(cls, errors):
    data = make_lc(errors=errors)
    original = data.y_data.copy()
    fit = cls()
    with patch.object(shared.logger, 'warning') as warning:
        lc_setup(fit, {V: data})
    assert warning.call_count == (1 if errors is None else 0)
    if errors is None:
        message = warning.call_args.args[0]
        assert 'heuristic weights' in message and 'credible intervals depend' in message
    assert_allclose(fit.y_data[V], original / 3.)
    assert_allclose(fit.y_err[V], .05 if errors is None else np.asarray(errors) / 3.)
    assert_array_equal(data.y_data, original)
    assert data.y_err is None if errors is None else np.array_equal(data.y_err, errors)
    replacement = make_lc([8., 12.], [.2, .4])
    lc_setup(fit, {V: replacement})
    assert_allclose(fit.y_err[V], [.02, .04])
    assert fit.num_of_points == {V: 2}
    assert fit.fit_xs is None
    assert len(fit.x_data_reducer[V]) == 2


@pytest.mark.parametrize('cls', RV_CLASSES)
@pytest.mark.parametrize('supplied', [False, True])
def test_rv_component_scales_and_repeated_setup(cls, supplied):
    cfg, _ = rv_problem()
    values = np.array([-4., -2., 0., 2., 4.])
    data = {component: RVData(x_data=np.arange(5) / 5., y_data=velocities,
                             y_err=np.full(5, .3) if supplied else None,
                             x_unit=u.dimensionless_unscaled, y_unit=u.km/u.s)
            for component, velocities in [('primary', values), ('secondary', 12000. + 3. * values)]}
    fit = cls()
    for component in ('secondary', 'primary'):
        subset = {component: data[component]}
        fit.set_up(BinaryInitialParameters(**cfg), subset, observer_system_cls=RadialVelocitySystem)
        assert set(fit.y_err) == {component}
        expected = np.full(5, 300.) if supplied else np.full(5, radialcurves_mean_error(data[component].y_data))
        assert_allclose(fit.y_err[component], expected)
    fit.set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    if not supplied:
        assert_allclose(fit.y_err['secondary'], 3. * fit.y_err['primary'])


@pytest.mark.parametrize('values', [[0., 0.], [12000.] * 3, [1.], [], [0., np.nan], [0., np.inf]])
def test_uninformative_rv_requires_errors(values):
    with pytest.raises(ValueError, match='supply explicit positive y_err'):
        radialcurves_mean_error(values)


@pytest.mark.parametrize('amplitude', [1e-200, 1e-10, 1e200])
def test_rv_scales_small_and_large_variations(amplitude):
    assert_allclose(radialcurves_mean_error(amplitude * np.array([-1., 0., 1.])),
                    .05 * 1.482602218505602 * amplitude, atol=0., rtol=2e-14)


def test_rv_translation_mad_outliers_and_tied_fallback():
    values = np.array([-300., -100., 0., 100., 300., 10000.])
    expected = .05 * 1.482602218505602 * np.median(np.abs(values - np.median(values)))
    for offset in (0., -1000., 1e9):
        assert_allclose(radialcurves_mean_error(values + offset), expected, rtol=2e-14)
    tied = np.array([0., 0., 0., 0., 10.])
    assert_allclose(radialcurves_mean_error(tied), .05 * np.std(tied, ddof=1))
    near_constant = 12000. + np.arange(5) * 2. ** -30
    assert radialcurves_mean_error(near_constant) > 0
    assert_allclose(radialcurves_mean_error(near_constant), radialcurves_mean_error(near_constant - 12000.))
    core = [-2., -1., 0., 1., 2.]
    assert_allclose(radialcurves_mean_error([*core, 100.]), radialcurves_mean_error([*core, 1e6]))


@pytest.mark.parametrize('cls', RV_CLASSES)
def test_constant_rv_with_explicit_errors(cls):
    cfg, data = rv_problem()
    data['primary'].y_data[:] = 12000.
    fit = cls()
    fit.set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    assert_allclose(fit.y_err['primary'], 100.)
    data['primary'].y_err = None
    with pytest.raises(ValueError, match='supply explicit positive y_err'):
        fit.set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)


@pytest.mark.parametrize('dataset_cls,unit', [(LCData, u.dimensionless_unscaled), (RVData, u.m/u.s)])
@pytest.mark.parametrize('errors', [[], [0., .1], [-.1, .1], [np.nan, .1], [np.inf, .1], [.1], [[.1, .1]]])
def test_invalid_supplied_errors(dataset_cls, unit, errors):
    with pytest.raises(ValueError):
        dataset_cls(x_data=[0., .5], y_data=[1., 2.], y_err=errors,
                    x_unit=u.dimensionless_unscaled, y_unit=unit)


@pytest.mark.parametrize('values', [[], [np.nan], [np.inf], [[1., 2.]], 1.])
def test_invalid_observation_arrays(values):
    with pytest.raises(ValueError):
        LCData(x_data=values, y_data=values, x_unit=u.dimensionless_unscaled, y_unit=u.dimensionless_unscaled)


@pytest.mark.parametrize('cls', (*LC_CLASSES, *RV_CLASSES))
def test_setup_rejects_mutated_dataset(cls):
    if cls in LC_CLASSES:
        data = make_lc(errors=[.1] * 3)
        data.y_err[0] = 0.
        with pytest.raises(ValueError):
            lc_setup(cls(), {V: data})
    else:
        cfg, data = rv_problem()
        data['primary'].y_err[0] = 0.
        with pytest.raises(ValueError):
            cls().set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)


@pytest.mark.parametrize('values', [[-1., 1.], [-2., -1.]])
def test_lc_invalid_normalization(values):
    with pytest.raises(ValueError):
        lc_setup(least_squares.DetachedLightCurveFit(), {V: make_lc(values, [.1, .1])})


def test_file_validation_precedes_downselection_and_smoothing_preserves_none(tmp_path):
    path = tmp_path / 'observations.dat'
    path.write_text('0 1 .1\n.1 1 0\n.2 1 .1\n')
    with pytest.raises(ValueError, match='strictly positive'):
        LCData.load_from_file(str(path), downselect_ratio=.5)
    for bad in ('nan', 'inf', 'bad'):
        path.write_text(f'0 1\n.2 {bad}\n')
        with pytest.raises(ValueError):
            LCData.load_from_file(str(path), data_columns=(0, 1))
    path.write_text('0 1\n.2 2\n.4 3\n')
    data = LCData.load_from_file(str(path), data_columns=(0, 1))
    data.smooth(n_bins=2, radius=1)
    assert data.y_err is None


@pytest.mark.parametrize('base', [shared.AbstractLCFit, shared.AbstractRVFit])
def test_slotted_subclasses_use_declared_attributes(base):
    class Fit(base):
        __slots__ = ()

        def fit(self, *args, **kwargs):
            raise NotImplementedError

    fit = Fit()
    assert not hasattr(fit, '__dict__')
    fit.atmosphere_model = {'primary': 'bb'}
    fit.limb_darkening_coefficients = None
    if base is shared.AbstractLCFit:
        lc_setup(fit, {V: make_lc()})
        assert fit._validated_reducers is not None
    else:
        cfg, data = rv_problem()
        fit.set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    slots = [name for cls in type(fit).__mro__ for name in cls.__dict__.get('__slots__', ())]
    assert len(slots) == len(set(slots))
    with pytest.raises(AttributeError):
        fit.undeclared = 1


@pytest.mark.parametrize('cls', RV_CLASSES)
@pytest.mark.parametrize('supplied', [False, True])
def test_actual_rv_evaluations_do_not_validate_inputs_again(cls, supplied):
    cfg, data = rv_problem()
    if not supplied:
        for dataset in data.values():
            dataset.y_err = None
    fit = cls()
    fit.set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    with patch.object(shared.DataSet, 'check_data_validity', side_effect=AssertionError('Repeated validation')):
        for _ in range(2):
            result = fit.likelihood(normalized(fit)) if cls is mcmc.CentralRadialVelocity else fit.central_rv_model_to_fit(normalized(fit))
            assert np.all(np.isfinite(result))


@pytest.mark.parametrize('supplied', [False, True])
def test_real_lc_evaluations_with_optional_errors(supplied):
    from tests.tests_analytics.test_lc_prediction_consistency import LCScientificConsistencyTestCase, X

    science = LCScientificConsistencyTestCase()
    try:
        science.setUp()
        fits = science.prepare({V: X}, missing_errors=not supplied)
        assert_allclose(fits[0].y_data[V], fits[1].y_data[V])
        assert_allclose(fits[0].y_err[V], fits[1].y_err[V])
        with (patch.object(shared.DataSet, 'check_data_validity', side_effect=AssertionError('Repeated validation')),
              patch.object(shared, 'validate_observations', side_effect=AssertionError('Repeated mapping validation'))):
            for _ in range(2):
                assert np.all(np.isfinite(fits[0].model_to_fit(normalized(fits[0]))))
                assert np.isfinite(fits[1].likelihood(normalized(fits[1])))
    finally:
        science.doCleanups()


@pytest.mark.parametrize('cls', RV_CLASSES)
def test_empty_and_invalid_component_datasets(cls):
    cfg, data = rv_problem()
    for bad in ({}, {'tertiary': data['primary']}):
        with pytest.raises(ValueError):
            cls().set_up(BinaryInitialParameters(**cfg), bad, observer_system_cls=RadialVelocitySystem)


def test_normalization_fraction_validated_during_setup_operations():
    fit = least_squares.DetachedLightCurveFit()
    lc_setup(fit, {V: make_lc()})
    for fraction in (0., 1.1, -.1, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match='fraction'):
            fit.normalize_data(top_fraction_to_average=fraction)


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('supplied', [False, True])
def test_smoothing_preserves_bin_boundaries_and_weighted_means(cyclic, supplied):
    data = LCData(x_data=[0., .25, .5, .75, 1.], y_data=[2., 4., 8., 16., 32.],
                  y_err=[1., 2., 1., 2., 1.] if supplied else None,
                  x_unit=u.dimensionless_unscaled, y_unit=u.dimensionless_unscaled)
    # Exact independent expectations include points on the right bin boundaries.
    if supplied:
        expected = [172./13., 6., 176./9., 15.6] if cyclic else [44./9., 6., 176./9., 28.8]
        expected_errors = [1.25, 1.5, 4./3., 1.5] if cyclic else [4./3., 1.5, 4./3., 1.5]
    else:
        expected = [11.5, 7.5, 56./3., 13.5] if cyclic else [14./3., 7.5, 56./3., 24.]
    data.smooth(n_bins=4, radius=1, cyclic_boundaries=cyclic)
    assert_allclose(data.x_data, [.125, .375, .625, .875])
    assert_allclose(data.y_data, expected)
    if supplied:
        assert_allclose(data.y_err, expected_errors)
    else:
        assert data.y_err is None


def test_mean_normalization_ignores_unused_fraction():
    from elisa.observer.utils import normalize_light_curve

    values = {'V': np.array([2., 4.]), 'B': np.array([6.])}
    for kind, scales in [('average', {'V': 3., 'B': 6.}),
                         ('global_average', {'V': 4., 'B': 4.})]:
        for fraction in (0., 1., 1.1, np.nan, np.inf, -np.inf):
            shared._validate_lc_normalization(values, kind, fraction)
            normalized, _ = normalize_light_curve(values, kind=kind, top_fraction_to_average=fraction)
            for band, scale in scales.items():
                assert_allclose(normalized[band], values[band] / scale)


def test_extrema_normalization_preserves_selection_and_error_scaling():
    from elisa.observer.utils import normalize_light_curve

    values = {'V': np.array([2., 4., 8., 16.]), 'B': np.array([3., 6.])}
    errors = {band: np.ones_like(flux) for band, flux in values.items()}
    expected_scales = {'maximum': {'V': 28./3., 'B': 4.5},
                       'global_maximum': {'V': 12., 'B': 12.},
                       'minimum': {'V': 3., 'B': 3.}}
    for kind, scales in expected_scales.items():
        with pytest.raises(ValueError, match='fraction'):
            shared._validate_lc_normalization(values, kind, np.nan)
        normalized, normalized_errors = normalize_light_curve(values, errors, kind, .5)
        for band, scale in scales.items():
            assert_allclose(normalized[band], values[band] / scale)
            assert_allclose(normalized_errors[band], errors[band] / scale)


def test_extrema_selection_handles_small_and_full_selections():
    from elisa.observer.utils import normalize_light_curve

    values = {'V': np.array([2., 4., 8.])}
    normalized, _ = normalize_light_curve(values, kind='minimum', top_fraction_to_average=.1)
    assert_allclose(normalized['V'], [1., 2., 4.])
    for kind in ('maximum', 'global_maximum', 'minimum'):
        singleton, _ = normalize_light_curve({'V': np.array([2.])}, kind=kind)
        assert_allclose(singleton['V'], [1.])
        # Public callers can request all points; partition indices must remain valid.
        normalized, _ = normalize_light_curve(values, kind=kind, top_fraction_to_average=1.)
        assert_allclose(normalized['V'], values['V'] / (14./3.))


@pytest.mark.parametrize('kind', ['maximum', 'global_maximum', 'minimum'])
def test_normalization_fraction_boundaries(kind):
    from elisa.observer.utils import normalize_light_curve

    values = {'V': np.array([2., 4., 8.]), 'B': np.array([6., 12.])}
    for fraction in (0., np.nextafter(1., np.inf), 1.1, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match='fraction'):
            shared._validate_lc_normalization(values, kind, fraction)
    shared._validate_lc_normalization(values, kind, 1.)
    normalized, _ = normalize_light_curve(values, kind=kind, top_fraction_to_average=1.)
    # Global maximum retains floor(total_points / passbands) + 1, even at 1.0.
    scales = {'V': 26./3., 'B': 26./3.} if kind == 'global_maximum' else {'V': 14./3., 'B': 9.}
    for band, scale in scales.items():
        assert_allclose(normalized[band], values[band] / scale)

    fit = least_squares.DetachedLightCurveFit()
    lc_setup(fit, {V: make_lc(errors=[.1, .2, .3])})
    observed = fit.y_data[V].copy()
    errors = fit.y_err[V].copy()
    fit.normalize_data(kind=kind, top_fraction_to_average=1.)
    assert_allclose(fit.y_data[V], observed)
    assert_allclose(fit.y_err[V], errors)
