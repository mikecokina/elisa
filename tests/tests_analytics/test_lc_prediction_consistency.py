"""Scientific regression tests: actual binary synthesis, deterministic data.

Requires the real ELISA linear LD and CK04 support tables. By default these
are read from ~/.elisa; ELISA_SCIENCE_SUPPORT can select another installation.
No synthesis, interpolation, parameter, reducer or cost functions are mocked.
"""
from tests import set_astropy_units

from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from parameterized import parameterized
from scipy.interpolate import interp1d

from elisa import const, settings, units as u
from elisa.analytics import LCData, SamplingMethod
from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.binary_fit.lc_prediction import PredictionAlignmentError
from elisa.analytics.binary_fit.shared import lc_r_squared
from elisa.analytics.models import cost_fns, lc
from elisa.analytics.params import parameters as p
from elisa.analytics.tools.utils import time_layer_resolver
from elisa.binary_system.system import BinarySystem
from elisa.observer.observer import Observer
from tests.utils import ElisaTestCase

set_astropy_units()

V, B = 'Generic.Bessell.V', 'Generic.Bessell.B'
X = [0., .04, .12, .25, .47, .5, .6, .8]
Z = [0., .1, .3, .5, .7]
MANUAL = [-.25, 0., .04, .12, .25, .47, .5, .6, .8, 1., 1.25]
COLLISION = [-.25, 0., .1, .25, .5, .75, 1., 1.25]
BASE = {
    'system': dict(inclination=85., eccentricity=0., argument_of_periastron=0.,
                   period=3., mass_ratio=.7, semi_major_axis=10.),
    'primary': dict(t_eff=6500., surface_potential=6., gravity_darkening=.32,
                    albedo=.5, metallicity=0.),
    'secondary': dict(t_eff=5000., surface_potential=6., gravity_darkening=.32,
                      albedo=.5, metallicity=0.),
}


class LCScientificConsistencyTestCase(ElisaTestCase):
    def setUp(self):
        super().setUp()
        support = Path(os.environ.get('ELISA_SCIENCE_SUPPORT', Path.home() / '.elisa'))
        ld, atm = support / 'limbdarkening' / 'ld', support / 'atmosphere' / 'ck04'
        if not (ld / f'lin.{V}.p00.csv').is_file() or not (atm / 'ckp00').is_dir():
            self.skipTest('Real ELISA LD/CK04 tables required; set ELISA_SCIENCE_SUPPORT')
        settings.configure(
            LD_TABLES=str(ld), CK04_ATM_TABLES=str(atm), ATM_ATLAS='bb',
            LIMB_DARKENING_LAW='linear', USE_SINGLE_LD_COEFFICIENTS=True,
            NUMBER_OF_PROCESSES=1, MESH_GENERATOR='auto',
            REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2,
            MAX_CURVE_DATAPOINTS=12,
        )

    def physical(self, standard=False):
        base = deepcopy(BASE)
        if standard:
            total = ((10 * u.solRad).to(u.m).value ** 3 * (2*np.pi/(3*86400))**2
                     / const.G / u.solMass.to(u.kg))
            del base['system']['mass_ratio'], base['system']['semi_major_axis']
            base['primary']['mass'], base['secondary']['mass'] = total/1.7, total*.7/1.7
        cfg = {g: {k: {'value': v, 'fixed': True} for k, v in d.items()} for g, d in base.items()}
        cfg['system']['inclination'] = dict(value=85., fixed=False, min=80., max=90.)
        cfg['primary']['t_eff'] = dict(value=6500., fixed=False, min=6000., max=7000.)
        cfg['secondary']['t_eff'] = {'constraint': 'primary@t_eff * 0.7692307692307693'}
        cfg['nuisance'] = {'ln_f': dict(value=-2., fixed=True)}
        flat = {f'{g}@{k}': v for g, d in base.items() for k, v in d.items()}
        return cfg, flat

    @staticmethod
    def synth(xs, bands, physical):
        obs = Observer(passband=list(bands), system=None)
        obs.system_cls = BinarySystem
        return lc.synthetic_binary(np.asarray(xs), 10., obs, **physical)

    def prepare(self, grids, threshold=100, samples=SamplingMethod.UNIFORM, jd=False, standard=False,
                atmosphere=None, custom_ld=False, free_noise=False, missing_errors=False):
        cfg, physical = self.physical(standard)
        if jd:
            cfg['system']['primary_minimum_time'] = dict(value=2450000., fixed=True)
        if free_noise:
            cfg['nuisance']['ln_f'] = dict(value=0., fixed=False, min=-5., max=1.)
        data = {}
        for band, xs in grids.items():
            raw = self.synth(xs, [band], physical)[band]
            data[band] = LCData(
                x_data=2450000.+3*np.asarray(xs) if jd else np.asarray(xs),
                y_data=raw*(1+.002*np.arange(len(xs))),
                y_err=None if missing_errors else np.full(len(xs), .01),
                x_unit=u.d if jd else u.dimensionless_unscaled, y_unit=u.dimensionless_unscaled,
            )
        ld = ({c: {b: [.1] for b in (*grids, 'bolometric')} for c in ('primary', 'secondary')}
              if custom_ld else None)
        fits = []
        for cls in (least_squares.DetachedLightCurveFit, mcmc.DetachedLightCurveFit):
            fit = cls()
            fit.atmosphere_model, fit.limb_darkening_coefficients = atmosphere, ld
            fit.set_up(p.BinaryInitialParameters(**deepcopy(cfg)), data, passband=data.keys(),
                       discretization=10., interp_treshold=threshold, observer_system_cls=BinarySystem,
                       samples=samples)
            fits.append(fit)
        return fits

    @staticmethod
    def capture(callback, *args, **kwargs):
        """Observe the actual arrays without substituting any computation."""
        cap = {}
        def profile(frame, event, result):
            if frame.f_code is lc.synthetic_binary.__code__:
                if event == 'call':
                    cap['support'] = frame.f_locals['phases'].copy()
                    cap['physical'] = deepcopy(frame.f_locals['kwargs'])
                elif event == 'return' and result is not None:
                    cap['raw'] = deepcopy(result)
            if event == 'call' and frame.f_code in (cost_fns.wssr.__code__, cost_fns.likelihood_fn.__code__):
                cap['prediction'] = deepcopy(frame.f_locals['synthetic'])
            if event == 'return' and frame.f_code.co_name == 'predict_light_curves' and isinstance(result, dict):
                cap['prediction'] = deepcopy(result)
        previous = sys.getprofile()
        try:
            sys.setprofile(profile)
            value = callback(*args, **kwargs)
        finally:
            sys.setprofile(previous)
        return value, cap

    def check(self, fits, *, change_vector=False):
        caps, values = [], []
        for fit in fits:
            xn = np.asarray(p.vector_normalizer(fit.initial_vector, fit.fitable.keys(), fit.normalization))
            if change_vector:
                xn[list(fit.fitable).index('system@inclination')] = .4
                xn[list(fit.fitable).index('primary@t_eff')] = .6
            physical = p.prepare_properties_set(
                np.asarray(p.vector_renormalizer(xn, fit.fitable.keys(), fit.normalization)),
                fit.fitable.keys(), fit.constrained, fit.fixed)
            assert_allclose(physical['secondary@t_eff'], physical['primary@t_eff'] * 0.7692307692307693)
            callback = fit.model_to_fit if isinstance(fit, least_squares.LightCurveFit) else fit.likelihood
            value, cap = self.capture(callback, xn)
            caps.append(cap)
            values.append(value)
            phases, _ = time_layer_resolver(fit.x_data_reduced, **physical)
            if fit.fit_xs is not None:
                assert_array_equal(cap['support'], fit.fit_xs)
            # Independent mapping/interpolation on the actual raw synthesized support.
            for band, curve in cap['raw'].items():
                target = phases[fit.x_data_reducer[band]]
                if np.array_equal(cap['support'], phases):
                    mapped = curve[fit.x_data_reducer[band]]
                    own = self.synth(target, [band], cap['physical'])[band]
                    assert_allclose(mapped / mapped.mean(), own / own.mean(), atol=2e-12, rtol=2e-12)
                else:
                    mapped = interp1d(cap['support'], curve, kind='cubic')(target)
                predicted = cap['prediction'][band]
                self.assertEqual(predicted.shape, fit.y_data[band].shape)
                assert_allclose(predicted, mapped / mapped.mean(), atol=2e-12, rtol=2e-12)
                assert_allclose(predicted.mean(), 1., atol=3e-15)
            chi2 = sum(np.dot((fit.y_data[b]-m)/fit.y_err[b], (fit.y_data[b]-m)/fit.y_err[b])
                       for b, m in cap['prediction'].items())
            if isinstance(fit, least_squares.LightCurveFit):
                independent = np.concatenate([(fit.y_data[b]-cap['prediction'][b])/fit.y_err[b]
                                              for b in fit.y_data])
                self.assertEqual(value.ndim, 1)
                assert_allclose(value, independent, atol=2e-12)
                assert_allclose(np.dot(value, value), chi2, rtol=2e-13, atol=2e-12)
            else:
                scale = 1+np.exp(2*physical['nuisance@ln_f'])
                expected = -.5*(chi2/scale + sum(np.log(2*np.pi*scale*np.broadcast_to(fit.y_err[b], y.shape)**2).sum()
                                                     for b, y in fit.y_data.items()))
                assert_allclose(value, expected, rtol=2e-13, atol=2e-12)
                fit.norm_init_vector, fit.sigmas = xn, np.full(len(xn), np.nan)
                assert_allclose(fit.ln_probability(xn), value, atol=2e-12)
            atm_physical = p.extend_json_with_atm_params(
                physical, atmosphere_model=fit.atmosphere_model,
                limb_darkening_coefficients=fit.limb_darkening_coefficients)
            r2, rcap = self.capture(
                lc_r_squared, lc.synthetic_binary, fit.x_data_reduced, fit.y_data,
                fit.observer.passband, fit.discretization, fit.x_data_reducer,
                1/fit.interp_treshold, fit.interp_treshold, BinarySystem,
                fit_xs=fit.fit_xs, x_data=fit.x_data, **atm_physical)
            residual = sum(np.sum((fit.y_data[b]-m)**2) for b, m in cap['prediction'].items())
            variance = sum(np.sum((y-y.mean())**2) for y in fit.y_data.values())
            assert_allclose(r2, 1-residual/variance, atol=2e-12)
            for band in fit.y_data:
                assert_allclose(rcap['prediction'][band], cap['prediction'][band], atol=2e-12, rtol=2e-12)
            if fit.atmosphere_model:
                self.assertEqual(cap['physical']['primary@atmosphere'], 'ck04')
            if fit.limb_darkening_coefficients:
                self.assertEqual(cap['physical']['primary@limb_darkening_coefficients'][V], [.1])
        for band in fits[0].y_data:
            assert_allclose(caps[0]['prediction'][band], caps[1]['prediction'][band], atol=2e-12, rtol=2e-12)
        return caps

    @parameterized.expand([
        ('sorted', {V: X}),
        ('unsorted', {V: [.25, 0., .5, .12]}),
        ('repeated', {V: [0., .1, .1, .5]}),
        ('different', {V: X, B: Z}),
        ('different_equal_lengths', {V: [0., .1, .25, .5], B: [0., .12, .3, .5]}),
        ('singleton', {V: X, B: [.25]}),
        ('unsorted_repeated_multiband', {V: [.5, 0., .1, .1], B: [.3, .1, 0.]}),
    ])
    def test_direct_observation_mapping(self, name, grids):
        self.check(self.prepare(grids))

    @parameterized.expand([
        ('uniform', 4, SamplingMethod.UNIFORM), ('adaptive', 4, SamplingMethod.ADAPTIVE),
        ('manual', 4, MANUAL), ('manual_array', 4, np.asarray(MANUAL)),
        ('manual_exact_coordinates', 4, X), ('manual_equal_length', 4, COLLISION), ('auto_equal_length', 6, SamplingMethod.UNIFORM),
        ('threshold_boundary', 8, SamplingMethod.UNIFORM),
    ])
    def test_prepared_support(self, name, threshold, samples):
        self.check(self.prepare({V: X}, threshold, samples))

    @parameterized.expand([(False, 100), (False, 4), (True, 100), (True, 4)])
    def test_multiband_phase_and_jd(self, jd, threshold):
        self.check(self.prepare({V: [.6, 0., .1, .1, .5], B: [.25]}, threshold, jd=jd))

    @parameterized.expand([(100, False, True), (100, True, False), (4, True, True)])
    def test_explicit_atmosphere_and_ld(self, threshold, atmosphere, custom_ld):
        grids = {V: X, B: Z}
        fits = self.prepare(grids, threshold, atmosphere={'primary': 'ck04', 'secondary': 'ck04'}
                            if atmosphere else None, custom_ld=custom_ld)
        caps = self.check(fits)
        baseline = self.prepare(grids, threshold)
        physical = p.prepare_properties_set(baseline[0].initial_vector, baseline[0].fitable.keys(),
                                             baseline[0].constrained, baseline[0].fixed)
        control = baseline[0].predict_light_curves(physical)
        self.assertGreater(max(np.max(np.abs(caps[0]['prediction'][b]-control[b])) for b in grids), 1e-5)

    @parameterized.expand([(False, False), (True, False), (False, True), (True, True)])
    def test_standard_community_constraints_and_free_noise(self, standard, jd):
        self.check(self.prepare({V: X, B: Z}, 4, jd=jd, standard=standard, free_noise=True), change_vector=True)

    @parameterized.expand([(100,), (4,)])
    def test_missing_error_scalar_noise_model(self, threshold):
        fits = self.prepare({V: X, B: [.25]}, threshold, missing_errors=True)
        for fit in fits:
            for error in fit.y_err.values():
                self.assertEqual(np.shape(error), ())
        self.check(fits)

    def test_invalid_physical_synthesis_retains_penalties(self):
        fits = self.prepare({V: X})
        for fit in fits:
            fit.fixed['system@eccentricity'].value = 1.1
            xn = np.asarray(p.vector_normalizer(fit.initial_vector, fit.fitable.keys(), fit.normalization))
            if isinstance(fit, least_squares.LightCurveFit):
                penalty = fit.model_to_fit(xn)
                self.assertEqual(penalty.shape, (len(X),))
                self.assertTrue(np.all(np.isfinite(penalty)))
                assert_allclose(np.dot(penalty, penalty), const.MAX_USABLE_FLOAT, rtol=2e-15)
            else:
                fit.norm_init_vector, fit.sigmas = xn, np.full(len(xn), np.nan)
                last = fit._last_known_lhood
                with self.assertRaises(ValueError):
                    fit.likelihood(xn)
                self.assertEqual(fit.ln_probability(xn), last * 1e3)

    def test_jd_period_epoch_changes(self):
        fits = self.prepare({V: [1.6, 1., 1.1, 1.1, 1.5], B: [1.25]}, 4, jd=True)
        for fit in fits:
            fit.fixed['system@period'].value = 3.1
            fit.fixed['system@primary_minimum_time'].value = 2450000.02
        self.check(fits)

    def test_default_threshold_equal_length(self):
        self.check(self.prepare({V: np.linspace(0., .99, 302)}, 300))

    def test_dense_support_approximation(self):
        fits = self.prepare({V: np.linspace(0., .99, 41), B: Z}, 4,
                            np.linspace(-.01, 1.01, 302).tolist())
        caps = self.check(fits)
        for band, raw_x in fits[0].x_data.items():
            direct = self.synth(raw_x, [band], caps[0]['physical'])[band]
            assert_allclose(caps[0]['prediction'][band], direct/direct.mean(), atol=.005, rtol=0)

    @parameterized.expand(['coordinates', 'reducer_length', 'reducer_bounds', 'reducer_type',
                           'uncertainty_length', 'passbands', 'support_duplicate', 'support_short', 'support_uncovered'])
    def test_structural_errors_escape_both_penalty_handlers(self, defect):
        fits = self.prepare({V: X, B: [.25]})
        for fit in fits:
            if defect == 'coordinates':
                fit.x_data_reducer[V] = fit.x_data_reducer[V][::-1]
            elif defect == 'reducer_length':
                fit.x_data_reducer[V] = fit.x_data_reducer[V][:-1]
            elif defect == 'reducer_bounds':
                fit.x_data_reducer[V][0] = len(fit.x_data_reduced)
            elif defect == 'reducer_type':
                fit.x_data_reducer[V] = fit.x_data_reducer[V].astype(float)
            elif defect == 'uncertainty_length':
                fit.y_err[B] = np.ones(8)
            elif defect == 'passbands':
                fit.observer.passband.pop(B)
            elif defect == 'support_duplicate':
                fit.fit_xs = np.array([-.1, 0., .5, .5, 1.1])
            elif defect == 'support_short':
                fit.fit_xs = np.array([0., .5, 1.])
            else:
                fit.fit_xs = np.linspace(.1, .7, 8)
            xn = np.asarray(p.vector_normalizer(fit.initial_vector, fit.fitable.keys(), fit.normalization))
            if isinstance(fit, mcmc.LightCurveFit):
                fit.norm_init_vector, fit.sigmas = xn, np.full(len(xn), np.nan)
                callback = fit.ln_probability
            else:
                callback = fit.model_to_fit
            with self.assertRaises(PredictionAlignmentError):
                if defect in ('coordinates', 'reducer_length', 'reducer_bounds', 'reducer_type', 'uncertainty_length'):
                    # Manual edits to static prepared data require explicit revalidation.
                    fit.validate_observations()
                callback(xn)
