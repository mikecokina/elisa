"""Real proposal construction and synthesis through LC/RV fitting boundaries."""

import multiprocessing

import numpy as np
import pytest

from elisa import const, settings, units as u
from elisa.analytics import RVData
from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.models import lc
from elisa.analytics.params import parameters
from elisa.base.error import PulsationModeError, SpotError
from elisa.binary_system.system import BinarySystem
from tests.tests_analytics.fitting_helpers import V, lc_parameters, make_lc, normalized


@pytest.fixture(autouse=True)
def synthesis_settings():
    previous = settings.settings_serializer()
    settings.configure(ATM_ATLAS='bb', LIMB_DARKENING_LAW='linear', NUMBER_OF_PROCESSES=1,
                       REFLECTION_EFFECT=True, MESH_GENERATOR='auto', RV_METHOD='kinematic')
    yield
    settings.configure(**previous)


def make_fit(kind, method, phenomenon, invalid_case='axis'):
    cfg, _ = lc_parameters(standard=kind == 'rv')
    if phenomenon == 'pulsation':
        values = dict(l=1, m=1, frequency=16., amplitude=50., start_phase=0.,
                      mode_axis_phi=0., mode_axis_theta=30.)
        metadata = {key: dict(value=value, fixed=True) for key, value in values.items()}
        metadata['mode_axis_theta'] = dict(value=30., fixed=False, min=0., max=200.)
        if invalid_case == 'fractional_l':
            metadata['l'] = dict(constraint='system@inclination - 84')
        elif invalid_case == 'm':
            metadata['m'] = dict(constraint='system@inclination - 84')
        elif invalid_case in ('amplitude', 'frequency'):
            metadata[invalid_case] = dict(constraint=f'(system@inclination - 84) * {50 if invalid_case == "amplitude" else 16}')
        cfg['primary']['pulsations'] = [dict(label='mode', **metadata)]
        proposal_key = 'primary@pulsation@mode@mode_axis_theta'
        invalid = 190.
        if invalid_case != 'axis':
            proposal_key = 'system@inclination'
            invalid = dict(fractional_l=85.5, m=86., amplitude=83., frequency=83.)[invalid_case]
    else:
        metadata = {key: dict(value=value, fixed=True) for key, value in
                    dict(longitude=90., latitude=60., angular_radius=25., temperature_factor=.9).items()}
        metadata['angular_radius'] = dict(value=25., fixed=False, min=0., max=40.)
        cfg['primary']['spots'] = [dict(label='spot', **metadata)]
        proposal_key, invalid = 'primary@spot@spot@angular_radius', 0.
    if kind == 'lc':
        fit = (least_squares.DetachedLightCurveFit if method == 'lsq' else mcmc.DetachedLightCurveFit)()
        fit.atmosphere_model = dict(primary='bb', secondary='bb')
        fit.limb_darkening_coefficients = {c: {V: [.1], 'bolometric': [.1]} for c in ('primary', 'secondary')}
        fit.set_up(parameters.BinaryInitialParameters(**cfg), {V: make_lc([1., .9, 1.], [.01] * 3)},
                   passband=[V], discretization=10., interp_treshold=100, samples='uniform',
                   observer_system_cls=BinarySystem)
    else:
        fit = (least_squares.CentralRadialVelocity if method == 'lsq' else mcmc.CentralRadialVelocity)()
        data = {c: RVData(x_data=np.array([0., .25, .5]), y_data=np.zeros(3), y_err=np.ones(3) * 100.,
                          x_unit=u.dimensionless_unscaled, y_unit=u.m / u.s) for c in ('primary', 'secondary')}
        fit.set_up(parameters.BinaryInitialParameters(**cfg), data, observer_system_cls=BinarySystem)
    if method == 'mcmc':
        fit.prepare_priors()
    xn = normalized(fit)
    bad = xn.copy()
    bad[list(fit.fitable).index(proposal_key)] = parameters.normalize_value(invalid, *fit.normalization[proposal_key])
    return fit, xn, bad


def evaluate(fit, xn):
    if isinstance(fit, mcmc.MCMCFit):
        return fit.ln_probability(xn)
    if isinstance(fit, least_squares.LightCurveFit):
        return fit.model_to_fit(xn)
    return fit.central_rv_model_to_fit(xn)


def assert_rejected(fit, value):
    if isinstance(fit, mcmc.MCMCFit):
        assert value == -np.inf
    else:
        count = sum(fit.num_of_points.values())
        np.testing.assert_array_equal(value, np.full(count, np.sqrt(const.MAX_USABLE_FLOAT / count)))
        assert value.shape == (count,)
        assert np.isfinite(value).all()


@pytest.mark.parametrize('kind', ['lc', 'rv'])
@pytest.mark.parametrize('method', ['lsq', 'mcmc'])
@pytest.mark.parametrize('invalid_case', ['axis', 'fractional_l', 'm', 'amplitude', 'frequency'])
def test_real_pulsation_proposal_valid_invalid_valid(kind, method, invalid_case):
    fit, xn, bad = make_fit(kind, method, 'pulsation', invalid_case)
    before = evaluate(fit, xn)
    assert np.isfinite(before).all()
    assert_rejected(fit, evaluate(fit, bad))
    np.testing.assert_array_equal(evaluate(fit, xn), before)


@pytest.mark.parametrize('method', ['lsq', 'mcmc'])
def test_real_spot_mesh_proposal_valid_invalid_valid(method):
    fit, xn, bad = make_fit('lc', method, 'spot')
    before = evaluate(fit, xn)
    assert np.isfinite(before).all()
    assert_rejected(fit, evaluate(fit, bad))
    np.testing.assert_array_equal(evaluate(fit, xn), before)


@pytest.mark.parametrize('kind', ['lc', 'rv'])
@pytest.mark.parametrize('method', ['lsq', 'mcmc'])
@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, TypeError, IndexError])
def test_unexpected_error_in_real_mode_constructor_terminates_fit(monkeypatch, kind, method, error_type):
    fit, xn, _ = make_fit(kind, method, 'pulsation')

    def broken(*args, **kwargs):
        raise error_type('mode implementation defect')

    monkeypatch.setattr('elisa.pulse.mode.PulsationMode.validate_mode', broken)
    with pytest.raises(error_type, match='mode implementation defect'):
        evaluate(fit, xn)


def _worker_evaluate(fit, xn):
    return evaluate(fit, xn)


@pytest.mark.parametrize('method', ['lsq', 'mcmc'])
def test_real_fitting_rejection_through_process_pool(method):
    fit, xn, bad = make_fit('lc', method, 'pulsation', 'm')
    before = evaluate(fit, xn)
    with multiprocessing.get_context('fork').Pool(1) as pool:
        assert_rejected(fit, pool.apply_async(_worker_evaluate, (fit, bad)).get(timeout=30))
        np.testing.assert_array_equal(pool.apply_async(_worker_evaluate, (fit, xn)).get(timeout=30), before)


@pytest.mark.parametrize('phenomenon,error_type', [('pulsation', PulsationModeError), ('spot', SpotError)])
def test_direct_synthesis_keeps_meaningful_domain_error(phenomenon, error_type):
    fit, _, bad = make_fit('lc', 'lsq', phenomenon)
    kwargs = parameters.prepare_properties_set(
        parameters.vector_renormalizer(bad, fit.fitable, fit.normalization), fit.fitable, fit.constrained, fit.fixed)
    with pytest.raises(error_type):
        lc.synthetic_binary(fit.x_data_reduced, fit.discretization, fit.observer, **kwargs)


@pytest.mark.parametrize('method', ['lsq', 'mcmc'])
def test_radiometric_rv_rejects_real_spot_mesh_failure(monkeypatch, method):
    from elisa.analytics.models import rv
    settings.configure(RV_METHOD='radiometric', USE_SINGLE_LD_COEFFICIENTS=False)
    fit, xn, bad = make_fit('rv', method, 'spot')
    # Supply trusted physical/table metadata at the existing construction boundary;
    # the RV helper's placeholder metallicity has no scientific table coverage.
    original = rv.prepare_central_rv_binary

    def configured_binary(**kwargs):
        binary = original(**kwargs)
        for component in (binary.primary, binary.secondary):
            component.atmosphere = 'bb'
            component.limb_darkening_coefficients = {'rv_band': [.1], 'bolometric': [.1]}
            component.discretization_factor = np.radians(10.)
        return binary

    monkeypatch.setattr(rv, 'prepare_central_rv_binary', configured_binary)
    before = evaluate(fit, xn)
    assert np.isfinite(before).all()
    assert_rejected(fit, evaluate(fit, bad))
    np.testing.assert_array_equal(evaluate(fit, xn), before)


@pytest.mark.parametrize('method', ['lsq', 'mcmc'])
def test_unexpected_constructor_error_crosses_fitting_process_boundary(monkeypatch, method):
    fit, xn, _ = make_fit('lc', method, 'pulsation')

    def broken(*args, **kwargs):
        raise RuntimeError('remote mode implementation defect')

    monkeypatch.setattr('elisa.pulse.mode.PulsationMode.validate_mode', broken)
    with multiprocessing.get_context('fork').Pool(1) as pool:
        with pytest.raises(RuntimeError, match='remote mode implementation defect') as caught:
            pool.apply_async(_worker_evaluate, (fit, xn)).get(timeout=30)
    assert 'broken' in str(caught.value.__cause__)
