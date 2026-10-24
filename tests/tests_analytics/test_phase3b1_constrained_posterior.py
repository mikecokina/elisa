"""Direct NumPy posterior references; no MCMC sampling or forward synthesis."""
import json
from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa.analytics.binary_fit import io_tools, mcmc
from elisa.analytics.binary_fit.lc_fit import LCFitMCMC
from elisa.analytics.binary_fit.rv_fit import RVFitMCMC
from elisa.analytics.binary_fit.mixins import MCMCMixin
from elisa.analytics.params import parameters
from elisa.analytics.params.parameters import BinaryInitialParameters, InitialParameter
from elisa.base.error import InitialParamsError

X = 'system@gamma'
Y = 'system@mass_ratio'
Z = 'system@asini'
NORM = {X: (-4., 4.)}
PHYSICAL = np.array([-3., -2., -1., 0., 1., 2., 3., 3.5])
CHAIN = ((PHYSICAL + 4.) / 8.)[:, None]


def initial(expression='system@gamma**2 + system@period'):
    return BinaryInitialParameters(system={
        'gamma': {'value': 0., 'fixed': False, 'min': -4., 'max': 4.},
        'period': {'value': 2., 'fixed': True},
        'mass_ratio': {'constraint': expression},
    })


def summary(values, expression, fixed=None, percentiles=None):
    init = initial(expression)
    return MCMCMixin.resolve_constrained_result(
        values, init.get_fitable(), NORM, init.get_constrained(),
        init.get_fixed() if fixed is None else fixed, percentiles,
    )[Y]


def check(summary, samples, percentiles=(16, 50, 84)):
    expected = np.percentile(samples, percentiles)
    assert_allclose([summary['confidence_interval']['min'], summary['value'],
                     summary['confidence_interval']['max']], expected, rtol=0, atol=1e-14)


def test_linear_and_nonlinear_fixed_dependencies_and_precision():
    check(summary(CHAIN, '3*system@gamma + system@period'), 3*PHYSICAL + 2)
    result = summary(CHAIN, 'system@gamma**2 + system@period')
    check(result, PHYSICAL**2 + 2)
    assert result['value'] == 6.
    assert np.median(PHYSICAL)**2 + 2 == 2.25
    check(summary(CHAIN, 'system@gamma**2', percentiles=[5, 50, 95]), PHYSICAL**2, (5, 50, 95))
    assert result['unit'] == initial().get_constrained()[Y].to_dict()['unit']


def test_joint_correlations_and_multiple_constraints():
    labels = {X: initial().get_fitable()[X], Z: initial().get_fitable()[X]}
    # Perfect anticorrelation: summing marginal intervals would give a false width.
    chain = np.column_stack((CHAIN[:, 0], 1 - CHAIN[:, 0]))
    constraints = {
        Y: InitialParameter(None, param='mass_ratio', constraint=f'{X}+{Z}', unit=None),
        'system@eccentricity': InitialParameter(None, param='eccentricity', constraint=f'{X}*{Z}', unit=None),
    }
    before = chain.copy()
    chain.setflags(write=False)
    results = MCMCMixin.resolve_constrained_result(chain, labels, {X: (-4., 4.), Z: (-4., 4.)}, constraints, {})
    check(results[Y], np.zeros(len(PHYSICAL)))
    check(results['system@eccentricity'], -PHYSICAL**2)
    assert_array_equal(chain, before)


def test_existing_engine_does_not_support_constrained_to_constrained_references():
    # Preserve this existing limitation rather than adding a new dependency engine.
    constraints = {Y: f'{X}**2', Z: f'{Y}+1'}
    with pytest.raises(InitialParamsError, match='non-variable parameter'):
        parameters.prepare_properties_set(np.array([-1.]), [X], constraints, {})


def test_negative_power_substitution_and_constant_samples():
    chain = np.full((5, 1), 3/8)  # physical gamma = -1
    with np.errstate(all='raise'):
        result = summary(chain, 'system@gamma**2')
        free = MCMCMixin.resolve_mcmc_result(chain, initial().get_fitable(), NORM)[X]
    check(result, np.ones(5))
    check(free, -np.ones(5))
    exact = 0.123456789012345
    chain = np.full((5, 1), (exact + 4)/8)
    check(MCMCMixin.resolve_mcmc_result(chain, initial().get_fitable(), NORM)[X], np.full(5, exact))


@pytest.mark.parametrize('expression', ['exp(1000*system@gamma)', 'log(system@gamma)'])
def test_nonfinite_constraints_reject_instead_of_dropping_samples(expression):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        with pytest.raises(ValueError, match='Nonfinite posterior constraint'):
            summary(CHAIN, expression)


@pytest.mark.parametrize('fit_cls', [LCFitMCMC, RVFitMCMC])
@pytest.mark.parametrize('structured', [False, True])
def test_result_chain_reload_discard_filter_and_reserialize(tmp_path, fit_cls, structured):
    fit = fit_cls('detached', None, None) if fit_cls is LCFitMCMC else fit_cls()
    init = initial()
    base = MCMCMixin.resolve_mcmc_result(CHAIN, init.get_fitable(), NORM)
    base.update({key: {'value': item.value, 'fixed': True, 'unit': item.to_dict()['unit']}
                 for key, item in init.get_fixed().items()})
    base[Y] = summary(CHAIN, 'system@gamma**2 + system@period')
    fit.set_result(parameters.serialize_result(base))
    check(fit.flat_result[Y], PHYSICAL**2 + 2)
    result_path = tmp_path / 'result.json'
    fit.save_result(result_path)
    fit.load_result(result_path)
    check(fit.flat_result[Y], PHYSICAL**2 + 2)
    payload = {'flat_chain': CHAIN.tolist(), 'fitable_parameters': [X], 'normalization': NORM,
               'fitable': {key: val.to_dict() for key, val in init.get_fitable().items()}}
    if structured:
        payload.update(chain_shape=[4, 2, 1], chain_order='step,walker,parameter')
    path = tmp_path / 'chain.json'
    path.write_text(json.dumps(payload))
    fit.load_chain(str(path), discard=1)
    retained = PHYSICAL[2 if structured else 1:]
    check(fit.flat_result[Y], retained**2 + 2)
    io_tools.filter_chain(fit, **{X: (-1.5, 3.1)})
    retained = retained[(retained > -1.5) & (retained < 3.1)]
    check(fit.flat_result[Y], retained**2 + 2)
    fit.save_result(result_path)
    fit.load_result(result_path)
    check(fit.flat_result[Y], retained**2 + 2)


@pytest.mark.parametrize('fit_cls', [mcmc.DetachedLightCurveFit, mcmc.CentralRadialVelocity])
def test_fit_completion_uses_samplewise_summary(fit_cls):
    init = initial()
    fit = fit_cls()
    fit.fitable, fit.fixed, fit.constrained = init.get_fitable(), init.get_fixed(), init.get_constrained()
    fit.normalization, fit.initial_vector = NORM, [0.]
    fit.x_data_reduced, fit.y_data, fit.x_data_reducer = np.array([]), {}, {}
    fit.observer = Mock()
    sampler = Mock()
    sampler.get_chain.return_value = CHAIN
    with patch.object(fit_cls, 'set_up'), patch.object(fit_cls, '_fit', return_value=sampler), \
         patch.object(fit_cls, 'prediction_r_squared', return_value=1., create=True), \
         patch.object(mcmc, 'rv_r_squared', return_value=1.), \
         patch.object(mcmc, 'check_for_boundary_surface_potentials', side_effect=lambda result, _: result):
        result = fit.fit(x0=init, data={}, nsteps=4, save=False)
    check(parameters.deserialize_result(result)[Y], PHYSICAL**2 + 2)


@pytest.mark.parametrize('missing', ['flat_result', 'flat_chain', 'normalization'])
def test_update_rejects_missing_state(missing):
    fit = RVFitMCMC()
    fit.flat_result = {}
    fit.flat_chain = CHAIN
    fit.normalization = NORM
    setattr(fit, missing, None)
    with pytest.raises(ValueError, match='requires loaded results, chain and normalization'):
        io_tools.update_solution(fit, {}, None)
