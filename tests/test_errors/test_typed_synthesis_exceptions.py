"""Origin markers retain public catches and existing failure diagnostics."""

import multiprocessing
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from elisa import settings
from elisa.base import error
from elisa.base.error import (
    ElisaError, ElisaNumericError, ElisaMaxIterationError,
    ElisaParametersError, MorphologyError,
    ElisaPredictionAlignmentError,
)
from elisa.base.surface import coverage, temperature
from elisa.binary_system import radius
from elisa.binary_system.surface import clipping
from elisa.binary_system.system import BinarySystem
from elisa.binary_system.transform import BinarySystemProperties


MARKERS = [
    (error.ElisaUnexpectedError, RuntimeError),
    (ElisaNumericError, FloatingPointError),
    (ElisaMaxIterationError, ElisaError),
]


def _worker_failure(error_type):
    raise error_type('worker diagnostics')


@pytest.mark.parametrize('marker, public_base', MARKERS)
def test_public_catch_and_pickle_compatibility(marker, public_base):
    error = marker('original diagnostics')
    with pytest.raises(public_base) as caught:
        raise error
    assert caught.value is error
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is marker
    assert restored.args == error.args
    assert not isinstance(error, MorphologyError)


@pytest.mark.parametrize('marker, public_base', MARKERS)
def test_worker_failure_keeps_type_and_remote_diagnostics(marker, public_base):
    with multiprocessing.get_context('fork').Pool(1) as pool:
        result = pool.apply_async(_worker_failure, (marker,))
        with pytest.raises(public_base, match='worker diagnostics') as caught:
            result.get(timeout=30)
    assert type(caught.value) is marker
    assert 'worker_failure' in str(caught.value.__cause__)


def test_markers_do_not_broaden_existing_semantic_hierarchies():
    assert not issubclass(error.ElisaUnexpectedError, ElisaError)
    assert issubclass(ElisaPredictionAlignmentError, ValueError)


def test_only_elisa_iteration_failure_is_a_synthesis_penalty():
    from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES

    assert ElisaMaxIterationError in SYNTHESIS_FAILURES
    assert ElisaMaxIterationError.__bases__ == (ElisaError,)
    assert isinstance(ElisaMaxIterationError("no convergence"), SYNTHESIS_FAILURES)
    assert not isinstance(ElisaError("generic failure"), SYNTHESIS_FAILURES)


def test_static_physical_range_check_retains_message():
    with pytest.raises(ElisaParametersError) as caught:
        BinarySystemProperties.eccentricity(1.0)
    assert type(caught.value) is ElisaParametersError
    assert str(caught.value)


def test_radius_domain_failure_retains_message(monkeypatch):
    monkeypatch.setattr('elisa.binary_system._radius_solver.brentq',
                        lambda *a, **k: (-0.5, SimpleNamespace(converged=True)))
    with pytest.raises(ElisaNumericError, match='Roche solver returned a radius outside its geometric bracket') as caught:
        radius.calculate_radius(1.0, 1.0, 4.0, 'primary', 1.0, 0.0, 0.0)
    assert type(caught.value) is ElisaNumericError


@pytest.mark.parametrize('component', ['primary', 'secondary'])
def test_critical_potential_nan_is_explicit_elisa_failure(monkeypatch, component):
    monkeypatch.setattr('elisa.binary_system.system.optimize.newton', lambda *a, **k: (np.nan, SimpleNamespace(converged=True)))
    with pytest.raises(ElisaNumericError, match='critical potential solver has failed') as caught:
        # noinspection PyTypeChecker
        BinarySystem.critical_potential_static(component, 1.0, 1.0, 1.0)
    assert type(caught.value) is ElisaNumericError


@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, FloatingPointError])
def test_dependency_solver_errors_are_not_relabelled(monkeypatch, error_type):
    original = error_type('dependency diagnostics')

    # noinspection PyUnusedLocal
    def fail(*args, **kwargs):
        raise original
    monkeypatch.setattr('elisa.binary_system.system.optimize.newton', fail)
    with pytest.raises(error_type) as caught:
        BinarySystem.critical_potential_static('primary', 1.0, 1.0, 1.0)
    assert caught.value is original


def test_contract_conversion_failure_keeps_cause():
    with pytest.raises(error.ElisaUnexpectedError, match='could not be interpreted as integers') as caught:
        # noinspection PyTypeChecker
        coverage._selector_to_mask(['bad-index'], 1, 'visible')
    assert type(caught.value) is error.ElisaUnexpectedError
    assert type(caught.value.__cause__) is ValueError


def test_missing_temperature_table_is_unexpected_and_keeps_cause(monkeypatch, tmp_path):
    monkeypatch.setattr(type(settings), 'PATH_TO_BETA', str(tmp_path / 'absent.json'))
    with pytest.raises(error.ElisaUnexpectedError, match='Cannot load the fixed gravity-darkening table') as caught:
        temperature.interpolate_bolometric_gravity_darkening(6000.0)
    assert type(caught.value) is error.ElisaUnexpectedError
    assert isinstance(caught.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize('kernel', [clipping.convex_polygon_triangle_intersection_area,
                                    clipping.optimized_intersection_area])
def test_clipping_overflow_retains_floating_point_catch(kernel):
    polygon = np.array([[0., 0.], [1., 0.], [0., 1.]])
    triangle = polygon * 1e200
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(FloatingPointError, match='Triangle determinant overflow') as caught:
            kernel(polygon, triangle)
    assert type(caught.value) is ElisaNumericError


def test_lagrange_solver_still_skips_isolated_failed_guesses(monkeypatch):
    from elisa.binary_system import system
    baseline = BinarySystem.lagrangian_points_static(1.0, 0.7)
    original = system.fsolve
    attempts = 0

    def isolated_failure(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ElisaNumericError('isolated Lagrange guess failure')
        return original(*args, **kwargs)

    BinarySystem._lagrangian_points_static.cache_clear()
    monkeypatch.setattr(system, 'fsolve', isolated_failure)
    np.testing.assert_allclose(BinarySystem.lagrangian_points_static(1.0, 0.7), baseline)
    assert attempts > 1


@pytest.mark.parametrize('cls', [error.MorphologyError])
def test_physical_errors_have_explicit_policy_and_pickle(cls):
    from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES

    assert cls.__bases__ == (error.ElisaError,)
    assert cls in SYNTHESIS_FAILURES
    assert type(pickle.loads(pickle.dumps(cls('diagnostic')))) is cls
    assert not issubclass(error.ElisaSpotError, error.PulsationModeError)
    assert not issubclass(error.PulsationModeError, error.ElisaSpotError)


def test_atmosphere_error_is_explicitly_penalized():
    from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES

    assert error.ElisaAtmosphereError.__bases__ == (error.ElisaError,)
    assert error.ElisaAtmosphereError in SYNTHESIS_FAILURES
    failure = error.ElisaAtmosphereError('diagnostic')
    assert type(pickle.loads(pickle.dumps(failure))) is error.ElisaAtmosphereError


def test_limb_darkening_error_is_explicitly_penalized():
    from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES

    assert error.ElisaLimbDarkeningError.__bases__ == (error.ElisaError,)
    assert error.ElisaLimbDarkeningError in SYNTHESIS_FAILURES
    failure = error.ElisaLimbDarkeningError('diagnostic')
    assert type(pickle.loads(pickle.dumps(failure))) is error.ElisaLimbDarkeningError


def test_pulsation_mode_error_is_explicitly_penalized():
    from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES

    assert error.PulsationModeError.__bases__ == (error.ElisaError,)
    assert error.PulsationModeError in SYNTHESIS_FAILURES
    failure = error.PulsationModeError("diagnostic")
    assert type(pickle.loads(pickle.dumps(failure))) is error.PulsationModeError


def test_spot_error_is_explicitly_penalized():
    from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES

    assert error.ElisaSpotError.__bases__ == (error.ElisaError,)
    assert error.ElisaSpotError in SYNTHESIS_FAILURES
    failure = error.ElisaSpotError("diagnostic")
    assert type(pickle.loads(pickle.dumps(failure))) is error.ElisaSpotError


def test_temperature_error_is_explicitly_penalized():
    from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES

    assert error.ElisaTemperatureError.__bases__ == (error.ElisaError,)
    assert error.ElisaTemperatureError in SYNTHESIS_FAILURES
    failure = error.ElisaTemperatureError("diagnostic")
    assert type(pickle.loads(pickle.dumps(failure))) is error.ElisaTemperatureError


def test_unexpected_error_has_no_legacy_runtime_parent():
    assert error.ElisaUnexpectedError.__bases__ == (RuntimeError,)
