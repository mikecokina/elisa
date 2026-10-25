"""Expected failures at real synthesis sources, with local recovery preserved."""

import multiprocessing
from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import QhullError

from elisa.base import error
from elisa.base.spot import remove_overlaped_spots_by_spot_index, remove_overlaped_spots_by_vertex_map
from elisa.base.surface import temperature
from elisa.binary_system import system
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.orbit.orbit import Orbit
from elisa.binary_system.surface import faces
from elisa.const import Position
from elisa.pulse import pulsations
from elisa.pulse.mode import PulsationMode
from elisa.single_system.surface.faces import single_surface
from tests import utils


MODE = dict(l=1, m=1, frequency=16., amplitude=50.)


def spotted_container():
    binary = utils.prepare_binary_system(utils.BINARY_SYSTEM_PARAMS['detached'],
                                         spots_primary=utils.SPOTS_META['primary'])
    return OrbitalPositionContainer.from_binary_system(binary, Position(0, 1., 0., 0., 0.))


@pytest.mark.parametrize('changes', [dict(l=-1), dict(m=2), dict(l=1.001), dict(m=np.float32(.5)),
                                     dict(l=np.nan), dict(m=np.inf), dict(l=Decimal('1.5')), dict(m=Fraction(1, 2)), dict(amplitude=-1), dict(frequency=-1),
                                     dict(mode_axis_theta=-1), dict(mode_axis_theta=180),
                                     dict(mode_axis_phi=np.inf)])
def test_real_mode_valid_invalid_valid(changes):
    before = PulsationMode(**MODE)
    with pytest.raises(error.PulsationModeError):
        PulsationMode(**(MODE | changes))
    after = PulsationMode(**MODE)
    for key in MODE:
        assert getattr(before, key) == getattr(after, key)
    assert after.frequency == 16 / 86400  # Direct constructor conversion remains unchanged.


@pytest.mark.parametrize('index', ['l', 'm'])
@pytest.mark.parametrize('value', [object(), '1', [1], 1j])
def test_incompatible_mode_index_types_remain_fatal(index, value):
    with pytest.raises(TypeError):
        PulsationMode(**(MODE | {index: value}))


@pytest.mark.parametrize('changes', [dict(frequency=0), dict(l=0, m=0)])
def test_amplitude_source_failure(changes):
    star = SimpleNamespace(equivalent_radius=1e9, mass=2e30,
                           pulsations={0: PulsationMode(**(MODE | changes))})
    with pytest.raises(error.PulsationModeError):
        pulsations.assign_amplitudes(star)


def test_zero_amplitude_is_valid_and_finite():
    star = SimpleNamespace(equivalent_radius=1e9, mass=2e30,
                           pulsations={0: PulsationMode(**(MODE | dict(amplitude=0)))})
    pulsations.assign_amplitudes(star)
    mode = star.pulsations[0]
    assert mode.radial_amplitude == mode.horizontal_amplitude == mode.temperature_amplitude_factor == 0


def test_pulsation_zero_denominators():
    with pytest.raises(error.PulsationModeError, match='zero denominator'):
        pulsations.horizontal_displacement_normalization([np.zeros(3)] * 2, [np.ones(3)] * 2)
    with pytest.raises(error.PulsationModeError, match='nonzero horizontal'):
        pulsations.temp_amplitude(SimpleNamespace(horizontal_to_radial_amplitude_ratio=0))


@pytest.mark.parametrize('by_vertices', [True, False])
def test_missing_retained_spot_is_rejected(by_vertices):
    container = spotted_container().primary
    count = len(container.spots)
    with pytest.raises(error.SpotError):
        if by_vertices:
            remove_overlaped_spots_by_vertex_map(container, [{'enum': -1}])
        else:
            remove_overlaped_spots_by_spot_index(container, [])
    assert len(container.spots) == count


def test_real_complete_spot_overlap_rejected():
    spot = utils.SPOTS_META['primary'][0]
    binary = utils.prepare_binary_system(utils.BINARY_SYSTEM_PARAMS['detached'], spots_primary=[spot, spot.copy()])
    container = utils.prepare_orbital_position_container(binary)
    with pytest.raises(error.SpotError, match='have no points'):
        container.build(components_distance=1.)


def test_surface_boundary_skips_isolated_bad_roots(monkeypatch):
    binary = utils.prepare_binary_system(utils.BINARY_SYSTEM_PARAMS['over-contact'])
    expected = binary.compute_equipotential_boundary(1., 'xy')
    original = system.fsolve
    calls = 0

    def isolated_bad_root(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 3:
            return np.array([np.nan if calls == 1 else -1.]), {}, 4 if calls == 2 else 1, 'unusable'
        return original(*args, **kwargs)

    monkeypatch.setattr(system, 'fsolve', isolated_bad_root)
    actual = binary.compute_equipotential_boundary(1., 'xy')
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_lagrange_terminal_failure_after_all_attempts(monkeypatch):
    calls = 0

    def no_root(*args, **kwargs):
        nonlocal calls
        calls += 1
        return np.array([np.nan]), {}, 4, 'no root'

    monkeypatch.setattr(system, 'fsolve', no_root)
    with pytest.raises(error.ElisaFloatingPointError, match='Lagrange solver'):
        system.BinarySystem.lagrangian_points_static(1., .7)
    assert calls > 1


@pytest.mark.parametrize('kind', ['single', 'detached', 'over-contact'])
def test_real_degenerate_triangulation_has_numerical_type_and_cause(kind):
    points = np.array([[x, 0., 0.] for x in range(1, 6)])
    with pytest.raises(error.ElisaFloatingPointError) as caught:
        if kind == 'single':
            single_surface(points=points)
        else:
            component = SimpleNamespace(name='primary', points=points, critical_surface_potential=3., surface_potential=6.)
            container = SimpleNamespace(primary=component)
            if kind == 'detached':
                faces.detached_system_surface(container, 1.)
            else:
                faces.over_contact_system_surface(container)
    assert isinstance(caught.value.__cause__, QhullError)


@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, TypeError, IndexError])
def test_triangulation_programming_errors_propagate(monkeypatch, error_type):
    def broken(*args):
        raise error_type('broken triangulator')

    monkeypatch.setattr('elisa.single_system.surface.faces.Delaunay', broken)
    with pytest.raises(error_type, match='broken triangulator'):
        single_surface(points=np.ones((5, 3)))


def test_kepler_function_runtime_error_propagates(monkeypatch):
    orbit = Orbit(period=1., eccentricity=.1, argument_of_periastron=0., inclination=1.)

    def broken(*args):
        raise RuntimeError('broken Kepler function')

    monkeypatch.setattr(orbit, 'mean_anomaly_fn', broken)
    with pytest.raises(RuntimeError, match='broken Kepler function'):
        orbit.mean_anomaly_to_eccentric_anomaly(.5)


def test_temperature_explicit_zero_flux_failures():
    star = SimpleNamespace(areas=np.ones(3), potential_gradient_magnitudes=np.zeros(3),
                           polar_potential_gradient_magnitude=1., gravity_darkening=.32,
                           has_spots=lambda: False)
    with pytest.raises(error.ElisaFloatingPointError, match='polar effective temperature'):
        temperature.calculate_polar_effective_temperature(star)
    star.spots, star.temperatures, star.t_eff = {}, np.zeros(3), 6000.
    with pytest.raises(error.ElisaFloatingPointError, match='bolometric flux'):
        temperature.renormalize_temperatures(star)


def _real_worker_mode_failure():
    PulsationMode(**(MODE | dict(m=2)))


def test_real_domain_failure_crosses_multiprocessing_boundary():
    with multiprocessing.get_context('fork').Pool(1) as pool:
        with pytest.raises(error.PulsationModeError, match='cannot be higher') as caught:
            pool.apply_async(_real_worker_mode_failure).get(timeout=30)
    assert '_real_worker_mode_failure' in str(caught.value.__cause__)
