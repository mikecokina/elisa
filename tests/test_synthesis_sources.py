"""Expected failures at real synthesis sources, with local recovery preserved."""

from __future__ import annotations

import multiprocessing
from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    import decimal
    import fractions
    from pathlib import Path

    from numpy.typing import NDArray


MODE = {"l": 1, "m": 1, "frequency": 16.0, "amplitude": 50.0}


def spotted_container() -> OrbitalPositionContainer:
    binary = utils.prepare_binary_system(
        utils.BINARY_SYSTEM_PARAMS["detached"],
        spots_primary=utils.SPOTS_META["primary"],
    )
    return OrbitalPositionContainer.from_binary_system(binary, Position(0, 1.0, 0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "changes",
    [
        {"l": -1},
        {"m": 2},
        {"l": 1.001},
        {"m": np.float32(0.5)},
        {"l": np.nan},
        {"m": np.inf},
        {"l": Decimal("1.5")},
        {"m": Fraction(1, 2)},
        {"amplitude": -1},
        {"frequency": -1},
        {"mode_axis_theta": -1},
        {"mode_axis_theta": 180},
        {"mode_axis_phi": np.inf},
    ],
)
def test_real_mode_valid_invalid_valid(
    changes: dict[str, decimal.Decimal]
    | dict[str, float]
    | dict[str, fractions.Fraction]
    | dict[str, int]
    | dict[str, np.float32],
) -> None:
    before = PulsationMode(**MODE)
    with pytest.raises(error.ElisaParametersError):
        PulsationMode(**(MODE | changes))
    after = PulsationMode(**MODE)
    for key in MODE:
        assert getattr(before, key) == getattr(after, key)
    assert after.frequency == 16 / 86400  # Direct constructor conversion remains unchanged.


@pytest.mark.parametrize("index", ["l", "m"])
@pytest.mark.parametrize("value", [object(), "1", [1], 1j])
def test_incompatible_mode_index_types_are_model_parameter_errors(
    index: str,
    value: complex | list[int] | object | str,
) -> None:
    with pytest.raises(error.ElisaParametersError):
        PulsationMode(**(MODE | {index: value}))


@pytest.mark.parametrize("changes", [{"frequency": 0}, {"l": 0, "m": 0}])
def test_amplitude_source_failure(changes: dict[str, int]) -> None:
    star = SimpleNamespace(equivalent_radius=1e9, mass=2e30, pulsations={0: PulsationMode(**(MODE | changes))})
    with pytest.raises(error.PulsationModeError):
        pulsations.assign_amplitudes(star)


def test_zero_amplitude_is_valid_and_finite() -> None:
    star = SimpleNamespace(equivalent_radius=1e9, mass=2e30, pulsations={0: PulsationMode(**(MODE | {"amplitude": 0}))})
    pulsations.assign_amplitudes(star)
    mode = star.pulsations[0]
    assert mode.radial_amplitude == mode.horizontal_amplitude == mode.temperature_amplitude_factor == 0


def test_pulsation_zero_denominators() -> None:
    with pytest.raises(error.PulsationModeError, match="zero denominator"):
        pulsations.horizontal_displacement_normalization([np.zeros(3)] * 2, [np.ones(3)] * 2)
    with pytest.raises(error.PulsationModeError, match="nonzero horizontal"):
        pulsations.temp_amplitude(SimpleNamespace(horizontal_to_radial_amplitude_ratio=0))


@pytest.mark.parametrize("by_vertices", [True, False])
def test_missing_retained_spot_is_rejected(by_vertices: bool) -> None:
    container = spotted_container().primary
    count = len(container.spots)
    with pytest.raises(error.ElisaSpotError):  # noqa: PT012 - scenario includes context cleanup or selection of the operation under test
        if by_vertices:
            remove_overlaped_spots_by_vertex_map(container, [{"enum": -1}])
        else:
            remove_overlaped_spots_by_spot_index(container, [])
    assert len(container.spots) == count


def test_real_complete_spot_overlap_rejected() -> None:
    spot = utils.SPOTS_META["primary"][0]
    binary = utils.prepare_binary_system(utils.BINARY_SYSTEM_PARAMS["detached"], spots_primary=[spot, spot.copy()])
    container = utils.prepare_orbital_position_container(binary)
    with pytest.raises(error.ElisaSpotError, match="have no mesh points"):
        container.build(components_distance=1.0)


def test_surface_boundary_skips_isolated_bad_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    binary = utils.prepare_binary_system(utils.BINARY_SYSTEM_PARAMS["over-contact"])
    expected = binary.compute_equipotential_boundary(1.0, "xy")
    original = system.fsolve
    calls = 0

    def isolated_bad_root(
        *args,
        **kwargs,
    ) -> (
        tuple[NDArray[np.float64], dict[object, object], int, str]
        | tuple[NDArray[np.float64], dict[str, int | NDArray[np.float64]], int, str]
    ):
        nonlocal calls
        calls += 1
        if calls <= 3:
            return np.array([np.nan if calls == 1 else -1.0]), {}, 4 if calls == 2 else 1, "unusable"
        return original(*args, **kwargs)

    monkeypatch.setattr(system, "fsolve", isolated_bad_root)
    actual = binary.compute_equipotential_boundary(1.0, "xy")
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_lagrange_terminal_failure_after_all_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    system.BinarySystem._lagrangian_points_static.cache_clear()
    calls = 0

    def no_root(*_args, **_kwargs) -> tuple[NDArray[np.float64], dict[object, object], int, str]:
        nonlocal calls
        calls += 1
        return np.array([np.nan]), {}, 4, "no root"

    monkeypatch.setattr(system, "fsolve", no_root)
    with pytest.raises(error.ElisaMaxIterationError, match="Lagrange solver"):
        system.BinarySystem.lagrangian_points_static(1.0, 0.7)
    assert calls > 1


@pytest.mark.parametrize("kind", ["single", "detached", "over-contact"])
def test_real_degenerate_triangulation_has_numerical_type_and_cause(kind: str) -> None:
    points = np.array([[x, 0.0, 0.0] for x in range(1, 6)])
    with pytest.raises(error.ElisaNumericError) as caught:  # noqa: PT012 - scenario includes context cleanup or selection of the operation under test
        if kind == "single":
            single_surface(points=points)
        else:
            component = SimpleNamespace(
                name="primary",
                points=points,
                critical_surface_potential=3.0,
                surface_potential=6.0,
            )
            container = SimpleNamespace(primary=component)
            if kind == "detached":
                faces.detached_system_surface(container, 1.0)
            else:
                faces.over_contact_system_surface(container)
    assert isinstance(caught.value.__cause__, QhullError)


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, TypeError, IndexError])
def test_triangulation_programming_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[IndexError | RuntimeError | TypeError | ValueError],
) -> None:
    def broken(*_args) -> None:
        raise error_type("broken triangulator")

    monkeypatch.setattr("elisa.single_system.surface.faces.Delaunay", broken)
    with pytest.raises(error_type, match="broken triangulator"):
        single_surface(points=np.ones((5, 3)))


def test_kepler_function_runtime_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    orbit = Orbit(period=1.0, eccentricity=0.1, argument_of_periastron=0.0, inclination=1.0)

    def broken(*_args) -> None:
        raise RuntimeError("broken Kepler function")

    monkeypatch.setattr(orbit, "mean_anomaly_fn", broken)
    with pytest.raises(RuntimeError, match="broken Kepler function"):
        orbit.mean_anomaly_to_eccentric_anomaly(0.5)


def test_temperature_explicit_zero_flux_failures() -> None:
    star = SimpleNamespace(
        areas=np.ones(3),
        potential_gradient_magnitudes=np.zeros(3),
        polar_potential_gradient_magnitude=1.0,
        gravity_darkening=0.32,
        has_spots=lambda: False,
    )
    with pytest.raises(error.ElisaTemperatureError, match="polar effective temperature"):
        temperature.calculate_polar_effective_temperature(star)
    star.spots, star.temperatures, star.t_eff = {}, np.zeros(3), 6000.0
    with pytest.raises(error.ElisaTemperatureError, match="bolometric flux"):
        temperature.renormalize_temperatures(star)


def _real_worker_mode_failure() -> None:
    PulsationMode(**(MODE | {"m": 2}))


def test_real_domain_failure_crosses_multiprocessing_boundary() -> None:
    with (
        multiprocessing.get_context("fork").Pool(1) as pool,
        pytest.raises(error.ElisaParametersError, match="cannot be higher") as caught,
    ):
        pool.apply_async(_real_worker_mode_failure).get(timeout=30)
    assert "_real_worker_mode_failure" in str(caught.value.__cause__)


@pytest.mark.parametrize("value", [0.0, -1.0])
@pytest.mark.parametrize("interpolation", ["albedo", "gravity_darkening"])
def test_temperature_interpolation_rejects_nonpositive_candidates(value: float, interpolation: str) -> None:
    from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
    from elisa.binary_system.surface.temperature import interpolate_albedo

    fn = interpolate_albedo if interpolation == "albedo" else temperature.interpolate_bolometric_gravity_darkening
    with pytest.raises(error.ElisaTemperatureError, match="requires a positive effective temperature") as caught:
        fn(value)
    assert isinstance(caught.value, SYNTHESIS_FAILURES)


@pytest.mark.parametrize("resource", ["beta", "albedo"])
@pytest.mark.parametrize("failure", ["missing", "invalid_json", "missing_keys"])
def test_fixed_temperature_tables_fail_as_unexpected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resource: str,
    failure: str,
) -> None:
    from elisa import settings
    from elisa.binary_system.surface.temperature import interpolate_albedo

    path = tmp_path / f"{resource}.json"
    if failure == "invalid_json":
        path.write_text("{broken")
    elif failure == "missing_keys":
        path.write_text('{"x": [3.0, 4.0]}')
    field = "PATH_TO_BETA" if resource == "beta" else "PATH_TO_ALBEDOS"
    monkeypatch.setattr(type(settings), field, path)
    fn = temperature.interpolate_bolometric_gravity_darkening if resource == "beta" else interpolate_albedo
    with pytest.raises(error.ElisaUnexpectedError, match=r"fixed .*table") as caught:
        fn(6000.0)
    cause_type = {"missing": FileNotFoundError, "invalid_json": ValueError, "missing_keys": KeyError}[failure]
    assert isinstance(caught.value.__cause__, cause_type)


@pytest.mark.parametrize(("points", "symmetry_count"), [(np.empty((0, 3)), 1), (np.ones((1, 3)), 0)])
def test_detached_mesh_reuse_requires_prepared_symmetric_mesh(points: NDArray[np.generic], symmetry_count: int) -> None:
    from elisa.binary_system.surface import mesh

    star = SimpleNamespace(
        points=points,
        base_symmetry_points_number=symmetry_count,
        synchronicity=1.0,
        surface_potential=5.0,
    )
    container = SimpleNamespace(primary=star, mass_ratio=1.0)
    with pytest.raises(error.ElisaUnexpectedError, match="Cannot rebuild the detached mesh"):
        mesh.rebuild_mesh_detached(container, 1.0, "primary")


def test_surface_preparation_must_return_requested_values(monkeypatch: pytest.MonkeyPatch) -> None:
    from elisa.binary_system.curves import utils as curve_utils

    monkeypatch.setattr(curve_utils, "prep_surface_params", lambda *_args, **_kwargs: None)
    with pytest.raises(error.ElisaUnexpectedError, match="unexpectedly returned None"):
        curve_utils.update_surface_params(None, {}, {}, require_rebuild=True)
