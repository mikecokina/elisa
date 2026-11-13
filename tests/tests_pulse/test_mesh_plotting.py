"""Public pulsating mesh plots must use prepared geometry at the requested time."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa import BinarySystem, Observer, SingleSystem
from elisa import units as u
from elisa.binary_system import dynamic
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.graphic import graphics
from elisa.single_system.container import SinglePositionContainer
from tests.tests_pulse.test_temperature_diagnostics import make_system, thermal_modes

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.types import Float
    from tests.tests_pulse.test_temperature_diagnostics import SystemKind


def expected_vertices(equilibrium_star: StarContainer, length_scale: Float, centre_x: Float) -> NDArray:
    """Reconstruct displacement from harmonic fields without using displacement routines."""
    equilibrium_points = cast("NDArray", equilibrium_star.points)
    relative_points = equilibrium_points - [centre_x, 0, 0]
    radii = np.linalg.norm(relative_points, axis=1)
    mode_rotation = equilibrium_star.pulsation_rotation
    assert mode_rotation is not None
    mode_directions = (relative_points / radii[:, None]) @ mode_rotation
    predicted_points = equilibrium_points.copy()
    for harmonic_mode in equilibrium_star.pulsations.values():
        assert harmonic_mode.point_harmonics is not None
        assert harmonic_mode.point_harmonics_gradient is not None
        mode_vector = (
            harmonic_mode.radial_amplitude / length_scale * harmonic_mode.point_harmonics[:, None] * mode_directions
        )
        if harmonic_mode.l != 0 and harmonic_mode.horizontal_amplitude != 0:
            weighted_gradient = radii[:, None] * harmonic_mode.point_harmonics_gradient
            divisor = np.sqrt(np.mean(np.sum(np.abs(weighted_gradient) ** 2, axis=1)))
            mode_vector += harmonic_mode.horizontal_amplitude / length_scale * weighted_gradient / divisor
        predicted_points += (mode_vector @ mode_rotation.T).real
    return predicted_points


def forbid_observer(*_observer_args: Any, **_observer_options: Any) -> None:
    pytest.fail("A geometry plot must not prepare an Observer or radiance")


@pytest.mark.requires_ld
@pytest.mark.parametrize("kind", ["single", "binary"])
@pytest.mark.parametrize("spotted", [False, True])
@pytest.mark.parametrize("scenario", ["one", "multiple"])
@pytest.mark.parametrize("requested_phase", [0.2, 1.2])
def test_pulsating_mesh_vertices(
    kind: SystemKind,
    spotted: bool,
    scenario: str,
    requested_phase: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = make_system(kind, thermal_modes(scenario), spotted, secondary_modes=True)
    input_state = pickle.dumps(system)
    reference = system.build_container(phase=requested_phase, build_pulsations=False)
    reference.build_harmonics()
    if isinstance(reference, SinglePositionContainer):
        predicted = {"mesh": expected_vertices(reference.star, 1.0, 0.0)}
    else:
        predicted = {
            "points_primary": expected_vertices(reference.primary, reference.semi_major_axis, 0.0),
            "points_secondary": expected_vertices(
                reference.secondary,
                reference.semi_major_axis,
                reference.position.distance,
            ),
        }
    monkeypatch.setattr(Observer, "__init__", forbid_observer)
    renderer_name = "single_star_mesh" if kind == "single" else "binary_mesh"
    monkeypatch.setattr(graphics, renderer_name, lambda **mesh_options: mesh_options)
    call_options: dict[str, Any] = {"axis_unit": u.m} if kind == "single" else {}
    for _ in range(2):
        rendered = cast(
            "dict[str, Any]",
            system.plot.mesh(
                phase=requested_phase,
                return_figure_instance=True,
                **call_options,
            ),
        )
        for vertex_key, predicted_vertices in predicted.items():
            # Bound roundoff on the surface scale, including cancellation near
            # Cartesian nodes when independent mode contributions are combined.
            coordinate_tolerance = 4 * np.finfo(float).eps * np.max(np.abs(predicted_vertices))
            assert_allclose(rendered[vertex_key], predicted_vertices, rtol=0, atol=coordinate_tolerance)
        assert rendered["phase"] == requested_phase
    assert pickle.dumps(system) == input_state


@pytest.mark.parametrize("kind", ["single", "binary"])
@pytest.mark.parametrize("spotted", [False, True])
@pytest.mark.parametrize("requested_phase", [0.2, 1.2])
def test_non_pulsating_mesh_remains_lightweight(
    kind: SystemKind,
    spotted: bool,
    requested_phase: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = make_system(kind, [], spotted)
    input_state = pickle.dumps(system)
    if isinstance(system, SingleSystem):
        position = system.calculate_lines_of_sight(requested_phase)[0]
        raw_container = SinglePositionContainer.from_single_system(system, position)
        raw_container.build_mesh()
        predicted = {"mesh": raw_container.star.get_flatten_parameter("points") / u.solRad.to(u.m)}
    else:
        binary_position = system.calculate_orbital_motion(requested_phase)[0]
        raw_binary = OrbitalPositionContainer.from_binary_system(system, binary_position)
        dynamic.assign_spot_longitudes(
            raw_binary,
            dynamic.calculate_spot_longitudes(system, requested_phase),
            index=None,
            component="all",
        )
        raw_binary.build_mesh(components_distance=binary_position.distance)
        predicted = {
            "points_primary": raw_binary.primary.get_flatten_parameter("points").copy(),
            "points_secondary": raw_binary.secondary.get_flatten_parameter("points").copy(),
        }
    container_class = SinglePositionContainer if kind == "single" else OrbitalPositionContainer
    monkeypatch.setattr(container_class, "build", forbid_observer)
    monkeypatch.setattr(Observer, "__init__", forbid_observer)
    renderer_name = "single_star_mesh" if kind == "single" else "binary_mesh"
    monkeypatch.setattr(graphics, renderer_name, lambda **mesh_options: mesh_options)
    rendered = cast("dict[str, Any]", system.plot.mesh(phase=requested_phase, return_figure_instance=True))
    for vertex_key, predicted_vertices in predicted.items():
        assert_allclose(rendered[vertex_key], predicted_vertices, rtol=2e-16, atol=0)
    assert pickle.dumps(system) == input_state


@pytest.mark.requires_ld
@pytest.mark.parametrize("selection", ["primary", "secondary", "both"])
def test_binary_mesh_preserves_component_selection(
    selection: Literal["primary", "secondary", "both"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = cast("BinarySystem", make_system("binary", thermal_modes("multiple"), True, secondary_modes=True))
    reference = binary.build_container(phase=0.2)
    monkeypatch.setattr(graphics, "binary_mesh", lambda **mesh_options: mesh_options)
    rendered = cast("dict[str, Any]", binary.plot.mesh(phase=0.2, components_to_plot=selection))
    for selected_name in ("primary", "secondary"):
        vertex_key = f"points_{selected_name}"
        if selection in (selected_name, "both"):
            assert_array_equal(rendered[vertex_key], getattr(reference, selected_name).points)
        else:
            assert vertex_key not in rendered


def test_unselected_binary_pulsator_does_not_force_full_build(monkeypatch: pytest.MonkeyPatch) -> None:
    binary_model = cast("BinarySystem", make_system("binary", thermal_modes("one"), True))
    binary_position = binary_model.calculate_orbital_motion(0.2)[0]
    raw_reference = OrbitalPositionContainer.from_binary_system(binary_model, binary_position)
    dynamic.assign_spot_longitudes(
        raw_reference, dynamic.calculate_spot_longitudes(binary_model, 0.2), index=None, component="all",
    )
    raw_reference.build_mesh(components_distance=binary_position.distance)
    expected_secondary = raw_reference.secondary.get_flatten_parameter("points").copy()
    monkeypatch.setattr(OrbitalPositionContainer, "build", forbid_observer)
    monkeypatch.setattr(Observer, "__init__", forbid_observer)
    monkeypatch.setattr(graphics, "binary_mesh", lambda **mesh_options: mesh_options)
    mesh_data = cast("dict[str, Any]", binary_model.plot.mesh(phase=0.2, components_to_plot="secondary"))
    assert_array_equal(mesh_data["points_secondary"], expected_secondary)
    assert "points_primary" not in mesh_data
