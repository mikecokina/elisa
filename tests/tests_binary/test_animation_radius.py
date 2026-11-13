"""Binary animation coordinates and diagnostics must share the component frame."""

from __future__ import annotations

import pickle
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa.base.graphics import plot
from elisa.graphic import graphics
from tests.tests_pulse.test_temperature_diagnostics import make_system, thermal_modes

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa import BinarySystem
    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.types import Float


def observer_rotation(azimuth: Float, inclination: Float) -> NDArray:
    """Construct the handbook's active rotation without using graphics utilities."""
    z_angle = azimuth + np.pi / 2
    y_angle = inclination - np.pi / 2
    rotate_z = np.array(
        [
            [np.cos(z_angle), -np.sin(z_angle), 0],
            [np.sin(z_angle), np.cos(z_angle), 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    rotate_y = np.array(
        [
            [np.cos(y_angle), 0, np.sin(y_angle)],
            [0, 1, 0],
            [-np.sin(y_angle), 0, np.cos(y_angle)],
        ],
        dtype=float,
    )
    return rotate_y @ rotate_z


@pytest.mark.requires_ld
@pytest.mark.requires_atmosphere
@pytest.mark.parametrize("pulsating", [False, True])
@pytest.mark.parametrize("spotted", [False, True])
@pytest.mark.parametrize(
    "colormap",
    ["radius", "temperature", "velocity", "radial_velocity", "gravity_acceleration", "radiance"],
)
def test_binary_animation_preserves_radius_and_rendered_geometry(
    pulsating: bool,
    spotted: bool,
    colormap: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = cast(
        "BinarySystem",
        make_system(
            "binary",
            thermal_modes("multiple") if pulsating else [],
            spotted,
            secondary_modes=True,
        ),
    )
    model_before = pickle.dumps(binary)
    original_build = plot.build_surface_with_temperature_reference
    original_colormap = plot.add_colormap_to_plt_kwargs
    animation_payloads: list[dict[str, Any]] = []
    # Each frame's independent reference comes from physical component coordinates
    # before the animation performs any display-coordinate transformation.
    component_points: dict[tuple[float, str], NDArray] = {}
    expected_radii: dict[tuple[float, str], NDArray] = {}
    expected_render_points: dict[tuple[float, str], NDArray] = {}

    def record_surface_build(
        frame_container: OrbitalPositionContainer,
        *build_args: Any,
        **build_options: Any,
    ) -> dict[str, NDArray]:
        references = original_build(frame_container, *build_args, **build_options)
        frame_position = frame_container.position
        transform = observer_rotation(frame_position.azimuth, binary.inclination)
        barycentre = frame_position.distance * binary.mass_ratio / (1 + binary.mass_ratio)
        for surface_name, centre_x in [("primary", 0.0), ("secondary", frame_position.distance)]:
            surface_star = getattr(frame_container, surface_name)
            reference_key = (float(frame_position.phase), str(surface_star.name))
            physical_points = surface_star.points.copy()
            component_points[reference_key] = physical_points @ transform.T
            vertex_radii = np.linalg.norm(physical_points - [centre_x, 0, 0], axis=1)
            expected_radii[reference_key] = vertex_radii[surface_star.faces].mean(axis=1) * binary.semi_major_axis
            # This is exactly the old renderer geometry: translate, then rotate.
            expected_render_points[reference_key] = (physical_points - [barycentre, 0, 0]) @ transform.T
        return references

    def check_colormap(*map_args: Any, **map_options: Any) -> NDArray | None:
        diagnostic_name, observed_star, requested_phase = map_args[:3]
        physical_state_before = pickle.dumps(observed_star)
        actual_values = cast("NDArray", original_colormap(*map_args, **map_options))
        assert pickle.dumps(observed_star) == physical_state_before
        reference_key = (float(requested_phase), str(observed_star.name))
        # Observer rotation is expected; a renderer-only translation is not.
        assert_allclose(observed_star.points, component_points[reference_key], rtol=0, atol=5e-16)
        if diagnostic_name == "radius":
            assert_allclose(actual_values, expected_radii[reference_key], rtol=2e-15, atol=2e-6)
        else:
            # Recreate the old translated input explicitly. Every other tested
            # diagnostic must retain its exact numerical values after the fix.
            legacy_star = deepcopy(observed_star)
            legacy_star.points -= (
                np.array(
                    [
                        map_args[6].distance * binary.mass_ratio / (1 + binary.mass_ratio),
                        0,
                        0,
                    ],
                    dtype=float,
                )
                @ observer_rotation(map_args[6].azimuth, binary.inclination).T
            )
            assert_array_equal(
                actual_values,
                original_colormap(diagnostic_name, legacy_star, *map_args[2:], **map_options),
            )
        return actual_values

    def capture_animation(**animation_options: Any) -> None:
        animation_payloads.append(animation_options)

    monkeypatch.setattr(plot, "build_surface_with_temperature_reference", record_surface_build)
    monkeypatch.setattr(plot, "add_colormap_to_plt_kwargs", check_colormap)
    monkeypatch.setattr(graphics, "binary_surface_anim", capture_animation)
    monkeypatch.setattr(graphics, "binary_surface", lambda **surface_options: surface_options)
    binary.animation.orbital_motion(start_phase=0.2, stop_phase=0.31, phase_step=0.05, colormap=colormap)
    [animation_data] = animation_payloads
    for frame_index, frame_phase in enumerate(animation_data["phases"]):
        static_data = cast("dict[str, Any]", binary.plot.surface(phase=frame_phase, colormap=colormap))
        # The two stars have separate references and renderer coordinate arrays.
        for component_label in ("primary", "secondary"):
            frame_key = (float(frame_phase), str(getattr(binary, component_label).name))
            rendered_vertices = animation_data[f"points_{component_label}"][frame_index]
            assert_allclose(rendered_vertices, expected_render_points[frame_key], rtol=0, atol=5e-16)
            assert_allclose(rendered_vertices, static_data[f"points_{component_label}"], rtol=0, atol=5e-16)
            # Static plots may reverse face winding. Reordered vertex means can
            # differ by a few floating-point rounding units without changing physics.
            assert_allclose(
                animation_data[f"{component_label}_cmap"][frame_index],
                static_data[f"{component_label}_cmap"],
                rtol=3e-15,
                atol=1e-12,
            )
            assert_array_equal(
                np.sort(animation_data[f"faces_{component_label}"][frame_index], axis=1),
                np.sort(static_data[f"{component_label}_triangles"], axis=1),
            )
    assert pickle.dumps(binary) == model_before
