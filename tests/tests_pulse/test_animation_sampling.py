"""Animation phase grids must reach the renderer with synchronized physical time."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa.base.error import ElisaGraphicsError
from elisa.base.graphics import plot
from elisa.graphic import graphics
from tests.tests_pulse.test_temperature_diagnostics import make_system, thermal_modes

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa import BinarySystem, SingleSystem
    from elisa.types import Float
    from tests.tests_pulse.test_temperature_diagnostics import PositionContainer, SystemKind


@pytest.mark.requires_ld
@pytest.mark.requires_atmosphere
@pytest.mark.parametrize("kind", ["single", "binary"])
@pytest.mark.parametrize(
    ("start_phase", "stop_phase", "phase_step", "expected_phases"),
    [
        (0.0, 0.2, 0.1, [0.0, 0.1, 0.2]),
        (0.2, 0.31, 0.05, [0.2, 0.25, 0.3]),
        (0.2, 0.2, 0.05, [0.2]),
        (0.2, 0.21, 0.05, [0.2]),
        (0.1, 0.3, 0.1, [0.1, 0.2, 0.3]),
        (0.2, 0.3, 0.1, [0.2, 0.3]),
        (-0.3, 0.0, 0.1, [-0.3, -0.2, -0.1, 0.0]),
        (0.0, 0.1 + 0.2, 0.1, [0.0, 0.1, 0.2, 0.1 + 0.2]),
        (0.9, 1.1, 0.1, [0.9, 1.0, 1.1]),
        (1.2, 1.31, 0.05, [1.2, 1.25, 1.3]),
    ],
)
def test_public_animation_sampling_and_time(
    kind: SystemKind,
    start_phase: float,
    stop_phase: float,
    phase_step: float,
    expected_phases: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = make_system(kind, thermal_modes("multiple"), secondary_modes=True)
    model_before = pickle.dumps(system)
    original_prepare = plot.build_surface_with_temperature_reference
    prepared_times: list[Float] = []
    prepared_phases: list[Float] = []
    payloads: list[dict[str, Any]] = []

    def record_preparation(
        frame_container: PositionContainer,
        *prepare_args: Any,
        **prepare_options: Any,
    ) -> dict[str, NDArray]:
        prepared_phases.append(frame_container.position.phase)
        prepared_times.append(frame_container.time)
        return original_prepare(frame_container, *prepare_args, **prepare_options)

    def capture_animation(**render_options: Any) -> None:
        payloads.append(render_options)

    monkeypatch.setattr(plot, "build_surface_with_temperature_reference", record_preparation)
    monkeypatch.setattr(graphics, f"{kind}_surface_anim", capture_animation)
    animate_system = (
        cast("SingleSystem", system).animation.rotational_motion
        if kind == "single"
        else cast("BinarySystem", system).animation.orbital_motion
    )
    animate_system(start_phase=start_phase, stop_phase=stop_phase, phase_step=phase_step, colormap="temperature")
    [animation_data] = payloads
    actual_phases = animation_data["phases"]
    rounding = 2 * np.finfo(float).eps * max(abs(start_phase), abs(stop_phase), phase_step)
    assert_allclose(actual_phases, expected_phases, rtol=0, atol=rounding)
    assert actual_phases[0] == start_phase
    assert np.all(actual_phases <= stop_phase)
    assert_allclose(np.diff(actual_phases), phase_step, rtol=0, atol=rounding)
    assert animation_data["n_frames"] == len(expected_phases)
    assert_array_equal(prepared_phases, actual_phases)
    period = cast("SingleSystem", system).rotation_period if kind == "single" else cast("BinarySystem", system).period
    assert_array_equal(prepared_times, 86400 * period * actual_phases)
    assert pickle.dumps(system) == model_before


@pytest.mark.parametrize("kind", ["single", "binary"])
@pytest.mark.parametrize(
    ("invalid_start", "invalid_stop", "invalid_step"),
    [
        (0.0, 0.2, 0.0),
        (0.0, 0.2, -0.1),
        (0.0, 0.2, np.nan),
        (0.0, 0.2, np.inf),
        (0.0, 0.2, -np.inf),
        (np.nan, 0.2, 0.1),
        (-np.inf, 0.2, 0.1),
        (0.0, np.inf, 0.1),
        (0.0, np.nan, 0.1),
        (0.2, 0.0, 0.1),
        (1.0, 1.2, np.nextafter(0.0, 1.0)),
        (-1e308, 1e308, 1.0),
    ],
)
def test_public_animation_rejects_invalid_grid(
    kind: SystemKind,
    invalid_start: float,
    invalid_stop: float,
    invalid_step: float,
) -> None:
    system = make_system(kind, [])
    animate_system = (
        cast("SingleSystem", system).animation.rotational_motion
        if kind == "single"
        else cast("BinarySystem", system).animation.orbital_motion
    )
    with pytest.raises(ElisaGraphicsError):
        animate_system(start_phase=invalid_start, stop_phase=invalid_stop, phase_step=invalid_step)


@pytest.mark.parametrize("large_start", [1e8, 1e12, -1e12])
@pytest.mark.parametrize("stop_in_steps", [4.0, 4.25, 4.75])
def test_large_phase_endpoint_does_not_stretch_grid(large_start: float, stop_in_steps: float) -> None:
    """All increments and quarter-step offsets are exactly representable here."""
    requested_step = 8 * np.spacing(abs(large_start))
    requested_stop = large_start + stop_in_steps * requested_step
    expected_grid = large_start + np.array([0, 1, 2, 3, 4]) * requested_step
    actual_grid = plot.animation_phases(large_start, requested_stop, requested_step)
    assert_array_equal(actual_grid, expected_grid)
    assert_array_equal(np.diff(actual_grid), np.full(4, requested_step))
    assert actual_grid[-1] <= requested_stop


@pytest.mark.parametrize("large_origin", [1e8, 1e12, -1e12])
@pytest.mark.parametrize("step_ulps", [1, 2])
def test_small_representable_step_keeps_each_grid_point_once(large_origin: float, step_ulps: int) -> None:
    representable_step = step_ulps * np.spacing(abs(large_origin))
    expected_samples = large_origin + np.array([0, 1, 2, 3, 4]) * representable_step
    sampled_phases = plot.animation_phases(large_origin, expected_samples[-1], representable_step)
    assert_array_equal(sampled_phases, expected_samples)
    assert_array_equal(np.diff(sampled_phases), np.full(4, representable_step))


def test_rounded_large_phase_grid_endpoint_is_included() -> None:
    """Subtraction loses relative precision, but the generated grid reaches the stop exactly."""
    grid_start = 1e8
    grid_step = 0.01
    expected_points = grid_start + np.arange(11) * grid_step
    assert_array_equal(plot.animation_phases(grid_start, expected_points[-1], grid_step), expected_points)


@pytest.mark.parametrize(
    ("precision_start", "precision_stop", "precision_step"),
    [(np.nextafter(1.0, 0.0), 1.0 + 2**-51, 2**-53), (1e12, 1e12 + 2**-10, 0.0001)],
)
def test_grid_rejects_steps_that_collapse_to_duplicate_phases(
    precision_start: float, precision_stop: float, precision_step: float,
) -> None:
    assert precision_start + precision_step > precision_start
    with pytest.raises(ElisaGraphicsError, match="distinct"):
        plot.animation_phases(precision_start, precision_stop, precision_step)
