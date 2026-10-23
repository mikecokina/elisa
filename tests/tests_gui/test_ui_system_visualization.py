from __future__ import annotations

# ruff: noqa: S101

from elisa.ui.tabs.system_visualization.components import observer_inputs
from elisa.ui.tabs.system_visualization.logic import compute


class _DummyStar:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _DummyPlot:
    def __init__(self) -> None:
        self.wireframe_calls: list[dict[str, object]] = []

    def wireframe(self, **kwargs: object) -> str:
        self.wireframe_calls.append(kwargs)
        return "wireframe_fig"


class _DummyBinarySystem:
    last_instance: _DummyBinarySystem | None = None

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.plot = _DummyPlot()
        _DummyBinarySystem.last_instance = self


def test_update_ui_enables_expected_controls_for_wireframe() -> None:
    updates = observer_inputs.update_ui("wireframe")

    assert updates[0].get("interactive") is True  # phase
    assert updates[1].get("interactive") is True  # components_to_plot
    assert updates[2].get("interactive") is False  # plane
    assert updates[3].get("interactive") is False  # frame_of_reference
    assert updates[4].get("interactive") is False  # colormap
    assert updates[5].get("interactive") is True  # elevation
    assert updates[6].get("interactive") is True  # azimuth


def test_run_visualization_dispatches_wireframe_with_camera_params(monkeypatch) -> None:
    monkeypatch.setattr(compute, "Star", _DummyStar)
    monkeypatch.setattr(compute, "BinarySystem", _DummyBinarySystem)
    monkeypatch.setattr(compute, "parse_pulsation_modes", lambda _: [])
    monkeypatch.setattr(compute, "parse_spots", lambda _: [])
    monkeypatch.setattr(compute, "figure_to_pil", lambda figure: figure)

    result = compute.run_visualization(
        primary_params={"mass": "2.1", "t_eff": "10000", "surface_potential": "3.6", "synchronicity": "1.0"},
        secondary_params={"mass": "1.2", "t_eff": "7000", "surface_potential": "4.0", "synchronicity": "1.0"},
        system_params={"inclination": "87", "period": "2.5", "eccentricity": "0.0", "argument_of_periastron": "90"},
        observer_params={
            "visualization_mode": "wireframe",
            "phase": "0.25",
            "components_to_plot": "both",
            "plane": "xy",
            "frame_of_reference": "primary",
            "colormap": None,
            "elevation": "20",
            "azimuth": "110",
        },
    )

    assert result == ("wireframe_fig", None, None, None)

    binary = _DummyBinarySystem.last_instance
    assert binary is not None
    assert len(binary.plot.wireframe_calls) == 1

    wireframe_call = binary.plot.wireframe_calls[0]
    assert wireframe_call["phase"] == 0.25
    assert wireframe_call["components_to_plot"] == "both"
    assert wireframe_call["inclination"] == 20.0
    assert wireframe_call["azimuth"] == 110.0
    assert wireframe_call["return_figure_instance"] is True

