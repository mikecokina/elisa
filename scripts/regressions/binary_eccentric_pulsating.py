# ruff: noqa: INP001 -- Independently executable script, not a package.
"""Standalone Eccentric pulsating visual regression.

Uses the standard mass-based JSON schema (solar masses, days, kelvin,
degrees and SI velocities).
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends import BackendFilter, backend_registry
from PIL import Image, ImageSequence

from elisa import BinarySystem, Observer
from elisa.base.graphics.plot import animation_phases
from elisa.graphic import graphics
from elisa.graphic.graphics import binary_surface_anim as render_native_animation

DISCRETIZATION_FACTOR = 5.0
BINARY_DEFINITION = {
    "system": {
        "inclination": 85.0,
        "period": 2.5,
        "argument_of_periastron": 58.0,
        "gamma": -30700.0,
        "eccentricity": 0.2,
        "primary_minimum_time": 2440000.0,
        "phase_shift": 0.0,
        "distance": 155.0,
    },
    "primary": {
        "mass": 2.15,
        "discretization_factor": DISCRETIZATION_FACTOR,
        "surface_potential": 3.6,
        "synchronicity": 1.0,
        "t_eff": 10000.0,
        "gravity_darkening": 1.0,
        "albedo": 0.6,
        "metallicity": 0.0,
        "atmosphere": "ck04",
        "pulsations": [
            {
                "l": 2,
                "m": 1,
                "amplitude": 500.0,
                "frequency": 2.3,
                "start_phase": 0.2,
                "mode_axis_theta": 25.0,
                "mode_axis_phi": 90.0,
                "temperature_amplitude_factor": 0.03,
                "tidally_locked": False,
            },
        ],
    },
    "secondary": {
        "mass": 1.2,
        "surface_potential": 4.0,
        "synchronicity": 1.0,
        "t_eff": 7000.0,
        "gravity_darkening": 1.0,
        "albedo": 0.6,
        "metallicity": 0.0,
        "atmosphere": "bb",
    },
}
PASSBANDS = ["Generic.Bessell.U"]
LC_START_PHASE = -0.6
LC_STOP_PHASE = 0.6
LC_SAMPLE_COUNT = 401
ANIMATION_START_PHASE = LC_START_PHASE
ANIMATION_PHASE_STEP = 0.05
# One orbit without a duplicate endpoint keeps the wraparound step uniform.
ANIMATION_STOP_PHASE = ANIMATION_START_PHASE + 1.0 - ANIMATION_PHASE_STEP
ANIMATION_FRAME_DURATION_MS = 150
ANIMATION_LC_HEIGHT = 650
ANIMATION_LC_DPI = 300
# Additional viewing elevation in degrees; None preserves the observer view.
CAMERA_ELEVATION: float | None = None
ANIMATION_LIMIT_MARGIN = 1.1
MINIMUM_ANIMATION_FRAMES = 2
ANIMATION_FORMAT = "gif"
FIGURE_SIZE = (9, 5)
PNG_DPI = 150
TITLE = "Eccentric pulsating"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs" / "eccentric_pulsating"
PNG_PATH = OUTPUT_DIRECTORY / "lightcurve.png"
GIF_PATH = OUTPUT_DIRECTORY / f"temperature_animation.{ANIMATION_FORMAT}"


def render_animation_view(**kwargs: Any) -> None:
    """Optionally tilt display vertices relative to the physical observer view."""
    if CAMERA_ELEVATION is not None:
        angle = np.deg2rad(CAMERA_ELEVATION)
        cosine, sine = np.cos(angle), np.sin(angle)
        # Native camera is at azimuth 180 degrees and elevation zero.
        rotation = np.array([[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]])
        for key in ("points_primary", "points_secondary"):
            kwargs[key] = [vertices @ rotation.T for vertices in kwargs[key]]
        extent = max(
            np.abs(vertices).max()
            for key in ("points_primary", "points_secondary")
            for vertices in kwargs[key]
        )
        kwargs["axis_lim"] = max(kwargs["axis_lim"], ANIMATION_LIMIT_MARGIN * extent)
        # Hide axes because these display coordinates differ from the physical frame.
        kwargs["plot_axis"] = False
    render_native_animation(**kwargs)


def add_lightcurve_panel(path: Path, phases: np.ndarray, fluxes: dict[str, np.ndarray]) -> None:
    """Append a synchronized LC panel to each native frame and set GIF timing."""
    frame_phases = animation_phases(ANIMATION_START_PHASE, ANIMATION_STOP_PHASE, ANIMATION_PHASE_STEP)
    frames: list[Image.Image] = []
    figure, axis = plt.subplots()
    try:
        markers = []
        for passband, values in fluxes.items():
            line, = axis.plot(phases, values, label=passband, linewidth=1.3)
            marker, = axis.plot([], [], "o", color=line.get_color(), markersize=7)
            markers.append((marker, values))
        cursor = axis.axvline(frame_phases[0], color="black", alpha=0.4, linewidth=1)
        axis.set(
            xlim=(LC_START_PHASE, LC_STOP_PHASE),
            xlabel="Orbital phase (elapsed cycles)",
            ylabel="Normalized flux",
        )
        axis.legend(loc="best")
        axis.set_title(f"Orbital phase = {frame_phases[0]:.2f}")
        axis.grid(alpha=0.25)
        with Image.open(path) as animation:
            if animation.n_frames != len(frame_phases):
                message = "Native GIF frame count does not match the animation phase grid"
                raise ValueError(message)
            figure.set_size_inches(
                animation.width / ANIMATION_LC_DPI,
                ANIMATION_LC_HEIGHT / ANIMATION_LC_DPI,
            )
            figure.set_dpi(ANIMATION_LC_DPI)
            figure.tight_layout()
            for frame, phase in zip(ImageSequence.Iterator(animation), frame_phases, strict=True):
                index = int(np.argmin(np.abs(phases - phase)))
                if not np.isclose(phases[index], phase, rtol=0.0, atol=1e-12):
                    message = "Animation phase is missing from the calculated light curve"
                    raise ValueError(message)
                for marker, values in markers:
                    marker.set_data([phase], [values[index]])
                cursor.set_xdata([phase, phase])
                axis.set_title(f"Orbital phase = {phase:.2f}")
                figure.canvas.draw()
                with Image.fromarray(np.asarray(figure.canvas.buffer_rgba()).copy()) as panel:
                    composite = Image.new("RGB", (animation.width, animation.height + panel.height), "white")
                    with frame.convert("RGB") as surface:
                        composite.paste(surface, (0, 0))
                    composite.paste(panel, (0, animation.height))
                frames.append(composite)
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=ANIMATION_FRAME_DURATION_MS,
            loop=0,
            disposal=2,
        )
    finally:
        plt.close(figure)
        for frame in frames:
            frame.close()


def main() -> None:
    """Calculate an unfolded LC and save the native temperature animation."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    binary = BinarySystem.from_json(BINARY_DEFINITION)
    if binary.morphology != "detached":
        message = f"Expected detached morphology, got {binary.morphology}"
        raise ValueError(message)
    observer = Observer(passband=PASSBANDS, system=binary)
    phases, fluxes = observer.observe.lc(
        phases=np.unique(np.concatenate((
            np.linspace(LC_START_PHASE, LC_STOP_PHASE, LC_SAMPLE_COUNT),
            animation_phases(ANIMATION_START_PHASE, ANIMATION_STOP_PHASE, ANIMATION_PHASE_STEP),
        ))),
        normalize=True,
    )
    if not all(np.isfinite(values).all() for values in fluxes.values()):
        message = "Light curve contains nonfinite fluxes"
        raise ValueError(message)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    try:
        for passband, values in fluxes.items():
            axis.plot(phases, values, label=passband, linewidth=1.3)
        axis.set(title=TITLE, xlabel="Orbital phase (elapsed cycles)", ylabel="Normalized flux (dimensionless)")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(PNG_PATH, dpi=PNG_DPI)
        existing_figures = set(plt.get_fignums())
        try:
            with patch.object(graphics, "binary_surface_anim", render_animation_view):
                binary.animation.orbital_motion(
                    start_phase=ANIMATION_START_PHASE,
                    stop_phase=ANIMATION_STOP_PHASE,
                    phase_step=ANIMATION_PHASE_STEP,
                    colormap="temperature",
                    subtract_equilibrium=False,
                    separate_colormaps=binary.morphology != "over-contact",
                    scale="linear",
                    savepath=str(GIF_PATH),
                )
        finally:
            for number in set(plt.get_fignums()) - existing_figures:
                plt.close(number)
        add_lightcurve_panel(GIF_PATH, phases, fluxes)
        with Image.open(GIF_PATH) as animation:
            if not animation.is_animated or animation.n_frames < MINIMUM_ANIMATION_FRAMES:
                message = "Temperature GIF must contain multiple frames"
                raise ValueError(message)
        if plt.get_backend() in backend_registry.list_builtin(BackendFilter.INTERACTIVE):
            plt.show()
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
