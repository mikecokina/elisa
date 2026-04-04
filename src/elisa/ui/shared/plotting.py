"""Shared matplotlib helpers for ELISa UI plots."""

from __future__ import annotations

import contextlib
from io import BytesIO
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.figure import Figure
    from numpy.typing import NDArray

# Ensure a non-interactive backend suitable for server-side rendering.
mpl.use("Agg")

# Curated color palette - colorblind-friendly, used as fallback for unknown passbands.
_PALETTE: list[str] = [
    "#0C5DA5",  # blue
    "#FF6B35",  # orange
    "#00B945",  # green
    "#FF2C00",  # red
    "#845B97",  # purple
    "#474747",  # dark grey
    "#9E9E00",  # olive
    "#00B2CC",  # teal
]

# Conventional per-filter colors - mirrors ELISa's own datapoint_clrs mapping
# in elisa.graphic.graphics.binary_lc_fit_plot so the UI is consistent with
# the rest of the library.
_PASSBAND_COLORS: dict[str, str] = {
    "bolometric":          "#808080",
    "Generic.Bessell.U":   "#cc0099",
    "Generic.Bessell.B":   "#00007f",
    "Generic.Bessell.V":   "#008000",
    "Generic.Bessell.R":   "#ff0000",
    "Generic.Bessell.I":   "#800000",
    "SLOAN.SDSS.u":        "#000099",
    "SLOAN.SDSS.g":        "#009900",
    "SLOAN.SDSS.r":        "#e60000",
    "SLOAN.SDSS.i":        "#800080",
    "SLOAN.SDSS.z":        "#00cccc",
    "Generic.Stromgren.u": "#990099",
    "Generic.Stromgren.v": "#cc00cc",
    "Generic.Stromgren.b": "#0000cc",
    "Generic.Stromgren.y": "#00b300",
    "Kepler":              "#890000",
    "Gaia.2010.G":         "#006400",
    "Gaia.2010.BP":        "#0000cd",
    "Gaia.2010.RP":        "#cd0000",
    "TESS":                "#006989",
}

# rcParams applied only within the figure context - no global side effects.
_RC: dict[str, object] = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.major.size": 5,
    "ytick.minor.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 10,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#cccccc",
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "grid.linestyle": "--",
    "figure.dpi": 130,
}


@contextlib.contextmanager
def _capture_figure() -> Iterator[list[Figure]]:
    """Temporarily capture figures shown via ``plt.show()``.

    Replaces :func:`matplotlib.pyplot.show` within the context and appends
    the current figure to the yielded list each time ``show()`` is called.

    :yields: List populated with captured figures.
    :rtype: list[matplotlib.figure.Figure]
    """
    captured: list[Figure] = []
    original_show = plt.show

    def _mock_show(*_args: object, **_kwargs: object) -> None:
        fig = plt.gcf()
        if fig is not None:
            captured.append(fig)

    plt.show = _mock_show  # type: ignore[assignment]
    try:
        yield captured
    finally:
        plt.show = original_show  # type: ignore[assignment]


def capture_figure() -> contextlib.AbstractContextManager[list[Figure]]:
    """Return a context manager that captures figures shown via ``plt.show()``.

    This public wrapper provides an ``AbstractContextManager`` return type
    for static checkers while delegating the implementation to
    :func:`_capture_figure`.

    :returns: Context manager yielding captured figures.
    :rtype: contextlib.AbstractContextManager[list[matplotlib.figure.Figure]]
    """
    return _capture_figure()


def figure_to_pil(fig: Figure | None) -> Image.Image | None:
    """Convert a matplotlib Figure to a PIL Image for efficient Gradio rendering.

    Matplotlib figures cause expensive serialisation in Gradio when used with
    ``gr.Plot``.  Converting to a PIL ``Image`` and using ``gr.Image`` instead
    reduces the WebSocket payload size and avoids Gradio's internal figure
    processing overhead.

    The figure is always closed after conversion so matplotlib's memory is
    released.  Pass ``None`` to skip conversion and get ``None`` back, which
    ``gr.Image`` accepts as a *clear* signal.

    :param fig: Matplotlib figure to convert, or ``None``.
    :type fig: matplotlib.figure.Figure | None
    :returns: PIL Image containing the rendered figure, or ``None``.
    :rtype: PIL.Image.Image | None
    """
    if fig is None:
        return None

    try:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img.load()
        img_copy = img.copy()
        buf.close()
    finally:
        plt.close(fig)

    return img_copy


def render_lc_figure(
    phases: NDArray,
    fluxes: dict[str, NDArray],
    *,
    normalize: bool = False,
) -> Figure:
    """Render a publication-quality light curve figure.

    Creates a Matplotlib figure with one line per passband using a
    curated color palette and a clean, scientific style.  All rc
    customisations are scoped to a context manager so no global
    Matplotlib state is modified.

    :param phases: 1-D array of orbital phases.
    :type phases: NDArray
    :param fluxes: Mapping of passband name to 1-D flux array.
    :type fluxes: dict[str, NDArray]
    :param normalize: When ``True``, the y-axis is labeled
        *Normalized flux*; otherwise the flux unit is shown in LaTeX.
    :type normalize: bool
    :returns: Matplotlib figure ready for display.
    :rtype: matplotlib.figure.Figure
    """
    # noinspection PyTypeChecker
    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(10, 5))

        for (passband, flux), fallback in zip(
            fluxes.items(), _PALETTE, strict=False,
        ):
            color = _PASSBAND_COLORS.get(passband, fallback)
            ax.plot(phases, flux, label=passband, linewidth=1.8, color=color)


        # add a small breathing margin on all sides so lines do not
        # touch the axis frame
        ax.margins(x=0.02, y=0.06)

        # vertical markers at primary (0) and secondary (0.5) eclipse
        # only when those phases fall within the plotted data range
        x_lo, x_hi = float(np.min(phases)), float(np.max(phases))
        for eclipse_phase, style in ((0.0, "-"), (0.5, "--")):
            if x_lo <= eclipse_phase <= x_hi:
                ax.axvline(
                    eclipse_phase,
                    color="#aaaaaa",
                    linewidth=0.9,
                    linestyle=style,
                    zorder=0,
                )

        ax.set_xlabel("Phase")
        y_label = (
            "Normalized flux"
            if normalize
            else r"Flux  (W m$^{-2}$)"
        )
        ax.set_ylabel(y_label)
        ax.set_title("Synthetic Light Curve", pad=10)

        ax.grid(visible=True)
        ax.legend(loc="best")

        fig.tight_layout()

    return fig


def render_rv_figure(
    phases: NDArray,
    rvs: dict[str, NDArray],
) -> Figure:
    """Render a publication-quality radial velocity curve figure.

    Creates a Matplotlib figure with one line per component (primary,
    secondary) using the same curated color palette and scientific style
    as the light-curve plotter.

    :param phases: 1-D array of orbital phases.
    :type phases: NDArray
    :param rvs: Mapping of component name (``"primary"``, ``"secondary"``)
        to 1-D radial velocity array in km/s.
    :type rvs: dict[str, NDArray]
    :returns: Matplotlib figure ready for display.
    :rtype: matplotlib.figure.Figure
    """
    # noinspection PyTypeChecker
    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(10, 5))

        for (component, rv), color in zip(
            rvs.items(), _PALETTE, strict=False,
        ):
            ax.plot(phases, rv, label=component.capitalize(), linewidth=1.8, color=color)

        # add breathing margin
        ax.margins(x=0.02, y=0.06)

        # vertical markers at primary (0) and secondary (0.5) eclipse
        x_lo, x_hi = float(np.min(phases)), float(np.max(phases))
        for eclipse_phase, style in ((0.0, "-"), (0.5, "--")):
            if x_lo <= eclipse_phase <= x_hi:
                ax.axvline(
                    eclipse_phase,
                    color="#aaaaaa",
                    linewidth=0.9,
                    linestyle=style,
                    zorder=0,
                )

        ax.set_xlabel("Phase")
        ax.set_ylabel(r"Radial velocity  (km s$^{-1}$)")
        ax.set_title("Synthetic Radial Velocity Curve", pad=10)

        ax.grid(visible=True)
        ax.legend(loc="best")

        fig.tight_layout()

    return fig

