"""Shared matplotlib helpers for ELISa UI plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

# Ensure a non-interactive backend suitable for server-side rendering.
mpl.use("Agg")

# Curated color palette - colorblind-friendly, distinguishable on white.
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

# rcParams applied only within the figure context - no global side-effects.
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

        for (passband, flux), color in zip(
            fluxes.items(), _PALETTE, strict=False,
        ):
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
