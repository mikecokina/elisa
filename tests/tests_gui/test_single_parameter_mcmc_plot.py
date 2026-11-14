"""Regression for posterior display after a one-parameter GUI MCMC fit."""

import matplotlib.pyplot as plt
import numpy as np

from elisa.graphic.mcmc_graphics import Plot
from elisa.ui.shared.plotting import capture_figure, figure_to_pil


def test_single_parameter_corner_with_truth_marker() -> None:
    """Produce a displayable posterior with the installed corner backend."""
    with capture_figure() as captured:
        Plot.corner(
            flat_chain=np.linspace(0.1, 0.9, 100).reshape(-1, 1),
            fit_params={
                "system@gamma": {
                    "value": 0.5,
                    "confidence_interval": {"min": 0.2, "max": 0.8},
                    "unit": None,
                },
            },
            variable_labels=["system@gamma"],
            labels=["gamma"],
            truths=[0.5],
        )
    assert len(captured) == 1
    axes = captured[0].axes
    assert len(axes) == 1
    assert any(np.array_equal(line.get_xdata(), [0.5, 0.5]) for line in axes[0].lines)
    assert "gamma" in axes[0].get_title()
    assert figure_to_pil(captured[0]).size[0] > 0
    plt.close(captured[0])
