from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from elisa import units as u
from elisa.graphic import graphics
from elisa.single_system.graphic.plot import Plot
from tests import utils as testutils
from tests.utils import ElisaTestCase, prepare_single_system


@pytest.mark.parametrize("colorbar", [False, True])
def test_surface_axes_are_attached_to_figure(colorbar: bool) -> None:
    """The stellar surface must render, including when a colorbar is added."""
    points = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]])
    faces = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]])
    figure = graphics.single_star_surface(
        points=points,
        triangles=faces,
        cmap=np.array([5000., 5100., 5200., 5300.]),
        colormap="temperature",
        colorbar=colorbar,
        colorbar_size=0.7,
        colorbar_orientation="vertical",
        colorbar_separation=0.05,
        unit="default",
        scale="linear",
        surface_color="g",
        elevation=0,
        azimuth=180,
        equatorial_radius=2.,
        normals=False,
        edges=True,
        plot_axis=True,
        axis_unit=u.solRad,
        return_figure_instance=True,
    )
    try:
        axes = [axis for axis in figure.axes if axis.name == "3d"]
        assert len(axes) == 1
        assert axes[0] in figure.get_children()
        assert len(axes[0].collections) == 1
        np.testing.assert_array_equal(axes[0].collections[0].get_array(), [5000, 5100, 5200, 5300])
        figure.canvas.draw()
    finally:
        plt.close(figure)


class PlotTestCase(ElisaTestCase):
    """Unit tests for `elisa.single_system.graphic.plot.Plot`.

    These tests exercise the plotting helpers in a non-interactive way by
    requesting the figure instance instead of showing the plot. The tests
    remain lightweight and verify that the plotting entry points complete
    without raising exceptions and return a figure-like object.
    """

    def test_equipotential_returns_figure(self) -> None:
        """Ensure `equipotential` returns a figure instance.

        :returns: None
        :rtype: None
        """
        single = prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS["spherical"])
        single.init()

        p = Plot(single)

        fig_eq = p.equipotential(axis_unit=u.solRad, return_figure_instance=True)
        assert (fig_eq) is not None

    def test_mesh_returns_figure(self) -> None:
        """Ensure `mesh` returns a figure instance.

        :returns: None
        :rtype: None
        """
        single = prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS["spherical"])
        single.init()

        p = Plot(single)

        fig_mesh = p.mesh(phase=0.0, plot_axis=False, axis_unit=u.solRad, return_figure_instance=True)
        assert (fig_mesh) is not None

    def test_wireframe_returns_figure(self) -> None:
        """Ensure `wireframe` returns a figure instance.

        :returns: None
        :rtype: None
        """
        single = prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS["spherical"])
        single.init()

        p = Plot(single)

        fig_wire = p.wireframe(phase=0.0, plot_axis=False, axis_unit=u.solRad, return_figure_instance=True)
        assert (fig_wire) is not None
