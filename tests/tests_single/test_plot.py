from __future__ import annotations

from elisa import units as u
from elisa.single_system.graphic.plot import Plot
from tests import utils as testutils
from tests.utils import ElisaTestCase, prepare_single_system


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
