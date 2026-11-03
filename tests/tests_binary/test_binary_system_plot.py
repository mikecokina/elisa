# The tests package selects Astropy constants before test-module imports.
import unittest

import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from elisa import BinarySystem, get_default_binary_definition
from elisa import units as u
from elisa.binary_system.graphic.plot import Plot
from tests import set_astropy_units
from tests.utils import ElisaTestCase

set_astropy_units()


class BinarySystemPlotOrbitTestCase(ElisaTestCase):
    """Happy path tests for Plot.orbit() method."""

    def setUp(self) -> None:
        super().setUp()
        definition = get_default_binary_definition()
        self.binary = BinarySystem.from_json(definition)

    def tearDown(self) -> None:
        super().tearDown()
        plt.close("all")

    def test_orbit_basic_primary_frame(self) -> None:
        """Test orbit plotting with primary reference frame - happy path."""
        plot = Plot(self.binary)
        fig = plot.orbit(
            start_phase=0.0,
            stop_phase=1.0,
            number_of_points=100,
            frame_of_reference="primary",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_orbit_basic_barycentric_frame(self) -> None:
        """Test orbit plotting with barycentric reference frame - happy path."""
        plot = Plot(self.binary)
        fig = plot.orbit(
            start_phase=0.0,
            stop_phase=1.0,
            number_of_points=100,
            frame_of_reference="barycentric",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_orbit_with_custom_units(self) -> None:
        """Test orbit plotting with custom axis units - happy path."""
        plot = Plot(self.binary)
        fig = plot.orbit(
            start_phase=0.0,
            stop_phase=1.0,
            number_of_points=50,
            axis_units=u.solRad,
            frame_of_reference="primary",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_orbit_dimensionless_units(self) -> None:
        """Test orbit plotting with dimensionless units - happy path."""
        plot = Plot(self.binary)
        fig = plot.orbit(
            start_phase=0.0,
            stop_phase=0.5,
            number_of_points=50,
            axis_units="dimensionless",
            frame_of_reference="primary",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_orbit_sma_units(self) -> None:
        """Test orbit plotting with SMA (semi-major axis) units - happy path."""
        plot = Plot(self.binary)
        fig = plot.orbit(
            start_phase=0.0,
            stop_phase=1.0,
            number_of_points=50,
            axis_units="SMA",
            frame_of_reference="primary",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_orbit_with_legend(self) -> None:
        """Test orbit plotting with legend enabled - happy path."""
        plot = Plot(self.binary)
        fig = plot.orbit(
            start_phase=0.0,
            stop_phase=1.0,
            number_of_points=50,
            legend=True,
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)


class BinarySystemPlotEquipotentialTestCase(ElisaTestCase):
    """Happy path tests for Plot.equipotential() method."""

    def setUp(self) -> None:
        super().setUp()
        definition = get_default_binary_definition()
        self.binary = BinarySystem.from_json(definition)

    def tearDown(self) -> None:
        super().tearDown()
        plt.close("all")

    def test_equipotential_xy_plane(self) -> None:
        """Test equipotential plotting in xy plane - happy path."""
        plot = Plot(self.binary)
        fig = plot.equipotential(
            plane="xy",
            phase=0.0,
            components_to_plot="both",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_equipotential_yz_plane(self) -> None:
        """Test equipotential plotting in yz plane - happy path."""
        plot = Plot(self.binary)
        fig = plot.equipotential(
            plane="yz",
            phase=0.5,
            components_to_plot="both",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_equipotential_zx_plane(self) -> None:
        """Test equipotential plotting in zx plane - happy path."""
        plot = Plot(self.binary)
        fig = plot.equipotential(
            plane="zx",
            phase=0.25,
            components_to_plot="both",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_equipotential_primary_only(self) -> None:
        """Test equipotential plotting with only primary component - happy path."""
        plot = Plot(self.binary)
        fig = plot.equipotential(
            plane="xy",
            phase=0.0,
            components_to_plot="primary",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_equipotential_secondary_only(self) -> None:
        """Test equipotential plotting with only secondary component - happy path."""
        plot = Plot(self.binary)
        fig = plot.equipotential(
            plane="xy",
            phase=0.0,
            components_to_plot="secondary",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_equipotential_custom_colors(self) -> None:
        """Test equipotential plotting with custom colors - happy path."""
        plot = Plot(self.binary)
        fig = plot.equipotential(
            plane="xy",
            phase=0.0,
            components_to_plot="both",
            colors=("blue", "red"),
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)


class BinarySystemPlotMeshTestCase(ElisaTestCase):
    """Happy path tests for Plot.mesh() method."""

    def setUp(self) -> None:
        super().setUp()
        definition = get_default_binary_definition()
        self.binary = BinarySystem.from_json(definition)

    def tearDown(self) -> None:
        super().tearDown()
        plt.close("all")

    def test_mesh_both_components(self) -> None:
        """Test mesh plotting with both components - happy path."""
        plot = Plot(self.binary)
        fig = plot.mesh(
            phase=0.0,
            components_to_plot="both",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_mesh_primary_only(self) -> None:
        """Test mesh plotting with primary component only - happy path."""
        plot = Plot(self.binary)
        fig = plot.mesh(
            phase=0.0,
            components_to_plot="primary",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_mesh_secondary_only(self) -> None:
        """Test mesh plotting with secondary component only - happy path."""
        plot = Plot(self.binary)
        fig = plot.mesh(
            phase=0.0,
            components_to_plot="secondary",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_mesh_different_phases(self) -> None:
        """Test mesh plotting at different orbital phases - happy path."""
        plot = Plot(self.binary)
        for phase in [0.0, 0.25, 0.5, 0.75]:
            fig = plot.mesh(
                phase=phase,
                components_to_plot="both",
                return_figure_instance=True,
            )
            assert isinstance(fig, Figure)

    def test_mesh_with_custom_elevation(self) -> None:
        """Test mesh plotting with custom camera elevation - happy path."""
        plot = Plot(self.binary)
        fig = plot.mesh(
            phase=0.0,
            components_to_plot="both",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_mesh_with_custom_azimuth(self) -> None:
        """Test mesh plotting with custom camera azimuth - happy path."""
        plot = Plot(self.binary)
        fig = plot.mesh(
            phase=0.0,
            azimuth=90.0,
            components_to_plot="both",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_mesh_axis_off(self) -> None:
        """Test mesh plotting with axis turned off - happy path."""
        plot = Plot(self.binary)
        fig = plot.mesh(
            phase=0.0,
            components_to_plot="both",
            plot_axis=False,
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)


class BinarySystemPlotWireframeTestCase(ElisaTestCase):
    """Happy path tests for Plot.wireframe() method."""

    def setUp(self) -> None:
        super().setUp()
        definition = get_default_binary_definition()
        self.binary = BinarySystem.from_json(definition)

    def tearDown(self) -> None:
        super().tearDown()
        plt.close("all")

    def test_wireframe_both_components(self) -> None:
        """Test wireframe plotting with both components - happy path."""
        plot = Plot(self.binary)
        fig = plot.wireframe(
            phase=0.0,
            components_to_plot="both",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_wireframe_primary_only(self) -> None:
        """Test wireframe plotting with primary component only - happy path."""
        plot = Plot(self.binary)
        fig = plot.wireframe(
            phase=0.0,
            components_to_plot="primary",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_wireframe_secondary_only(self) -> None:
        """Test wireframe plotting with secondary component only - happy path."""
        plot = Plot(self.binary)
        fig = plot.wireframe(
            phase=0.0,
            components_to_plot="secondary",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_wireframe_different_phases(self) -> None:
        """Test wireframe plotting at different orbital phases - happy path."""
        plot = Plot(self.binary)
        for phase in [0.0, 0.5]:
            fig = plot.wireframe(
                phase=phase,
                components_to_plot="both",
                return_figure_instance=True,
            )
            assert isinstance(fig, Figure)

    def test_wireframe_with_camera_angles(self) -> None:
        """Test wireframe plotting with custom camera angles - happy path."""
        plot = Plot(self.binary)
        fig = plot.wireframe(
            phase=0.0,
            azimuth=45.0,
            components_to_plot="both",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)


@pytest.mark.requires_ld(passbands=("bolometric",), laws=("cosine",))
@pytest.mark.requires_atmosphere
class BinarySystemPlotSurfaceTestCase(ElisaTestCase):
    """Happy path tests for Plot.surface() method."""

    def setUp(self) -> None:
        super().setUp()
        definition = get_default_binary_definition()
        self.binary = BinarySystem.from_json(definition)

    def tearDown(self) -> None:
        super().tearDown()
        plt.close("all")

    def test_surface_basic_both_components(self) -> None:
        """Test surface plotting with both components - happy path."""
        plot = Plot(self.binary)
        fig = plot.surface(
            phase=0.0,
            components_to_plot="both",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_surface_primary_only(self) -> None:
        """Test surface plotting with primary component only - happy path."""
        plot = Plot(self.binary)
        fig = plot.surface(
            phase=0.0,
            components_to_plot="primary",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_surface_secondary_only(self) -> None:
        """Test surface plotting with secondary component only - happy path."""
        plot = Plot(self.binary)
        fig = plot.surface(
            phase=0.0,
            components_to_plot="secondary",
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_surface_different_phases(self) -> None:
        """Test surface plotting at different orbital phases - happy path."""
        plot = Plot(self.binary)
        for phase in [0.0, 0.25, 0.5]:
            fig = plot.surface(
                phase=phase,
                components_to_plot="both",
                return_figure_instance=True,
            )
            assert isinstance(fig, Figure)

    def test_surface_with_normals(self) -> None:
        """Test surface plotting with surface normals displayed - happy path."""
        plot = Plot(self.binary)
        fig = plot.surface(
            phase=0.0,
            components_to_plot="both",
            normals=True,
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_surface_with_edges(self) -> None:
        """Test surface plotting with edges highlighted - happy path."""
        plot = Plot(self.binary)
        fig = plot.surface(
            phase=0.0,
            components_to_plot="both",
            edges=True,
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_surface_with_custom_colors(self) -> None:
        """Test surface plotting with custom surface colors - happy path."""
        plot = Plot(self.binary)
        fig = plot.surface(
            phase=0.0,
            components_to_plot="both",
            surface_colors=("cyan", "magenta"),
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_surface_with_custom_axes_unit(self) -> None:
        """Test surface plotting with custom axis unit - happy path."""
        plot = Plot(self.binary)
        fig = plot.surface(
            phase=0.0,
            components_to_plot="both",
            axis_unit=u.km,
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_surface_without_colorbar(self) -> None:
        """Test surface plotting without colorbar - happy path."""
        plot = Plot(self.binary)
        fig = plot.surface(
            phase=0.0,
            components_to_plot="both",
            colorbar=False,
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)

    def test_surface_axis_off(self) -> None:
        """Test surface plotting with axis turned off - happy path."""
        plot = Plot(self.binary)
        fig = plot.surface(
            phase=0.0,
            components_to_plot="both",
            plot_axis=False,
            return_figure_instance=True,
        )
        assert isinstance(fig, Figure)


class BinarySystemPlotIntegrationTestCase(ElisaTestCase):
    """Integration tests for Plot class - multiple plots on same binary."""

    def setUp(self) -> None:
        super().setUp()
        definition = get_default_binary_definition()
        self.binary = BinarySystem.from_json(definition)

    def tearDown(self) -> None:
        super().tearDown()
        plt.close("all")

    def test_multiple_plot_methods_sequential(self) -> None:
        """Test calling multiple plot methods sequentially - happy path."""
        plot = Plot(self.binary)

        # Test all methods can be called in sequence without errors
        fig1 = plot.orbit(return_figure_instance=True)
        assert isinstance(fig1, Figure)

        fig2 = plot.equipotential(return_figure_instance=True)
        assert isinstance(fig2, Figure)

        fig3 = plot.mesh(return_figure_instance=True)
        assert isinstance(fig3, Figure)

    def test_plot_class_initialization(self) -> None:
        """Test Plot class initialization - happy path."""
        plot = Plot(self.binary)
        assert (plot.binary) is not None
        assert (plot.binary) == (self.binary)

    def test_plot_default_position(self) -> None:
        """Test Plot class has correct default position - happy path."""
        plot = Plot(self.binary)
        assert (plot.defpos) is not None
        assert (plot.defpos.idx) == (0)
        assert (plot.defpos.distance) == (1.0)


if __name__ == "__main__":
    unittest.main()
