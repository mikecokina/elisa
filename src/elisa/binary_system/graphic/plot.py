from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from elisa import const, utils
from elisa import umpy as up
from elisa import units as u
from elisa.base import transform
from elisa.base.graphics import plot
from elisa.base.surface.faces import correct_face_orientation
from elisa.binary_system import dynamic
from elisa.binary_system import utils as butils
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.curves import utils as crv_utils
from elisa.const import Position
from elisa.graphic import graphics

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from elisa import BinarySystem
    from elisa.types import NP_BOOL_, Float, Int
    from elisa.units import Unit


class Plot:
    """Universal plot interface for binary system clas.

    More detailed documentation for each value of descriptor is
    available in graphics library.

    available plot methods::
        `orbit` - plots orbit in orbital plane
        `equipotential` - plots crossections of surface Hill planes in xy,yz,zx planes
        `mesh` - plot surface points
        `wireframe` - wire frame model of the binary
        `surface` - plot stellar surfaces
    """

    defpos = Position(*(0, 1.0, 0.0, 0.0, 0.0))

    def __init__(self, instance: BinarySystem) -> None:
        """Init of plot."""
        self.binary = instance

    def orbit(
        self,
        start_phase: Float = 0.0,
        stop_phase: Float = 1.0,
        number_of_points: Int = 300,
        axis_units: Unit = u.solRad,
        frame_of_reference: Literal["primary", "barycentric"] = "primary",
        *,
        legend: bool = True,
        return_figure_instance: bool = False,
    ) -> Figure | None:
        """Quick 2D plot of the orbital motion in the orbital plane.

        :param start_phase: float; starting phase for the plot
        :param stop_phase: float; finishing phase for the plot
        :param number_of_points: int; number of points in the plot
        :param axis_units: Union[astropy.unit, str]; specifying axis unit, use astropy
                           units or `dimensionless` or `SMA` (semi-major axis) units for axis scale
        :param frame_of_reference: str; `barycentric` or `primary`
        :param return_figure_instance: bool; if True, the Figure instance is returned instead of displaying the
                                             produced figure
        :param legend: bool;
        """
        orbit_kwargs = {}
        if axis_units in ["dimensionless", "SMA"]:
            axis_units = u.dimensionless_unscaled

        # orbit calculation for given phases
        phases = np.linspace(start_phase, stop_phase, number_of_points)
        ellipse = self.binary.orbit.orbital_motion(phase=phases)

        # if axis are without unit a = 1
        if axis_units != u.dimensionless_unscaled:
            a = self.binary.semi_major_axis * u.DefaultBinarySystemUnits.system.semi_major_axis.to(axis_units)
            radius = a * ellipse[:, 0]
        else:
            radius = ellipse[:, 0]
        azimuth = ellipse[:, 1]
        x, y = utils.polar_to_cartesian(radius=radius, phi=azimuth + const.PI / 2.0)

        if frame_of_reference == "barycentric":
            orbit_kwargs.update(
                {
                    "x1_data": -self.binary.mass_ratio * x / (1 + self.binary.mass_ratio),
                    "y1_data": -self.binary.mass_ratio * y / (1 + self.binary.mass_ratio),
                    "x2_data": x / (1 + self.binary.mass_ratio),
                    "y2_data": y / (1 + self.binary.mass_ratio),
                },
            )
        elif frame_of_reference == "primary":
            orbit_kwargs.update(
                {
                    "x_data": x,
                    "y_data": y,
                },
            )
        orbit_kwargs.update(
            {
                "return_figure_instance": return_figure_instance,
                "axis_units": axis_units,
                "start_phase": start_phase,
                "stop_phase": stop_phase,
                "number_of_points": number_of_points,
                "frame_of_reference": frame_of_reference,
                "legend": legend,
            },
        )
        return graphics.orbit(**orbit_kwargs)

    def equipotential(
        self,
        plane: Literal["xy", "yz", "zx"] = "xy",
        phase: Float = 0.0,
        components_to_plot: Literal["both", "primary", "secondary"] = "both",
        colors: tuple[str, str] = ("b", "r"),
        legend_loc: int = 1,
        *,
        legend: bool = True,
        return_figure_instance: bool = False,
    ) -> Figure | None:
        """Quick 2D plot of equipotential cross-section.

        :param plane: str; (`xy`, `yz` or `xz`) specifying what plane cross-section to display default is `xy`
        :param phase: float; phase at which to plot cross-section
        :param components_to_plot: str; component to plot `primary`, `secondary` or `both` (default)
        :param colors: Tuple; tuple of colors for primary and secondary component equipotentials
        :param legend: bool; legend display on/off
        :param legend_loc: int; location of the legend
        :param return_figure_instance: bool; if True, the Figure instance is returned instead of displaying the
                                             produced figure
        """
        equipotential_kwargs = {}
        # relative distance between components (a = 1)
        is_xy = utils.is_plane(plane, "xy")
        is_yz = utils.is_plane(plane, "yz")
        is_zx = utils.is_plane(plane, "zx")

        if is_xy or is_yz or is_zx:
            components_distance = self.binary.orbit.orbital_motion(phase=phase)[0][0]
            points_primary, points_secondary = self.binary.compute_equipotential_boundary(components_distance, plane)
        else:
            msg = "Invalid choice of crossection plane, use only: `xy`, `yz`, `zx`."
            raise ValueError(msg)

        equipotential_kwargs.update(
            {
                "return_figure_instance": return_figure_instance,
                "plane": plane,
                "phase": phase,
                "points_primary": points_primary,
                "points_secondary": points_secondary,
                "components_to_plot": components_to_plot,
                "colors": colors,
                "legend": legend,
                "legend_loc": legend_loc,
            },
        )
        return graphics.equipotential(**equipotential_kwargs)

    def mesh(
        self,
        phase: Float = 0.0,
        components_to_plot: Literal["both", "primary", "secondary"] = "both",
        inclination: Float | None = None,
        azimuth: Float | None = None,
        *,
        plot_axis: bool = True,
        return_figure_instance: bool = False,
    ) -> Figure | None:
        """Plot 3D scatter plot of the surface points.

        :param plot_axis: bool; switch the plot axis on/off
        :param inclination: Union[float, astropy.Quantity]; elevation of the camera (in degrees if float)
        :param azimuth: Union[float, astropy.Quantity]; azimuth of the camera (in degrees if float)
        :param components_to_plot: str; component to plot `primary`, `secondary` or `both` (default)
        :param phase: float; phase at which to construct plot
        :param return_figure_instance: bool; if True, the Figure instance is
                                             returned instead of displaying the produced figure
        """
        binary_mesh_kwargs = {}
        inclination = (
            transform.deg_transform(
                inclination, u.deg, transform.WHEN_FLOAT64, u.DefaultBinarySystemInputUnits.system.inclination,
            )
            if inclination is not None
            else up.degrees(self.binary.inclination)
        )
        components_distance, azim = self.binary.orbit.orbital_motion(phase=phase)[0][:2]

        azimuth = (
            transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64)
            if azimuth is not None
            else up.degrees(azim) - 90
        )

        orbital_position_container = OrbitalPositionContainer.from_binary_system(self.binary, self.defpos)
        orbital_position_container.build_mesh(components_distance=components_distance)
        orbital_position_container.build_perturbations(
            component=components_to_plot, components_distance=components_distance,
        )

        if components_to_plot in ["primary", "both"]:
            binary_mesh_kwargs.update(
                {
                    "points_primary": orbital_position_container.primary.get_flatten_parameter("points"),
                },
            )

        if components_to_plot in ["secondary", "both"]:
            binary_mesh_kwargs.update(
                {
                    "points_secondary": orbital_position_container.secondary.get_flatten_parameter("points"),
                },
            )

        binary_mesh_kwargs.update(
            {
                "return_figure_instance": return_figure_instance,
                "phase": phase,
                "components_to_plot": components_to_plot,
                "plot_axis": plot_axis,
                "inclination": inclination,
                "azimuth": azimuth,
            },
        )
        return graphics.binary_mesh(**binary_mesh_kwargs)

    def wireframe(
        self,
        phase: Float = 0.0,
        components_to_plot: Literal["both", "primary", "secondary"] = "both",
        inclination: Float | None = None,
        azimuth: Float | None = None,
        *,
        plot_axis: bool = True,
        return_figure_instance: bool = False,
    ) -> Figure | None:
        """Display wireframe model of the stellar surface.

        :param phase: float; phase at which to construct plot
        :param components_to_plot: str; component to plot `primary`, `secondary` or `both` (default)
        :param plot_axis: bool; switch the plot axis on/off
        :param inclination: Union[float, astropy.Quantity]; elevation of the camera (in degrees if float)
        :param azimuth: Union[float, astropy.Quantity]; azimuth of the camera (in degrees if float)
        :param return_figure_instance: bool; if True, the Figure instance is returned instead of displaying the
                                             produced figure
        """
        wireframe_kwargs = {}
        inclination = (
            transform.deg_transform(
                inclination, u.deg, transform.WHEN_FLOAT64, u.DefaultBinarySystemInputUnits.system.inclination,
            )
            if inclination is not None
            else up.degrees(self.binary.inclination)
        )
        components_distance, azim = self.binary.orbit.orbital_motion(phase=phase)[0][:2]
        azimuth = (
            transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64)
            if azimuth is not None
            else up.degrees(azim) - 90
        )

        orbital_position_container = OrbitalPositionContainer.from_binary_system(self.binary, self.defpos)

        # recalculating spot latitudes
        spots_longitudes = dynamic.calculate_spot_longitudes(self.binary, phase, component="all")
        dynamic.assign_spot_longitudes(orbital_position_container, spots_longitudes, index=None, component="all")

        orbital_position_container.build_mesh(components_distance=components_distance)
        orbital_position_container.build_faces(components_distance=components_distance)

        if components_to_plot in ["primary", "both"]:
            points, faces = orbital_position_container.primary.surface_serializer()
            wireframe_kwargs.update(
                {
                    "points_primary": points,
                    "primary_triangles": faces,
                },
            )
        if components_to_plot in ["secondary", "both"]:
            points, faces = orbital_position_container.secondary.surface_serializer()
            wireframe_kwargs.update(
                {
                    "points_secondary": points,
                    "secondary_triangles": faces,
                },
            )
        wireframe_kwargs.update(
            {
                "return_figure_instance": return_figure_instance,
                "phase": phase,
                "components_to_plot": components_to_plot,
                "plot_axis": plot_axis,
                "inclination": inclination,
                "azimuth": azimuth,
            },
        )
        return graphics.binary_wireframe(**wireframe_kwargs)

    def surface(
        self,
        phase: Float = 0.0,
        components_to_plot: Literal["both", "primary", "secondary"] = "both",
        colormap: str | None = None,
        face_mask_primary: NDArray[NP_BOOL_] | None = None,
        face_mask_secondary: NDArray[NP_BOOL_] | None = None,
        elevation: Float | None = None,
        azimuth: Float | None = None,
        colorbar_unit: str = "default",
        axis_unit: Unit = u.dimensionless_unscaled,
        colorbar_orientation: str = "vertical",
        scale: str = "linear",
        surface_colors: tuple[str, str] = ("g", "r"),
        colorbar_separation: Float = 0.0,
        colorbar_size: Float = 0.7,
        *,
        normals: bool = False,
        edges: bool = False,
        colorbar: bool = True,
        plot_axis: bool = True,
        separate_colormaps: bool | None = None,
        return_figure_instance: bool = False,
        subtract_equilibrium: bool = False,
    ) -> Figure | None:
        """Create plot of binary system components.

        :param phase: float; phase at which plot the system
        :param components_to_plot: str; `primary`, `secondary` or `both` (default),
        :param normals: bool; plot normals of the surface phases as arrows
        :param edges: bool; highlight edges of surface faces
        :param colormap: str;
        :param plot_axis: bool; if False, axis will be hidden
        :param face_mask_primary: array[bool]; mask to select which faces to display
        :param face_mask_secondary: array[bool]: mask to select which faces to display
        :param elevation: Union[float, astropy.Quantity]; in degrees - elevation of camera
        :param azimuth: Union[float, astropy.Quantity]; camera azimuth
        :param colorbar_unit: str; colorbar unit
        :param axis_unit: Union[astropy.unit, dimensionless]; - axis units
        :param colorbar_orientation: str; 'horizontal' or 'vertical' (default)
        :param colorbar: bool; colorbar on/off switch
        :param scale: str; 'linear' or 'log'
        :param surface_colors: Tuple; tuple of colors for components if `colormap` are not specified
        :param separate_colormaps: bool; if True, figure will contain separate colormap for each component
        :param colorbar_separation: float; shifting position of the colorbar from its default postition, default is 0.0
        :param colorbar_size: float; relative size of the colorbar, default 0.7
        :param subtract_equilibrium: bool; if True; equilibrium values are subtracted from surface colormap
        :param return_figure_instance: bool; if True, the Figure instance is returned instead of displaying the
                                             produced figure
        :**available colormap options**:
            * :'gravity_acceleration': surface distribution of gravity acceleration,
            * :'temperature': surface distribution of the effective temperature,
            * :'velocity': absolute values of surface elements velocities with respect to the observer,
            * :'radial_velocity': radial component of the surface element velocities relative to the observer,
            * :'normal_radiance': surface element radiance perpendicular to the surface element,
            * :'radiance': radiance of the surface element in a direction towards the observer,
            * :'radius': distance of the surface elements from the centre of mass
            * :'horizontal_displacement': distribution of the horizontal component of surface displacement
            * :'horizontal_acceleration': distribution of horizontal component surface acceleration
            * :'v_r_perturbed': radial component of the pulsation velocity (perpendicular towards
            * :'v_horizontal_perturbed': horizontal component of the pulsation  velocity

        :return: Union[None, matplotlib.pyplot.figure]
        """
        surface_kwargs = {}

        elevation = (
            transform.deg_transform(elevation, u.deg, when_float64=transform.WHEN_FLOAT64)
            if elevation is not None
            else 0
        )

        orbital_position = self.binary.orbit.orbital_motion(phase=phase)[0]
        components_distance, azim = orbital_position[:2]
        orbital_position = Position(
            0, orbital_position[0], orbital_position[1], orbital_position[2], orbital_position[3],
        )

        azimuth = (
            transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64)
            if azimuth is not None
            else up.degrees(azim) - 90
        )

        if separate_colormaps is None:
            separate_colormaps = (
                self.binary.morphology != "over-contact"
                and colormap not in ["velocity", "radial_velocity"]
                and components_to_plot == "both"
            )

        potentials = self.binary.correct_potentials([phase], component="all", iterations=2)

        orbital_position_container = OrbitalPositionContainer.from_binary_system(self.binary, self.defpos)
        # recalculating spot latitudes
        spots_longitudes = dynamic.calculate_spot_longitudes(self.binary, phase, component="all")
        dynamic.assign_spot_longitudes(orbital_position_container, spots_longitudes, index=None, component="all")

        orbital_position_container.set_on_position_params(
            orbital_position, potentials["primary"][0], potentials["secondary"][0],
        )

        orbital_position_container.set_time()
        orbital_position_container.build(
            components_distance=components_distance, components="both", build_pulsations=True,
        )
        components = butils.component_to_list(components_to_plot)

        com = {"primary": 0.0, "secondary": components_distance}
        for component in components:
            correct_face_orientation(getattr(orbital_position_container, str(component)), com=com[component])

        # calculating radiances
        # Local import to avoid circular import at module import time
        from elisa.observer.observer import Observer  # noqa: PLC0415

        o = Observer(passband=["bolometric"], system=self.binary)
        atm_kwargs = {
            "passband": o.passband,
            "left_bandwidth": o.left_bandwidth,
            "right_bandwidth": o.right_bandwidth,
        }

        crv_utils.prep_surface_params(
            system=orbital_position_container,
            write_to_containers=True,
            **atm_kwargs,
        )

        orbital_position_container = butils.move_sys_onpos(orbital_position_container, orbital_position, on_copy=True)
        args = (orbital_position.distance, self.binary.mass_ratio, orbital_position_container.secondary.com)
        pos_correction = butils.correction_to_com(*args)

        for component in components:
            star = getattr(orbital_position_container, str(component))

            args = (
                colormap,
                star,
                phase,
                com[component],
                self.binary.semi_major_axis,
                self.binary.inclination,
                orbital_position_container.position,
            )
            kwargs = {"scale": scale, "unit": colorbar_unit, "subtract_equilibrium": subtract_equilibrium}

            surface_kwargs.update({f"{component}_cmap": plot.add_colormap_to_plt_kwargs(*args, **kwargs)})

            surface_kwargs.update(
                {
                    f"points_{component}": star.points - pos_correction[None, :],
                    f"{component}_triangles": star.faces,
                },
            )

            face_mask = locals().get(f"face_mask_{component}")
            face_mask = np.ones(star.faces.shape[0], dtype=bool) if face_mask is None else face_mask
            surface_kwargs[f"{component}_triangles"] = surface_kwargs[f"{component}_triangles"][face_mask]
            if colormap is not None:
                surface_kwargs[f"{component}_cmap"] = surface_kwargs[f"{component}_cmap"][face_mask]

            if normals:
                surface_kwargs.update(
                    {
                        f"{component}_centres": star.face_centres[face_mask] - pos_correction[None, :],
                        f"{component}_arrows": star.normals[face_mask],
                    },
                )

            if axis_unit != u.dimensionless_unscaled:
                sma = (
                    (self.binary.semi_major_axis * u.DefaultBinarySystemUnits.system.semi_major_axis)
                    .to(axis_unit)
                    .value
                )
                surface_kwargs[f"points_{component}"] *= sma

                if normals:
                    surface_kwargs[f"{component}_centres"] *= sma
                    surface_kwargs[f"{component}_arrows"] *= sma

        surface_kwargs.update(
            {
                "phase": phase,
                "components_to_plot": components_to_plot,
                "normals": normals,
                "edges": edges,
                "colormap": colormap,
                "plot_axis": plot_axis,
                "face_mask_primary": face_mask_primary,
                "face_mask_secondary": face_mask_secondary,
                "elevation": elevation,
                "azimuth": azimuth,
                "unit": colorbar_unit,
                "axis_unit": axis_unit,
                "colorbar_orientation": colorbar_orientation,
                "colorbar": colorbar,
                "scale": scale,
                "morphology": self.binary.morphology,
                "surface_colors": surface_colors,
                "separate_colormaps": separate_colormaps,
                "colorbar_separation": colorbar_separation,
                "colorbar_size": colorbar_size,
                "return_figure_instance": return_figure_instance,
            },
        )

        return graphics.binary_surface(**surface_kwargs)
