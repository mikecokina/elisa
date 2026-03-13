from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import umpy as up
from elisa import units as u
from elisa.base import transform
from elisa.base.graphics import plot
from elisa.base.surface.faces import correct_face_orientation
from elisa.const import Position
from elisa.graphic import graphics
from elisa.observer.observer import Observer
from elisa.single_system import utils as sutils
from elisa.single_system.container import SinglePositionContainer
from elisa.single_system.curves import utils as crv_utils

# TYPE_CHECKING block at the end of import header
if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from elisa.single_system.system import SingleSystem
    from elisa.types import AstropyQuantity as Quantity
    from elisa.types import AstropyUnit as Unit
    from elisa.types import Float


class Plot:
    """Universal plot interface for single-system visualisation.

    The class provides convenience entry points that wrap the lower-level
    plotting helpers in :mod:`elisa.graphic.graphics`.
    """

    defpos = Position(*(0, np.nan, 0.0, np.nan, 0.0))

    def __init__(self, instance: SingleSystem) -> None:
        """Create a :class:`Plot` wrapper for a :class:`SingleSystem`.

        :param instance: SingleSystem instance to visualise.
        :type instance: elisa.single_system.system.SingleSystem
        """
        self.single = instance

    def equipotential(
        self,
        axis_unit: Unit = u.solRad,
        *,
        return_figure_instance: bool = False,
    ) -> Figure | None:
        """Plot equipotential cross-section in XZ plane.

        :param axis_unit: Axis unit for the returned coordinates.
        :type axis_unit: astropy.unit.Unit
        :param return_figure_instance: If True return the Matplotlib Figure
            instance instead of displaying the plot.
        :type return_figure_instance: bool

        :returns: Figure instance or ``None`` when the plot is shown.
        :rtype: matplotlib.figure.Figure | None
        """
        equipotential_kwargs = {}

        points = self.single.calculate_equipotential_boundary()
        points = (points * u.DefaultSingleSystemUnits.star.equivalent_radius).to(axis_unit)

        equipotential_kwargs.update(
            {
                "return_figure_instance": return_figure_instance,
                "points": points,
                "axis_unit": axis_unit,
            },
        )
        return graphics.equipotential_single_star(**equipotential_kwargs)

    def mesh(
        self,
        phase: Float = 0.0,
        axis_unit: Unit = u.solRad,
        inclination: Quantity | Float | None = None,
        azimuth: Quantity | Float | None = None,
        *,
        plot_axis: bool = True,
        return_figure_instance: bool = False,
    ) -> Figure | None:
        """Plot 3D scatter of surface points.

        :param phase: Photometric phase at which to render the mesh.
        :type phase: elisa.types.Float
        :param axis_unit: Axis unit for the returned coordinates.
        :type axis_unit: astropy.unit.Unit
        :param inclination: Camera elevation in degrees or Quantity; if ``None``
            the system's inclination is used.
        :type inclination: float | astropy.units.Quantity | None
        :param azimuth: Camera azimuth in degrees or Quantity.
        :type azimuth: float | astropy.units.Quantity | None
        :param plot_axis: If True display plot axes.
        :type plot_axis: bool
        :param return_figure_instance: If True return the Figure instance.
        :type return_figure_instance: bool

        :returns: Figure instance or ``None`` when the plot is shown.
        :rtype: matplotlib.figure.Figure | None
        """
        single_mesh_kwargs: dict = {}

        inclination = (
            transform.deg_transform(inclination, u.deg, transform.WHEN_FLOAT64,
                                     u.DefaultSingleSystemInputUnits.system.inclination)
            if inclination is not None
            else up.degrees(self.single.inclination)
        )
        azim = self.single.orbit.rotational_motion(phase=phase)[0][0]
        azimuth = (
            transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64)
            if azimuth is not None
            else up.degrees(azim) - 90
        )

        position_container: SinglePositionContainer = SinglePositionContainer.from_single_system(
            self.single,
            self.defpos,
        )
        position_container.build_mesh()
        position_container.build_perturbations()

        mesh = position_container.star.get_flatten_parameter("points")
        denominator = 1 * axis_unit.to(u.DefaultSingleSystemUnits.star.equivalent_radius)
        mesh /= denominator
        equatorial_radius = (
            position_container.star.equatorial_radius
            * u.DefaultSingleSystemUnits.star.equivalent_radius.to(axis_unit)
        )

        single_mesh_kwargs.update(
            {
                "return_figure_instance": return_figure_instance,
                "phase": phase,
                "axis_unit": axis_unit,
                "plot_axis": plot_axis,
                "inclination": inclination,
                "azimuth": azimuth,
                "mesh": mesh,
                "equatorial_radius": equatorial_radius,
            },
        )

        return graphics.single_star_mesh(**single_mesh_kwargs)

    def wireframe(
        self,
        phase: Float = 0.0,
        axis_unit: Unit = u.solRad,
        inclination: Quantity | Float | None = None,
        azimuth: Quantity | Float | None = None,
        *,
        plot_axis: bool = True,
        return_figure_instance: bool = False,
    ) -> Figure | None:
        """Return a 3D wireframe of the object.

        :param phase: Photometric phase at which to render the wireframe.
        :type phase: elisa.types.Float
        :param axis_unit: Axis unit for the returned coordinates.
        :type axis_unit: astropy.unit.Unit
        :param inclination: Camera elevation in degrees or Quantity; if ``None``
            the system's inclination is used.
        :type inclination: float | astropy.units.Quantity | None
        :param azimuth: Camera azimuth in degrees or Quantity.
        :type azimuth: float | astropy.units.Quantity | None
        :param plot_axis: If True display plot axes.
        :type plot_axis: bool
        :param return_figure_instance: If True return the Figure instance.
        :type return_figure_instance: bool

        :returns: Figure instance or ``None`` when the plot is shown.
        :rtype: matplotlib.figure.Figure | None
        """
        wireframe_kwargs: dict = {}

        inclination = (
            transform.deg_transform(inclination, u.deg, transform.WHEN_FLOAT64,
                                     u.DefaultSingleSystemInputUnits.system.inclination)
            if inclination is not None
            else up.degrees(self.single.inclination)
        )
        azim = self.single.orbit.rotational_motion(phase=phase)[0][0]
        azimuth = (
            transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64)
            if azimuth is not None
            else up.degrees(azim) - 90
        )

        position_container: SinglePositionContainer = SinglePositionContainer.from_single_system(
            self.single,
            self.defpos,
        )
        position_container.build_mesh()
        position_container.build_faces()

        points, faces = position_container.star.surface_serializer()
        denominator = 1 * axis_unit.to(u.DefaultSingleSystemUnits.star.equivalent_radius)
        points /= denominator
        equatorial_radius = (
            position_container.star.equatorial_radius
            * u.DefaultSingleSystemUnits.star.equivalent_radius.to(axis_unit)
        )

        wireframe_kwargs.update(
            {
                "return_figure_instance": return_figure_instance,
                "phase": phase,
                "axis_unit": axis_unit,
                "plot_axis": plot_axis,
                "inclination": inclination,
                "azimuth": azimuth,
                "mesh": points,
                "triangles": faces,
                "equatorial_radius": equatorial_radius,
            },
        )

        return graphics.single_star_wireframe(**wireframe_kwargs)

    def surface(
        self,
        phase: Float = 0.0,
        colormap: str | None = None,
        face_mask: NDArray[np.bool] | None = None,
        elevation: Quantity | Float | None = None,
        azimuth: Quantity | Float | None = None,
        colorbar_unit: str = "default",
        axis_unit: Unit = u.solRad,
        colorbar_orientation: str = "vertical",
        scale: str = "linear",
        surface_color: str = "g",
        colorbar_separation: Float = 0.0,
        colorbar_size: Float = 0.7,
        *,
        normals: bool = False,
        edges: bool = False,
        plot_axis: bool = True,
        colorbar: bool = True,
        return_figure_instance: bool = False,
        subtract_equilibrium: bool = False,
    ) -> Figure | None:
        """Create a surface plot of the single system.

        The function computes radiances and prepares plotting kwargs which are
        forwarded to the low-level surface painter. Only a minimal set of
        options is validated here; all plotting-specific parameters are
        forwarded unchanged.

        :param phase: Photometric phase at which to render the surface.
        :type phase: elisa.types.Float
        :param colormap: Colormap name or ``None`` to use defaults.
        :type colormap: str | None
        :param face_mask: Optional boolean mask selecting faces to include.
        :type face_mask: numpy.ndarray | None
        :param elevation: Camera elevation in degrees or Quantity.
        :type elevation: float | astropy.units.Quantity | None
        :param azimuth: Camera azimuth in degrees or Quantity.
        :type azimuth: float | astropy.units.Quantity | None
        :param colorbar_unit: Unit label used for the colorbar.
        :type colorbar_unit: str
        :param axis_unit: Axis unit for the returned coordinates.
        :type axis_unit: astropy.unit.Unit
        :param colorbar_orientation: 'horizontal' or 'vertical'.
        :type colorbar_orientation: str
        :param scale: Color scale, either 'linear' or 'log'.
        :type scale: str
        :param surface_color: Fallback surface color when colormap is not set.
        :type surface_color: str
        :param colorbar_separation: Horizontal separation of the colorbar.
        :type colorbar_separation: elisa.types.Float
        :param colorbar_size: Relative size of the colorbar.
        :type colorbar_size: elisa.types.Float
        :param normals: If True draw normals as arrows.
        :type normals: bool
        :param edges: If True draw triangle edges.
        :type edges: bool
        :param plot_axis: If True display plot axes.
        :type plot_axis: bool
        :param colorbar: If True display a colorbar.
        :type colorbar: bool
        :param return_figure_instance: If True return the Figure instance.
        :type return_figure_instance: bool
        :param subtract_equilibrium: If True subtract equilibrium values from colormap.
        :type subtract_equilibrium: bool

        :returns: Figure instance or ``None`` when the plot is shown.
        :rtype: matplotlib.figure.Figure | None
        """
        surface_kwargs: dict = {}

        elevation = (
            transform.deg_transform(elevation, u.deg, when_float64=transform.WHEN_FLOAT64)
            if elevation is not None
            else 0
        )
        azimuth = (
            transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64)
            if azimuth is not None
            else 180
        )

        single_position = self.single.orbit.rotational_motion(phase=phase)[0]
        single_position = Position(0, np.nan, single_position[0], single_position[1], single_position[2])

        position_container: SinglePositionContainer = SinglePositionContainer.from_single_system(
            self.single, self.defpos,
        )
        position_container.set_on_position_params(single_position)
        position_container.set_time()
        position_container.build(phase=phase, build_pulsations=True)
        correct_face_orientation(position_container.star, com=0)

        # calculating radiances
        o = Observer(passband=["bolometric"], system=self.single)
        atm_kwargs = {
            "passband": o.passband,
            "left_bandwidth": o.left_bandwidth,
            "right_bandwidth": o.right_bandwidth,
        }

        crv_utils.prep_surface_params(system=position_container, write_to_containers=True, **atm_kwargs)

        position_container = sutils.move_sys_onpos(position_container, single_position)

        star_container = position_container.star

        args = (colormap, star_container, phase, 0.0, 1.0, self.single.inclination, position_container.position)
        kwargs = {"scale": scale, "unit": colorbar_unit, "subtract_equilibrium": subtract_equilibrium}
        surface_kwargs.update({"cmap": plot.add_colormap_to_plt_kwargs(*args, **kwargs)})

        surface_kwargs.update(
            {
                "points": star_container.points,
                "triangles": star_container.faces,
            },
        )

        face_mask = np.ones(star_container.faces.shape[0], dtype=bool) if face_mask is None else face_mask
        surface_kwargs["triangles"] = surface_kwargs["triangles"][face_mask]
        if "cmap" in surface_kwargs:
            surface_kwargs["cmap"] = surface_kwargs["cmap"][face_mask]

        if normals:
            face_centres = star_container.get_flatten_parameter("face_centres")
            norm = star_container.get_flatten_parameter("normals")
            surface_kwargs.update(
                {
                    "centres": face_centres[face_mask],
                    "arrows": norm[face_mask],
                },
            )

        # normals
        unit_mult = (1 * u.DefaultSingleSystemUnits.star.equivalent_radius).to(axis_unit).value
        surface_kwargs["points"] *= unit_mult

        if normals:
            surface_kwargs["centres"] *= unit_mult

        equatorial_radius = (
            star_container.equatorial_radius * u.DefaultSingleSystemUnits.star.equivalent_radius
        ).to(axis_unit).value

        surface_kwargs.update(
            {
                "phase": phase,
                "normals": normals,
                "edges": edges,
                "colormap": colormap,
                "plot_axis": plot_axis,
                "face_mask": face_mask,
                "elevation": elevation,
                "azimuth": azimuth,
                "unit": colorbar_unit,
                "axis_unit": axis_unit,
                "colorbar_orientation": colorbar_orientation,
                "colorbar": colorbar,
                "scale": scale,
                "equatorial_radius": equatorial_radius,
                "surface_color": surface_color,
                "colorbar_separation": colorbar_separation,
                "colorbar_size": colorbar_size,
                "return_figure_instance": return_figure_instance,
            },
        )
        return graphics.single_star_surface(**surface_kwargs)
