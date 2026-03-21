from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np

from elisa.base.graphics import plot
from elisa.binary_system import dynamic
from elisa.binary_system import utils as butils
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.curves import utils as crv_utils
from elisa.const import Position
from elisa.graphic import graphics
from elisa.logger import getLogger

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from elisa.binary_system.system import BinarySystem
    from elisa.types import Float

logger = getLogger("binary_system.graphic.animation")


class Animation:
    """Animation helper for binary-system visualizations."""

    defpos = Position(*(0, 1.0, 0.0, 0.0, 0.0))

    def __init__(self, instance: BinarySystem) -> None:
        """Initialize the animation helper.

        :param instance: Binary system instance to animate.
        :type instance: BinarySystem
        :return: ``None``.
        :rtype: None
        """
        self.binary = instance

    # noinspection PyUnusedLocal
    def orbital_motion(
        self,
        start_phase: Float = -0.5,
        stop_phase: Float = 0.5,
        phase_step: Float = 0.01,
        units: str = "cgs",
        scale: str = "linear",
        colormap: str | None = None,
        savepath: str | Path | None = None,
        *,
        separate_colormaps: bool | None = None,
        subtract_equilibrium: bool = False,
        plot_axis: bool = True,
        edges: bool = False,
    ) -> None:
        """Create an animation of the orbital motion.

        :param start_phase: Starting orbital phase of the animation.
        :type start_phase: Float
        :param stop_phase: Ending orbital phase of the animation.
        :type stop_phase: Float
        :param phase_step: Phase step between animation frames.
        :type phase_step: Float
        :param units: Unit system of the surface colormap, for example
            ``"SI"`` or ``"cgs"``.
        :type units: str
        :param scale: Colormap scale, either ``"linear"`` or ``"log"``.
        :type scale: str
        :param colormap: Surface quantity to visualize.
        :type colormap: str | None
        :param savepath: Output path where the animation should be stored.
        :type savepath: str | Path | None
        :param separate_colormaps: If ``True``, use separate colormaps for each
            component.
        :type separate_colormaps: bool | None
        :param subtract_equilibrium: If ``True``, subtract the equilibrium part
            of the quantity, which is useful for pulsation-related quantities.
        :type subtract_equilibrium: bool
        :param plot_axis: If ``False``, hide the axis.
        :type plot_axis: bool
        :param edges: If ``True``, highlight edges of surface faces.
        :type edges: bool
        :return: ``None``.
        :rtype: None

        Available colormap options include:

        - ``"gravity_acceleration"``: surface distribution of gravity acceleration
        - ``"temperature"``: surface distribution of the effective temperature
        - ``"velocity"``: absolute value of surface element velocities with
          respect to the observer
        - ``"radial_velocity"``: radial component of surface element velocities
          relative to the observer
        - ``"normal_radiance"``: surface element radiance perpendicular to the
          surface element
        - ``"radiance"``: radiance of the surface element in the direction of
          the observer
        - ``"radius"``: distance of surface elements from the centre of mass
        - ``"horizontal_displacement"``: distribution of the horizontal
          component of surface displacement
        - ``"horizontal_acceleration"``: distribution of the horizontal
          component of surface acceleration
        - ``"v_r_perturbed"``: radial component of the pulsation velocity
        - ``"v_horizontal_perturbed"``: horizontal component of the pulsation
          velocity
        """
        del units

        if stop_phase < start_phase:
            message = (
                f"Starting phase {start_phase} is greater than stop phase "
                f"{stop_phase}."
            )
            raise ValueError(message)

        components = butils.component_to_list("both")

        if separate_colormaps is None:
            separate_colormaps = (
                self.binary.morphology != "over-contact"
                and colormap not in ["velocity", "radial_velocity"]
            )

        n_frames = int((stop_phase - start_phase) / phase_step)
        phases: NDArray[np.float64] = np.linspace(
            start_phase,
            stop_phase,
            num=n_frames,
        )

        none = [None for _ in range(n_frames)]
        points = {component: copy(none) for component in components}
        faces = {component: copy(none) for component in components}
        cmap = {component: copy(none) for component in components}

        orbital_motion = self.binary.calculate_orbital_motion(
            input_argument=phases,
            return_nparray=False,
            calculate_from="phase",
        )

        # in case of asynchronous component rotation and spots, the positions of spots are recalculated
        spots_longitudes = dynamic.calculate_spot_longitudes(
            self.binary,
            phases,
            component="all",
        )
        potentials = self.binary.correct_potentials(
            phases,
            component="all",
            iterations=2,
        )
        distances_to_com = {"primary": np.nan, "secondary": np.nan}

        logger.info("calculating surface parameters (points, faces, colormap)")
        for pos_idx, position in enumerate(orbital_motion):
            from_this = {
                "binary_system": self.binary,
                "position": self.defpos,
            }
            on_pos = OrbitalPositionContainer.from_binary_system(**from_this)
            on_pos.time = 86400 * self.binary.period * position.phase
            dynamic.assign_spot_longitudes(
                on_pos,
                spots_longitudes,
                index=pos_idx,
                component="all",
            )
            on_pos.set_on_position_params(
                position,
                potentials["primary"][pos_idx],
                potentials["secondary"][pos_idx],
            )
            on_pos.build(components_distance=position.distance)

            # calculating radiances
            # Local import to avoid circular import at module import time
            from elisa.observer.observer import Observer  # noqa: PLC0415

            observer = Observer(passband=["bolometric"], system=self.binary)
            atm_kwargs = {
                "passband": observer.passband,
                "left_bandwidth": observer.left_bandwidth,
                "right_bandwidth": observer.right_bandwidth,
            }

            crv_utils.prep_surface_params(
                system=on_pos,
                write_to_containers=True,
                **atm_kwargs,
            )

            com = {"primary": 0.0, "secondary": position.distance}
            distances_to_com = {
                "primary": position.distance * self.binary.mass_ratio
                / (1 + self.binary.mass_ratio),
                "secondary": position.distance / (1 + self.binary.mass_ratio),
            }
            on_pos.primary.points[:, 0] -= distances_to_com["primary"]
            on_pos.secondary.points[:, 0] -= distances_to_com["primary"]

            on_pos = butils.move_sys_onpos(
                on_pos,
                position,
                potentials["primary"][pos_idx],
                potentials["secondary"][pos_idx],
                on_copy=False,
            )

            for component in components:
                star = getattr(on_pos, component)
                points[component][pos_idx] = star.points
                faces[component][pos_idx] = star.faces

                args = (
                    colormap,
                    star,
                    position.phase,
                    com[component],
                    self.binary.semi_major_axis,
                    self.binary.inclination,
                    on_pos.position,
                )
                kwargs = {
                    "scale": scale,
                    "unit": "default",
                    "subtract_equilibrium": subtract_equilibrium,
                }
                cmap[component][pos_idx] = plot.add_colormap_to_plt_kwargs(
                    *args,
                    **kwargs,
                )

        anim_kwargs = {
            "morphology": self.binary.morphology,
            "start_phase": start_phase,
            "stop_phase": stop_phase,
            "n_frames": n_frames,
            "phases": phases,
            "points_primary": points["primary"],
            "points_secondary": points["secondary"],
            "faces_primary": faces["primary"],
            "faces_secondary": faces["secondary"],
            "primary_cmap": cmap["primary"],
            "secondary_cmap": cmap["secondary"],
            "axis_lim": 1.3
            * np.max(
                (
                    distances_to_com["primary"],
                    distances_to_com["secondary"],
                ),
            ),
            "savepath": savepath,
            "colormap": colormap,
            "separate_colormaps": separate_colormaps,
            "plot_axis": plot_axis,
            "edges": edges,
        }

        logger.debug("Passing parameters to graphics module")
        graphics.binary_surface_anim(**anim_kwargs)
