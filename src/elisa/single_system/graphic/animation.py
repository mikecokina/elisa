from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa.base.graphics import plot
from elisa.const import Position
from elisa.graphic import graphics
from elisa.logger import getLogger
from elisa.observer.observer import Observer
from elisa.single_system import utils as sutils
from elisa.single_system.container import SinglePositionContainer
from elisa.single_system.curves import utils as crv_utils

if TYPE_CHECKING:
    import pathlib

    from elisa.single_system.system import SingleSystem
    from elisa.types import Float


logger = getLogger("single_system.graphic.animation")


class Animation:
    """Helper for creating animations of a single-star system's rotational motion.

    :cvar defpos: Default position used when preparing temporary containers.
    :type defpos: Position
    """

    defpos: Position = Position(*(0, np.nan, 0.0, np.nan, 0.0))

    def __init__(self, instance: SingleSystem) -> None:
        """Initialise an :class:`Animation` helper for a single-system instance.

        :param instance: Single-system instance for which animations will be produced.
        :type instance: elisa.single_system.system.SingleSystem
        :return: None
        :rtype: None
        """
        self.single = instance

    def rotational_motion(
        self,
        start_phase: Float = -0.5,
        stop_phase: Float = 0.5,
        phase_step: Float = 0.01,
        scale: str = "linear",
        colormap: str | None = None,
        savepath: pathlib.Path | str | None = None,
        *,
        plot_axis: bool = True,
        subtract_equilibrium: bool = False,
        edges: bool = False,
    ) -> None:
        """Create an animation of the rotational motion of the single system.

        The function steps the system through phases from ``start_phase`` to
        ``stop_phase`` (inclusive) with step ``phase_step`` and prepares frame
        data (points, faces and colormap) which is then passed to the graphics
        layer.

        :param start_phase: Starting phase of the animation.
        :type start_phase: elisa.types.Float
        :param stop_phase: Ending phase of the animation.
        :type stop_phase: elisa.types.Float
        :param phase_step: Phase increment between frames.
        :type phase_step: elisa.types.Float
        :param scale: Colormap scale, typically ``"linear"`` or ``"log"``.
        :type scale: str
        :param colormap: Name of the requested colormap or ``None`` to use default.
        :type colormap: str | None
        :param savepath: If provided, animation will be stored to this path.
        :type savepath: pathlib.Path | str | None
        :param plot_axis: If ``False``, axes will be hidden in the produced plot.
        :type plot_axis: bool
        :param subtract_equilibrium: Remove equilibrium part of quantity (useful for pulsations).
        :type subtract_equilibrium: bool
        :param edges: Highlight edges of surface faces when rendering.
        :type edges: bool
        :return: None
        :rtype: None
        """
        anim_kwargs: dict = {}

        if stop_phase < start_phase:
            msg = f"Starting phase {start_phase} is greater than stop phase {stop_phase}"
            raise ValueError(msg)

        n_frames = int((stop_phase - start_phase) / phase_step)
        phases = np.linspace(start_phase, stop_phase, num=n_frames)
        points: list = []
        faces: list = []
        cmap: list = []

        orbital_motion = self.single.calculate_lines_of_sight(
            input_argument=phases, return_nparray=False, calculate_from="phase",
        )

        # calculating radiances
        o = Observer(passband=["bolometric"], system=self.single)
        atm_kwargs = {
            "passband": o.passband,
            "left_bandwidth": o.left_bandwidth,
            "right_bandwidth": o.right_bandwidth,
        }

        logger.info("calculating surface parameters (points, faces, colormap)")
        for position in orbital_motion:
            from_this = {"single_system": self.single, "position": self.defpos}
            on_pos = SinglePositionContainer.from_single_system(**from_this)
            on_pos.set_on_position_params(position)
            on_pos.set_time()
            on_pos.build()

            crv_utils.prep_surface_params(
                system=on_pos, write_to_containers=True, **atm_kwargs,
            )

            on_pos = sutils.move_sys_onpos(on_pos, position, on_copy=False)

            star = on_pos.star
            points.append(star.points)
            faces.append(star.faces)

            args = (colormap, star, position.phase, 0.0, 1.0, self.single.inclination, on_pos.position)
            kwargs = {"scale": scale, "unit": "default", "subtract_equilibrium": subtract_equilibrium}
            cmap.append(plot.add_colormap_to_plt_kwargs(*args, **kwargs))

        anim_kwargs.update({
            "start_phase": start_phase,
            "stop_phase": stop_phase,
            "n_frames": n_frames,
            "phases": phases,
            "points": points,
            "faces": faces,
            "cmap": cmap,
            "axis_lim": 1.2 * np.max(points[0]),
            "savepath": savepath,
            "colormap": colormap,
            "plot_axis": plot_axis,
            "edges": edges,
        })
        logger.debug("Passing parameters to graphics module")
        graphics.single_surface_anim(**anim_kwargs)
