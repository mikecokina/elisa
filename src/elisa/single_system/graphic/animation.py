import numpy as np

from elisa.logger import getLogger

from ... const import Position
from ... graphic import graphics
from .. container import SinglePositionContainer
from .. import utils as sutils
from ... base.graphics import plot

logger = getLogger('single_system.graphic.animation')


class Animation(object):
    defpos = Position(*(0, np.nan, 0.0, np.nan, 0.0))

    def __init__(self, instance):
        self.single = instance

    def rotational_motion(self, start_phase=-0.5, stop_phase=0.5, phase_step=0.01, units='cgs', scale='linear',
                          colormap=None, savepath=None, subtract_equilibrium=False):
        """
        Function creates animation of the rotational motion.

        :param start_phase: float; starting phase of the animation
        :param stop_phase: float; end phase of the animation
        :param phase_step: float; phase step between animation frames
        :param units: str; unit type of surface colormap `SI` or `cgs`
        :param scale: str; `linear` or `log`, scale of the colormap
        :param colormap: str; `temperature`, `gravity_acceleration`, `velocity`, `radial_velocity` or None
        :param savepath: str; animation will be stored to `savepath`
        :param subtract_equilibrium: bool; equilibrium part of the quantity is removed (for pulsations)
        """
        anim_kwargs = dict()

        if stop_phase < start_phase:
            raise ValueError(f'Starting phase {start_phase} is greater than stop phase {stop_phase}')

        n_frames = int((stop_phase - start_phase) / phase_step)
        phases = np.linspace(start_phase, stop_phase, num=n_frames)
        points = []
        faces = []
        cmap = []

        orbital_motion = \
            self.single.calculate_lines_of_sight(input_argument=phases, return_nparray=False, calculate_from='phase')

        logger.info('calculating surface parameters (points, faces, colormap)')
        mult = np.array([-1, -1, 1.0])[None, :]
        for pos_idx, position in enumerate(orbital_motion):
            from_this = dict(single_system=self.single, position=self.defpos)
            on_pos = SinglePositionContainer.from_single_system(**from_this)
            on_pos.set_on_position_params(position)
            on_pos.set_time()
            on_pos.build()
            on_pos = sutils.move_sys_onpos(on_pos, position, on_copy=False)

            star = getattr(on_pos, 'star')
            points.append(mult * star.points)
            faces.append(star.faces)

            args = (colormap, star, position.phase, 0.0, 1.0, self.single.inclination, on_pos.position)
            kwargs = dict(scale=scale, unit='default', subtract_equilibrium=subtract_equilibrium)
            cmap.append(plot.add_colormap_to_plt_kwargs(*args, **kwargs))

        anim_kwargs.update({
            'start_phase': start_phase,
            'stop_phase': stop_phase,
            'n_frames': n_frames,
            'phases': phases,
            'points': points,
            'faces': faces,
            'cmap': cmap,
            'axis_lim': 1.2 * np.max(points[0]),
            'savepath': savepath,
            'colormap': colormap
        })
        logger.debug('Passing parameters to graphics module')
        graphics.single_surface_anim(**anim_kwargs)


