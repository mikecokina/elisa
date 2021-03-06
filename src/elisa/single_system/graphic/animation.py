import numpy as np

from elisa.logger import getLogger

from ... const import Position
from ... graphic import graphics
from .. container import SystemContainer
from .. import utils as sutils

logger = getLogger('single_system.graphic.animation')


class Animation(object):
    defpos = Position(*(0, np.nan, 0.0, np.nan, 0.0))

    def __init__(self, instance):
        self.single = instance

    def rotational_motion(self, start_phase=-0.5, stop_phase=0.5, phase_step=0.01, units='cgs', scale='linear',
                          colormap=None, savepath=None):
        """
        Function creates animation of the rotational motion.

        :param start_phase: float; starting phase of the animation
        :param stop_phase: float; end phase of the animation
        :param phase_step: float; phase step between animation frames
        :param units: str; unit type of surface colormap `SI` or `cgs`
        :param scale: str; `linear` or `log`, scale of the colormap
        :param colormap: str; `temperature`, `gravity_acceleration`, `velocity`, `radial_velocity` or None
        :param savepath: str; animation will be stored to `savepath`
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
            on_pos = SystemContainer.from_single_system(**from_this)
            on_pos.time = 86400 * self.single.rotation_period * position.phase
            on_pos.build()
            on_pos = sutils.move_sys_onpos(on_pos, position, on_copy=False)

            star = getattr(on_pos, 'star')
            points.append(mult * star.points)
            faces.append(star.faces)

            if colormap == 'gravity_acceleration':
                log_g = star.log_g
                value = log_g if units == 'SI' else log_g + 2
                val_to_append = value if scale == 'log' else np.power(10, value)
                cmap.append(val_to_append)

            elif colormap == 'temperature':
                temperatures = star.temperatures
                val_to_append = temperatures if scale == 'linear' else np.log10(temperatures)
                cmap.append(val_to_append)

            elif colormap == 'velocity':
                velocities = np.linalg.norm(getattr(star, 'velocities'), axis=1)
                velocities = velocities / 1000.0 if units == 'SI' else velocities * 1000.0
                val_to_append = velocities if scale == 'linear' else np.log10(velocities)
                cmap.append(val_to_append)

            elif colormap == 'radial_velocity':
                velocities = getattr(star, 'velocities')[:, 0]
                velocities = velocities / 1000.0 if units == 'SI' else velocities * 1000.0
                cmap.append(velocities)
                if scale == 'log':
                    raise Warning("`log` scale is not allowed for radial velocity colormap.")

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


