import numpy as np

from elisa.logger import getLogger

from ... const import Position
from ... graphic import graphics
from .. container import SinglePositionContainer
from .. import utils as sutils
from ... base.graphics import plot
from .. curves import utils as crv_utils
from ... observer.observer import Observer


logger = getLogger('single_system.graphic.animation')


class Animation(object):
    defpos = Position(*(0, np.nan, 0.0, np.nan, 0.0))

    def __init__(self, instance):
        self.single = instance

    def rotational_motion(self, start_phase=-0.5, stop_phase=0.5, phase_step=0.01, scale='linear',
                          colormap=None, savepath=None, plot_axis=True, subtract_equilibrium=False, edges=False):
        """
        Function creates animation of the rotational motion.

        :param start_phase: float; starting phase of the animation
        :param stop_phase: float; end phase of the animation
        :param phase_step: float; phase step between animation frames
        :param scale: str; `linear` or `log`, scale of the colormap
        :param colormap: str;
        :param savepath: str; animation will be stored to `savepath`
        :param subtract_equilibrium: bool; equilibrium part of the quantity is removed (for pulsations)
        :param plot_axis: bool; if False, axis will be hidden
        :param edges: bool; highlight edges of surface faces
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

        # calculating radiances
        o = Observer(passband=['bolometric', ], system=self.single)
        atm_kwargs = dict(
            passband=o.passband,
            left_bandwidth=o.left_bandwidth,
            right_bandwidth=o.right_bandwidth,
        )

        logger.info('calculating surface parameters (points, faces, colormap)')
        for pos_idx, position in enumerate(orbital_motion):
            from_this = dict(single_system=self.single, position=self.defpos)
            on_pos = SinglePositionContainer.from_single_system(**from_this)
            on_pos.set_on_position_params(position)
            on_pos.set_time()
            on_pos.build()

            crv_utils.prep_surface_params(
                system=on_pos, write_to_containers=True, **atm_kwargs
            )

            on_pos = sutils.move_sys_onpos(on_pos, position, on_copy=False)

            star = getattr(on_pos, 'star')
            points.append(star.points)
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
            'colormap': colormap,
            "plot_axis": plot_axis,
            'edges': edges
        })
        logger.debug('Passing parameters to graphics module')
        graphics.single_surface_anim(**anim_kwargs)


