import numpy as np

from copy import copy
from .. container import OrbitalPositionContainer
from .. import utils as butils, dynamic
from ... const import Position
from ... logger import getLogger
from ... graphic import graphics
from ... base.graphics import plot
from .. curves import utils as crv_utils
from ... observer.observer import Observer

logger = getLogger('binary_system.graphic.animation')


class Animation(object):
    defpos = Position(*(0, 1.0, 0.0, 0.0, 0.0))

    def __init__(self, instance):
        self.binary = instance

    def orbital_motion(self, start_phase=-0.5, stop_phase=0.5, phase_step=0.01, units='cgs', scale='linear',
                       colormap=None, savepath=None, separate_colormaps=None, subtract_equilibrium=False,
                       plot_axis=True, edges=False):
        """
        Function creates animation of the orbital motion.

        :param start_phase: float; starting phase of the animation
        :param stop_phase: float; end phase of the animation
        :param phase_step: float; phase step between animation frames
        :param units: str; unit type of surface colormap `SI` or `cgs`
        :param scale: str; `linear` or `log`, scale of the colormap
        :param colormap: str;
        :param savepath: str; animation will be stored to `savepath`
        :param separate_colormaps: bool; if True, figure will contain separate colormap for each component
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

        components = butils.component_to_list('both')

        if separate_colormaps is None:
            separate_colormaps = self.binary.morphology != 'over-contact' and \
                                 colormap not in ['velocity', 'radial_velocity']

        n_frames = int((stop_phase - start_phase) / phase_step)
        phases = np.linspace(start_phase, stop_phase, num=n_frames)
        none = [None for _ in range(n_frames)]
        points = {component: copy(none) for component in components}
        faces = {component: copy(none) for component in components}
        cmap = {component: copy(none) for component in components}

        orbital_motion = \
            self.binary.calculate_orbital_motion(input_argument=phases, return_nparray=False, calculate_from='phase')

        # in case of asynchronous component rotation and spots, the positions of spots are recalculated
        spots_longitudes = dynamic.calculate_spot_longitudes(self.binary, phases, component="all")
        potentials = self.binary.correct_potentials(phases, component="all", iterations=2)

        logger.info('calculating surface parameters (points, faces, colormap)')
        for pos_idx, position in enumerate(orbital_motion):
            from_this = dict(binary_system=self.binary, position=self.defpos)
            on_pos = OrbitalPositionContainer.from_binary_system(**from_this)
            on_pos.time = 86400 * self.binary.period * position.phase
            dynamic.assign_spot_longitudes(on_pos, spots_longitudes, index=pos_idx, component="all")
            on_pos.set_on_position_params(position, potentials["primary"][pos_idx],
                                          potentials["secondary"][pos_idx])
            on_pos.build(components_distance=position.distance)

            # calculating radiances
            o = Observer(passband=['bolometric', ], system=self.binary)
            atm_kwargs = dict(
                passband=o.passband,
                left_bandwidth=o.left_bandwidth,
                right_bandwidth=o.right_bandwidth,
            )

            crv_utils.prep_surface_params(
                system=on_pos, write_to_containers=True, **atm_kwargs
            )

            com = {'primary': 0.0, 'secondary': position.distance}
            distances_to_com = {'primary': position.distance * self.binary.mass_ratio / (1 + self.binary.mass_ratio),
                                'secondary': position.distance / (1 + self.binary.mass_ratio)}
            on_pos.primary.points[:, 0] -= distances_to_com['primary']
            on_pos.secondary.points[:, 0] -= distances_to_com['primary']

            on_pos = butils.move_sys_onpos(on_pos, position, potentials["primary"][pos_idx],
                                           potentials["secondary"][pos_idx], on_copy=False)

            for component in components:
                star = getattr(on_pos, component)
                points[component][pos_idx], faces[component][pos_idx] = star.points, star.faces

                args = (colormap, star, position.phase, com[component], self.binary.semi_major_axis,
                        self.binary.inclination, on_pos.position)
                kwargs = dict(scale=scale, unit='default', subtract_equilibrium=subtract_equilibrium)
                cmap[component][pos_idx] = plot.add_colormap_to_plt_kwargs(*args, **kwargs)

        anim_kwargs.update({
            'morphology': self.binary.morphology,
            'start_phase': start_phase,
            'stop_phase': stop_phase,
            'n_frames': n_frames,
            'phases': phases,
            'points_primary': points['primary'],
            'points_secondary': points['secondary'],
            'faces_primary': faces['primary'],
            'faces_secondary': faces['secondary'],
            'primary_cmap': cmap['primary'],
            'secondary_cmap': cmap['secondary'],
            'axis_lim': 1.3*np.max((distances_to_com['primary'], distances_to_com['secondary'])),
            'savepath': savepath,
            'colormap': colormap,
            "separate_colormaps": separate_colormaps,
            "plot_axis": plot_axis,
            "edges": edges
        })
        logger.debug('Passing parameters to graphics module')
        graphics.binary_surface_anim(**anim_kwargs)
