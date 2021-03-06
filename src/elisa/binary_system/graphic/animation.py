import numpy as np

from copy import copy
from .. container import OrbitalPositionContainer
from .. import utils as butils, dynamic
from ... import umpy as up
from ... const import Position
from ... logger import getLogger
from ... graphic import graphics

logger = getLogger('binary_system.graphic.animation')


class Animation(object):
    defpos = Position(*(0, 1.0, 0.0, 0.0, 0.0))

    def __init__(self, instance):
        self.binary = instance

    def orbital_motion(self, start_phase=-0.5, stop_phase=0.5, phase_step=0.01, units='cgs', scale='linear',
                       colormap=None, savepath=None, separate_colormaps=None):
        """
        Function creates animation of the orbital motion.

        :param start_phase: float; starting phase of the animation
        :param stop_phase: float; end phase of the animation
        :param phase_step: float; phase step between animation frames
        :param units: str; unit type of surface colormap `SI` or `cgs`
        :param scale: str; `linear` or `log`, scale of the colormap
        :param colormap: str; `temperature`, `gravity_acceleration`, `velocity`, `radial_velocity` or None
        :param savepath: str; animation will be stored to `savepath`
        :param separate_colormaps: bool; if True, figure will contain separate colormap for each component
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
        mult = np.array([-1, -1, 1.0])[None, :]
        for pos_idx, position in enumerate(orbital_motion):
            from_this = dict(binary_system=self.binary, position=self.defpos)
            on_pos = OrbitalPositionContainer.from_binary_system(**from_this)
            on_pos.time = 86400 * self.binary.period * position.phase
            dynamic.assign_spot_longitudes(on_pos, spots_longitudes, index=pos_idx, component="all")
            on_pos.set_on_position_params(position, potentials["primary"][pos_idx],
                                          potentials["secondary"][pos_idx])
            on_pos.build(components_distance=position.distance)
            distances_to_com = {'primary': position.distance * self.binary.mass_ratio / (1 + self.binary.mass_ratio),
                                'secondary': position.distance / (1 + self.binary.mass_ratio)}
            on_pos.flatt_it()
            on_pos.primary.points[:, 0] -= distances_to_com['primary']
            on_pos.secondary.points[:, 0] -= distances_to_com['secondary']

            on_pos = butils.move_sys_onpos(on_pos, position, potentials["primary"][pos_idx],
                                           potentials["secondary"][pos_idx], on_copy=False)

            for component in components:
                star = getattr(on_pos, component)
                points[component][pos_idx], faces[component][pos_idx] = mult * star.points, star.faces

                if colormap == 'gravity_acceleration':
                    log_g = star.log_g
                    value = log_g if units == 'SI' else log_g + 2
                    cmap[component][pos_idx] = value if scale == 'log' else up.power(10, value)

                elif colormap == 'temperature':
                    temperatures = star.temperatures
                    cmap[component][pos_idx] = temperatures if scale == 'linear' else up.log10(temperatures)

                elif colormap == 'velocity':
                    velocities = np.linalg.norm(getattr(star, 'velocities'), axis=1)
                    velocities = velocities / 1000.0 if units == 'SI' else velocities * 1000.0
                    cmap[component][pos_idx] = velocities if scale == 'linear' else up.log10(velocities)

                elif colormap == 'radial_velocity':
                    velocities = getattr(star, 'velocities')[:, 0]
                    velocities = velocities / 1000.0 if units == 'SI' else velocities * 1000.0
                    cmap[component][pos_idx] = velocities
                    if scale == 'log':
                        raise Warning("`log` scale is not allowed for radial velocity colormap.")

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
            'axis_lim': 0.7,
            'savepath': savepath,
            'colormap': colormap,
            "separate_colormaps": separate_colormaps,
        })
        logger.debug('Passing parameters to graphics module')
        graphics.binary_surface_anim(**anim_kwargs)
