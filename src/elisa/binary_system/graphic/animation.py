import numpy as np

from copy import copy
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.const import Position
from elisa.logger import getLogger
from elisa.graphic import graphics

from elisa.binary_system import (
    utils as butils,
    dynamic
)
from elisa import (
    umpy as up,
    utils,
    const
)

logger = getLogger('binary_system.graphic.animation')


class Animation(object):
    defpos = Position(*(0, 1.0, 0.0, 0.0, 0.0))

    def __init__(self, instance):
        self.binary = instance

    def orbital_motion(self, start_phase=-0.5, stop_phase=0.5, phase_step=0.01, units='cgs', scale='linear',
                       colormap=None, savepath=None):
        """
        Function creates animation of the orbital motion.

        :param start_phase: float; starting phase of the animation
        :param stop_phase: float; end phase of the animation
        :param phase_step: float; phase step between animation frames
        :param units: str; unit type of surface colormap `SI` or `cgs`
        :param scale: str; `linear` or `log`, scale of the colormap
        :param colormap: str; `temperature`, `gravity_acceleration` or None
        :param savepath: str; animation will be stored to `savepath`
        """
        anim_kwargs = dict()

        if stop_phase < start_phase:
            raise ValueError(f'Starting phase {start_phase} is greater than stop phase {stop_phase}')

        components = butils.component_to_list('both')

        n_frames = int((stop_phase - start_phase) / phase_step)
        phases = np.linspace(start_phase, stop_phase, num=n_frames)
        none = [None for _ in range(n_frames)]
        points = {component: copy(none) for component in components}
        faces = {component: copy(none) for component in components}
        cmap = {component: copy(none) for component in components}

        result = self.binary.orbit.orbital_motion(phase=phases)[:, :2]
        components_distance, azimuth = result[:, 0], result[:, 1]

        # in case of assynchronous component rotation and spots, the positions of spots are recalculated
        spots_longitudes = dynamic.calculate_spot_longitudes(self.binary, phases, component="all")
        orbital_position_container = OrbitalPositionContainer.from_binary_system(self.binary, self.defpos)
        logger.info('calculating surface parameters (points, faces, colormap)')
        for idx, phase in enumerate(phases):
            # assigning new longitudes for each spot
            dynamic.assign_spot_longitudes(orbital_position_container, spots_longitudes, index=idx, component="both")
            orbital_position_container.build(components_distance=components_distance[idx], components='both')

            for component in components:
                star = getattr(orbital_position_container, component)
                points[component][idx], faces[component][idx] = star.surface_serializer()

                if colormap == 'gravity_acceleration':
                    log_g = star.get_flatten_parameter('log_g')
                    value = log_g if units == 'SI' else log_g + 2
                    cmap[component][idx] = value if scale == 'log' else up.power(10, value)

                elif colormap == 'temperature':
                    temperatures = star.get_flatten_parameter('temperatures')
                    cmap[component][idx] = temperatures if scale == 'linear' else up.log10(temperatures)

                # rotating pints to correct place
                # rotation by azimuth
                points[component][idx] = utils.around_axis_rotation(azimuth[idx] - const.HALF_PI,
                                                                    points[component][idx],
                                                                    "z", False, False)
                # tilting the orbit due to inclination
                points[component][idx] = utils.around_axis_rotation(const.HALF_PI - self.binary.inclination,
                                                                    points[component][idx],
                                                                    "y", False, False)

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
        })
        logger.debug('Passing parameters to graphics module')
        graphics.binary_surface_anim(**anim_kwargs)

