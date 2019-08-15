import numpy as np

from copy import copy

from elisa import utils, logger, graphics, const
from elisa.binary_system import geo


class Animation(object):
    def __init__(self, instance):
        self._logger = logger.getLogger(name=self.__class__.__name__)
        self._self = instance

    def orbital_motion(self, **kwargs):
        all_kwargs = ['start_phase', 'stop_phase', 'phase_step', 'inclination', 'units', 'plot_axis', 'colormap',
                      'savepath']
        utils.invalid_kwarg_checker(kwargs, all_kwargs, self.orbital_motion)

        kwargs['start_phase'] = kwargs.get('start_phase', -0.5)
        kwargs['stop_phase'] = kwargs.get('stop_phase', 0.5)
        kwargs['phase_step'] = kwargs.get('phase_step', 0.01)
        kwargs['inclination'] = kwargs.get('inclination', np.degrees(self._self.inclination))
        kwargs['units'] = kwargs.get('units', 'logg_cgs')
        kwargs['plot_axis'] = kwargs.get('plot_axis', True)
        kwargs['colormap'] = kwargs.get('colormap', None)
        kwargs['savepath'] = kwargs.get('savepath', None)

        kwargs['morphology'] = self._self.morphology

        if kwargs['stop_phase'] < kwargs['start_phase']:
            raise ValueError('Starting phase {0} is greater than stop phase {1}'.format(kwargs['start_phase'],
                                                                                        kwargs['stop_phase']))
        kwargs['Nframes'] = int((kwargs['stop_phase'] - kwargs['start_phase'])/kwargs['phase_step'])
        kwargs['phases'] = np.linspace(kwargs['start_phase'], kwargs['stop_phase'], num=kwargs['Nframes'])
        kwargs['points_primary'] = [None for _ in range(kwargs['Nframes'])]
        kwargs['points_secondary'] = [None for _ in range(kwargs['Nframes'])]
        kwargs['faces_primary'] = [None for _ in range(kwargs['Nframes'])]
        kwargs['faces_secondary'] = [None for _ in range(kwargs['Nframes'])]
        kwargs['primary_cmap'] = [None for _ in range(kwargs['Nframes'])]
        kwargs['secondary_cmap'] = [None for _ in range(kwargs['Nframes'])]

        kwargs['axis_lim'] = 0.7

        result = self._self.orbit.orbital_motion(phase=kwargs['phases'])[:, :2]
        components_distance, azimuth = result[:, 0], result[:, 1]
        com = components_distance * self._self.mass_ratio / (1 + self._self.mass_ratio)

        # in case of assynchronous component rotation and spots, the positions of spots are recalculated
        spots_longitudes = geo.calculate_spot_longitudes(self._self, kwargs['phases'], component=None)

        self._logger.info('Calculating surface parameters (points, faces, colormap)')
        for idx, phase in enumerate(kwargs['phases']):
            # assigning new longitudes for each spot
            geo.assign_spot_longitudes(self._self, spots_longitudes, index=idx, component=None)

            points, faces = self._self.build_surface(component=None,
                                                     components_distance=components_distance[idx],
                                                     return_surface=True)
            if kwargs['colormap']:
                cmap = self._self.build_surface_map(colormap=kwargs['colormap'], component=None,
                                                    components_distance=components_distance[idx], return_map=True,
                                                    phase=phase)

            kwargs['points_primary'][idx] = copy(points['primary'])
            kwargs['points_secondary'][idx] = copy(points['secondary'])
            kwargs['faces_primary'][idx] = copy(faces['primary'])
            kwargs['faces_secondary'][idx] = copy(faces['secondary'])
            if kwargs['colormap']:
                kwargs['primary_cmap'][idx] = copy(cmap['primary'])
                kwargs['secondary_cmap'][idx] = copy(cmap['secondary'])

            # correcting to barycentre reference frame
            kwargs['points_primary'][idx][:, 0] -= com[idx]
            kwargs['points_secondary'][idx][:, 0] -= com[idx]

            # rotating pints to correct place
            # rotation by azimuth
            kwargs['points_primary'][idx] = utils.around_axis_rotation(azimuth[idx] - const.HALF_PI,
                                                                       kwargs['points_primary'][idx],
                                                                       "z", False, False)
            kwargs['points_secondary'][idx] = utils.around_axis_rotation(azimuth[idx] - const.HALF_PI,
                                                                         kwargs['points_secondary'][idx],
                                                                         "z", False, False)
            # tilting the orbit due to inclination
            kwargs['points_primary'][idx] = utils.around_axis_rotation(const.HALF_PI - self._self.inclination,
                                                                       kwargs['points_primary'][idx],
                                                                       "y", False, False)
            kwargs['points_secondary'][idx] = utils.around_axis_rotation(const.HALF_PI - self._self.inclination,
                                                                         kwargs['points_secondary'][idx],
                                                                         "y", False, False)

        self._logger.debug('Passing parameters to graphics module')
        graphics.binary_surface_anim(**kwargs)

