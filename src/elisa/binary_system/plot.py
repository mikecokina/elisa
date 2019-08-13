import numpy as np
from astropy import units as u
from elisa.binary_system import geo

from elisa import utils, const, graphics, units


class Plot(object):
    """
    universal plot interface for binary system class, more detailed documentation for each value of descriptor is
    available in graphics library

                        `orbit` - plots orbit in orbital plane
                        `equipotential` - plots crossections of surface Hill planes in xy,yz,zx planes
                        `mesh` - plot surface points
                        `surface` - plot stellar surfaces
    :return:
    """

    def __init__(self, instance):
        self._self = instance

    def orbit(self, **kwargs):
        all_kwargs = ['start_phase', 'stop_phase', 'number_of_points', 'axis_unit', 'frame_of_reference']
        utils.invalid_kwarg_checker(kwargs, all_kwargs, self._self)

        start_phase = kwargs.get('start_phase', 0.0)
        stop_phase = kwargs.get('stop_phase', 1.0)
        number_of_points = kwargs.get('number_of_points', 300)

        kwargs['axis_unit'] = kwargs.get('axis_unit', u.solRad)
        kwargs['frame_of_reference'] = kwargs.get('frame_of_reference', 'primary_component')

        if kwargs['axis_unit'] == 'dimensionless':
            kwargs['axis_unit'] = u.dimensionless_unscaled

        # orbit calculation for given phases
        phases = np.linspace(start_phase, stop_phase, number_of_points)
        ellipse = self._self.orbit.orbital_motion(phase=phases)
        # if axis are without unit a = 1
        if kwargs['axis_unit'] != u.dimensionless_unscaled:
            a = self._self.semi_major_axis * units.DISTANCE_UNIT.to(kwargs['axis_unit'])
            radius = a * ellipse[:, 0]
        else:
            radius = ellipse[:, 0]
        azimuth = ellipse[:, 1]
        x, y = utils.polar_to_cartesian(radius=radius, phi=azimuth - const.PI / 2.0)
        if kwargs['frame_of_reference'] == 'barycentric':
            kwargs['x1_data'] = - self._self.mass_ratio * x / (1 + self._self.mass_ratio)
            kwargs['y1_data'] = - self._self.mass_ratio * y / (1 + self._self.mass_ratio)
            kwargs['x2_data'] = x / (1 + self._self.mass_ratio)
            kwargs['y2_data'] = y / (1 + self._self.mass_ratio)
        elif kwargs['frame_of_reference'] == 'primary_component':
            kwargs['x_data'], kwargs['y_data'] = x, y
        graphics.orbit(**kwargs)

    def equipotential(self, **kwargs):
        all_kwargs = ['plane', 'phase']
        utils.invalid_kwarg_checker(kwargs, all_kwargs, self.equipotential)

        kwargs['phase'] = kwargs.get('phase', 0.0)
        kwargs['plane'] = kwargs.get('plane', 'xy')

        # relative distance between components (a = 1)
        if utils.is_plane(kwargs['plane'], 'xy') or utils.is_plane(
                kwargs['plane'], 'yz') or utils.is_plane(kwargs['plane'], 'zx'):
            components_distance = self._self.orbit.orbital_motion(phase=kwargs['phase'])[0][0]
            points_primary, points_secondary = \
                self._self.compute_equipotential_boundary(components_distance=components_distance,
                                                          plane=kwargs['plane'])
        else:
            raise ValueError('Invalid choice of crossection plane, use only: `xy`, `yz`, `zx`.')

        kwargs['points_primary'] = points_primary
        kwargs['points_secondary'] = points_secondary

        graphics.equipotential(**kwargs)

    def mesh(self, **kwargs):
        all_kwargs = ['phase', 'components_to_plot', 'plot_axis', 'inclination', 'azimuth']
        utils.invalid_kwarg_checker(kwargs, all_kwargs, self.mesh)

        kwargs['phase'] = kwargs.get('phase', 0)
        kwargs['components_to_plot'] = kwargs.get('components_to_plot', 'both')
        kwargs['plot_axis'] = kwargs.get('plot_axis', True)
        kwargs['inclination'] = kwargs.get('inclination', np.degrees(self._self.inclination))

        components_distance, azim = self._self.orbit.orbital_motion(phase=kwargs['phase'])[0][:2]
        kwargs['azimuth'] = kwargs.get('azimuth', np.degrees(azim) - 90)

        if kwargs['components_to_plot'] in ['primary', 'both']:
            points, _ = self._self.build_surface(component='primary', components_distance=components_distance,
                                                 return_surface=True)
            kwargs['points_primary'] = points['primary']

        if kwargs['components_to_plot'] in ['secondary', 'both']:
            points, _ = self._self.build_surface(component='secondary', components_distance=components_distance,
                                                 return_surface=True)
            kwargs['points_secondary'] = points['secondary']

        graphics.binary_mesh(**kwargs)

    def wireframe(self, **kwargs):
        all_kwargs = ['phase', 'components_to_plot', 'plot_axis', 'inclination', 'azimuth']
        utils.invalid_kwarg_checker(kwargs, all_kwargs, self.wireframe)

        kwargs['phase'] = kwargs.get('phase', 0)
        kwargs['components_to_plot'] = kwargs.get('components_to_plot', 'both')
        kwargs['plot_axis'] = kwargs.get('plot_axis', True)
        kwargs['inclination'] = kwargs.get('inclination', np.degrees(self._self.inclination))

        components_distance, azim = self._self.orbit.orbital_motion(phase=kwargs['phase'])[0][:2]
        kwargs['azimuth'] = kwargs.get('azimuth', np.degrees(azim) - 90)

        if kwargs['components_to_plot'] in ['primary', 'both']:
            points, faces = self._self.build_surface(component='primary', components_distance=components_distance,
                                                     return_surface=True)
            kwargs['points_primary'] = points['primary']
            kwargs['primary_triangles'] = faces['primary']
        if kwargs['components_to_plot'] in ['secondary', 'both']:
            points, faces = self._self.build_surface(component='secondary', components_distance=components_distance,
                                                     return_surface=True)
            kwargs['points_secondary'] = points['secondary']
            kwargs['secondary_triangles'] = faces['secondary']

        graphics.binary_wireframe(**kwargs)

    def surface(self, **kwargs):
        all_kwargs = ['phase', 'components_to_plot', 'normals', 'edges', 'colormap', 'plot_axis', 'face_mask_primary',
                      'face_mask_secondary', 'inclination', 'azimuth', 'units']
        utils.invalid_kwarg_checker(kwargs, all_kwargs, self.surface)

        kwargs['phase'] = kwargs.get('phase', 0)
        kwargs['components_to_plot'] = kwargs.get('components_to_plot', 'both')
        kwargs['normals'] = kwargs.get('normals', False)
        kwargs['edges'] = kwargs.get('edges', False)
        kwargs['colormap'] = kwargs.get('colormap', None)
        kwargs['plot_axis'] = kwargs.get('plot_axis', True)
        kwargs['face_mask_primary'] = kwargs.get('face_mask_primary', None)
        kwargs['face_mask_secondary'] = kwargs.get('face_mask_secondary', None)
        kwargs['inclination'] = kwargs.get('inclination', np.degrees(self._self.inclination))
        kwargs['units'] = kwargs.get('units', 'logg_cgs')

        components_distance, azim = self._self.orbit.orbital_motion(phase=kwargs['phase'])[0][:2]
        kwargs['azimuth'] = kwargs.get('azimuth', np.degrees(azim) - 90)
        kwargs['morphology'] = self._self.morphology
        kwg = {'suppress_parallelism': False}

        # recalculating spot latitudes
        spots_longitudes = geo.calculate_spot_longitudes(self._self, kwargs['phase'], component=None)
        geo.assign_spot_longitudes(self._self, spots_longitudes, index=None, component=None)

        # this part decides if both components need to be calculated at once (due to reflection effect)
        if kwargs['colormap'] == 'temperature':
            points, faces = self._self.build_surface(components_distance=components_distance,
                                                     return_surface=True, **kwg)
            kwargs['points_primary'] = points['primary']
            kwargs['primary_triangles'] = faces['primary']
            kwargs['points_secondary'] = points['secondary']
            kwargs['secondary_triangles'] = faces['secondary']

            cmap = self._self.build_surface_map(colormap=kwargs['colormap'],
                                                components_distance=components_distance,
                                                return_map=True,
                                                phase=kwargs['phase'])
            kwargs['primary_cmap'] = cmap['primary']
            kwargs['secondary_cmap'] = cmap['secondary']

            if kwargs['face_mask_primary'] is not None:
                kwargs['primary_triangles'] = kwargs['primary_triangles'][kwargs['face_mask_primary']]
                kwargs['primary_cmap'] = kwargs['primary_cmap'][kwargs['face_mask_primary']]
            if kwargs['face_mask_secondary'] is not None:
                kwargs['secondary_triangles'] = kwargs['secondary_triangles'][kwargs['face_mask_secondary']]
                kwargs['secondary_cmap'] = kwargs['secondary_cmap'][kwargs['face_mask_secondary']]

            if kwargs['normals']:
                kwargs['primary_centres'] = self._self.primary.calculate_surface_centres(
                    kwargs['points_primary'], kwargs['primary_triangles'])
                kwargs['primary_arrows'] = self._self.primary.calculate_normals(
                    kwargs['points_primary'], kwargs['primary_triangles'], com=0)
                kwargs['secondary_centres'] = self._self.secondary.calculate_surface_centres(
                    kwargs['points_secondary'], kwargs['secondary_triangles'])
                kwargs['secondary_arrows'] = self._self.secondary.calculate_normals(
                    kwargs['points_secondary'], kwargs['secondary_triangles'], com=components_distance)
        else:
            if kwargs['components_to_plot'] in ['primary', 'both']:
                points, faces = self._self.build_surface(component='primary', components_distance=components_distance,
                                                         return_surface=True, **kwg)
                kwargs['points_primary'] = points['primary']
                kwargs['primary_triangles'] = faces['primary']

                if kwargs['colormap']:
                    cmap = self._self.build_surface_map(colormap=kwargs['colormap'], component='primary',
                                                        components_distance=components_distance, return_map=True)
                    kwargs['primary_cmap'] = cmap['primary']
                    if kwargs['colormap'] == 'gravity_acceleration':
                        kwargs['primary_cmap'] = \
                            utils.convert_gravity_acceleration_array(kwargs['primary_cmap'], kwargs['units'])

                if kwargs['normals']:
                    kwargs['primary_centres'] = self._self.primary.calculate_surface_centres(
                        kwargs['points_primary'], kwargs['primary_triangles'])
                    kwargs['primary_arrows'] = self._self.primary.calculate_normals(
                        kwargs['points_primary'], kwargs['primary_triangles'], com=0)

                if kwargs['face_mask_primary'] is not None:
                    kwargs['primary_triangles'] = kwargs['primary_triangles'][kwargs['face_mask_primary']]
                    kwargs['primary_cmap'] = kwargs['primary_cmap'][kwargs['face_mask_primary']]

            if kwargs['components_to_plot'] in ['secondary', 'both']:
                points, faces = self._self.build_surface(component='secondary', components_distance=components_distance,
                                                         return_surface=True, **kwg)
                kwargs['points_secondary'] = points['secondary']
                kwargs['secondary_triangles'] = faces['secondary']

                if kwargs['colormap']:
                    cmap = self._self.build_surface_map(colormap=kwargs['colormap'], component='secondary',
                                                        components_distance=components_distance, return_map=True)
                    kwargs['secondary_cmap'] = cmap['secondary']
                    if kwargs['colormap'] == 'gravity_acceleration':
                        kwargs['secondary_cmap'] = \
                            utils.convert_gravity_acceleration_array(kwargs['secondary_cmap'], kwargs['units'])

                if kwargs['normals']:
                    kwargs['secondary_centres'] = self._self.secondary.calculate_surface_centres(
                        kwargs['points_secondary'], kwargs['secondary_triangles'])
                    kwargs['secondary_arrows'] = self._self.secondary.calculate_normals(
                        kwargs['points_secondary'], kwargs['secondary_triangles'], com=components_distance)

                if kwargs['face_mask_secondary'] is not None:
                    kwargs['secondary_triangles'] = kwargs['secondary_triangles'][kwargs['face_mask_secondary']]
                    kwargs['secondary_cmap'] = kwargs['secondary_cmap'][kwargs['face_mask_secondary']]

        graphics.binary_surface(**kwargs)
