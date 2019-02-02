from copy import copy

import numpy as np
from astropy import units as u

from elisa.engine import utils, graphics, units


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

    def equipotential(self, **kwargs):
        if 'axis_unit' not in kwargs:
            kwargs['axis_unit'] = u.solRad

        all_kwargs = ['axis_unit']
        utils.invalid_kwarg_checker(kwargs, all_kwargs, self.equipotential)
        points = self._self.calculate_equipotential_boundary()
        kwargs['points'] = (points * units.DISTANCE_UNIT).to(kwargs['axis_unit'])
        graphics.equipotential_single_star(**kwargs)

    def mesh(self, **kwargs):
        if 'axis_unit' not in kwargs:
            kwargs['axis_unit'] = u.solRad

        all_kwargs = ['axis_unit', 'plot_axis', 'inclination', 'azimuth']
        utils.invalid_kwarg_checker(kwargs, all_kwargs, self.mesh)

        kwargs['plot_axis'] = kwargs.get('plot_axis', True)
        kwargs['inclination'] = kwargs.get('inclination', np.degrees(self._self.inclination))

        kwargs['mesh'], _ = self._self.build_surface(return_surface=True)  # potom tu daj ked bude vediet skvrny
        denominator = (1 * kwargs['axis_unit'].to(units.DISTANCE_UNIT))
        kwargs['mesh'] /= denominator
        kwargs['equatorial_radius'] = self._self.star.equatorial_radius * units.DISTANCE_UNIT.to(kwargs['axis_unit'])
        kwargs['azimuth'] = kwargs.get('azimuth', 0)
        graphics.single_star_mesh(**kwargs)

    def wireframe(self, **kwargs):
        if 'axis_unit' not in kwargs:
            kwargs['axis_unit'] = u.solRad

        all_kwargs = ['axis_unit', 'plot_axis', 'inclination', 'azimuth']
        utils.invalid_kwarg_checker(kwargs, all_kwargs, self.wireframe)

        kwargs['plot_axis'] = kwargs.get('plot_axis', True)

        kwargs['mesh'], kwargs['triangles'] = self._self.build_surface(return_surface=True)
        denominator = (1 * kwargs['axis_unit'].to(units.DISTANCE_UNIT))
        kwargs['mesh'] /= denominator
        kwargs['equatorial_radius'] = self._self.star.equatorial_radius * units.DISTANCE_UNIT.to(kwargs['axis_unit'])
        kwargs['inclination'] = kwargs.get('inclination', np.degrees(self._self.inclination))
        kwargs['azimuth'] = kwargs.get('azimuth', 0)

        graphics.single_star_wireframe(**kwargs)

    def surface(self, **kwargs):
        if 'axis_unit' not in kwargs:
            kwargs['axis_unit'] = u.solRad

        all_kwargs = ['axis_unit', 'edges', 'normals', 'colormap', 'plot_axis', 'inclination', 'azimuth', 'units']
        utils.invalid_kwarg_checker(kwargs, all_kwargs, self.surface)

        kwargs['edges'] = kwargs.get('edges', False)
        kwargs['normals'] = kwargs.get('normals', False)
        kwargs['colormap'] = kwargs.get('colormap', None)
        kwargs['plot_axis'] = kwargs.get('plot_axis', True)
        kwargs['inclination'] = kwargs.get('inclination', np.degrees(self._self.inclination))
        kwargs['azimuth'] = kwargs.get('azimuth', 0)
        kwargs['units'] = kwargs.get('units', 'logg_cgs')

        output = self._self.build_surface(return_surface=True)
        kwargs['mesh'], kwargs['triangles'] = copy(output[0]), copy(output[1])
        denominator = (1 * kwargs['axis_unit'].to(units.DISTANCE_UNIT))
        kwargs['mesh'] /= denominator
        kwargs['equatorial_radius'] = self._self.star.equatorial_radius * units.DISTANCE_UNIT.to(kwargs['axis_unit'])

        if kwargs['colormap'] is not None:
            kwargs['cmap'] = self._self.build_surface_map(colormap=kwargs['colormap'], return_map=True)
            if kwargs['colormap'] == 'gravity_acceleration':
                kwargs['cmap'] = utils.convert_gravity_acceleration_array(kwargs['cmap'], kwargs['units'])
        if kwargs['normals']:
            kwargs['arrows'] = self._self.star.calculate_normals(points=kwargs['mesh'], faces=kwargs['triangles'],
                                                                 com=0)
            kwargs['centres'] = self._self.star.calculate_surface_centres(points=kwargs['mesh'],
                                                                          faces=kwargs['triangles'])

        graphics.single_star_surface(**kwargs)
