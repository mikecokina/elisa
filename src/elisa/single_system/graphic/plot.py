from copy import copy
import numpy as np

from elisa.base.container import StarContainer
from elisa.base import transform
from elisa.single_system.container import SystemContainer
from elisa import (
    utils,
    graphics,
    units as eu,
)


class Plot(object):
    """
    universal plot interface for binary system class, more detailed documentation for each value of descriptor is
    available in graphics library

                        `orbit` - plots orbit in orbital plane
                        `equipotential` - plots crossection of surface Hill planes in xz plane
                        `mesh` - plot surface points
                        `surface` - plot stellar surfaces
    :return:
    """

    def __init__(self, instance):
        self.single = instance

    def equipotential(self, axis_unit=eu.solRad):
        """
        Function for quick 2D plot of equipotential cross-section in xz plane
        :param axis_unit: Union[astropy.unit, dimensionless]; - axis units
        :return:
        """
        equipotential_kwargs = dict()

        points = self.single.calculate_equipotential_boundary()
        points = (points * eu.DISTANCE_UNIT).to(axis_unit)

        equipotential_kwargs.update({
            'points': points,
            'axis_unit': axis_unit,
        })
        graphics.equipotential_single_star(**equipotential_kwargs)

    def mesh(self, phase=0.0, plot_axis=True, axis_unit=eu.dimensionless_unscaled, inclination=None, azimuth=None):
        """
        Function plots 3D scatter plot of the surface points

        :param phase: float;
        :param plot_axis: bool; switch the plot axis on/off
        :param axis_unit: Union[astropy.unit, dimensionless]; - axis units
        :param inclination: Union[float, astropy.Quantity]; in degree - elevation of camera
        :param azimuth: Union[float, astropy.Quantity]; camera azimuth
        :return:
        """
        single_mesh_kwargs = dict()

        inclination = transform.deg_transform(inclination, eu.deg, when_float64=transform.WHEN_FLOAT64) \
            if inclination is not None else np.degrees(self.single.inclination)
        azim = self.single.orbit.orbital_motion(phase=phase)[0][0]
        azimuth = transform.deg_transform(azimuth, eu.deg, when_float64=transform.WHEN_FLOAT64) \
            if azimuth is not None else np.degrees(azim) - 90

        position_container = SystemContainer(
            star=StarContainer.from_properties_container(self.single.star.to_properties_container()),
            **self.single.properties_serializer()
        )

        # kwargs['mesh'], _ = self._self.build_surface(return_surface=True)  # potom tu daj ked bude vediet skvrny
        # denominator = (1 * kwargs['axis_unit'].to(eu.DISTANCE_UNIT))
        # kwargs['mesh'] /= denominator
        # kwargs['equatorial_radius'] = self._self.star.equatorial_radius * eu.DISTANCE_UNIT.to(kwargs['axis_unit'])

        single_mesh_kwargs.update({
            'phase': phase,
            'axis_unit': axis_unit,
            'plot_axis': plot_axis,
            "inclination": inclination,
            "azimuth": azimuth
        })

        graphics.single_star_mesh(**kwargs)

    def wireframe(self, **kwargs):
        if 'axis_unit' not in kwargs:
            kwargs['axis_unit'] = units.solRad

        all_kwargs = ['axis_unit', 'plot_axis', 'inclination', 'azimuth']
        utils.invalid_kwarg_checker(kwargs, all_kwargs, self.wireframe)

        kwargs['plot_axis'] = kwargs.get('plot_axis', True)

        kwargs['mesh'], kwargs['triangles'] = self._self.build_surface(return_surface=True)
        denominator = (1 * kwargs['axis_unit'].to(units.DISTANCE_UNIT))
        kwargs['mesh'] /= denominator
        kwargs['equatorial_radius'] = self._self.star.equatorial_radius * units.DISTANCE_UNIT.to(kwargs['axis_unit'])
        kwargs['inclination'] = np.degrees(kwargs.get('inclination', self._self.inclination))
        kwargs['azimuth'] = kwargs.get('azimuth', 0)

        graphics.single_star_wireframe(**kwargs)

    def surface(self, **kwargs):
        if 'axis_unit' not in kwargs:
            kwargs['axis_unit'] = units.solRad

        all_kwargs = ['axis_unit', 'edges', 'normals', 'colormap', 'plot_axis', 'inclination', 'azimuth', 'units']
        utils.invalid_kwarg_checker(kwargs, all_kwargs, self.surface)

        kwargs['edges'] = kwargs.get('edges', False)
        kwargs['normals'] = kwargs.get('normals', False)
        kwargs['colormap'] = kwargs.get('colormap', None)
        kwargs['plot_axis'] = kwargs.get('plot_axis', True)
        kwargs['inclination'] = np.degrees(kwargs.get('inclination', self._self.inclination))
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
