import numpy as np

from .. container import SystemContainer
from ... import units as u
from ... base import transform
from ... const import SinglePosition
from ... graphic import graphics
from ... utils import is_empty
from ... base.surface.faces import correct_face_orientation
from .. import utils as sutils


class Plot(object):
    """
    Universal plot interface for binary system class, more detailed documentation for each value of descriptor is
    available in graphics library::

        `orbit` - plots orbit in orbital plane
        `equipotential` - plots crossection of surface Hill planes in xz plane
        `mesh` - plot surface points
        `surface` - plot stellar surfaces
    """

    defpos = SinglePosition(*(0, 0.0, 0.0))

    def __init__(self, instance):
        self.single = instance

    def equipotential(self, axis_unit=u.solRad):
        """
        Function for quick 2D plot of equipotential cross-section in xz plane.

        :param axis_unit: Union[astropy.unit, dimensionless]; - axis units
        """
        equipotential_kwargs = dict()

        points = self.single.calculate_equipotential_boundary()
        points = (points * u.DISTANCE_UNIT).to(axis_unit)

        equipotential_kwargs.update({
            'points': points,
            'axis_unit': axis_unit,
        })
        graphics.equipotential_single_star(**equipotential_kwargs)

    def mesh(self, phase=0.0, plot_axis=True, axis_unit=u.solRad, inclination=None, azimuth=None):
        """
        Function plots 3D scatter plot of the surface points.

        :param phase: float;
        :param plot_axis: bool; switch the plot axis on/off
        :param axis_unit: Union[astropy.unit, dimensionless]; - axis units
        :param inclination: Union[float, astropy.Quantity]; in degree - elevation of camera
        :param azimuth: Union[float, astropy.Quantity]; camera azimuth
        """
        single_mesh_kwargs = dict()

        inclination = transform.deg_transform(inclination, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if inclination is not None else np.degrees(self.single.inclination)
        azim = self.single.orbit.rotational_motion(phase=phase)[0][0]
        azimuth = transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if azimuth is not None else np.degrees(azim) - 90

        position_container = SystemContainer.from_single_system(self.single, self.defpos)
        position_container.build_mesh()
        position_container.build_pulsations_on_mesh()

        mesh = position_container.star.get_flatten_parameter('points')
        denominator = (1 * axis_unit.to(u.DISTANCE_UNIT))
        mesh /= denominator
        equatorial_radius = position_container.star.equatorial_radius * u.DISTANCE_UNIT.to(axis_unit)

        single_mesh_kwargs.update({
            'phase': phase,
            'axis_unit': axis_unit,
            'plot_axis': plot_axis,
            "inclination": inclination,
            "azimuth": azimuth,
            "mesh": mesh,
            'equatorial_radius': equatorial_radius,
        })

        graphics.single_star_mesh(**single_mesh_kwargs)

    def wireframe(self, phase=0.0, plot_axis=True, axis_unit=u.solRad, inclination=None, azimuth=None):
        """
        Returns 3D wireframe of the object.

        :param phase: float;
        :param plot_axis: bool; switch the plot axis on/off
        :param axis_unit: Union[astropy.unit, dimensionless]; - axis units
        :param inclination: Union[float, astropy.Quantity]; in degree - elevation of camera
        :param azimuth: Union[float, astropy.Quantity]; camera azimuth
        """
        wireframe_kwargs = dict()

        inclination = transform.deg_transform(inclination, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if inclination is not None else np.degrees(self.single.inclination)
        azim = self.single.orbit.rotational_motion(phase=phase)[0][0]
        azimuth = transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if azimuth is not None else np.degrees(azim) - 90

        position_container = SystemContainer.from_single_system(self.single, self.defpos)
        position_container.build_mesh()
        position_container.build_faces()

        points, faces = position_container.star.surface_serializer()
        denominator = (1 * axis_unit.to(u.DISTANCE_UNIT))
        points /= denominator
        equatorial_radius = position_container.star.equatorial_radius * u.DISTANCE_UNIT.to(axis_unit)

        wireframe_kwargs.update({
            'phase': phase,
            'axis_unit': axis_unit,
            'plot_axis': plot_axis,
            "inclination": inclination,
            "azimuth": azimuth,
            "mesh": points,
            "triangles": faces,
            'equatorial_radius': equatorial_radius,
        })

        graphics.single_star_wireframe(**wireframe_kwargs)

    def surface(self, phase=0.0, normals=False, edges=False, colormap=None, plot_axis=True, face_mask=None,
                elevation=None, azimuth=None, units='SI', axis_unit=u.solRad,
                colorbar_orientation='vertical', colorbar=True, scale='linear', surface_color='g'):
        """
        Function creates plot of single system components.

        :param phase: float; phase at which plot the system, important for eccentric orbits
        :param normals: bool; plot normals of the surface phases as arrows
        :param edges: bool; highlight edges of surface faces
        :param colormap: str; 'gravity_acceleration`, `temperature`, `velocity`, `radial_velocity` or None(default)
        :param plot_axis: bool; if False, axis will be hidden
        :param face_mask: array[bool]; mask to select which faces to display
        :param elevation: Union[float, astropy.Quantity]; in degree - elevation of camera
        :param azimuth: Union[float, astropy.Quantity]; camera azimuth
        :param units: str; unit system of colormap  `SI` or `cgs`
        :param axis_unit: Union[astropy.unit, dimensionless]; - axis units
        :param colorbar_orientation: `horizontal` or `vertical` (default)
        :param colorbar: bool; colorbar on/off switch
        :param scale: str; `linear` or `log`
        :param surface_color: tuple; tuple of colors for components if `colormap` is not specified
        """
        surface_kwargs = dict()

        elevation = transform.deg_transform(elevation, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if elevation is not None else 0
        azimuth = transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if azimuth is not None else 180

        single_position = self.single.orbit.rotational_motion(phase=phase)[0]
        single_position = SinglePosition(0, single_position[0], single_position[1])

        position_container = SystemContainer.from_single_system(self.single, self.defpos)
        position_container.set_on_position_params(single_position)
        position_container.build(phase=phase)

        position_container = sutils.move_sys_onpos(position_container, single_position)

        mult = np.array([-1, -1, 1.0])[None, :]
        star_container = getattr(position_container, 'star')

        correct_face_orientation(star_container, com=0)
        points, faces = star_container.points, star_container.faces

        surface_kwargs.update({
            'points': mult * points,
            'triangles': faces
        })

        if colormap == 'gravity_acceleration':
            log_g = getattr(star_container, 'log_g')
            value = log_g if units == 'SI' else log_g + 2
            surface_kwargs.update({
                'cmap': value if scale == 'log' else np.power(10, value)
            })

        elif colormap == 'temperature':
            temperatures = getattr(star_container, 'temperatures')
            surface_kwargs.update({
                'cmap': temperatures if scale == 'linear' else np.log10(temperatures)
            })

        elif colormap == 'velocity':
            velocities = np.linalg.norm(getattr(star_container, 'velocities'), axis=1)
            velocities = velocities / 1000.0 if units == 'SI' else velocities * 1000.0
            surface_kwargs.update({
                f'cmap': velocities if scale == 'linear' else np.log10(velocities)
            })

        elif colormap == 'radial_velocity':
            velocities = getattr(star_container, 'velocities')[:, 0]
            velocities = velocities / 1000.0 if units == 'SI' else velocities * 1000.0
            surface_kwargs.update({
                f'cmap': velocities
            })
            if scale == 'log':
                raise Warning("`log` scale is not allowed for radial velocity colormap.")

        if not is_empty(face_mask):
            surface_kwargs['triangles'] = surface_kwargs['triangles'][face_mask]
            if 'cmap' in surface_kwargs.keys():
                surface_kwargs['cmap'] = surface_kwargs['cmap'][face_mask]

        if normals:
            face_centres = star_container.get_flatten_parameter('face_centres')
            norm = star_container.get_flatten_parameter('normals')
            surface_kwargs.update({
                'centres': mult * face_centres,
                'arrows': mult * norm
            })

        # normals
        unit_mult = (1*u.DISTANCE_UNIT).to(axis_unit).value
        surface_kwargs['points'] *= unit_mult

        if normals:
            surface_kwargs['centres'] *= unit_mult

        surface_kwargs.update({
            'phase': phase,
            'normals': normals,
            'edges': edges,
            'colormap': colormap,
            'plot_axis': plot_axis,
            'face_mask': face_mask,
            "elevation": elevation,
            "azimuth": azimuth,
            'units': units,
            'axis_unit': axis_unit,
            'colorbar_orientation': colorbar_orientation,
            'colorbar': colorbar,
            'scale': scale,
            'equatorial_radius': (star_container.equatorial_radius*u.DISTANCE_UNIT).to(axis_unit).value,
            'surface_color': surface_color
        })
        graphics.single_star_surface(**surface_kwargs)
