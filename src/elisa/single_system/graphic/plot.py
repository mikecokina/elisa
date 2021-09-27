import numpy as np

from .. container import SinglePositionContainer
from ... import units as u
from ... base import transform
from ... base.graphics import plot
from ... const import Position
from ... graphic import graphics
from ... base.surface.faces import correct_face_orientation
from .. import utils as sutils
from .. curves import utils as crv_utils
from ... observer.observer import Observer


class Plot(object):
    """
    Universal plot interface for binary system class, more detailed documentation for each value of descriptor is
    available in graphics library::

        `equipotential` - plots crossection of surface Hill planes in xz plane
        `wireframe` - wire frame model of the star
        `mesh` - plot surface points
        `surface` - plot stellar surfaces
    """

    defpos = Position(*(0, np.nan, 0.0, np.nan, 0.0))

    def __init__(self, instance):
        self.single = instance

    def equipotential(self, axis_unit=u.solRad, return_figure_instance=False):
        """
        Function for quick 2D plot of equipotential cross-section in xz plane.

        :param axis_unit: Union[astropy.unit, dimensionless]; - axis units
        :param return_figure_instance: bool; if True, the Figure instance is returned instead of displaying the
                                             produced figure
        """
        equipotential_kwargs = dict()

        points = self.single.calculate_equipotential_boundary()
        points = (points * u.DISTANCE_UNIT).to(axis_unit)

        equipotential_kwargs.update({
            'return_figure_instance': return_figure_instance,
            'points': points,
            'axis_unit': axis_unit,
        })
        return graphics.equipotential_single_star(**equipotential_kwargs)

    def mesh(self, phase=0.0, plot_axis=True, axis_unit=u.solRad, inclination=None, azimuth=None,
             return_figure_instance=False):
        """
        Function plots 3D scatter plot of the surface points.

        :param phase: float;
        :param plot_axis: bool; switch the plot axis on/off
        :param axis_unit: Union[astropy.unit, dimensionless]; - axis units
        :param inclination: Union[float, astropy.Quantity]; in degree - elevation of camera
        :param azimuth: Union[float, astropy.Quantity]; camera azimuth
        :param return_figure_instance: bool; if True, the Figure instance is returned instead of displaying the
                                             produced figure
        """
        single_mesh_kwargs = dict()

        inclination = transform.deg_transform(inclination, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if inclination is not None else np.degrees(self.single.inclination)
        azim = self.single.orbit.rotational_motion(phase=phase)[0][0]
        azimuth = transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if azimuth is not None else np.degrees(azim) - 90

        position_container = SinglePositionContainer.from_single_system(self.single, self.defpos)
        position_container.build_mesh()
        position_container.build_perturbations()

        mesh = position_container.star.get_flatten_parameter('points')
        denominator = (1 * axis_unit.to(u.DISTANCE_UNIT))
        mesh /= denominator
        equatorial_radius = position_container.star.equatorial_radius * u.DISTANCE_UNIT.to(axis_unit)

        single_mesh_kwargs.update({
            'return_figure_instance': return_figure_instance,
            'phase': phase,
            'axis_unit': axis_unit,
            'plot_axis': plot_axis,
            "inclination": inclination,
            "azimuth": azimuth,
            "mesh": mesh,
            'equatorial_radius': equatorial_radius,
        })

        return graphics.single_star_mesh(**single_mesh_kwargs)

    def wireframe(self, phase=0.0, plot_axis=True, axis_unit=u.solRad, inclination=None, azimuth=None,
                  return_figure_instance=False):
        """
        Returns 3D wireframe of the object.

        :param phase: float;
        :param plot_axis: bool; switch the plot axis on/off
        :param axis_unit: Union[astropy.unit, dimensionless]; - axis units
        :param inclination: Union[float, astropy.Quantity]; in degree - elevation of camera
        :param azimuth: Union[float, astropy.Quantity]; camera azimuth
        :param return_figure_instance: bool; if True, the Figure instance is returned instead of displaying the
                                             produced figure
        """
        wireframe_kwargs = dict()

        inclination = transform.deg_transform(inclination, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if inclination is not None else np.degrees(self.single.inclination)
        azim = self.single.orbit.rotational_motion(phase=phase)[0][0]
        azimuth = transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if azimuth is not None else np.degrees(azim) - 90

        position_container = SinglePositionContainer.from_single_system(self.single, self.defpos)
        position_container.build_mesh()
        position_container.build_faces()

        points, faces = position_container.star.surface_serializer()
        denominator = (1 * axis_unit.to(u.DISTANCE_UNIT))
        points /= denominator
        equatorial_radius = position_container.star.equatorial_radius * u.DISTANCE_UNIT.to(axis_unit)

        wireframe_kwargs.update({
            'return_figure_instance': return_figure_instance,
            'phase': phase,
            'axis_unit': axis_unit,
            'plot_axis': plot_axis,
            "inclination": inclination,
            "azimuth": azimuth,
            "mesh": points,
            "triangles": faces,
            'equatorial_radius': equatorial_radius,
        })

        return graphics.single_star_wireframe(**wireframe_kwargs)

    def surface(self, phase=0.0, normals=False, edges=False, colormap=None, plot_axis=True, face_mask=None,
                elevation=None, azimuth=None, colorbar_unit='default', axis_unit=u.solRad,
                colorbar_orientation='vertical', colorbar=True, scale='linear', surface_color='g',
                colorbar_separation=0.0, colorbar_size=0.7, return_figure_instance: bool=False,
                subtract_equilibrium: bool=False):
        """
        Function creates plot of single system components.

        :param phase: float; phase at which plot the system, important for eccentric orbits
        :param normals: bool; plot normals of the surface phases as arrows
        :param edges: bool; highlight edges of surface faces
        :param colormap: str;
        :param plot_axis: bool; if False, axis will be hidden
        :param face_mask: array[bool]; mask to select which faces to display
        :param elevation: Union[float, astropy.Quantity]; in degree - elevation of camera
        :param azimuth: Union[float, astropy.Quantity]; camera azimuth
        :param colorbar_unit: str; colormap unit
        :param axis_unit: Union[astropy.unit, dimensionless]; - axis units
        :param colorbar_orientation: `horizontal` or `vertical` (default)
        :param colorbar: bool; colorbar on/off switch
        :param scale: str; `linear` or `log`
        :param surface_color: Tuple; tuple of colors for components if `colormap` is not specified
        :param colorbar_separation: float; shifting position of the colorbar from its default postition, default is 0.0
        :param colorbar_size: float; relative size of the colorbar, default 0.7
        :param return_figure_instance: bool; if True, the Figure instance is returned instead of displaying the
                                             produced figure
        :param subtract_equilibrium: bool; if True, equilibrium values are subtracted from the colormap
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
        surface_kwargs = dict()

        elevation = transform.deg_transform(elevation, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if elevation is not None else 0
        azimuth = transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if azimuth is not None else 180

        single_position = self.single.orbit.rotational_motion(phase=phase)[0]
        single_position = Position(0, np.nan, single_position[0], single_position[1], single_position[2])

        position_container = SinglePositionContainer.from_single_system(self.single, self.defpos)
        position_container.set_on_position_params(single_position)
        position_container.set_time()
        position_container.build(phase=phase, build_pulsations=True)
        correct_face_orientation(position_container.star, com=0)

        # calculating radiances
        o = Observer(passband=['bolometric', ], system=self.single)
        atm_kwargs = dict(
            passband=o.passband,
            left_bandwidth=o.left_bandwidth,
            right_bandwidth=o.right_bandwidth,
        )

        crv_utils.prep_surface_params(
            system=position_container, write_to_containers=True, **atm_kwargs
        )

        position_container = sutils.move_sys_onpos(position_container, single_position)

        star_container = getattr(position_container, 'star')

        args = (colormap, star_container, phase, 0.0, 1.0, self.single.inclination, position_container.position)
        kwargs = dict(scale=scale, unit=colorbar_unit, subtract_equilibrium=subtract_equilibrium)
        surface_kwargs.update({'cmap': plot.add_colormap_to_plt_kwargs(*args, **kwargs)})

        surface_kwargs.update({
            'points': star_container.points,
            'triangles': star_container.faces
        })

        face_mask = np.ones(star_container.faces.shape[0], dtype=bool) if face_mask is None else face_mask
        surface_kwargs['triangles'] = surface_kwargs['triangles'][face_mask]
        if 'colormap' in surface_kwargs.keys():
            surface_kwargs['cmap'] = surface_kwargs['cmap'][face_mask]

        if normals:
            face_centres = star_container.get_flatten_parameter('face_centres')
            norm = star_container.get_flatten_parameter('normals')
            surface_kwargs.update({
                'centres': face_centres[face_mask],
                'arrows': norm[face_mask]
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
            'unit': colorbar_unit,
            'axis_unit': axis_unit,
            'colorbar_orientation': colorbar_orientation,
            'colorbar': colorbar,
            'scale': scale,
            'equatorial_radius': (star_container.equatorial_radius*u.DISTANCE_UNIT).to(axis_unit).value,
            'surface_color': surface_color,
            'colorbar_separation': colorbar_separation,
            'colorbar_size': colorbar_size,
            'return_figure_instance': return_figure_instance
        })
        return graphics.single_star_surface(**surface_kwargs)
