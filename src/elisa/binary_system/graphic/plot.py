import numpy as np

from .. import utils as butils, dynamic
from ... base import transform
from .. container import OrbitalPositionContainer
from ... const import Position
from ... utils import is_empty
from ... graphic import graphics
from ... base.surface.faces import correct_face_orientation
from .. curves import utils as crv_utils
from ... observer.observer import Observer

from ... import (
    umpy as up,
    units as u,
    utils,
    const
)
from ... import settings
from ... ld import limb_darkening_factor


class Plot(object):
    """
    Universal plot interface for binary system class, more detailed documentation for each value of descriptor is
    available in graphics library.
    
    available plot methods::
                        
        `orbit` - plots orbit in orbital plane
        `equipotential` - plots crossections of surface Hill planes in xy,yz,zx planes
        `mesh` - plot surface points
        `surface` - plot stellar surfaces
    """

    defpos = Position(*(0, 1.0, 0.0, 0.0, 0.0))

    def __init__(self, instance):
        self.binary = instance

    def orbit(self, start_phase=0.0, stop_phase=1.0, number_of_points=300,
              axis_units=u.solRad, frame_of_reference='primary', legend=True):
        """
        Function for quick 2D plot of the orbital motion in the orbital plane.

        :param start_phase: float; starting phase for the plot
        :param stop_phase: float; finishing phase for the plot
        :param number_of_points: int; number of points in the plot
        :param axis_units: Union[astropy.unit, str]; specifying axis unit, use astropy
                           units or `dimensionless` or `SMA` (semi-major axis) units for axis scale
        :param frame_of_reference: str; `barycentric` or `primary`
        :param legend: bool;
        """
        orbit_kwargs = dict()
        if axis_units in ['dimensionless', 'SMA']:
            axis_units = u.dimensionless_unscaled

        # orbit calculation for given phases
        phases = np.linspace(start_phase, stop_phase, number_of_points)
        ellipse = self.binary.orbit.orbital_motion(phase=phases)

        # if axis are without unit a = 1
        if axis_units != u.dimensionless_unscaled:
            a = self.binary.semi_major_axis * u.DISTANCE_UNIT.to(axis_units)
            radius = a * ellipse[:, 0]
        else:
            radius = ellipse[:, 0]
        azimuth = ellipse[:, 1]
        x, y = utils.polar_to_cartesian(radius=radius, phi=azimuth - const.PI / 2.0)

        if frame_of_reference == 'barycentric':
            orbit_kwargs.update({
                'x1_data': - self.binary.mass_ratio * x / (1 + self.binary.mass_ratio),
                'y1_data': - self.binary.mass_ratio * y / (1 + self.binary.mass_ratio),
                'x2_data': x / (1 + self.binary.mass_ratio),
                'y2_data': y / (1 + self.binary.mass_ratio)
            })
        elif frame_of_reference == 'primary':
            orbit_kwargs.update({
                'x_data': x,
                'y_data': y
            })
        orbit_kwargs.update({
            "axis_units": axis_units,
            "start_phase": start_phase,
            "stop_phase": stop_phase,
            "number_of_points": number_of_points,
            "frame_of_reference": frame_of_reference,
            "legend": legend
        })
        graphics.orbit(**orbit_kwargs)

    def equipotential(self, plane='xy', phase=0.0, components_to_plot='both', colors=('b', 'r'), legend=True,
                      legend_loc=1):
        """
        Function for quick 2D plot of equipotential cross-section.

        :param plane: str; (`xy`, `yz` or `xz`) specifying what plane cross-section to display default is `xy`
        :param phase: float; phase at which to plot cross-section
        :param components_to_plot: str; component to plot `primary`, `secondary` or `both` (default)
        :param colors: tuple; tuple of colors for primary and secondary component equipotentials
        :param legend: bool; legend display on/off
        :param legend_loc: int; location of the legend
        """
        equipotential_kwargs = dict()
        # relative distance between components (a = 1)
        if utils.is_plane(plane, 'xy') or utils.is_plane(plane, 'yz') or utils.is_plane(plane, 'zx'):
            components_distance = self.binary.orbit.orbital_motion(phase=phase)[0][0]
            points_primary, points_secondary = self.binary.compute_equipotential_boundary(components_distance, plane)
        else:
            raise ValueError('Invalid choice of crossection plane, use only: `xy`, `yz`, `zx`.')

        equipotential_kwargs.update({
            'plane': plane,
            'phase': phase,
            'points_primary': points_primary,
            'points_secondary': points_secondary,
            'components_to_plot': components_to_plot,
            'colors': colors,
            'legend': legend,
            "legend_loc": legend_loc
        })
        graphics.equipotential(**equipotential_kwargs)

    def mesh(self, phase=0.0, components_to_plot='both', plot_axis=True, inclination=None, azimuth=None):
        """
        Function plots 3D scatter plot of the surface points.

        :param plot_axis: bool; switch the plot axis on/off
        :param inclination: Union[float, astropy.Quantity]; elevation of the camera (in degrees if float)
        :param azimuth: Union[float, astropy.Quantity]; azimuth of the camera (in degrees if float)
        :param components_to_plot: str; component to plot `primary`, `secondary` or `both` (default)

        :param phase: float; phase at which to construct plot
        """

        binary_mesh_kwargs = dict()
        inclination = transform.deg_transform(inclination, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if inclination is not None else up.degrees(self.binary.inclination)
        components_distance, azim = self.binary.orbit.orbital_motion(phase=phase)[0][:2]

        azimuth = transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if azimuth is not None else up.degrees(azim) - 90

        orbital_position_container = OrbitalPositionContainer.from_binary_system(self.binary, self.defpos)
        orbital_position_container.build_mesh(components_distance=components_distance)
        orbital_position_container.build_pulsations_on_mesh(component=components_to_plot,
                                                            components_distance=components_distance)

        if components_to_plot in ['primary', 'both']:
            binary_mesh_kwargs.update({
                'points_primary': orbital_position_container.primary.get_flatten_parameter('points')
            })

        if components_to_plot in ['secondary', 'both']:
            binary_mesh_kwargs.update({
                'points_secondary': orbital_position_container.secondary.get_flatten_parameter('points')
            })

        binary_mesh_kwargs.update({
            "phase": phase,
            "components_to_plot": components_to_plot,
            "plot_axis": plot_axis,
            "inclination": inclination,
            "azimuth": azimuth,
        })
        graphics.binary_mesh(**binary_mesh_kwargs)

    def wireframe(self, phase=0.0, components_to_plot='both', plot_axis=True, inclination=None, azimuth=None):
        """
        Function displays wireframe model of the stellar surface.

        :param phase: float; phase at which to construct plot
        :param components_to_plot: str; component to plot `primary`, `secondary` or `both` (default)
        :param plot_axis: bool; switch the plot axis on/off
        :param inclination: Union[float, astropy.Quantity]; elevation of the camera (in degrees if float)
        :param azimuth: Union[float, astropy.Quantity]; azimuth of the camera (in degrees if float)
        """

        binary_wireframe_kwargs = dict()
        inclination = transform.deg_transform(inclination, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if inclination is not None else up.degrees(self.binary.inclination)
        components_distance, azim = self.binary.orbit.orbital_motion(phase=phase)[0][:2]
        azimuth = transform.deg_transform(azimuth, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if azimuth is not None else up.degrees(azim) - 90

        orbital_position_container = OrbitalPositionContainer.from_binary_system(self.binary, self.defpos)

        # recalculating spot latitudes
        spots_longitudes = dynamic.calculate_spot_longitudes(self.binary, phase, component="all")
        dynamic.assign_spot_longitudes(orbital_position_container, spots_longitudes, index=None, component="all")

        orbital_position_container.build_mesh(components_distance=components_distance)
        orbital_position_container.build_faces(components_distance=components_distance)

        if components_to_plot in ['primary', 'both']:
            points, faces = orbital_position_container.primary.surface_serializer()
            binary_wireframe_kwargs.update({
                "points_primary": points,
                "primary_triangles": faces
            })
        if components_to_plot in ['secondary', 'both']:
            points, faces = orbital_position_container.secondary.surface_serializer()
            binary_wireframe_kwargs.update({
                "points_secondary": points,
                "secondary_triangles": faces
            })
        binary_wireframe_kwargs.update({
            "phase": phase,
            "components_to_plot": components_to_plot,
            "plot_axis": plot_axis,
            "inclination": inclination,
            "azimuth": azimuth
        })
        graphics.binary_wireframe(**binary_wireframe_kwargs)

    def surface(self, phase=0.0, components_to_plot='both', normals=False, edges=False, colormap=None, plot_axis=True,
                face_mask_primary=None, face_mask_secondary=None, elevation=None, azimuth=None, units='SI',
                axis_unit=u.dimensionless_unscaled, colorbar_orientation='vertical', colorbar=True, scale='linear',
                surface_colors=('g', 'r'), separate_colormaps=None, colorbar_separation=0.0, colorbar_size=0.7):
        """
        Function creates plot of binary system components

        :param phase: float; phase at which plot the system, important for eccentric orbits
        :param components_to_plot: str; `primary`, `secondary` or `both` (default),
        :param normals: bool; plot normals of the surface phases as arrows
        :param edges: bool; highlight edges of surface faces
        :param colormap: str; 'gravity_acceleration`, `temperature`, `velocity`, `radial_velocity`, 'radiance',
        `normal_radiance` or None(default)
        :param plot_axis: bool; if False, axis will be hidden
        :param face_mask_primary: array[bool]; mask to select which faces to display
        :param face_mask_secondary: array[bool]: mask to select which faces to display
        :param elevation: Union[float, astropy.Quantity]; in degree - elevation of camera
        :param azimuth: Union[float, astropy.Quantity]; camera azimuth
        :param units: str; unit system of colormap  `SI` or `cgs`
        :param axis_unit: Union[astropy.unit, dimensionless]; - axis units
        :param colorbar_orientation: str; `horizontal` or `vertical` (default)
        :param colorbar: bool; colorbar on/off switch
        :param scale: str; `linear` or `log`
        :param surface_colors: tuple; tuple of colors for components if `colormap` are not specified
        :param separate_colormaps: bool; if True, figure will contain separate colormap for each component
        :param colorbar_separation: float; shifting position of the colorbar from its default postition, default is 0.0
        :param colorbar_size: float; relative size of the colorbar, default 0.7
        """
        surface_kwargs = dict()

        elevation = transform.deg_transform(elevation, u.deg, when_float64=transform.WHEN_FLOAT64) \
            if elevation is not None else 0
        
        orbital_position = self.binary.orbit.orbital_motion(phase=phase)[0]
        components_distance, azim = orbital_position[:2]
        orbital_position = Position(0, orbital_position[0], orbital_position[1],
                                    orbital_position[2], orbital_position[3])
        
        azimuth = azimuth if azimuth is not None else 180

        available_colormaps = ['gravity_acceleration', 'temperature', 'velocity', 'radial_velocity', 'radiance',
                               'normal_radiance', None]
        if separate_colormaps is None:
            separate_colormaps = self.binary.morphology != 'over-contact' and \
                                 colormap not in ['velocity', 'radial_velocity'] and components_to_plot == 'both'

        potentials = self.binary.correct_potentials([phase, ], component="all", iterations=2)
        
        orbital_position_container = OrbitalPositionContainer.from_binary_system(self.binary, self.defpos)
        # recalculating spot latitudes
        spots_longitudes = dynamic.calculate_spot_longitudes(self.binary, phase, component="all")
        dynamic.assign_spot_longitudes(orbital_position_container, spots_longitudes, index=None, component="all")
        
        orbital_position_container.set_on_position_params(orbital_position, potentials["primary"][0],
                                                          potentials["secondary"][0])
        orbital_position_container.build(components_distance=components_distance, components='both')

        orbital_position_container.flatt_it()

        # calculating radiances
        o = Observer(passband=['bolometric', ], system=self.binary)
        atm_kwargs = dict(
            passband=o.passband,
            left_bandwidth=o.left_bandwidth,
            right_bandwidth=o.right_bandwidth,
        )

        crv_utils.prep_surface_params(
            system=orbital_position_container, write_to_containers=True, **atm_kwargs
        )

        distances_to_com = orbital_position.distance * self.binary.mass_ratio / (1 + self.binary.mass_ratio)
        orbital_position_container.primary.points[:, 0] -= distances_to_com
        orbital_position_container.secondary.points[:, 0] -= distances_to_com

        orbital_position_container = butils.move_sys_onpos(orbital_position_container, orbital_position, on_copy=True)
        components = butils.component_to_list(components_to_plot)

        com = {'primary': 0.0, 'secondary': components_distance}
        mult = np.array([-1, -1, 1.0])[None, :]
        for component in components:
            star = getattr(orbital_position_container, component)

            correct_face_orientation(star, com=com[component])
            points, faces = star.points, star.faces

            surface_kwargs.update({
                f'points_{component}': mult * points,
                f'{component}_triangles': faces
            })

            if colormap is None:
                pass
            elif colormap == 'gravity_acceleration':
                log_g = getattr(star, 'log_g')
                value = log_g if units == 'SI' else log_g + 2
                surface_kwargs.update({
                    f'{component}_cmap': value if scale == 'log' else up.power(10, value)
                })

            elif colormap == 'temperature':
                temperatures = getattr(star, 'temperatures')
                surface_kwargs.update({
                    f'{component}_cmap': temperatures if scale == 'linear' else up.log10(temperatures)
                })

            elif colormap == 'velocity':
                velocities = np.linalg.norm(getattr(star, 'velocities'), axis=1)
                velocities = velocities / 1000.0 if units == 'SI' else velocities * 1000.0
                surface_kwargs.update({
                    f'{component}_cmap': velocities if scale == 'linear' else up.log10(velocities)
                })

            elif colormap == 'radial_velocity':
                velocities = getattr(star, 'velocities')[:, 0]
                velocities = velocities / 1000.0 if units == 'SI' else velocities * 1000.0
                surface_kwargs.update({
                    f'{component}_cmap': velocities
                })
                if scale == 'log':
                    raise Warning("`log` scale is not allowed for radial velocity colormap.")
            elif colormap == 'normal_radiance':
                normal_radiance = getattr(star, 'normal_radiance')['bolometric']
                surface_kwargs.update({
                    f'{component}_cmap': normal_radiance if scale == 'linear' else up.log10(normal_radiance)
                })
            elif colormap == 'radiance':
                normal_radiance = getattr(star, 'normal_radiance')['bolometric']
                los_cosines = getattr(star, 'los_cosines')
                indices = getattr(star, 'indices')
                ld_cfs = getattr(star, 'ld_cfs')['bolometric'][
                    settings.LD_LAW_CFS_COLUMNS[settings.LIMB_DARKENING_LAW]
                ].values[indices]
                ld_cors = limb_darkening_factor(coefficients=ld_cfs,
                                                limb_darkening_law=settings.LIMB_DARKENING_LAW,
                                                cos_theta=los_cosines[indices])
                retval = np.zeros(normal_radiance.shape)
                retval[indices] = normal_radiance[indices] * los_cosines[indices] * ld_cors
                surface_kwargs.update({
                    f'{component}_cmap': retval
                })
                if scale == 'log':
                    raise Warning("`log` scale is not allowed for radiance colormap.")
            else:
                raise KeyError(f'Unknown `colormap` argument {colormap}. Options: {available_colormaps}')

            face_mask = locals().get(f'face_mask_{component}')
            if not is_empty(face_mask):
                surface_kwargs[f'{component}_triangles'] = surface_kwargs[f'{component}_triangles'][face_mask]
                # fixme: this is directly related to f'{component}_cmap' but it is not enforced to be set before
                surface_kwargs[f'{component}_cmap'] = surface_kwargs[f'{component}_cmap'][face_mask]

            if normals:
                surface_kwargs.update({
                    f'{component}_centres': mult * star.face_centres,
                    f'{component}_arrows': mult * star.normals
                })

            if axis_unit != u.dimensionless_unscaled:
                sma = (self.binary.semi_major_axis * u.DISTANCE_UNIT).to(axis_unit).value
                surface_kwargs[f'points_{component}'] *= sma

                if normals:
                    surface_kwargs[f'{component}_centres'] *= sma
                    surface_kwargs[f'{component}_arrows'] *= sma

        surface_kwargs.update({
            "phase": phase,
            "components_to_plot": components_to_plot,
            "normals": normals,
            "edges": edges,
            "colormap": colormap,
            "plot_axis": plot_axis,
            "face_mask_primary": face_mask_primary,
            "face_mask_secondary": face_mask_secondary,
            "elevation": elevation,
            "azimuth": azimuth,
            "units": units,
            "axis_unit": axis_unit,
            "colorbar_orientation": colorbar_orientation,
            "colorbar": colorbar,
            "scale": scale,
            "morphology": self.binary.morphology,
            "surface_colors": surface_colors,
            "separate_colormaps": separate_colormaps,
            'colorbar_separation': colorbar_separation,
            'colorbar_size': colorbar_size
        })

        graphics.binary_surface(**surface_kwargs)
