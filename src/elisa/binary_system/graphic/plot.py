import numpy as np

from astropy import units as au
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.const import BINARY_POSITION_PLACEHOLDER
from elisa.binary_system import utils as butils
from elisa.utils import is_empty

from elisa import (
    umpy as up,
    utils,
    const,
    graphics,
    units as eunits
)


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

    defpos = BINARY_POSITION_PLACEHOLDER(*(0, 1.0, 0.0, 0.0, 0.0))

    def __init__(self, instance):
        self.binary = instance

    def orbit(self, start_phase=0.0, stop_phase=1.0, number_of_points=300,
              axis_units=eunits.solRad, frame_of_reference='primary'):
        """
        Function for quick 2D plot of the orbital motion in the orbital plane.

        :param start_phase: float; starting phase for the plot
        :param stop_phase: float; finishing phase for the plot
        :param number_of_points: int; number of points in the plot
        :param axis_units: Union[astropy.unit, str]; specifying axis unit, use astropy
        units or `dimensionless` or `SMA` (semi-major axis) units for axis scale
        :param frame_of_reference: str; `barycentric` or `primary`
        """
        orbit_kwargs = dict()
        if axis_units == 'dimensionless' or 'SMA':
            axis_units = eunits.dimensionless_unscaled

        # orbit calculation for given phases
        phases = np.linspace(start_phase, stop_phase, number_of_points)
        ellipse = self.binary.orbit.orbital_motion(phase=phases)

        # if axis are without unit a = 1
        if axis_units != au.dimensionless_unscaled:
            a = self.binary.semi_major_axis * eunits.DISTANCE_UNIT.to(axis_units)
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
            "frame_of_reference": frame_of_reference
        })
        graphics.orbit(**orbit_kwargs)

    def equipotential(self, plane='xy', phase=0.0):
        """
        Function for quick 2D plot of equipotential cross-section.

        :param plane: str; (`xy`, `yz` or `xz`) specifying what plane cross-section to display default is `xy`
        :param phase: float; phase at which to plot cross-section
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
            'points_secondary': points_secondary
        })
        graphics.equipotential(**equipotential_kwargs)

    def mesh(self, phase=0.0, components_to_plot='both', plot_axis=True, inclination=None, azimuth=None):
        """
        Function plots 3D scatter plot of the surface points.

        :param phase: float; phase at which to construct plot
        :param components_to_plot: str; component to plot `primary`, `secondary` or `both`(default)
        :param plot_axis: bool; switch the plot axis on/off
        :param inclination: float; elevation of the camera (in degrees)
        :param azimuth: float; azimuth of the camera (in degrees)
        """

        binary_mesh_kwargs = dict()
        inclination = up.degrees(self.binary.inclination) if is_empty(inclination) else inclination
        components_distance, azim = self.binary.orbit.orbital_motion(phase=phase)[0][:2]
        azimuth = up.degrees(azim) - 90.0 if is_empty(azimuth) else azimuth
        orbital_position_container = OrbitalPositionContainer.from_binary_system(self.binary, self.defpos)
        orbital_position_container.build_mesh(components_distance=components_distance)

        if components_to_plot in ['primary', 'both']:
            binary_mesh_kwargs.update({
                'points_primary': orbital_position_container.primary.flatt_it().points
            })

        if components_to_plot in ['secondary', 'both']:
            binary_mesh_kwargs.update({
                'points_secondary': orbital_position_container.secondary.flatt_it().points
            })

        binary_mesh_kwargs.update({
            "phase": phase,
            "components_to_plot": components_to_plot,
            "plot_axis": plot_axis,
            "inclination": inclination,
            "azimuth": azimuth
        })
        graphics.binary_mesh(**binary_mesh_kwargs)

    def wireframe(self, phase=0.0, components_to_plot='both', plot_axis=True, inclination=None, azimuth=None):
        """
        Function displays wireframe model of the stellar surface.

        :param phase: float; phase at which to construct plot
        :param components_to_plot: str; component to plot `primary`, `secondary` or `both`(default)
        :param plot_axis: bool; switch the plot axis on/off
        :param inclination: float; elevation of the camera (in degrees)
        :param azimuth: float; azimuth of the camera (in degrees)
        """

        binary_wireframe_kwargs = dict()
        inclination = up.degrees(self.binary.inclination) if is_empty(inclination) else inclination
        components_distance, azim = self.binary.orbit.orbital_motion(phase=phase)[0][:2]
        azimuth = up.degrees(azim) - 90.0 if is_empty(azimuth) else azimuth
        orbital_position_container = OrbitalPositionContainer.from_binary_system(self.binary, self.defpos)
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
                face_mask_primary=None, face_mask_secondary=None, inclination=None, azimuth=None, units='cgs',
                axis_unit=eunits.dimensionless_unscaled, colorbar_orientation='vertical', colorbar=True, scale='linear'):
        """
        function creates plot of binary system components

        :param phase: float -- phase at which plot the system, important for eccentric orbits
        :param components_to_plot: str; `primary`, `secondary` or `both`(default),
        :param normals: bool; plot normals of the surface phases as arrows
        :param edges: bool; highlight edges of surface faces
        :param colormap: str; 'gravity_acceleration`, `temperature` or None(default)
        :param plot_axis: bool; if False, axis will be hidden
        :param face_mask_primary: array[bool]; mask to select which faces to display
        :param face_mask_secondary: array[bool]: mask to select which faces to display
        :param inclination: float; in degree - elevation of camera
        :param azimuth: float; camera azimuth
        :param units: str; units of gravity acceleration colormap  `SI` or `cgs`
        :param axis_unit: Union[astropy.unit, dimensionless]; - axis units
        :param colorbar_orientation: str; `horizontal` or `vertical`(default)
        :param colorbar: bool; colorabar on/off switchic
        :param scale: str; `linear` or `log`
        """
        surface_kwargs = dict()

        inclination = up.degrees(self.binary.inclination) if is_empty(inclination) else inclination
        components_distance, azim = self.binary.orbit.orbital_motion(phase=phase)[0][:2]
        azimuth = up.degrees(azim) - 90.0 if is_empty(azimuth) else azimuth

        # recalculating spot latitudes
        # TODO: implement latitude recalculation once the functions will be relocated
        # spots_longitudes = geo.calculate_spot_longitudes(self.binary, kwargs['phase'], component="all")
        # geo.assign_spot_longitudes(self.binary, spots_longitudes, index=None, component="all")

        orbital_position_container = OrbitalPositionContainer.from_binary_system(self.binary, self.defpos)

        orbital_position_container.build(components_distance=components_distance, components=components_to_plot)
        # this part decides if both components need to be calculated at once (due to reflection effect)
        components = butils.component_to_list(components_to_plot)

        for component in components:
            star = getattr(orbital_position_container, component)
            points, faces = star.surface_serializer()
            surface_kwargs.update({
                f'points_{component}': points,
                f'{component}_triangles': faces
            })

            if colormap == 'gravity_acceleration':
                log_g = star.flatten_parameter('log_g')
                value = log_g if 'units' == 'SI' else log_g + 2
                surface_kwargs.update({
                    f'{component}_cmap': value if scale == 'log' else up.power(10, value)
                })

            elif colormap == 'temperature':
                temperatures = star.flatten_parameter('temperatures')
                surface_kwargs.update({
                    f'{component}_cmap': temperatures if scale == 'linear' else up.log10(temperatures)
                })

            face_mask = locals().get(f'face_mask_{component}')
            if not is_empty(face_mask):
                surface_kwargs[f'{component}_triangles'] = surface_kwargs[f'{component}_triangles'][face_mask]
                # fixme: this is directly related to f'{component}_cmap' but it is not enforced to be set before
                surface_kwargs[f'{component}_cmap'] = surface_kwargs[f'{component}_cmap'][face_mask]

            if normals:
                surface_kwargs.update({
                    f'{component}_centres': star.face_centres,
                    f'{component}_arrows': star.normals
                })

            if axis_unit != au.dimensionless_unscaled and axis_unit != 'SMA':
                sma = (self.binary.semi_major_axis * eunits.DISTANCE_UNIT).to(axis_unit).value
                surface_kwargs[f'points_{component}'] *= sma

                if surface_kwargs['normals']:
                    surface_kwargs[f'{component}_centres'] *= sma

        surface_kwargs.update({
            "phase": phase,
            "components_to_plot": components_to_plot,
            "normals": normals,
            "edges": edges,
            "colormap": colormap,
            "plot_axis": plot_axis,
            "face_mask_primary": face_mask_primary,
            "face_mask_secondary": face_mask_secondary,
            "inclination": inclination,
            "azimuth": azimuth,
            "units": units,
            "axis_unit": axis_unit,
            "colorbar_orientation": colorbar_orientation,
            "colorbar": colorbar,
            "scale": scale,
            "morphology": self.binary.morphology
        })

        graphics.binary_surface(**surface_kwargs)
