import numpy as np
import mpl_toolkits.mplot3d.axes3d as axes3d

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

from matplotlib import cm
from .. import (
    umpy as up,
    utils,
    units
)
from .. import units as u
from . import utils as gutils


CMAPS = {'temperature': cm.jet_r,
         'velocity': cm.jet,
         'radial_velocity': cm.jet,
         'v_r_perturbed': cm.jet,
         'v_horizontal_perturbed': cm.jet,
         'gravity_acceleration': cm.jet,
         'horizontal_acceleration': cm.jet,
         'radius': cm.jet,
         'normal_radiance': cm.hot,
         'radiance': cm.hot,
         'horizontal_displacement': cm.jet,
         }


def orbit(**kwargs):
    """
    Graphics part of the function for quick 2D plot of the orbital motion in the orbital plane.

    :param kwargs: Dict;
    :**kwargs options**:
        * **start_phase** * -- float; starting phase for the plot
        * **stop_phase** * -- float; finishing phase for the plot
        * **number_of_points** * -- int; number of points in the plot
        * **axis_units** * -- Union[astropy.unit, 'str']; specifying axis unit, use astropy units or `dimensionless`
                              or `SMA` (semi-major axis) units for axis scale
        * **frame_of_reference** * -- str; `barycentric` or `primary`
    """
    unit = str(kwargs['axis_units'])
    if kwargs['axis_units'] == units.dimensionless_unscaled:
        x_label, y_label = 'x/[SMA]', 'y/[SMA]'
    else:
        x_label, y_label = r'x/' + unit, r'y/' + unit

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.grid()
    if kwargs['frame_of_reference'] == 'barycentric':
        x1, y1 = kwargs['x1_data'], kwargs['y1_data']
        x2, y2 = kwargs['x2_data'], kwargs['y2_data']
        ax.plot(x1, y1, label='primary')
        ax.plot(x2, y2, label='secondary')
        ax.scatter([0], [0], c='black', s=4)
    elif kwargs['frame_of_reference'] == 'primary':
        x, y = kwargs['x_data'], kwargs['y_data']
        ax.plot(x, y, label='secondary')
        # ax.scatter(x[0], y[0], c='r')
        ax.scatter([0], [0], c='b', label='primary')

    if kwargs['legend']:
        ax.legend(loc=1)
    ax.set_aspect('equal')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.subplots_adjust(top=0.98, right=0.98)
    return f if kwargs['return_figure_instance'] else plt.show()


def equipotential(**kwargs):
    """
    Plot function for descriptor = `equipotential` in function BinarySystem.plot(). This function plots crossections of
    surface Hill planes in xy, yz or zx plane

    :param kwargs: Dict;
    :**kwargs options**:
        * **plane** * -- str; 'xy' - plane in which surface Hill plane is calculated, planes: 'xy', 'yz', 'zx'
        * **phase** * float; photometric phase in which surface Hill plane is calculated
    """
    x_label, y_label = 'x', 'y'
    if utils.is_plane(kwargs['plane'], 'yz'):
        x_label, y_label = 'y', 'z'
    elif utils.is_plane(kwargs['plane'], 'zx'):
        x_label, y_label = 'x', 'z'

    x_primary, y_primary = kwargs['points_primary'][:, 0], kwargs['points_primary'][:, 1]
    x_secondary, y_secondary = kwargs['points_secondary'][:, 0], kwargs['points_secondary'][:, 1]

    f = plt.figure()
    ax = f.add_subplot(111)

    if kwargs['components_to_plot'] in ['primary', 'both']:
        ax.plot(x_primary, y_primary, label='primary', c=kwargs['colors'][0])
    if kwargs['components_to_plot'] in ['secondary', 'both']:
        ax.plot(x_secondary, y_secondary, label='secondary', c=kwargs['colors'][1])

    lims = ax.get_xlim() - np.mean(ax.get_xlim())
    ax.set_ylim(lims)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if kwargs['legend']:
        ax.legend(loc=kwargs['legend_loc'])
    ax.grid()

    return f if kwargs['return_figure_instance'] else plt.show()


def equipotential_single_star(**kwargs):
    """
    Plot function for descriptor = `equipotential` in function SingleSystem.plot(). Calculates zx plane crossection of
    equipotential surface.

    :param kwargs: Dict;
    :**kwargs options**:
        * **axis_unit** * -- astropy.units.solRad - unit in which axis will be displayed, please use
                             astropy.units format, default unit is solar radius
    """

    x, y = kwargs['points'][:, 0], kwargs['points'][:, 1]

    unit = str(kwargs['axis_unit'])
    x_label, y_label = r'x/' + unit, r'z/' + unit

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.grid()
    ax.plot(x, y)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return f if kwargs['return_figure_instance'] else plt.show()


def single_star_mesh(**kwargs):
    """
    Plot function for descriptor `mesh`, plots surface mesh of star in SingleStar system

    :param kwargs: Dict;
    :**kwargs options**:
        * **axis_unit** * -- astropy.units.solRad - unit in which axis will be displayed, please use
                            astropy.units format, default unit is solar radius
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.elev = 90 - kwargs['inclination']
    ax.azim = kwargs['azimuth']

    ax.scatter(kwargs['mesh'][:, 0], kwargs['mesh'][:, 1], kwargs['mesh'][:, 2], s=2)
    ax.set_xlim3d(-kwargs['equatorial_radius'], kwargs['equatorial_radius'])
    ax.set_ylim3d(-kwargs['equatorial_radius'], kwargs['equatorial_radius'])
    ax.set_zlim3d(-kwargs['equatorial_radius'], kwargs['equatorial_radius'])
    ax.set_box_aspect([1, 1, 1])
    if kwargs['plot_axis']:
        unit = str(kwargs['axis_unit'])
        x_label, y_label, z_label = r'x/' + unit, r'y/' + unit, r'z/' + unit
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
    else:
        ax.set_axis_off()
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    return fig if kwargs['return_figure_instance'] else plt.show()


def binary_mesh(**kwargs):
    """
    Function plots 3D scatter plot of the surface points

    :param kwargs: Dict;
    :**kwargs options**:
        * **phase** * -- float; phase at which to construct plot
        * **components_to_plot** * -- str; component to plot `primary`, `secondary` or `both` (default)
        * **plot_axis** * -- bool; switch the plot axis on/off
        * **inclination** * -- float; elevation of the camera (in degrees)
        * **azimuth** * -- float; azimuth of the camera (in degrees)
    """

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.elev = 90 - kwargs['inclination']
    ax.azim = kwargs['azimuth']
    if kwargs['components_to_plot'] in ['primary', 'both']:
        ax.scatter(kwargs['points_primary'][:, 0], kwargs['points_primary'][:, 1], kwargs['points_primary'][:, 2], s=5,
                   label='primary', alpha=1.0)
    if kwargs['components_to_plot'] in ['secondary', 'both']:
        ax.scatter(kwargs['points_secondary'][:, 0], kwargs['points_secondary'][:, 1], kwargs['points_secondary'][:, 2],
                   s=2, label='secondary', alpha=1.0)
    ax.legend(loc=1)

    x_min, x_max = 0, 0
    if kwargs['components_to_plot'] == 'both':
        x_min = np.min(kwargs['points_primary'][:, 0])
        x_max = np.max(kwargs['points_secondary'][:, 0])
    elif kwargs['components_to_plot'] == 'primary':
        x_min = np.min(kwargs['points_primary'][:, 0])
        x_max = np.max(kwargs['points_primary'][:, 0])
    elif kwargs['components_to_plot'] == 'secondary':
        x_min = np.min(kwargs['points_secondary'][:, 0])
        x_max = np.max(kwargs['points_secondary'][:, 0])

    d = (x_max - x_min) / 2
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(-d, d)
    ax.set_zlim3d(-d, d)

    if kwargs['plot_axis']:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    else:
        ax.set_axis_off()
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    return fig if kwargs['return_figure_instance'] else plt.show()


def single_star_surface(**kwargs):
    """
    Plot function for descriptor `surface` in SingleSystem plot function, plots surface of star in SingleStar system

    :param kwargs: Dict;
    :**kwargs options**:
        * **phase** * float -- phase at which plot the system, important for eccentric orbits
        * **components_to_plot** * -- str; `primary`, `secondary` or `both` (default),
        * **normals** * -- bool; plot normals of the surface phases as arrows
        * **edges** * -- bool; highlight edges of surface faces
        * **colormap** * -- str; `gravity_acceleration`, `temperature` or None(default)
        * **plot_axis** * -- bool; if False, axis will be hidden
        * **face_mask_primary** * -- array[bool]; mask to select which faces to display
        * **face_mask_secondary** * -- array[bool]: mask to select which faces to display
        * **elevation** * -- float; in degree - elevation of camera
        * **azimuth** * -- float; camera azimuth
        * **units** * -- str; units of gravity acceleration colormap  `SI` or `cgs`
        * **scale** * -- str; `linear` or `log`
        * **axis_unit** * -- Union[astropy.unit, dimensionless]; - axis units
        * **colorbar_orientation** * -- str; `horizontal` or `vertical` (default)
        * **colorbar** * -- bool; colorabar on/off switchic
        * **colorbar_separation** * -- float; shifting position of the colorbar from its default postition
        * **colorbar_size** * -- float; relative size of the colorbar, default 0.7
    """

    fig = plt.figure(figsize=(7, 7))
    ax = axes3d.Axes3D(fig)
    ax.set_box_aspect([1, 1, 1])
    ax.elev = kwargs['elevation']
    ax.azim = kwargs['azimuth']

    clr = kwargs['surface_color']

    star_plot = ax.plot_trisurf(kwargs['points'][:, 0], kwargs['points'][:, 1], kwargs['points'][:, 2],
                                triangles=kwargs['triangles'], antialiased=True, shade=True, color=clr)
    if kwargs['edges']:
        star_plot.set_edgecolor('black')

    if kwargs['normals']:
        ax.quiver(kwargs['centres'][:, 0], kwargs['centres'][:, 1], kwargs['centres'][:, 2],
                  kwargs['arrows'][:, 0], kwargs['arrows'][:, 1], kwargs['arrows'][:, 2], color='black',
                  length=0.1 * kwargs['equatorial_radius'])

    if kwargs.get('colormap', False):
        cmap = CMAPS[kwargs['colormap']]
        star_plot.set_cmap(cmap=cmap)
        star_plot.set_array(kwargs['cmap'])
        if kwargs['colorbar']:
            colorbar = fig.colorbar(star_plot, shrink=kwargs['colorbar_size'],
                                    orientation=kwargs['colorbar_orientation'],
                                    pad=kwargs['colorbar_separation'])
            set_colorbar_label(colorbar, kwargs['colormap'], kwargs['unit'], kwargs['scale'])

    ax.set_xlim3d(-kwargs['equatorial_radius'], kwargs['equatorial_radius'])
    ax.set_ylim3d(-kwargs['equatorial_radius'], kwargs['equatorial_radius'])
    ax.set_zlim3d(-kwargs['equatorial_radius'], kwargs['equatorial_radius'])

    if kwargs['plot_axis']:
        unit = str(kwargs['axis_unit'])
        x_label, y_label, z_label = r'x/' + unit, r'y/' + unit, r'z/' + unit
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
    else:
        ax.set_axis_off()
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    return fig if kwargs['return_figure_instance'] else plt.show()


def binary_surface(**kwargs):
    """
    Function creates plot of binary system components.

    :param kwargs: Dict;
    :**kwargs options**:
        * **phase** * float -- phase at which plot the system, important for eccentric orbits
        * **components_to_plot** * -- str; `primary`, `secondary` or `both` (default),
        * **normals** * -- bool; plot normals of the surface phases as arrows
        * **edges** * -- bool; highlight edges of surface faces
        * **colormap** * -- str; `gravity_acceleration`, `temperature` or None(default)
        * **plot_axis** * -- bool; if False, axis will be hidden
        * **face_mask_primary** * -- array[bool]; mask to select which faces to display
        * **face_mask_secondary** * -- array[bool]: mask to select which faces to display
        * **elevation** * -- float; in degree - elevation of camera
        * **azimuth** * -- float; camera azimuth
        * **unit** * -- str; units of gravity acceleration colormap  `SI` or `cgs`
        * **scale** * -- str; `linear` or `log`
        * **axis_unit** * -- Union[astropy.unit, dimensionless]; - axis units
        * **colorbar_orientation** * -- str; `horizontal` or `vertical` (default)
        * **colorbar** * -- bool; colorabar on/off switchic
        * **colorbar_separation** * -- float; shifting position of the colorbar from its default postition
        * **colorbar_size** * -- float; relative size of the colorbar, default 0.7
    """

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.elev = kwargs['elevation']
    ax.azim = kwargs['azimuth']

    clr = kwargs['surface_colors']

    plot, plot1, plot2 = None, None, None
    if kwargs['components_to_plot'] == 'primary':
        plot = ax.plot_trisurf(
            kwargs['points_primary'][:, 0], kwargs['points_primary'][:, 1],
            kwargs['points_primary'][:, 2], triangles=kwargs['primary_triangles'],
            antialiased=True, shade=True, color=clr[0])

        if kwargs.get('normals', False):
            ax.quiver(
                kwargs['primary_centres'][:, 0], kwargs['primary_centres'][:, 1], kwargs['primary_centres'][:, 2],
                kwargs['primary_arrows'][:, 0], kwargs['primary_arrows'][:, 1], kwargs['primary_arrows'][:, 2],
                color='black', length=0.05)

    elif kwargs['components_to_plot'] == 'secondary':
        plot = ax.plot_trisurf(kwargs['points_secondary'][:, 0], kwargs['points_secondary'][:, 1],
                               kwargs['points_secondary'][:, 2], triangles=kwargs['secondary_triangles'],
                               antialiased=True, shade=True, color=clr[1])

        if kwargs.get('normals', False):
            ax.quiver(kwargs['secondary_centres'][:, 0], kwargs['secondary_centres'][:, 1],
                      kwargs['secondary_centres'][:, 2],
                      kwargs['secondary_arrows'][:, 0], kwargs['secondary_arrows'][:, 1],
                      kwargs['secondary_arrows'][:, 2],
                      color='black', length=0.05)

    elif kwargs['components_to_plot'] == 'both':
        if not kwargs['separate_colormaps']:
            points = up.concatenate((kwargs['points_primary'], kwargs['points_secondary']), axis=0)
            triangles = up.concatenate((kwargs['primary_triangles'],
                                    kwargs['secondary_triangles'] + np.shape(kwargs['points_primary'])[0]), axis=0)

            plot = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, antialiased=True,
                                   shade=True, color=clr[0])
        else:
            plot1 = ax.plot_trisurf(kwargs['points_primary'][:, 0], kwargs['points_primary'][:, 1],
                                    kwargs['points_primary'][:, 2], triangles=kwargs['primary_triangles'],
                                    antialiased=True, shade=True, color=clr[0])
            plot2 = ax.plot_trisurf(kwargs['points_secondary'][:, 0], kwargs['points_secondary'][:, 1],
                                    kwargs['points_secondary'][:, 2], triangles=kwargs['secondary_triangles'],
                                    antialiased=True, shade=True, color=clr[1])
        if kwargs.get('normals', False):
            centres = up.concatenate((kwargs['primary_centres'], kwargs['secondary_centres']), axis=0)
            arrows = up.concatenate((kwargs['primary_arrows'], kwargs['secondary_arrows']), axis=0)

            ax.quiver(centres[:, 0], centres[:, 1], centres[:, 2],
                      arrows[:, 0], arrows[:, 1], arrows[:, 2],
                      color='black', length=0.05)

    else:
        raise ValueError('Invalid value of keyword argument `components_to_plot`. '
                         'Expected values are: `primary`, `secondary` or `both`')

    if kwargs.get('edges', False):
        if kwargs['separate_colormaps']:
            plot1.set_edgecolor('black')
            plot2.set_edgecolor('black')
        else:
            plot.set_edgecolor('black')

    if kwargs.get('colormap', False):
        cmap = CMAPS[kwargs['colormap']]
        if kwargs['separate_colormaps']:
            plot1.set_cmap(cmap=cmap)
            plot2.set_cmap(cmap=cmap)
        else:
            plot.set_cmap(cmap=cmap)

        if kwargs['components_to_plot'] == 'primary':
            plot.set_array(kwargs['primary_cmap'])
            if kwargs['colorbar']:
                colorbar = fig.colorbar(plot, shrink=kwargs['colorbar_size'],
                                        orientation=kwargs['colorbar_orientation'],
                                        pad=kwargs['colorbar_separation'])
                set_colorbar_label(colorbar, kwargs['colormap'], kwargs['unit'], kwargs['scale'], extra='primary')
        elif kwargs['components_to_plot'] == 'secondary':
            plot.set_array(kwargs['secondary_cmap'])
            if kwargs['colorbar']:
                colorbar = fig.colorbar(plot, shrink=kwargs['colorbar_size'],
                                        orientation=kwargs['colorbar_orientation'],
                                        pad=kwargs['colorbar_separation'])
                set_colorbar_label(colorbar, kwargs['colormap'], kwargs['unit'], kwargs['scale'], extra='secondary')
        elif kwargs['components_to_plot'] == 'both':
            if not kwargs['separate_colormaps']:
                both_cmaps = up.concatenate((kwargs['primary_cmap'], kwargs['secondary_cmap']), axis=0)
                plot.set_array(both_cmaps)
                if kwargs['colorbar']:
                    colorbar = fig.colorbar(plot, shrink=kwargs['colorbar_size'],
                                            orientation=kwargs['colorbar_orientation'],
                                            pad=kwargs['colorbar_separation'])
                    set_colorbar_label(colorbar, kwargs['colormap'], kwargs['unit'], kwargs['scale'])
            else:
                plot1.set_array(kwargs['primary_cmap'])
                plot2.set_array(kwargs['secondary_cmap'])
                if kwargs['colorbar']:
                    colorbar1 = fig.colorbar(plot1, shrink=kwargs['colorbar_size'],
                                             orientation=kwargs['colorbar_orientation'],
                                             pad=kwargs['colorbar_separation'])
                    set_colorbar_label(
                        colorbar1, kwargs['colormap'], kwargs['unit'], kwargs['scale'], extra='primary'
                    )
                    colorbar2 = fig.colorbar(plot2, shrink=kwargs['colorbar_size'],
                                             orientation=kwargs['colorbar_orientation'],
                                             pad=kwargs['colorbar_separation'])
                    set_colorbar_label(
                        colorbar2, kwargs['colormap'], kwargs['unit'], kwargs['scale'], extra='secondary'
                    )

    if kwargs['plot_axis']:
        unit = str(kwargs['axis_unit'])
        if kwargs['axis_unit'] == units.dimensionless_unscaled:
            x_label, y_label, z_label = 'x', 'y', 'z'
        else:
            x_label, y_label, z_label = r'x/' + unit, r'y/' + unit, r'z/' + unit
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
    else:
        ax.set_axis_off()

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    gutils.set_axes_equal(ax)

    return fig if kwargs['return_figure_instance'] else plt.show()


def set_colorbar_label(colorbar, colorbar_name, unit, scale, extra=''):
    lbl = {
        'temperature': 'T',
        'gravity_acceleration': 'g',
        'velocity': 'v',
        'radial_velocity': r'v$_{rad}$',
        'v_r_perturbed': r'$d v_r$',
        'v_horizontal_perturbed': r'$d v_{horizontal}$',
        'normal_radiance': r'I$_{norm}$',
        'radiance': 'I',
        'radius': '$r$',
        'horizontal_displacement': r'$d r_{horizontal}$',
        'horizontal_acceleration': r'$g_{horizontal}$',
    }
    def_unit = {
        'temperature': 'K',
        'gravity_acceleration': '$m\,s^{-2}$',
        'velocity': '$m\,s^{-1}$',
        'radial_velocity': '$m\,s^{-1}$',
        'v_r_perturbed': '$m\,s^{-1}$',
        'v_horizontal_perturbed': '$m\,s^{-1}$',
        'normal_radiance': '$W.sr^{-1}.m^{-2}$',
        'radiance': '$W.sr^{-1}.m^{-2}$',
        'radius': '$m$',
        'horizontal_displacement': r'$m$',
        'horizontal_acceleration': r'$m\,s^{-2}$',
    }
    unt = def_unit[colorbar_name] if unit == 'default' else unit
    if scale == 'linear':
        colorbar.set_label(extra + f' {lbl[colorbar_name]}/[{unt}]')
    elif scale == 'log':
        colorbar.set_label(extra + f' log({lbl[colorbar_name]}/[{unt}])')


def single_star_wireframe(**kwargs):
    """
    Plot function for descriptor `wireframe` in SingleSystem, plots wireframe model of single system star

    :param kwargs: Dict;
    :**kwargs options**:
        * **axis_unit** * -- astropy.units.solRad - unit in which axis will be displayed, please use
                                                    astropy.units format, default unit is solar radius
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.elev = 90 - kwargs['inclination']
    ax.azim = kwargs['azimuth']

    star_plot = ax.plot_trisurf(kwargs['mesh'][:, 0], kwargs['mesh'][:, 1], kwargs['mesh'][:, 2],
                                triangles=kwargs['triangles'], antialiased=True, color='none')
    star_plot.set_edgecolor('black')

    ax.set_xlim3d(-kwargs['equatorial_radius'], kwargs['equatorial_radius'])
    ax.set_ylim3d(-kwargs['equatorial_radius'], kwargs['equatorial_radius'])
    ax.set_zlim3d(-kwargs['equatorial_radius'], kwargs['equatorial_radius'])
    ax.set_box_aspect([1, 1, 1])
    if kwargs['plot_axis']:
        unit = str(kwargs['axis_unit'])
        x_label, y_label, z_label = r'x/' + unit, r'y/' + unit, r'z/' + unit
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
    else:
        ax.set_axis_off()
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    return fig if kwargs['return_figure_instance'] else plt.show()


def binary_wireframe(**kwargs):
    """
    Function displays wireframe model of the stellar surface

    :param kwargs:
    :**kwargs options**:
        * **phase** * -- float; phase at which to construct plot
        * **components_to_plot** * -- str; component to plot `primary`, `secondary` or `both` (default)
        * **plot_axis** * -- bool; switch the plot axis on/off
        * **inclination** * -- float; elevation of the camera (in degrees)
        * **azimuth** * -- float; azimuth of the camera (in degrees)
    :return:
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.elev = 90 - kwargs['inclination']
    ax.azim = kwargs['azimuth']

    if kwargs['components_to_plot'] == 'primary':
        plot = ax.plot_trisurf(
            kwargs['points_primary'][:, 0], kwargs['points_primary'][:, 1],
            kwargs['points_primary'][:, 2], triangles=kwargs['primary_triangles'],
            antialiased=True, shade=False, color='none')

    elif kwargs['components_to_plot'] == 'secondary':
        plot = ax.plot_trisurf(kwargs['points_secondary'][:, 0], kwargs['points_secondary'][:, 1],
                               kwargs['points_secondary'][:, 2], triangles=kwargs['secondary_triangles'],
                               antialiased=True, shade=False, color='none')

    elif kwargs['components_to_plot'] == 'both':
        points = up.concatenate((kwargs['points_primary'], kwargs['points_secondary']), axis=0)
        triangles = up.concatenate((kwargs['primary_triangles'],
                                    kwargs['secondary_triangles'] + np.shape(kwargs['points_primary'])[0]), axis=0)

        plot = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, antialiased=True,
                               shade=False, color='none')

    else:
        raise ValueError('Invalid value of keyword argument `components_to_plot`. '
                         'Expected values are: `primary`, `secondary` or `both`')

    plot.set_edgecolor('black')

    x_min, x_max = 0, 0
    if kwargs['components_to_plot'] == 'both':
        x_min = np.min(kwargs['points_primary'][:, 0])
        x_max = np.max(kwargs['points_secondary'][:, 0])
    elif kwargs['components_to_plot'] == 'primary':
        x_min = np.min(kwargs['points_primary'][:, 0])
        x_max = np.max(kwargs['points_primary'][:, 0])
    elif kwargs['components_to_plot'] == 'secondary':
        x_min = np.min(kwargs['points_secondary'][:, 0])
        x_max = np.max(kwargs['points_secondary'][:, 0])

    d = (x_max - x_min) / 2
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(-d, d)
    ax.set_zlim3d(-d, d)
    if kwargs['plot_axis']:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    else:
        ax.set_axis_off()
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    return fig if kwargs['return_figure_instance'] else plt.show()


def binary_surface_anim(**kwargs):
    """
    Function creates animation of the orbital motion.

    :param kwargs: Dict;
    :**kwargs options**:
        * **start_phase** * -- float;
        * **stop_phase** * -- float;
        * **phase_step** * -- float;
        * **units** * -- units for gravity acceleration colormap
        * **plot_axis** * -- bool, if False, axis will not be displayed
        * **colormap** * -- `temperature`, `gravity_acceleration` or None,
        * **savepath** * -- string or None, animation will be stored to `savepath`
        * **separate_colormaps** * -- bool; if True, figure will contain separate colormap for each component
    """
    def update_plot(frame_number, _points, _faces, _clr, _cmaps, _plot):
        for _, _ in enumerate(_plot):
            p = ax.clear()
        for ii, p in enumerate(_plot):
            ax.set_xlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
            ax.set_ylim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
            ax.set_zlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            p = ax.plot_trisurf(_points[ii][frame_number][:, 0],
                                _points[ii][frame_number][:, 1],
                                _points[ii][frame_number][:, 2],
                                triangles=_faces[ii][frame_number],
                                antialiased=True, shade=False, color=_clr[ii])
            if kwargs.get('colormap', False):
                p.set_cmap(cmap=CMAPS[kwargs.get('colormap', cm.jet)])
                p.set_array(_cmaps[ii][frame_number])

            if kwargs['edges']:
                p.set_edgecolor('black')

            ax.text(-kwargs['axis_lim'], 0.9*kwargs['axis_lim'], 0.8*kwargs['axis_lim'],
                    f"{kwargs['phases'][frame_number]%1.0:.2f}")

            if not kwargs['plot_axis']:
                ax.set_axis_off()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.elev = 0
    ax.azim = 180

    ax.set_xlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
    ax.set_ylim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
    ax.set_zlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])

    clr = ['g', 'r']
    cmaps = []

    if not kwargs['separate_colormaps']:
        points = [[up.concatenate((kwargs['points_primary'][ii], kwargs['points_secondary'][ii]), axis=0)
                  for ii in range(kwargs['n_frames'])]]
        faces = [[up.concatenate((kwargs['faces_primary'][ii],
                                      kwargs['faces_secondary'][ii] + np.shape(kwargs['points_primary'][ii])[0]),
                                      axis=0)
                     for ii in range(kwargs['n_frames'])]]

        plot = [ax.plot_trisurf(points[0][0][:, 0], points[0][0][:, 1], points[0][0][:, 2],
                                triangles=faces[0][0], antialiased=True,
                                shade=False, color=clr[0])]
        if kwargs.get('colormap', False):
            plot[0].set_cmap(cmap=CMAPS[kwargs['colormap']])
            cmaps = [[up.concatenate((kwargs['primary_cmap'][ii], kwargs['secondary_cmap'][ii]), axis=0)
                      for ii in range(kwargs['n_frames'])]]
            plot[0].set_array(cmaps[0][0])
    else:
        points = [kwargs['points_primary'], kwargs['points_secondary']]
        faces = [kwargs['faces_primary'], kwargs['faces_secondary']]
        plot = [ax.plot_trisurf(kwargs['points_primary'][0][:, 0], kwargs['points_primary'][0][:, 1],
                                kwargs['points_primary'][0][:, 2], triangles=kwargs['faces_primary'][0],
                                antialiased=True, shade=False, color=clr[0]),
                ax.plot_trisurf(kwargs['points_secondary'][0][:, 0], kwargs['points_secondary'][0][:, 1],
                                kwargs['points_secondary'][0][:, 2], triangles=kwargs['faces_secondary'][0],
                                antialiased=True, shade=False, color=clr[1])]
        if kwargs.get('colormap', False):
            plot[0].set_cmap(cmap=CMAPS[kwargs['colormap']])
            plot[1].set_cmap(cmap=CMAPS[kwargs['colormap']])
            cmaps = [kwargs['primary_cmap'], kwargs['secondary_cmap']]
            plot[0].set_array(cmaps[0][0])
            plot[1].set_array(cmaps[1][0])

    if kwargs.get('edges', False):
        if kwargs['separate_colormaps']:
            plot[0].set_edgecolor('black')
            plot[1].set_edgecolor('black')
        else:
            plot[0].set_edgecolor('black')

    args = (points, faces, clr, cmaps, plot)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ani = animation.FuncAnimation(fig, update_plot, kwargs['n_frames'], fargs=args, interval=20)
    plt.show() if not kwargs['savepath'] else ani.save(kwargs['savepath'], writer='ffmpeg', fps=30, dpi=300)


def single_surface_anim(**kwargs):
    def update_plot(frame_number, _points, _faces, _clr, _cmaps, _plot):
        for _, _ in enumerate(_plot):
            p = ax.clear()
        for ii, p in enumerate(_plot):
            ax.set_xlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
            ax.set_ylim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
            ax.set_zlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            p = ax.plot_trisurf(_points[ii][frame_number][:, 0],
                                _points[ii][frame_number][:, 1],
                                _points[ii][frame_number][:, 2],
                                triangles=_faces[ii][frame_number],
                                antialiased=True, shade=False, color=_clr[ii])
            if kwargs.get('colormap', False):
                p.set_cmap(cmap=CMAPS[kwargs.get('colormap', cm.jet)])
                p.set_array(_cmaps[ii][frame_number])
                p.set_clim(cmap_limits[0], cmap_limits[1])

            if kwargs['edges']:
                p.set_edgecolor('black')

            ax.text(-kwargs['axis_lim'], 0.9 * kwargs['axis_lim'], 0.8 * kwargs['axis_lim'],
                    f"{kwargs['phases'][frame_number]%1.0:.2f}")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.elev = 0
    ax.azim = 180

    ax.set_xlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
    ax.set_ylim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
    ax.set_zlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])

    clr = ['g', 'r']
    cmaps = []
    points = [kwargs['points']]
    faces = [kwargs['faces']]

    plot = [ax.plot_trisurf(points[0][0][:, 0], points[0][0][:, 1], points[0][0][:, 2],
                            triangles=faces[0][0], antialiased=True,
                            shade=False, color=clr[0])]
    if kwargs.get('colormap', False):
        cmap_limits = (np.min(kwargs['cmap']), np.max(kwargs['cmap']))
        plot[0].set_cmap(cmap=CMAPS[kwargs['colormap']])
        cmaps = [kwargs['cmap']]
        plot[0].set_array(cmaps[0][0])

    args = (points, faces, clr, cmaps, plot)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ani = animation.FuncAnimation(fig, update_plot, kwargs['n_frames'], fargs=args, interval=20)
    plt.show() if not kwargs['savepath'] else ani.save(kwargs['savepath'], writer='ffmpeg', fps=20, dpi=300)


def phase_curve(**kwargs):
    fig = plt.figure(figsize=(8, 6))
    if kwargs['unit'] in ['normalized', 'normalised']:
        C = np.max([kwargs['fluxes'][item] for item in kwargs['fluxes']])
    else:
        C = 1
    for item in kwargs['fluxes']:
        plt.plot(kwargs['phases'], kwargs['fluxes'][item] / C, label=item)
        plt.legend()

    plt.xlabel('Phase')
    if isinstance(kwargs['unit'], type(u.W/u.m**2)):
        uu = kwargs['unit']
        plt.ylabel(f'Flux/({uu:latex})')
    else:
        plt.ylabel('Flux')
    if kwargs['legend']:
        plt.legend(loc=kwargs['legend_location'])

    return fig if kwargs['return_figure_instance'] else plt.show()


def rv_curve(**kwargs):
    fig = plt.figure(figsize=(8, 6))
    phases, rvs = kwargs["phases"], kwargs["rvs"]
    for component in rvs.keys():
        plt.plot(phases, rvs[component], label=component)
    plt.legend()

    plt.xlabel('Phase')
    if isinstance(kwargs['unit'], type(u.m / u.s)):
        uu = kwargs['unit']
        plt.ylabel(f'Radial velocity/({uu:latex})')
    else:
        plt.ylabel('Radial velocity')
    if kwargs['legend']:
        plt.legend(loc=kwargs['legend_location'])

    return fig if kwargs['return_figure_instance'] else plt.show()


def binary_rv_fit_plot(**kwargs):
    """
    Plots the model and residuals described by fit params or calculated by last run of fitting procedure.

    :param kwargs: Dict;
    :**kwargs options**:
        * **fit_params** * -- Dict; {fit_parameter: {value: float, unit: astropy.unit.Unit}
        * **start_phase** * -- float;
        * **stop_phase** * -- float;
        * **number_of_points** * -- int;
        * **y_axis_unit** * -- astropy.unit.Unit;
    :return:
    """
    matplotlib.rcParams.update({'errorbar.capsize': 2})
    fig = plt.figure(figsize=(8, 6))

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(kwargs['synth_phases'], kwargs['rv_fit']['primary'], label='primary RV fit', color='cornflowerblue')
    ax1.plot(kwargs['synth_phases'], kwargs['rv_fit']['secondary'], label='secondary RV fit', color='firebrick',
             ls='dashed')
    if kwargs['y_err']['primary'] is None:
        ax1.scatter(kwargs['x_data']['primary'], kwargs['y_data']['primary'],
                    marker='o', color='blue', s=3, label='primary')
        ax2.scatter(kwargs['x_data']['primary'], kwargs['residuals']['primary'],
                    marker='o', color='blue', s=3, label='primary')
    else:
        ax1.errorbar(kwargs['x_data']['primary'], kwargs['y_data']['primary'], yerr=kwargs['y_err']['primary'],
                     linestyle='none', marker='o', color='blue', markersize=3, label='primary')
        ax2.errorbar(kwargs['x_data']['primary'], kwargs['residuals']['primary'], yerr=kwargs['y_err']['primary'],
                     linestyle='none', marker='o', color='blue', markersize=3, label='primary')

    if kwargs['y_err']['secondary'] is None:
        ax1.scatter(kwargs['x_data']['secondary'], kwargs['y_data']['secondary'],
                    marker='x', color='red', s=3, label='secondary')
        ax2.scatter(kwargs['x_data']['secondary'], kwargs['residuals']['secondary'],
                    marker='x', color='red', s=3, label='secondary')
    else:
        ax1.errorbar(kwargs['x_data']['secondary'], kwargs['y_data']['secondary'], yerr=kwargs['y_err']['secondary'],
                     linestyle='none', marker='x', color='red',
                     markersize=3, label='secondary')
        ax2.errorbar(kwargs['x_data']['secondary'], kwargs['residuals']['secondary'], yerr=kwargs['y_err']['secondary'],
                     linestyle='none', marker='x', color='red',
                     markersize=3, label='secondary')
    ax1.legend()
    unit = kwargs['y_unit']
    ax1.set_ylabel(f'Radial velocity/[{unit}]')

    ax2.axhline(0, ls='dashed', c='black', lw=0.5)

    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Residuals')

    plt.subplots_adjust(hspace=0.0, top=0.98, right=0.97)
    return fig if kwargs['return_figure_instance'] else plt.show()


def binary_lc_fit_plot(**kwargs):
    synthetic_clrs = {
        'bolometric': 'black',
        'Generic.Bessell.U': '#ff00bf',
        'Generic.Bessell.B': '#0000E5',
        'Generic.Bessell.V': '#00cc00',
        'Generic.Bessell.R': '#fd2b2b',
        'Generic.Bessell.I': '#b30000',
        'SLOAN.SDSS.u': '#0000ff',
        'SLOAN.SDSS.g': '#00cc00',
        'SLOAN.SDSS.r': '#ff1a1a',
        'SLOAN.SDSS.i': '#cc00cc',
        'SLOAN.SDSS.z': '#00ffff',
        'Generic.Stromgren.u': '#cc00cc',
        'Generic.Stromgren.v': '#ff00ff',
        'Generic.Stromgren.b': '#3333ff',
        'Generic.Stromgren.y': '#00e600',
        'Kepler': '#E50000',
        'GaiaDR2': 'black',
    }
    datapoint_clrs = {
        'bolometric': 'gray',
        'Generic.Bessell.U': '#cc0099',
        'Generic.Bessell.B': '#00007F',
        'Generic.Bessell.V': '#008000',
        'Generic.Bessell.R': '#ff0000',
        'Generic.Bessell.I': '#800000',
        'SLOAN.SDSS.u': '#000099',
        'SLOAN.SDSS.g': '#009900',
        'SLOAN.SDSS.r': '#e60000',
        'SLOAN.SDSS.i': '#800080',
        'SLOAN.SDSS.z': '#00cccc',
        'Generic.Stromgren.u': '#990099',
        'Generic.Stromgren.v': '#cc00cc',
        'Generic.Stromgren.b': '#0000cc',
        'Generic.Stromgren.y': '#00b300',
        'Kepler': '#890000',
        'GaiaDR2': 'gray',
        'TESS': '#006989'
    }

    matplotlib.rcParams.update({'errorbar.capsize': 2})
    fig = plt.figure(figsize=(8, 6))

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    for fltr, curve in kwargs['lcs'].items():
        rasterize = np.shape(kwargs['x_data'][fltr])[0] > 10000 if kwargs['rasterize'] is None else rasterize
        (dt_clr, clr) = (datapoint_clrs[fltr], datapoint_clrs[fltr]) if len(kwargs['lcs']) > 1 else ('blue', 'red')

        if kwargs['y_err'][fltr] is None:
            ax1.scatter(kwargs['x_data'][fltr], kwargs['y_data'][fltr], s=3, label=fltr + ' observed',
                        color=dt_clr)

            ax2.scatter(kwargs['x_data'][fltr], kwargs['residuals'][fltr], s=3, label=fltr + ' residual',
                        color=dt_clr)
        else:
            ax1.errorbar(kwargs['x_data'][fltr], kwargs['y_data'][fltr], yerr=kwargs['y_err'][fltr],
                         linestyle='none', markersize=3, label=fltr + ' observed', color=dt_clr, rasterized=rasterize)

            ax2.errorbar(kwargs['x_data'][fltr], kwargs['residuals'][fltr], yerr=kwargs['y_err'][fltr],
                         linestyle='none', markersize=3, label=fltr + ' residual', color=dt_clr, rasterized=rasterize)

        ax1.plot(kwargs['synth_phases'], curve, label=fltr + ' synthetic', color=clr, linewidth=2)

    ax2.axhline(0, ls='dashed', c='black', lw=0.5)

    if kwargs['legend']:
        ax1.legend(loc=kwargs['loc'])
    # ax2.legend(loc=1)

    ax1.set_ylabel(f'Normalized flux')

    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Residuals')

    plt.subplots_adjust(hspace=0.0, top=0.98, right=0.97)
    return fig if kwargs['return_figure_instance'] else plt.show()

