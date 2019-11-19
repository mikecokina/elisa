import numpy as np
import mpl_toolkits.mplot3d.axes3d as axes3d

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import cm
from elisa import (
    umpy as up,
    utils,
    units
)


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

    ax.legend(loc=1)
    ax.set_aspect('equal')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()


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
    ax.plot(x_primary, y_primary, label='primary')
    ax.plot(x_secondary, y_secondary, label='secondary')
    lims = ax.get_xlim() - np.mean(ax.get_xlim())
    ax.set_ylim(lims)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc=1)
    ax.grid()
    plt.show()


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
    plt.show()


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
    ax.set_aspect('equal', adjustable='box')
    if kwargs['plot_axis']:
        unit = str(kwargs['axis_unit'])
        x_label, y_label, z_label = r'x/' + unit, r'y/' + unit, r'z/' + unit
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
    else:
        ax.set_axis_off()
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.show()


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
    ax.set_aspect('equal')
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
    plt.show()


def single_star_surface(**kwargs):
    """
    Plot function for descriptor `surface` in SingleSystem plot function, plots surface of star in SingleStar system

    :param kwargs: Dict;
    :**kwargs options**:
        * **axis_unit** * -- astropy.units.solRad; unit in which axis will be displayed, please use
                                                astropy.units format, default unit is solar radius
        * **edges** * -- bool; if True edges of surface faces are visible
        * **normals** * -- bool; if True surface faces outward facing normals are visible
        * **colormap** * -- str; `temperature` - displays temperature surface colormap
        * **gravity_acceleration** * -- bool; displays gravity acceleration colormap
    """

    fig = plt.figure(figsize=(7, 7))
    ax = axes3d.Axes3D(fig)
    ax.set_aspect('equal')
    ax.elev = 90 - kwargs['inclination']
    ax.azim = kwargs['azimuth']

    star_plot = ax.plot_trisurf(kwargs['points'][:, 0], kwargs['points'][:, 1], kwargs['points'][:, 2],
                                triangles=kwargs['triangles'], antialiased=True, shade=False, color='g')
    if kwargs['edges']:
        star_plot.set_edgecolor('black')

    if kwargs['normals']:
        ax.quiver(kwargs['centres'][:, 0], kwargs['centres'][:, 1], kwargs['centres'][:, 2],
                  kwargs['arrows'][:, 0], kwargs['arrows'][:, 1], kwargs['arrows'][:, 2], color='black',
                  length=0.1 * kwargs['equatorial_radius'])

    if kwargs.get('colormap', False):
        if kwargs['colormap'] == 'temperature':
            star_plot.set_cmap(cmap=cm.jet_r)
            star_plot.set_array(kwargs['cmap'])
            if kwargs['colorbar']:
                colorbar = fig.colorbar(star_plot, shrink=0.7, orientation=kwargs['colorbar_orientation'], pad=0.0)
                set_t_colorbar_label(colorbar, kwargs['scale'])

        elif kwargs['colormap'] == 'gravity_acceleration':
            try:
                star_plot.set_cmap(cmap=cm.jet_r)
            except:
                pass
            star_plot.set_array(kwargs['cmap'])
            colorbar = fig.colorbar(star_plot, shrink=0.7, orientation=kwargs['colorbar_orientation'])
            set_g_colorbar_label(colorbar, kwargs['units'], kwargs['scale'])

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
    plt.show()


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
        * **inclination** * -- float; in degree - elevation of camera
        * **azimuth** * -- float; camera azimuth
        * **units** * -- str; units of gravity acceleration colormap  `SI` or `cgs`
        * **scale** * -- str; `linear` or `log`
        * **axis_unit** * -- Union[astropy.unit, dimensionless]; - axis units
        * **colorbar_orientation** * -- str; `horizontal` or `vertical` (default)
        * **colorbar** * -- bool; colorabar on/off switchic
    """

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.elev = 90 - kwargs['inclination']
    ax.azim = kwargs['azimuth']

    clr = ['g', 'r']

    if kwargs['components_to_plot'] == 'primary':
        plot = ax.plot_trisurf(
            kwargs['points_primary'][:, 0], kwargs['points_primary'][:, 1],
            kwargs['points_primary'][:, 2], triangles=kwargs['primary_triangles'],
            antialiased=True, shade=False, color=clr[0])

        if kwargs.get('normals', False):
            ax.quiver(
                kwargs['primary_centres'][:, 0], kwargs['primary_centres'][:, 1], kwargs['primary_centres'][:, 2],
                kwargs['primary_arrows'][:, 0], kwargs['primary_arrows'][:, 1], kwargs['primary_arrows'][:, 2],
                color='black', length=0.05)

    elif kwargs['components_to_plot'] == 'secondary':
        plot = ax.plot_trisurf(kwargs['points_secondary'][:, 0], kwargs['points_secondary'][:, 1],
                               kwargs['points_secondary'][:, 2], triangles=kwargs['secondary_triangles'],
                               antialiased=True, shade=False, color=clr[1])

        if kwargs.get('normals', False):
            ax.quiver(kwargs['secondary_centres'][:, 0], kwargs['secondary_centres'][:, 1],
                      kwargs['secondary_centres'][:, 2],
                      kwargs['secondary_arrows'][:, 0], kwargs['secondary_arrows'][:, 1],
                      kwargs['secondary_arrows'][:, 2],
                      color='black', length=0.05)

    elif kwargs['components_to_plot'] == 'both':
        if kwargs['morphology'] == 'over-contact':
            points = up.concatenate((kwargs['points_primary'], kwargs['points_secondary']), axis=0)
            triangles = up.concatenate((kwargs['primary_triangles'],
                                    kwargs['secondary_triangles'] + np.shape(kwargs['points_primary'])[0]), axis=0)

            plot = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, antialiased=True,
                                   shade=False, color=clr[0])
        else:
            plot1 = ax.plot_trisurf(kwargs['points_primary'][:, 0], kwargs['points_primary'][:, 1],
                                    kwargs['points_primary'][:, 2], triangles=kwargs['primary_triangles'],
                                    antialiased=True, shade=False, color=clr[0])
            plot2 = ax.plot_trisurf(kwargs['points_secondary'][:, 0], kwargs['points_secondary'][:, 1],
                                    kwargs['points_secondary'][:, 2], triangles=kwargs['secondary_triangles'],
                                    antialiased=True, shade=False, color=clr[1])
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
        if kwargs['components_to_plot'] == 'both' and kwargs['morphology'] != 'over-contact':
            plot1.set_edgecolor('black')
            plot2.set_edgecolor('black')
        else:
            plot.set_edgecolor('black')

    if kwargs.get('colormap', False):
        if kwargs['colormap'] == 'temperature':
            if kwargs['components_to_plot'] == 'both' and kwargs['morphology'] != 'over-contact':
                plot1.set_cmap(cmap=cm.jet_r)
                plot2.set_cmap(cmap=cm.jet_r)
            else:
                plot.set_cmap(cmap=cm.jet_r)
            if kwargs['components_to_plot'] == 'primary':
                plot.set_array(kwargs['primary_cmap'])
                if kwargs['colorbar']:
                    colorbar = fig.colorbar(plot, shrink=0.7, orientation=kwargs['colorbar_orientation'], pad=0.0)
                    set_t_colorbar_label(colorbar, kwargs['scale'], extra='primary')
            elif kwargs['components_to_plot'] == 'secondary':
                plot.set_array(kwargs['secondary_cmap'])
                if kwargs['colorbar']:
                    colorbar = fig.colorbar(plot, shrink=0.7, orientation=kwargs['colorbar_orientation'], pad=0.0)
                    set_t_colorbar_label(colorbar, kwargs['scale'], extra='secondary')
            elif kwargs['components_to_plot'] == 'both':
                if kwargs['morphology'] == 'over-contact':
                    both_cmaps = up.concatenate((kwargs['primary_cmap'], kwargs['secondary_cmap']), axis=0)
                    plot.set_array(both_cmaps)
                    if kwargs['colorbar']:
                        colorbar = fig.colorbar(plot, shrink=0.7)
                        set_t_colorbar_label(colorbar, kwargs['scale'])
                else:
                    plot1.set_array(kwargs['primary_cmap'])
                    plot2.set_array(kwargs['secondary_cmap'])
                    if kwargs['colorbar']:
                        colorbar1 = fig.colorbar(plot1, shrink=0.7, orientation=kwargs['colorbar_orientation'], pad=0.0)
                        set_t_colorbar_label(colorbar1, kwargs['scale'], extra='primary')
                        colorbar2 = fig.colorbar(plot2, shrink=0.7, orientation=kwargs['colorbar_orientation'], pad=0.0)
                        set_t_colorbar_label(colorbar2, kwargs['scale'], extra='secondary')
        elif kwargs['colormap'] == 'gravity_acceleration':
            try:
                plot1.set_cmap(cmap=cm.jet_r)
                plot2.set_cmap(cmap=cm.jet_r)
            except:
                pass
            try:
                plot.set_cmap(cmap=cm.jet_r)
            except:
                pass
            if kwargs['components_to_plot'] == 'primary':
                plot.set_array(kwargs['primary_cmap'])
                if kwargs['colorbar']:
                    colorbar1 = fig.colorbar(plot, shrink=0.7, orientation=kwargs['colorbar_orientation'])
                    set_g_colorbar_label(colorbar1, kwargs['units'], kwargs['scale'])
            elif kwargs['components_to_plot'] == 'secondary':
                plot.set_array(kwargs['secondary_cmap'])
                if kwargs['colorbar']:
                    colorbar1 = fig.colorbar(plot, shrink=0.7, orientation=kwargs['colorbar_orientation'])
                    set_g_colorbar_label(colorbar1, kwargs['units'], kwargs['scale'])
            elif kwargs['components_to_plot'] == 'both':
                if kwargs['morphology'] == 'over-contact':
                    both_cmaps = up.concatenate((kwargs['primary_cmap'], kwargs['secondary_cmap']), axis=0)
                    plot.set_array(both_cmaps)
                    if kwargs['colorbar']:
                        colorbar = fig.colorbar(plot, shrink=0.7, orientation=kwargs['colorbar_orientation'])
                        set_g_colorbar_label(colorbar, kwargs['units'], kwargs['scale'])
                else:
                    plot1.set_array(kwargs['primary_cmap'])
                    plot2.set_array(kwargs['secondary_cmap'])
                    if kwargs['colorbar']:
                        colorbar1 = fig.colorbar(plot1, shrink=0.7, orientation=kwargs['colorbar_orientation'])
                        set_g_colorbar_label(colorbar1, kwargs['units'], kwargs['scale'], extra='primary')
                        colorbar2 = fig.colorbar(plot2, shrink=0.7, orientation=kwargs['colorbar_orientation'])
                        set_g_colorbar_label(colorbar2, kwargs['units'], kwargs['scale'], extra='secondary')

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
    plt.show()


def set_g_colorbar_label(colorbar, unit, scale, extra=''):
    """
    Function sets label of the colorbar for gravity acceleration surface function.
    """

    if unit == 'cgs':
        if scale == 'linear':
            colorbar.set_label(extra + r' $g/[cm s^{-2}]$')
        elif scale == 'log':
            colorbar.set_label(extra + ' log(g/[cgs])')
    elif unit == 'SI':
        if scale == 'linear':
            colorbar.set_label(extra + r' $g/[m s^{-2}]$')
        elif scale == 'log':
            colorbar.set_label(extra + ' log(g/[SI])')


def set_t_colorbar_label(colorbar, scale, extra=''):
    """
    Function sets label of the colorbar for effective temperature surface function.
    """
    if scale == 'linear':
        colorbar.set_label(extra + r' $T_{eff}/[K]$')
    elif scale == 'log':
        colorbar.set_label(extra + r' $log(T_{eff})$')


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
    ax.set_aspect('equal', adjustable='box')
    if kwargs['plot_axis']:
        unit = str(kwargs['axis_unit'])
        x_label, y_label, z_label = r'x/' + unit, r'y/' + unit, r'z/' + unit
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
    else:
        ax.set_axis_off()
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.show()


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
    ax.set_aspect('equal')
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
    plt.show()


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
                p.set_cmap(cmap=cm.jet_r)
                p.set_array(_cmaps[ii][frame_number])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    ax.set_xlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
    ax.set_ylim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
    ax.set_zlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])

    clr = ['g', 'r']
    cmaps = []

    if kwargs['morphology'] == 'over-contact':
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
            plot[0].set_cmap(cmap=cm.jet_r)
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
            plot[0].set_cmap(cmap=cm.jet_r)
            plot[1].set_cmap(cmap=cm.jet_r)
            cmaps = [kwargs['primary_cmap'], kwargs['secondary_cmap']]
            plot[0].set_array(cmaps[0][0])
            plot[1].set_array(cmaps[1][0])

    args = (points, faces, clr, cmaps, plot)
    ani = animation.FuncAnimation(fig, update_plot, kwargs['n_frames'], fargs=args, interval=20)
    plt.show() if not kwargs['savepath'] else ani.save(kwargs['savepath'], writer='imagemagick', fps=20)


def phase_curve(**kwargs):
    plt.figure(figsize=(8, 6))
    for item in kwargs['fluxes']:
        plt.plot(kwargs['phases'], kwargs['fluxes'][item], label=item)
        plt.legend()

    plt.xlabel('Phase')
    if kwargs['flux_unit'] == units.W / units.m**2:
        plt.ylabel(r'Flux/($W/m^{2}$)')
    else:
        plt.ylabel('Flux')
    if kwargs['legend']:
        plt.legend(loc=kwargs['legend_location'])
    plt.show()


def rv_curve(**kwargs):
    plt.figure(figsize=(8, 6))
    phases, primary_rv, secondary_rv = kwargs["phases"], kwargs["primary_rv"], kwargs["secondary_rv"]
    plt.plot(phases, primary_rv, label="primary")
    plt.plot(phases, secondary_rv, label="secondary")
    plt.legend()

    plt.xlabel('Phase')
    if kwargs['unit'] == units.m / units.s:
        plt.ylabel(r'Radial Velocity/($m/s$)')
    else:
        plt.ylabel('Radial Velocity')
    if kwargs['legend']:
        plt.legend(loc=kwargs['legend_location'])
    plt.show()
