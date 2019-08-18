from elisa import utils
from matplotlib import cm
from astropy import units as u
import numpy as np
import mpl_toolkits.mplot3d.axes3d as axes3d

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib


def orbit(**kwargs):
    """
    Plot function for descriptor = `orbit` in function BinarySystem.plot(). This function plots orbit of the secondary
    component in the reference frame of the primary component.

    :param kwargs: dict:
                  keywords: start_phase = 0 - starting photometric phase (mean anomaly) of orbit, default value is 0
                            stop_phase = 1 - starting photometric phase (mean anomaly) of orbit, default value is 1
                            number_of_points = 300 - number of points where position is calculated, the more points,
                                                     the less coarse orbit plot is, default value is 300
                            axis_unit = astropy.units.solRad - unit in which axis will be displayed, please use
                                                               astropy.units format, default unit is solar radius
                                                               if you want dimensionless axis, use
                                                               astropy.units.dimensionless_unscaled or `dimensionless`
                            frame_or_reference = 'primary_component' - origin point for frame of reference in which
                                                                       orbit will be displayed, choices:
                                                                       primary_component - default
                                                                       barycentric
    :return:
    """
    unit = str(kwargs['axis_unit'])
    if kwargs['axis_unit'] == u.dimensionless_unscaled:
        x_label, y_label = 'x', 'y'
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
    elif kwargs['frame_of_reference'] == 'primary_component':
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

    :param kwargs: dict
                   keywords: plane = 'xy' - plane in which surface Hill plane is calculated, planes: 'xy', 'yz', 'zx'
                             phase = 0 - photometric phase in which surface Hill plane is calculated
    :return:
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

    :param kwargs: dict:
                   keywords: `axis_unit` = astropy.units.solRad - unit in which axis will be displayed, please use
                                                               astropy.units format, default unit is solar radius
    :return:
    """
    x, y = kwargs['points'][:, 0], kwargs['points'][:, 1]

    unit = str(kwargs['axis_unit'])
    x_label, y_label = r'x/' + unit, r'y/' + unit

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

    :param kwargs: dict
                   keywords:`axis_unit` = astropy.units.solRad - unit in which axis will be displayed, please use
                                                                 astropy.units format, default unit is solar radius
    :return:
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
    Plot function for descriptor `mesh`, plots surface mesh of binary star in BinaryStar system

    :param kwargs: dict
                   keywords: `phase`: np.float - phase in which system is plotted default value is 0
                             `components_to_plot`: str - decides which argument to plot, choices: `primary`, secondary`
                                                         , `both`, default is `both`
    :return:
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

    D = (x_max - x_min) / 2
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(-D, D)
    ax.set_zlim3d(-D, D)

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

    :param kwargs: `axis_unit` - astropy.units.solRad : unit in which axis will be displayed, please use
                                                        astropy.units format, default unit is solar radius
                   `edges` - bool: if True edges of surface faces are visible
                   `normals` - bool: if True surface faces outward facing normals are visible
                   `colormap` - string: `temperature` - displays temperature surface colormap
                                        `gravity_acceleration` - displays gravity acceleration colormap

    :return:
    """
    fig = plt.figure(figsize=(7, 7))
    ax = axes3d.Axes3D(fig)
    ax.set_aspect('equal')
    ax.elev = 90 - kwargs['inclination']
    ax.azim = kwargs['azimuth']

    star_plot = ax.plot_trisurf(kwargs['mesh'][:, 0], kwargs['mesh'][:, 1], kwargs['mesh'][:, 2],
                                triangles=kwargs['triangles'], antialiased=True, shade=False, alpha=1)
    if kwargs['edges']:
        star_plot.set_edgecolor('black')

    if kwargs['normals']:
        arrows = ax.quiver(kwargs['centres'][:, 0], kwargs['centres'][:, 1], kwargs['centres'][:, 2],
                           kwargs['arrows'][:, 0], kwargs['arrows'][:, 1], kwargs['arrows'][:, 2], color='black',
                           length=0.1 * kwargs['equatorial_radius'])

    if kwargs.get('colormap', False):
        star_plot.set_cmap(cmap=cm.jet_r)
        star_plot.set_array(kwargs['cmap'])
        colorbar = fig.colorbar(star_plot, shrink=0.7)
        if kwargs['colormap'] == 'temperature':
            colorbar.set_label('T/[K]')
        elif kwargs['colormap'] == 'gravity_acceleration':
            set_g_colorbar_label(colorbar, kwargs['units'])

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
        function creates plot of binary system components
        :param kwargs:
            phase: float - phase at which plot the system, important for eccentric orbits
            components_to_plot: str - `primary`, `secondary` or `both`(default),
            normals: bool - plot normals of the surface phases as arrows
            edges: bool - highlight edges of surface faces
            colormap: str - 'gravity_acceleration`, `temperature` or None(default)
            plot_axis: bool - if False, axis will be hidden
            face_mask_primary - bool array: mask to select which faces to display
            face_mask_secondary - bool array: mask to select which faces to display
            inclination: float in degree - elevation of camera
            azimuth: camera azimuth
            units: str - units of gravity acceleration colormap  `log_cgs`, `SI`, `cgs`, `log_SI`
            axis_units: astropy.unit or dimensionless - axis units
            colorbar_orientation: str - `horizontal` or `vertical`(default)
            colorbar: bool - colorabar on/off switch
        :return:
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
            points = np.concatenate((kwargs['points_primary'], kwargs['points_secondary']), axis=0)
            triangles = np.concatenate((kwargs['primary_triangles'],
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
            centres = np.concatenate((kwargs['primary_centres'], kwargs['secondary_centres']), axis=0)
            arrows = np.concatenate((kwargs['primary_arrows'], kwargs['secondary_arrows']), axis=0)

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
                    set_T_colorbar_label(colorbar, extra='primary')
            elif kwargs['components_to_plot'] == 'secondary':
                plot.set_array(kwargs['secondary_cmap'])
                if kwargs['colorbar']:
                    colorbar = fig.colorbar(plot, shrink=0.7, orientation=kwargs['colorbar_orientation'], pad=0.0)
                    set_T_colorbar_label(colorbar, extra='secondary')
            elif kwargs['components_to_plot'] == 'both':
                if kwargs['morphology'] == 'over-contact':
                    both_cmaps = np.concatenate((kwargs['primary_cmap'], kwargs['secondary_cmap']), axis=0)
                    plot.set_array(both_cmaps)
                    if kwargs['colorbar']:
                        colorbar = fig.colorbar(plot, shrink=0.7)
                        set_T_colorbar_label(colorbar)
                else:
                    plot1.set_array(kwargs['primary_cmap'])
                    plot2.set_array(kwargs['secondary_cmap'])
                    if kwargs['colorbar']:
                        colorbar1 = fig.colorbar(plot1, shrink=0.7, orientation=kwargs['colorbar_orientation'], pad=0.0)
                        set_T_colorbar_label(colorbar1, extra='primary')
                        colorbar2 = fig.colorbar(plot2, shrink=0.7, orientation=kwargs['colorbar_orientation'], pad=0.0)
                        set_T_colorbar_label(colorbar2, extra='secondary')
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
                    colorbar1 = fig.colorbar(plot, shrink=0.7)
                    set_g_colorbar_label(colorbar1, kwargs['units'])
            elif kwargs['components_to_plot'] == 'secondary':
                plot.set_array(kwargs['secondary_cmap'])
                if kwargs['colorbar']:
                    colorbar1 = fig.colorbar(plot, shrink=0.7)
                    set_g_colorbar_label(colorbar1, kwargs['units'])
            elif kwargs['components_to_plot'] == 'both':
                if kwargs['morphology'] == 'over-contact':
                    both_cmaps = np.concatenate((kwargs['primary_cmap'], kwargs['secondary_cmap']), axis=0)
                    plot.set_array(both_cmaps)
                    if kwargs['colorbar']:
                        colorbar = fig.colorbar(plot, shrink=0.7)
                        set_g_colorbar_label(colorbar, kwargs['units'])
                else:
                    plot1.set_array(kwargs['primary_cmap'])
                    plot2.set_array(kwargs['secondary_cmap'])
                    if kwargs['colorbar']:
                        colorbar1 = fig.colorbar(plot1, shrink=0.7)
                        set_g_colorbar_label(colorbar1, kwargs['units'], extra='primary')
                        colorbar2 = fig.colorbar(plot2, shrink=0.7)
                        set_g_colorbar_label(colorbar2, kwargs['units'], extra='secondary')

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

    D = (x_max - x_min) / 2
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(-D, D)
    ax.set_zlim3d(-D, D)

    if kwargs['plot_axis']:
        unit = str(kwargs['axis_unit'])
        if kwargs['axis_unit'] == u.dimensionless_unscaled:
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


def set_g_colorbar_label(colorbar, kwarg, extra=''):
    """
    function sets label of the colorbar for gravity acceleration surface function

    :param colorbar:
    :param kwarg:
    :return:
    """
    if kwarg == 'log_cgs':
        colorbar.set_label(extra + ' log(g/[cgs])')
    elif kwarg == 'log_SI':
        colorbar.set_label(extra + ' log(g/[SI])')
    elif kwarg == 'SI':
        colorbar.set_label(extra + r' $g/[m s^{-2}]$')
    elif kwarg == 'cgs':
        colorbar.set_label(extra + r' $g/[cm s^{-2}]$')


def set_T_colorbar_label(colorbar, extra=''):
    """
    function sets label of the colorbar for effective temperature surface function

    :param colorbar:
    :param kwarg:
    :return:
    """
    colorbar.set_label(extra + r' $T_{eff}/[K]$')


def single_star_wireframe(**kwargs):
    """
    Plot function for descriptor `wireframe` in SingleSystem, plots wireframe model of single system star

    :param kwargs: `axis_unit` = astropy.units.solRad - unit in which axis will be displayed, please use
                                                                 astropy.units format, default unit is solar radius

    :return:
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
        points = np.concatenate((kwargs['points_primary'], kwargs['points_secondary']), axis=0)
        triangles = np.concatenate((kwargs['primary_triangles'],
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

    D = (x_max - x_min) / 2
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(-D, D)
    ax.set_zlim3d(-D, D)
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
    function creates animation of the orbital motion
    :param kwargs: dict
        'start_phase' - float,
        'stop_phase' - float,
        'phase_step' - sloat,
        'units' - units for gravity acceleration colormap,
        'plot_axis' - bool - if False, axis will not be displayed,
        'colormap' - `temperature`, `gravity_acceleration` or None,
        'savepath' - string or None, animation will be stored to `savepath`
    :return:
    """
    def update_plot(frame_number, points, faces, clr, cmaps, plot):
        # plot.pop(0).remove()
        for ii, p in enumerate(plot):
            p = ax.clear()
        for ii, p in enumerate(plot):
            ax.set_xlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
            ax.set_ylim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
            ax.set_zlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            p = ax.plot_trisurf(points[ii][frame_number][:, 0],
                                points[ii][frame_number][:, 1],
                                points[ii][frame_number][:, 2],
                                triangles=faces[ii][frame_number],
                                antialiased=True, shade=False, color=clr[ii])
            if kwargs.get('colormap', False):
                p.set_cmap(cmap=cm.jet_r)
                p.set_array(cmaps[ii][frame_number])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    ax.set_xlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
    ax.set_ylim3d(-kwargs['axis_lim'], kwargs['axis_lim'])
    ax.set_zlim3d(-kwargs['axis_lim'], kwargs['axis_lim'])

    clr = ['g', 'r']

    if kwargs['morphology'] == 'over-contact':
        points = [[np.concatenate((kwargs['points_primary'][ii], kwargs['points_secondary'][ii]), axis=0)
                  for ii in range(kwargs['Nframes'])]]
        faces = [[np.concatenate((kwargs['faces_primary'][ii],
                                      kwargs['faces_secondary'][ii] + np.shape(kwargs['points_primary'][ii])[0]),
                                      axis=0)
                     for ii in range(kwargs['Nframes'])]]

        plot = [ax.plot_trisurf(points[0][0][:, 0], points[0][0][:, 1], points[0][0][:, 2],
                                triangles=faces[0][0], antialiased=True,
                                shade=False, color=clr[0])]
        if kwargs.get('colormap', False):
            plot[0].set_cmap(cmap=cm.jet_r)
            cmaps = [[np.concatenate((kwargs['primary_cmap'][ii], kwargs['secondary_cmap'][ii]), axis=0)
                    for ii in range(kwargs['Nframes'])]]
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
    ani = animation.FuncAnimation(fig, update_plot, kwargs['Nframes'], fargs=args, interval=20)
    plt.show() if not kwargs['savepath'] else ani.save(kwargs['savepath'], writer='imagemagick', fps=20)
