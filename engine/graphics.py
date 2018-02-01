from engine import utils
import matplotlib.pyplot as plt
from astropy import units as u
import numpy as np
import re
from mpl_toolkits.mplot3d import Axes3D


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
        ax.plot(x, y, label='primary')
        # ax.scatter(x[0], y[0], c='r')
        ax.scatter([0], [0], c='b', label='secondary')

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
    x_label, y_label = 'x', 'z'
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
                   keywords: `mesh` = surface points of the star in standard numpy array format:
                                      numpy.array([[x1 y1 z1],
                                                   [x2 y2 z2],
                                                   ...
                                                   [xN yN zN]])
                             `axis_unit` = astropy.units.solRad - unit in which axis will be displayed, please use
                                                                 astropy.units format, default unit is solar radius
                             `equatorial_radius': numpy.float - equatorial radius of the star in axis units
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(kwargs['mesh'][:, 0], kwargs['mesh'][:, 1], kwargs['mesh'][:, 2], s=2)
    ax.set_xlim3d(-kwargs['equatorial_radius'], kwargs['equatorial_radius'])
    ax.set_ylim3d(-kwargs['equatorial_radius'], kwargs['equatorial_radius'])
    ax.set_zlim3d(-kwargs['equatorial_radius'], kwargs['equatorial_radius'])
    ax.set_aspect('equal', adjustable='box')
    unit = str(kwargs['axis_unit'])
    x_label, y_label, z_label = r'x/' + unit, r'y/' + unit, r'z/' + unit
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.show()


def binary_mesh(**kwargs):
    """
    Plot function for descriptor `mesh`, plots surface mesh of binary star in BinaryStar system

    :param kwargs: dict
                   keywords: `phase`: np.float - phase in which system is plotted default value is 0
                             `components_to_plot`: str - decides which argument to plot, choices: `primary`, secondary`
                                                         , `both`, default is `both`
                             `alpha1`: np.float - discretization factor for primary component, equals to mean angular
                                                  spacing between points, default value is 5
                             `alpha2`: np.float - discretization factor for secondary component, equals to mean angular
                                                  spacing between points, default value is 5
    :return:
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    if kwargs['components_to_plot'] in ['primary', 'both']:
        ax.scatter(kwargs['points_primary'][:, 0], kwargs['points_primary'][:, 1], kwargs['points_primary'][:, 2], s=2,
                   label='primary')
    if kwargs['components_to_plot'] in ['secondary', 'both']:
        ax.scatter(kwargs['points_secondary'][:, 0], kwargs['points_secondary'][:, 1], kwargs['points_secondary'][:, 2],
                   s=2, label='secondary')
    ax.legend(loc=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if kwargs['components_to_plot'] == 'both':
        x_min = np.min(kwargs['points_primary'][:, 0])
        x_max = np.max(kwargs['points_secondary'][:, 0])
    elif kwargs['components_to_plot'] == 'primary':
        x_min = np.min(kwargs['points_primary'][:, 0])
        x_max = np.max(kwargs['points_primary'][:, 0])
    elif kwargs['components_to_plot'] == 'secondary':
        x_min = np.min(kwargs['points_secondary'][:, 0])
        x_max = np.max(kwargs['points_secondary'][:, 0])

    D = (x_max - x_min)/2
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(-D, D)
    ax.set_zlim3d(-D, D)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.show()
