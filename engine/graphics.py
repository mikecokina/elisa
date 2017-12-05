import matplotlib.pyplot as plt
from astropy import units as u


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
    :return:
    """
    x, y = kwargs['x_data'], kwargs['y_data']
    unit = str(kwargs['axis_unit'])
    if kwargs['axis_unit'] == u.dimensionless_unscaled:
        x_label, y_label = 'x', 'y'
    else:
        x_label, y_label = r'x/' + unit, r'y/' + unit

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(x, y)
    # ax.scatter(x[0], y[0], c='r')
    ax.scatter([0], [0], c='b')
    ax.set_aspect('equal')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()
    plt.show()


def equipotential(**kwargs):

    if kwargs['plane'] in ['xy', 'xy']:
        x_label, y_label = 'x', 'y'
    elif kwargs['plane'] in ['yz', 'zy']:
        pass
    elif kwargs['plane'] in ['zx', 'xz']:
        pass

    x_primary, y_primary = kwargs['points_primary'][:, 0], kwargs['points_primary'][:, 1]
    x_secondary, y_secondary = kwargs['points_secondary'][:, 0], kwargs['points_secondary'][:, 1]

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(x_primary, y_primary)
    ax.plot(x_secondary, y_secondary)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()
    plt.show()
