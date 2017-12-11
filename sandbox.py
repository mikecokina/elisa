import numpy as np
from matplotlib import pyplot as plt


mass_ratio = 0.5

def potential_dx_primary(x, *args):
    """
    first derivative of potential in frame of reference of primary component
    :param x: float - distance on x axis where origin is in primary component and x axis is pointing to
    secondary component
    :param args: tuple: (d,): d - periastron distance
    :return: float
    """
    d, = args
    r_sqr, rw_sqr = x ** 2, (d - x) ** 2
    return - np.power(x, -2) + ((mass_ratio * (d - x)) / rw_sqr ** (3.0 / 2.0)) + (mass_ratio + 1) * x - mass_ratio / d ** 2


def primary_potential_derivation_x(x, *args):
    actual_distance, synchronicity_parameter = args
    r_sqr, rw_sqr = x ** 2, (actual_distance - x) ** 2
    return - (x / r_sqr ** (3. / 2.)) + (
        (mass_ratio * (actual_distance - x)) / rw_sqr ** (3. / 2.)) + synchronicity_parameter ** 2 * (
        mass_ratio + 1) * x - mass_ratio / actual_distance ** 2


if False:
    args = 1.0,
    xss = np.linspace(-3, 3, 10000)
    ys = []
    xs = []
    # ys = [[x, potential_dx_primary(x, *args)] for x in xs if abs(x) > 0.01]

    for x in xss:
        y = potential_dx_primary(x, *args)
        if abs(y) < 10:
            ys.append(y)
            xs.append(x)

    plt.scatter(xs, ys, s=0.1)
    plt.grid(True)
    plt.show()

if True:
    args = 1.0, 1.0
    xss = np.linspace(-3, 3, 10000)
    ys = []
    xs = []
    # ys = [[x, potential_dx_primary(x, *args)] for x in xs if abs(x) > 0.01]

    for x in xss:
        y = primary_potential_derivation_x(x, *args)
        if abs(y) < 10:
            ys.append(y)
            xs.append(x)

    plt.scatter(xs, ys, s=0.1)
    plt.grid(True)
    plt.show()

