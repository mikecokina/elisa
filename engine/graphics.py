import matplotlib.pyplot as plt
from engine import binary_system
from engine import const as c
from engine import utils
import numpy as np

def orbit(**kwargs):
    if 'start_phase' not in kwargs:
        start_phase = 0
    else:
        start_phase = kwargs['start_phase']
    if 'stop_phase' not in kwargs:
        stop_phase = 1.0
    else:
        stop_phase = kwargs['stop_phase']
    if 'number_of_points' not in kwargs:
        number_of_points = 100
    else:
        number_of_points = kwargs['number_of_points']

    phases = np.linspace(start_phase, stop_phase, number_of_points)
    ellipse = binary_system.BinarySystem.orbit.orbital_motion(phase=phases)
    radius = ellipse[:, 0]
    azimut = ellipse[:, 1]
    x, y = utils.polar_to_cartesian(radius=radius, phi=azimut - c.PI / 2)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(x,y)
    ax.scatter(x[0], y[0], c='r')
    ax.scatter(0, 0, c='b')
    ax.set_aspect('equal')
    ax.grid()

    plt.show()