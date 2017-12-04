import matplotlib.pyplot as plt
from engine import binary_system
from engine import const as c
from engine import utils
import numpy as np

def orbit(**kwargs):
    x = kwargs['x_data']
    y = kwargs['y_data']

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(x,y)
    ax.scatter(x[0], y[0], c='r')
    ax.scatter(0, 0, c='b')
    ax.set_aspect('equal')
    ax.grid()

    plt.show()