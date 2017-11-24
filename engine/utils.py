import numpy as np


def polar_to_cartesian(radius=None, phi=None):
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    return x,y