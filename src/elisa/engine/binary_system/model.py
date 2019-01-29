import numpy as np


def static_potential_primary_fn(radius, *args):
    mass_ratio, surface_potential, b, c, d, e = args
    radius2 = np.power(radius, 2)
    a = 1 / radius + mass_ratio / np.sqrt(b + radius2 - c * radius) - d * radius + e * radius2
    return a - surface_potential


def static_potential_secondary_fn(radius, *args):
    mass_ratio, surface_potential, b, c, d, e, f = args
    radius2 = np.power(radius, 2)
    a = mass_ratio / radius + 1. / np.sqrt(b + radius2 - c * radius) - d * radius + e * radius2 + f
    return a - surface_potential
