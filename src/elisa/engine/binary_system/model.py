import numpy as np


def static_potential_primary_fn(radius: float, *args) -> float:
    mass_ratio, surface_potential, b, c, d, e = args
    radius2 = np.power(radius, 2)
    a = 1 / radius + mass_ratio / np.sqrt(b + radius2 - c * radius) - d * radius + e * radius2
    return a - surface_potential


def static_potential_secondary_fn(radius: float, *args) -> float:
    mass_ratio, surface_potential, b, c, d, e, f = args
    radius2 = np.power(radius, 2)
    a = mass_ratio / radius + 1. / np.sqrt(b + radius2 - c * radius) - d * radius + e * radius2 + f
    return a - surface_potential


def static_potential_primary_cylindrical_fn(radius: float, *args) -> float:
    mass_ratio, surface_potential, a, b, c, d, e, f = args
    radius2 = np.power(radius, 2)
    return 1 / np.sqrt(a + radius2) + mass_ratio / np.sqrt(b + radius2) - c + d * (e + f * radius2) - surface_potential


def static_potential_secondary_cylindrical_fn(radius: float, *args) -> float:
    mass_ratio, surface_potential, a, b, c, d, e, f = args
    radius2 = np.power(radius, 2)
    return mass_ratio / np.sqrt(a + radius2) + 1. / np.sqrt(b + radius2) + c * (d + e * radius2) + f - surface_potential
