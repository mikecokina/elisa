from engine.binary_system import BinarySystem
from engine.star import Star
from engine.planet import Planet
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt

# bs = BinarySystem(gamma=25, period=10.0, eccentricity=0.2)

primary = Star(mass=2.0, surface_potential=5.0)
secondary = Star(mass=1.0, surface_potential=5.0)
ur_anus = Planet(mass=500.2)

bs = BinarySystem(primary=primary, secondary=secondary)

