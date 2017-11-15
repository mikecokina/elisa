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
bs.phase_shift = 0.1
bs.eccentricity = 0.4
bs.periastron = 0
bs.phase_shift = 0.2
phs = np.linspace(0, 1, 100)
position = bs.orbit.orbital_motion(phase=phs)

theta = position[:, 1]
r = position[:, 0]

ax = plt.subplot(111, projection='polar')
ax.scatter(theta, r)
ax.scatter(theta[0], r[0], c='r')
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.grid(True)

plt.show()
