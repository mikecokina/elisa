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
bs.argument_of_periastron = 45*u.deg
bs.eccentricity = 0.3
bs.inclination = 85*u.deg

print(bs.orbit.get_conjuction()['primary_eclipse'])

phases = np.linspace(0,0.9,100)
ellipse = bs.orbit.orbital_motion(phase=phases)
radius = ellipse[:, 0]
azimut = ellipse[:, 1]

ax = plt.subplot(111, projection='polar')
ax.plot(azimut, radius)
ax.scatter(azimut[0], radius[0], color='r')
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.grid(True)

plt.show()
