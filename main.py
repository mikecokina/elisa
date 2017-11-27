from engine.binary_system import BinarySystem
from engine.star import Star
from engine.planet import Planet
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from engine import utils
from engine import const as c

# bs = BinarySystem(gamma=25, period=10.0, eccentricity=0.2)

primary = Star(mass=2.0, surface_potential=5.0, synchronicity=1.0)
secondary = Star(mass=1.0, surface_potential=5.0, synchronicity=1.0)
ur_anus = Planet(mass=500.2)

bs = BinarySystem(primary=primary,
                  secondary=secondary,
                  argument_of_periastron=0*u.deg,
                  gamma=0*u.km/u.s,
                  period=1.0*u.d,
                  eccentricity=0.6,
                  inclination=85*u.deg,
                  primary_minimum_time=0.0*u.d,
                  phase_shift=0.0)

# bs.argument_of_periastron = 135*u.deg
# bs.eccentricity = 0.3
# bs.inclination = 85*u.deg
# bs.init()
#
# phases = np.linspace(0.0,1.0,100)
# ellipse = bs.orbit.orbital_motion(phase=phases)
# radius = ellipse[:, 0]
# azimut = ellipse[:, 1]
# x, y = utils.polar_to_cartesian(radius=radius, phi=azimut-c.PI/2)
#
# f = plt.figure()
# ax = f.add_subplot(111)
# ax.plot(x,y)
# ax.scatter(x[0], y[0], c='r')
# ax.scatter(0, 0, c='b')
# ax.set_aspect('equal')
# ax.grid()
#
# plt.show()

# xs = np.arange(-1, 1, 0.001)
# ys = [[x, bs.secondary_potential_derivation_x(x, *(1.0, 1.0))] for x in xs
#       if abs(bs.secondary_potential_derivation_x(x, *(1.0, 1.0))) < 100]
# plt.scatter(list(zip(*ys))[0], list(zip(*ys))[1])
# plt.show()

print(bs.critical_potential(target='primary', component_distance=1.0))

