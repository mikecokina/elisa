from engine.binary_system import BinarySystem
from engine.single_system import SingleSystem
from engine.star import Star
from engine.planet import Planet
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from engine import utils
from engine import const as c


primary = Star(mass=3.0, surface_potential=4.55, synchronicity=2.5)
secondary = Star(mass=1.5, surface_potential=3.5, synchronicity=1.0)

bs = BinarySystem(primary=primary,
                  secondary=secondary,
                  argument_of_periastron=90*u.deg,
                  gamma=0*u.km/u.s,
                  period=1.0*u.d,
                  eccentricity=0.2,
                  inclination=80*u.deg,
                  primary_minimum_time=0.0*u.d,
                  phase_shift=0.0)

pc = bs.critical_potential(component="primary", phase=bs.orbit.periastron_phase)
sc = bs.critical_potential(component="secondary", phase=bs.orbit.periastron_phase)

print("{0:0.15f}".format(pc))
print("{0:0.15f}".format(sc))

# bs.plot('equipotential', plane="xy", phase=bs.orbit.periastron_phase)

# bs.argument_of_periastron = 135*u.deg
# bs.eccentricity = 0.3
# bs.inclination = 85*u.deg
# bs.init()

# xs = np.arange(-1, 1, 0.001)
# ys = [[x, bs.secondary_potential_derivation_x(x, *(1.0, 1.0))] for x in xs
#       if abs(bs.secondary_potential_derivation_x(x, *(1.0, 1.0))) < 100]
# plt.scatter(list(zip(*ys))[0], list(zip(*ys))[1])
# plt.show()
#
# print(bs.critical_potential(component='primary', phase=0))
# print(bs.critical_potential(component='secondary', phase=0))
# bs.plot('orbit', frame_of_reference='barycentric')
# bs.plot('equipotential', plane="zx", phase=bs.orbit.periastron_phase)
# print(bs.lagrangian_points())
bs.plot(descriptor='mesh', phase=0, components_to_plot='both', alpha1=5, alpha2=3)
