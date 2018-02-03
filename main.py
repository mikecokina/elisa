from engine.binary_system import BinarySystem
from engine.single_system import SingleSystem
from engine.star import Star
from engine.planet import Planet
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from engine import utils
from engine import const as c


primary = Star(mass=1.5, surface_potential=8.0, synchronicity=1.0)
secondary = Star(mass=1.0, surface_potential=8.0, synchronicity=1.0)

bs = BinarySystem(primary=primary,
                  secondary=secondary,
                  argument_of_periastron=90*u.deg,
                  gamma=0*u.km/u.s,
                  period=1*u.d,
                  eccentricity=0.0,
                  inclination=90*u.deg,
                  primary_minimum_time=0.0*u.d,
                  phase_shift=0.0)

phase = 0
pc = bs.critical_potential(component="primary", phase=phase)
sc = bs.critical_potential(component="secondary", phase=phase)

print("[{0:0.15f}, {1:0.15f}]".format(pc, sc))

# bs.plot('equipotential', plane="xy", phase=bs.orbit.periastron_phase)

# bs.argument_of_periastron = 135*u.deg
# bs.eccentricity = 0.3
# bs.inclination = 85*u.deg
# bs.init()

# plt.scatter(neck[:, 0], neck[:, 1])
# plt.plot(neck[:, 0], fit, c='r')
# plt.show()
#
# print(bs.critical_potential(component='primary', phase=0))
# print(bs.critical_potential(component='secondary', phase=0))
# bs.plot('orbit', frame_of_reference='barycentric')
# bs.plot('equipotential', plane="zx", phase=bs.orbit.periastron_phase)
bs.plot(descriptor='mesh', phase=0, components_to_plot='primary')
