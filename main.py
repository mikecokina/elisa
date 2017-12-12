from engine.binary_system import BinarySystem
from engine.single_system import SingleSystem
from engine.star import Star
from engine.planet import Planet
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from engine import utils
from engine import const as c

# bs = BinarySystem(gamma=25, period=10.0, eccentricity=0.2)

# primary = Star(mass=2.0, surface_potential=2.5772602683306705, synchronicity=1.0)
# secondary = Star(mass=1.0, surface_potential=2.5772602683306705, synchronicity=1.0)
primary = Star(mass=2.0, surface_potential=2.5, synchronicity=1.0)
secondary = Star(mass=1.0, surface_potential=2.5, synchronicity=1.0)
# ur_anus = Planet(mass=500.2)
#
# s = SingleSystem(star=primary,
#                  gamma=0*u.km/u.s,
#                  inclination=85*u.deg)
#
# s.plot(descriptor='equipotential')

bs = BinarySystem(primary=primary,
                  secondary=secondary,
                  argument_of_periastron=315*u.deg,
                  gamma=0*u.km/u.s,
                  period=1.0*u.d,
                  eccentricity=0.0,
                  inclination=90*u.deg,
                  primary_minimum_time=0.0*u.d,
                  phase_shift=0.0)

# bs.plot('orbit', frame_of_reference='primary_component')

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
bs.plot('equipotential', plane="zx")
# print(bs.lagrangian_points())
