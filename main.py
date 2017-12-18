from engine.binary_system import BinarySystem
from engine.single_system import SingleSystem
from engine.star import Star
from engine.planet import Planet
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from engine import utils
from engine import const as c


# primary = Star(mass=2.0, surface_potential=8.643878058931882, synchronicity=2.0)
# secondary = Star(mass=5.0, surface_potential=13.396945616139345, synchronicity=5.0)

single = Star(mass=2.0, surface_potential=1.48)

# ur_anus = Planet(mass=500.2)
#
s = SingleSystem(star=single,
                 gamma=0*u.km/u.s,
                 inclination=85*u.deg,
                 P_rot=1.2*u.d)

# s.plot(descriptor='equipotential')
print('Critical radius: {}'.format((s.critical_break_up_radius() * u.m).to(u.solRad)))
print('Critical velocity: {}'.format((s.critical_break_up_velocity() * u.m / u.s).to(u.km / u.s)))

# bs = BinarySystem(primary=primary,
#                   secondary=secondary,
#                   argument_of_periastron=41*u.deg,
#                   gamma=0*u.km/u.s,
#                   period=1.0*u.d,
#                   eccentricity=0.3,
#                   inclination=80*u.deg,
#                   primary_minimum_time=0.0*u.d,
#                   phase_shift=0.0)
#
#
# print(bs.libration_potentials())
# print(bs.orbit.periastron_phase)
# pc = bs.critical_potential(component="primary", phase=bs.orbit.periastron_phase)
# sc = bs.critical_potential(component="secondary", phase=bs.orbit.periastron_phase)
#
#
# print("{0:0.15f}".format(pc))
# print("{0:0.15f}".format(sc))
# print(bs.orbit.periastron_phase)

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
