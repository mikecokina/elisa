from elisa.binary_system.system import BinarySystem
from elisa.base.star import Star
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from time import time
import logging
from elisa.observer.observer import Observer
from elisa.conf import settings
import matplotlib.gridspec as gridspec

"""script aims to test the precision of various aproximations taken during eccentric lc calculation compared to exact 
solution"""

logger = logging.getLogger()
# logger.setLevel(level='WARNING')
logger.setLevel(level='DEBUG')
# contact_pot = 2.5
contact_pot = 4
start_time = time()

# N = 10
N = 1

star_time = time()
primary = Star(mass=1.514 * u.solMass,
               surface_potential=contact_pot,
               synchronicity=1.0,
               t_eff=10000 * u.K,
               gravity_darkening=1.0,
               discretization_factor=5,
               albedo=0.6,
               metallicity=0,
               # spots=spots_metadata['primary'],
               )
secondary = Star(mass=0.327 * u.solMass,
                 surface_potential=contact_pot,
                 synchronicity=1.0,
                 t_eff=4000 * u.K,
                 gravity_darkening=1.0,
                 albedo=0.6,
                 metallicity=0,
                 # spots=spots_metadata['secondary'],
                 )

bs = BinarySystem(primary=primary,
                  secondary=secondary,
                  argument_of_periastron=320 * u.deg,
                  gamma=-41.7 * u.km / u.s,
                  period=0.7949859 * u.d,
                  eccentricity=0.25,
                  # eccentricity=0,
                  inclination=85 * u.deg,
                  primary_minimum_time=2440862.60793 * u.d,
                  phase_shift=0.0,
                  )

print('Elapsed time during system build: {:.6f}'.format(time() - star_time))


o = Observer(passband=['Generic.Bessell.V',
                       # 'Generic.Bessell.B',
                       # 'Generic.Bessell.U',
                       # 'Generic.Bessell.R',
                       # 'Generic.Bessell.I',
                       ],
             system=bs)

start_phs = -0.6
stop_phs = 0.6
step = 0.005
settings.POINTS_ON_ECC_ORBIT = 50

start_time = time()
for _ in range(N):
    curves_approx1 = o.lc(from_phase=start_phs,
                          to_phase=stop_phs,
                          phase_step=step,
                          )
interp_time = (time() - start_time) / N
print('Elapsed time for approx one LC gen: {:.6f}'.format(interp_time))

settings.POINTS_ON_ECC_ORBIT = 9999
settings.MAX_RELATIVE_D_R_POINT = 0.005
start_time = time()

for _ in range(N):
    curves_approx2 = o.lc(from_phase=start_phs,
                          to_phase=stop_phs,
                          phase_step=step,
                          )
sim_geom_time = (time() - start_time) / N
print('Elapsed time for approx two LC gen: {:.6f}'.format(sim_geom_time))


settings.MAX_RELATIVE_D_R_POINT = 1e-8

start_time = time()
for _ in range(N):
    curves_exact = o.lc(from_phase=start_phs,
                        to_phase=stop_phs,
                        phase_step=step,
                        )
exact_time = (time() - start_time) / N
print('Elapsed time for exact LC gen: {:.6f}'.format(exact_time))

x = np.linspace(start_phs, stop_phs, int(round((stop_phs-start_phs)/step, 0)))

fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

text = r'$t_{approx}/t_{exact}=$'
for item in curves_approx1[1]:
    y_approx1 = curves_approx1[1][item]/max(curves_approx1[1][item])
    y_approx2 = curves_approx2[1][item]/max(curves_approx2[1][item])
    y_exact = curves_exact[1][item]/max(curves_exact[1][item])
    time_frac1 = np.round(interp_time/exact_time, 2)
    time_frac2 = np.round(sim_geom_time/exact_time, 2)
    ax1.plot(x, y_approx1, label='interpolation, '+text+str(time_frac1))
    ax1.plot(x, y_approx2, label='similar geometry, '+text+str(time_frac2))
    ax1.plot(x, y_exact, label='exact')
    ax2.plot(x, y_exact-y_approx1, label='exact - interpolation')
    ax2.plot(x, y_exact-y_approx2, label='exact - similar geometry')

ax1.legend()
ax2.legend()
ax2.ticklabel_format(scilimits=(-3, 5))
ax1.set_ylabel('Flux')
ax2.set_xlabel('Phase')
ax2.set_ylabel('Residual flux')
plt.subplots_adjust(hspace=0, right=0.98, top=0.98)
plt.show()

# bs.plot.orbit()
