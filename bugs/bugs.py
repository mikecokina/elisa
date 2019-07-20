from elisa.engine.binary_system.system import BinarySystem
from elisa.engine.base.star import Star
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from elisa.engine import utils
from elisa.engine import const as c
from time import time
import logging
from elisa.engine.binary_system import geo
from elisa.engine.observer.observer import Observer

logger = logging.getLogger()
logger.setLevel(level='WARNING')
# logger.setLevel(level='DEBUG')
# contact_pot = 2.5
contact_pot = 4
start_time = time()

spots_metadata = {
    "primary":
        [
            {"longitude": 90,
             "latitude": 58,
             # "angular_density": 1,
             "angular_diameter": 5,
             "temperature_factor": 0.98},
            # {"longitude": 90,
            #  "latitude": 57,
            #  # "angular_density": 2,
            #  "angular_diameter": 30,
            #  "temperature_factor": 0.65},
            # {"longitude": 60,
            #  "latitude": 90,
            #  # "angular_density": 2,
            #  "angular_diameter": 30,
            #  "temperature_factor": 0.7},
        ],

    "secondary":
        [
            {"longitude": 10,
             "latitude": 45,
             # "angular_density": 3,
             "angular_diameter": 28,
             "temperature_factor": 1.01},
            # {"longitude": 30,
            #  "latitude": 65,
            #  # "angular_density": 3,
            #  "angular_diameter": 45,
            #  "temperature_factor": 0.5},
        ]
}

star_time = time()
primary = Star(mass=1.514*u.solMass,
               surface_potential=contact_pot,
               synchronicity=1.0,
               t_eff=5000*u.K,
               gravity_darkening=1.0,
               discretization_factor=7,
               albedo=0.6,
               metallicity=0,
               spots=spots_metadata['primary'],
               )
secondary = Star(mass=0.327*u.solMass,
                 surface_potential=contact_pot,
                 synchronicity=1.0,
                 t_eff=1000*u.K,  # tu malo vypisat ze si mimo teplot, pouzivam sqrt limb-darkening
                 gravity_darkening=1.0,
                 albedo=0.6,
                 metallicity=0,
                 # spots=spots_metadata['secondary'],
                )

bs = BinarySystem(primary=primary,
                  secondary=secondary,
                  argument_of_periastron=250*u.deg,
                  gamma=-41.7*u.km/u.s,
                  period=0.7949859*u.d,
                  eccentricity=0.0,
                  # eccentricity=0.1,
                  inclination=85*u.deg,
                  primary_minimum_time=2440862.60793*u.d,
                  phase_shift=0.0,
                  )

# components_min_distance = 1
print('Elapsed time during system build: {:.6f}'.format(time()-star_time))

star_time = time()
# bs.build(components_distance=1.0)

# o = Observer(passband=['Generic.Bessell.V',
#                        # 'Generic.Bessell.B',
#                        # 'Generic.Bessell.U',
#                        # 'Generic.Bessell.R',
#                        # 'Generic.Bessell.I',
#                        ],
#              system=bs)
#
# start_phs = -0.6
# stop_phs = 0.6
# # step = 0.1
# step = 0.01
# curves = o.observe(from_phase=start_phs,
#                    to_phase=stop_phs,
#                    phase_step=step,
#                   )
#
# print('Elapsed time for LC gen: {:.6f}'.format(time()-star_time))
#
# x = np.linspace(start_phs, stop_phs, int(round((stop_phs-start_phs)/step, 0)))
# for item in curves:
#     plt.plot(x, curves[item]/max(curves[item]), label=item)
#     # plt.plot(x, curves[item], label=item)
#     # plt.plot(x, curves[item], label=item)
# plt.legend()
# plt.show()

# bs.plot.orbit()
bs.plot.surface(
        phase=0.2,
        components_to_plot='primary',
        # components_to_plot='secondary',
        # edges=True,
        # normals=True,
        # colormap='gravity_acceleration',
        colormap='temperature',
        # plot_axis=False,
        # face_mask_primary=a,
        # face_mask_secondary=b,
        # inclination=crit_incl,
        # azimuth=azim[0],
        units='log_cgs'
        )
