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


contact_pot = 4.0
start_time = time()

primary = Star(mass=1.514*u.solMass,
               surface_potential=contact_pot,
               synchronicity=1.0,
               t_eff=6900*u.K,
               gravity_darkening=1.0,
               discretization_factor=3,
               albedo=0.6,
               metallicity=0
               )
secondary = Star(mass=0.327*u.solMass,
                 surface_potential=contact_pot,
                 synchronicity=1.0,
                 t_eff=6969*u.K,
                 gravity_darkening=1.0,
                 albedo=0.6,
                 metallicity=0
                )

bs = BinarySystem(primary=primary,
                  secondary=secondary,
                  argument_of_periastron=0*u.deg,
                  gamma=-41.7*u.km/u.s,
                  period=0.7949859*u.d,
                  eccentricity=0.0,
                  inclination=86.39*u.deg,
                  primary_minimum_time=2440862.60793*u.d,
                  phase_shift=0.0,
                  )

components_min_distance = 1
kwargs = {'suppress_parallelism': False}
bs.build_surface(components_distance=components_min_distance, **kwargs)
# # bs.build_surface(components_distance=components_min_distance, component='primary')
# # bs.build_surface(components_distance=components_min_distance, component='secondary')
bs.build_surface_map(colormap='temperature', components_distance=components_min_distance)
# bs.build_surface_map(colormap='temperature', component='primary', components_distance=components_min_distance)
# bs.build_surface_map(colormap='temperature', component='secondary', components_distance=components_min_distance)

line_of_sight = np.array([1, 0, 0])

res = geo.darkside_filter(line_of_sight, primary.normals)
print(np.shape(res))
