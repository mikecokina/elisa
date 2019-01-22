from engine.binary_system import BinarySystem
from engine.single_system import SingleSystem
from engine.star import Star
from engine.planet import Planet
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from engine import utils
from engine import const as c
from time import time
import logging

logging.basicConfig(level=logging.DEBUG)

from conf import config

from scipy.spatial import distance_matrix

spots_metadata = {
    "primary":
        [
            {"longitude": 90,
             "latitude": 58,
             # "angular_density": 1,
             "angular_diameter": 5,
             "temperature_factor": 0.50},
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
             "temperature_factor": 0.55},
            {"longitude": 30,
             "latitude": 65,
             # "angular_density": 3,
             "angular_diameter": 45,
             "temperature_factor": 0.5},
            # {"longitude": 45,
            #  "latitude": 40,
            #  # "angular_density": 3,
            #  "angular_diameter": 40,
            #  "temperature_factor": 0.80},
            # {"longitude": 50,
            #  "latitude": 55,
            #  # "angular_density": 3,
            #  "angular_diameter": 28,
            #  "temperature_factor": 0.85},
            # {"longitude": 25,
            #  "latitude": 55,
            #  # "angular_density": 3,
            #  "angular_diameter": 15,
            #  "temperature_factor": 0.9},
            # {"longitude": 0,
            #  "latitude": 70,
            #  # "angular_density": 3,
            #  "angular_diameter": 45,
            #  "temperature_factor": 0.95}
        ]
}

pulsations_metadata = {'primary': [{'l': 4, 'm': 3, 'amplitude': 1000 * u.K, 'frequency': 15 / u.d},
                                   # {'l': 3, 'm': 2, 'amplitude': 50*u.K, 'frequency': 20/u.d},
                                   ],
                       'secondary': [{'l': 5, 'm': 5, 'amplitude': 300 * u.K, 'frequency': 15 / u.d},
                                     ]
                       }

# contact_pot = 2.875844632141054
contact_pot = 3.3
start_time = time()

primary = Star(mass=2.0*u.solMass,
               # surface_potential=2.7,
               surface_potential=contact_pot,
               # spots=spots_metadata['primary'],
               # pulsations=pulsations_metadata['primary'],
               synchronicity=1.0,
               t_eff=10000*u.K,
               gravity_darkening=1.0,
               discretization_factor=3,
               albedo=0.6
               )
secondary = Star(mass=1.0*u.solMass,
                 # surface_potential=5.0,
                 surface_potential=contact_pot,
                 synchronicity=1.0,
                 t_eff=6800*u.K,
                 gravity_darkening=1.0,
                 # discretization_factor=5,
                 # spots=spots_metadata['secondary'],
                 # pulsations=pulsations_metadata['primary'],
                 albedo=0.6
                )

bs = BinarySystem(primary=primary,
                  secondary=secondary,
                  argument_of_periastron=90*u.deg,
                  gamma=0*u.km/u.s,
                  period=1*u.d,
                  eccentricity=0.0,
                  inclination=80*u.deg,
                  primary_minimum_time=0.0*u.d,
                  phase_shift=0.0,
                  )


components_min_distance = 1 - bs.eccentricity
start_time = time()
bs.build_surface(components_distance=components_min_distance)
# # bs.build_surface(components_distance=components_min_distance, component='primary')
# # bs.build_surface(components_distance=components_min_distance, component='secondary')
bs.build_surface_map(colormap='temperature', components_distance=components_min_distance)
# bs.build_surface_map(colormap='temperature', component='primary', components_distance=components_min_distance)
# bs.build_surface_map(colormap='temperature', component='secondary', components_distance=components_min_distance)
# bs.build_temperature_distribution(components_distance=1.0)
# bs.evaluate_normals()
# bs.build_surface(components_distance=1)
# azim = bs.get_eclipse_boundaries(components_distance=components_min_distance)
# azim = np.degrees(azim)
# crit_incl = bs.get_critical_inclination(components_distance=components_min_distance)

# t1 = np.min(bs.primary.temperatures)
# a, b = bs.reflection_effect(iterations=2, components_distance=1)
# t2 = np.min(bs.primary.temperatures)
# print(t1, t2)
# print(np.shape(a))
# dists, dist_vect = utils.calculate_distance_matrix(points1=bs.primary.points, points2=bs.secondary.points,
#                                                    return_distance_vector_matrix=True)
# print(np.shape(dists), np.shape(dist_vect))
# dists = distance_matrix(bs.primary.points, bs.secondary.points)
# print(np.shape(dists))

print('Elapsed time: {0:.5f} s.'.format(time() - start_time))
crit_primary_potential = bs.critical_potential('primary', components_distance=components_min_distance)
print('Critical potential for primary component: {}'.format(crit_primary_potential))

crit_secondary_potential = bs.critical_potential('secondary', components_distance=components_min_distance)
print('Critical potential for secondary component: {}'.format(crit_secondary_potential))

# bs.plot('orbit', frame_of_reference='primary_component', axis_unit='dimensionless')
# bs.plot('orbit', frame_of_reference='barycentric')
# bs.plot('equipotential', plane="zx", phase=bs.orbit.periastron_phase)

# bs.plot(descriptor='mesh',
#         # components_to_plot='primary',
#         components_to_plot='secondary',
#         plot_axis=False
#         )
# bs.plot(descriptor='wireframe',
#         # components_to_plot='primary',
#         components_to_plot='secondary',
#         # plot_axis=False
#         )

# bs.plot(descriptor='surface',
#         phase=0.4,
#         # components_to_plot='primary',
#         components_to_plot='secondary',
#         # edges=True,
#         # normals=True,
#         # colormap='gravity_acceleration',
#         colormap='temperature',
#         # plot_axis=False,
#         # face_mask_primary=a,
#         # face_mask_secondary=b,
#         # inclination=crit_incl,
#         # azimuth=azim[0],
#         )

