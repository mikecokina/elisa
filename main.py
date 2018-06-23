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

spots_metadata = {
    "primary":
         [
         {"longitude": 90,
          "latitude": 58,
          # "angular_density": 1,
          "angular_diameter": 5,
          "temperature_factor": 0.9},
         {"longitude": 70,
          "latitude": 80,
          # "angular_density": 2,
          "angular_diameter": 30,
          "temperature_factor": 1.05},
         {"longitude": 45,
          "latitude": 90,
          # "angular_density": 2,
          "angular_diameter": 30,
          "temperature_factor": 0.95},
         ],

    "secondary":
        [
         {"longitude": 0,
          "latitude": 40,
          # "angular_density": 3,
          "angular_diameter": 30,
          "temperature_factor": 1.03},
         {"longitude": 45,
          "latitude": 3,
          # "angular_density": 3,
          "angular_diameter": 10,
          "temperature_factor": 0.98}
         ]
     }

pulsations_metadata = {'primary': [{'l': 4, 'm': 3, 'amplitude': 1000*u.K, 'frequency': 15/u.d},
                                   # {'l': 3, 'm': 2, 'amplitude': 50*u.K, 'frequency': 20/u.d},
                                   ],
                       'secondary': [{'l': 5, 'm': 5, 'amplitude': 3000*u.K, 'frequency': 15/u.d},
                                     ]
                       }

start_time = time()
primary = Star(mass=1.5*u.solMass,
               surface_potential=3.15,
               synchronicity=1.0,
               t_eff=7000*u.K,
               gravity_darkening=1.0,
               discretization_factor=3,
               spots=spots_metadata['primary'],
               pulsations=pulsations_metadata['primary']
               )
secondary = Star(mass=0.9*u.solMass,
                 surface_potential=3.15,
                 synchronicity=1.2,
                 t_eff=6000*u.K,
                 gravity_darkening=0.32,
                 discretization_factor=3,
                 spots=spots_metadata['secondary'],
                 pulsations=pulsations_metadata['secondary']
                 )

bs = BinarySystem(primary=primary,
                  secondary=secondary,
                  argument_of_periastron=90*u.deg,
                  gamma=0*u.km/u.s,
                  period=1*u.d,
                  eccentricity=0.0,
                  inclination=90*u.deg,
                  primary_minimum_time=0.0*u.d,
                  phase_shift=0.0)


# bs.build_mesh()
# bs.build_faces()
# bs.build_temperature_distribution(components_distance=1.0)
# bs.evaluate_normals()


print('Elapsed time: {0:.5f} s.'.format(time() - start_time))

# bs.plot('orbit', frame_of_reference='barycentric')
# bs.plot('equipotential', plane="zx", phase=bs.orbit.periastron_phase)

# bs.plot(descriptor='mesh',
#         # components_to_plot='primary',
#         # components_to_plot='secondary',
#         plot_axis=False
#         )
# bs.plot(descriptor='wireframe',
        # components_to_plot='primary',
        # components_to_plot='secondary',
        # plot_axis=False
        # )
bs.plot(descriptor='surface',
        phase=0,
        # components_to_plot='primary',
        components_to_plot='secondary',
        # edges=True,
        # normals=True,
        # colormap='gravity_acceleration',
        colormap='temperature',
        # plot_axis=False,
        )
