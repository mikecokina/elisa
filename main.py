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
        [{"longitude": 90,
          "latitude": 58,
          "angular_density": 1,
          "angular_diameter": 5,
          "temperature_factor": 0.9},
         {"longitude": 70,
          "latitude": 80,
          "angular_density": 2,
          "angular_diameter": 30,
          "temperature_factor": 1.05},
         {"longitude": 45,
          "latitude": 90,
          "angular_density": 2,
          "angular_diameter": 30,
          "temperature_factor": 0.95},
         ],

    "secondary":
        [{"longitude": 0,
          "latitude": 40,
          "angular_density": 3,
          "angular_diameter": 30,
          "temperature_factor": 1.1},
         {"longitude": 0,
          "latitude": 0,
          "angular_density": 1,
          "angular_diameter": 10,
          "temperature_factor": 0.9}
         ]
     }

start_time = time()
primary = Star(mass=1.5*u.solMass,
               surface_potential=3.15,
               synchronicity=1.0,
               t_eff=7000*u.K,
               gravity_darkening=1.0,
               spots=spots_metadata["primary"],
               discretization_factor=3)
secondary = Star(mass=1.0*u.solMass,
                 surface_potential=3.15,
                 synchronicity=1.0,
                 t_eff=6000*u.K,
                 gravity_darkening=0.32,
                 spots=spots_metadata["secondary"],
                 discretization_factor=3)

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
components_distance = bs.orbit.orbital_motion(phase=phase)[0][0]
pc = bs.critical_potential(component="primary", components_distance=components_distance)
sc = bs.critical_potential(component="secondary", components_distance=components_distance)
print('Critical potentials: {0}, {1}'.format(pc, sc))

component = 'primary'
phase = 0

# print(bs.primary.spots)
#
# bs.build_mesh(component=component)
# bs.evaluate_normals(component=component, component_distance=components_distance)
# bs.surface(component=component)
# component = 'secondary'
# bs.build_mesh(component=component)
# bs.evaluate_normals(component=component, component_distance=components_distance)
# bs.surface(component=component)

# if True:
#     component_instance = getattr(bs, component)
#     component_instance.points = bs.mesh_over_contact(component=component)
#     idx = np.argmax(component_instance.points[:, 2])
#     component_instance.faces = bs.over_contact_surface(component=component)
#     component_instance.polar_radius = bs.calculate_polar_radius(component=component,
#                                                                 components_distance=components_distance)
#     component_instance.areas = component_instance.calculate_areas()
#     component_instance.potential_gradients = bs.calculate_face_magnitude_gradient(component=component,
#                                                                                   components_distance=components_distance)
#     component_instance.polar_potential_gradient = \
#         bs.calculate_polar_potential_gradient_magnitude(component=component, components_distance=components_distance)
#     component_instance.temperatures = component_instance.calculate_effective_temperatures()
#     print(component_instance.temperatures)

# print(component_instance.temperatures)

# print(bs.morphology)
# print("[{0:0.15f}, {1:0.15f}]".format(pc, sc))

# bs.plot('equipotential', plane="xy", phase=bs.orbit.periastron_phase)

# bs.argument_of_periastron = 135*u.deg
# bs.eccentricity = 0.3
# bs.inclination = 85*u.deg
# bs.init()

print('Elapsed time: {0:.5f} s.'.format(time() - start_time))

# bs.plot('orbit', frame_of_reference='barycentric')
# bs.plot('equipotential', plane="zx", phase=bs.orbit.periastron_phase)

# bs.plot(descriptor='mesh', components_to_plot='both')
bs.plot(descriptor='surface',
        phase=0,
        # components_to_plot='primary',
        # components_to_plot='secondary',
        # edges=True,
        # normals=True,
        # colormap='gravity_acceleration')
        colormap='temperature')

