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
        [{"longitude": 0,
          "latitude": 0,
          "angular_density": 2,
          "angular_diameter": 20,
          "temperature_factor": 0.9}],

    "secondary":
        [{"longitude": 0,
          "latitude": 20,
          "angular_density": 2,
          "angular_diameter": 20,
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
               surface_potential=3.0,
               synchronicity=1.0,
               t_eff=7000*u.K,
               gravity_darkening=1.0,
               spots=spots_metadata["primary"],
               discretization_factor=10)
secondary = Star(mass=1.0*u.solMass,
                 surface_potential=3.0,
                 synchronicity=1.0,
                 t_eff=6000*u.K,
                 gravity_darkening=0.32,
                 discretization_factor=5)

bs = BinarySystem(primary=primary,
                  secondary=secondary,
                  argument_of_periastron=90 * u.deg,
                  gamma=0 * u.km / u.s,
                  period=1 * u.d,
                  eccentricity=0.0,
                  inclination=90 * u.deg,
                  primary_minimum_time=0.0 * u.d,
                  phase_shift=0.0)

phase = 0
components_distance = bs.orbit.orbital_motion(phase=phase)[0][0]
pc = bs.critical_potential(component="primary", components_distance=components_distance)
sc = bs.critical_potential(component="secondary", components_distance=components_distance)
print('Critical potentials: {0}, {1}'.format(pc, sc))

component = 'primary'
phase = 0

# bs.build_mesh(component=component)
# bs.surface(component=component)
component = 'primary'
component_instance = getattr(bs, component)
component_instance.points = bs.mesh_over_contact(component=component)
idx = np.argmax(component_instance.points[:, 2])
component_instance.faces = bs.over_contact_surface(component=component)
component_instance.polar_radius = bs.calculate_polar_radius(component=component,
                                                            components_distance=components_distance)
component_instance.areas = component_instance.calculate_areas()
component_instance.potential_gradients = bs.calculate_potential_gradient(component=component,
                                                                         components_distance=components_distance)
component_instance.polar_potential_gradient = bs.calculate_polar_potential_gradient(component=component,
                                                                                    components_distance=
                                                                                    components_distance)
# component_instance.temperatures = component_instance.calculate_effective_temperatures()

# print(component_instance.temperatures)

# print(bs.morphology)
# print("[{0:0.15f}, {1:0.15f}]".format(pc, sc))

# primary.normals = primary.calculate_normals()
# print(primary.normals)
# print(np.linalg.norm(primary.normals, axis=1))

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

# print('Elapsed time: {0:.5f} s.'.format(time() - start_time))

# bs.plot('orbit', frame_of_reference='barycentric')
# bs.plot('equipotential', plane="zx", phase=bs.orbit.periastron_phase)

# bs.plot(descriptor='mesh', components_to_plot='primary')
bs.plot(descriptor='surface',
        phase=0,
        components_to_plot='primary',
        edges=True,
        normals=False)

