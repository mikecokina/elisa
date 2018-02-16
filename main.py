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
primary = Star(mass=1.5, surface_potential=3.0, synchronicity=1.0, spots=spots_metadata["primary"])
secondary = Star(mass=1.0, surface_potential=3.0, synchronicity=1.0)


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
pc = bs.critical_potential(component="primary", phase=phase)
sc = bs.critical_potential(component="secondary", phase=phase)

component = 'primary'
# component = 'secondary'
component_instance = getattr(bs, component)
component_instance.points = bs.mesh_over_contact(component=component, alpha=5)
idx = np.argmax(component_instance.points[:, 2])
component_instance.faces = bs.over_contact_surface(points=component_instance.points)
component_instance.polar_radius = bs.calculate_polar_radius(component=component, phase=0.0)

print(component_instance.points[idx, :])
print(component_instance.polar_radius)

# print(bs.morphology)
# print("[{0:0.15f}, {1:0.15f}]".format(pc, sc))

print(max(bs.calculate_potential_gradient(component=component, component_distance=1.0)))
print(bs.calculate_polar_potential_gradient(component=component, component_distance=1.0))
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
# bs.plot('orbit', frame_of_reference='barycentric')
# bs.plot('equipotential', plane="zx", phase=bs.orbit.periastron_phase)
# bs.plot(descriptor='surface', phase=0, components_to_plot='primary', alpha1=10, alpha2=10)

print('Elapsed time: {0:.5f} s.'.format(time() - start_time))