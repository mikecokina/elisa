# from engine.binary_system import BinarySystem
# from engine.single_system import SingleSystem
# from engine.star import Star
# from engine.planet import Planet
# from astropy import units as u
# import numpy as np
# import matplotlib.pyplot as plt
# from engine import utils
# from engine import const as c
# from time import time
#
# # todo: spots order
# # spots_metadata = {
# #     "primary":
# #          [
# #          {"longitude": 90,
# #           "latitude": 58,
# #           "angular_density": 1,
# #           "angular_diameter": 5,
# #           "temperature_factor": 0.9},
# #          {"longitude": 70,
# #           "latitude": 80,
# #           "angular_density": 2,
# #           "angular_diameter": 30,
# #           "temperature_factor": 1.05},
# #          {"longitude": 45,
# #           "latitude": 90,
# #           "angular_density": 2,
# #           "angular_diameter": 30,
# #           "temperature_factor": 0.95},
# #          ],
# #
# #     "secondary":
# #         [
# #          {"longitude": 0,
# #           "latitude": 40,
# #           "angular_density": 2,
# #           "angular_diameter": 30,
# #           "temperature_factor": 1.01},
# #          {"longitude": 0,
# #           "latitude": 0,
# #           "angular_density": 1,
# #           "angular_diameter": 10,
# #           "temperature_factor": 0.98}
# #          ]
# #      }
# #
# # start_time = time()
# # primary = Star(mass=1.5*u.solMass,
# #                surface_potential=3.15,
# #                synchronicity=1.0,
# #                t_eff=7000*u.K,
# #                gravity_darkening=1.0,
# #                discretization_factor=5,
# #                spots=spots_metadata['primary'],
# #                )
# # secondary = Star(mass=0.9*u.solMass,
# #                  surface_potential=3.15,
# #                  synchronicity=1.2,
# #                  t_eff=6000*u.K,
# #                  gravity_darkening=0.32,
# #                  discretization_factor=5,
# #                  spots=spots_metadata['secondary'],
# #                  )
# #
# # bs = BinarySystem(primary=primary,
# #                   secondary=secondary,
# #                   argument_of_periastron=90*u.deg,
# #                   gamma=0*u.km/u.s,
# #                   period=1*u.d,
# #                   eccentricity=0.0,
# #                   inclination=90*u.deg,
# #                   primary_minimum_time=0.0*u.d,
# #                   phase_shift=0.0)
#
#
#
# spots_metadata = {
#     "primary":
#          [
#          {"longitude": 90,
#           "latitude": 58,
#           "angular_density": 1,
#           "angular_diameter": 5,
#           "temperature_factor": 0.9},
#          {"longitude": 70,
#           "latitude": 80,
#           "angular_density": 2,
#           "angular_diameter": 30,
#           "temperature_factor": 1.05},
#          {"longitude": 45,
#           "latitude": 90,
#           "angular_density": 2,
#           "angular_diameter": 30,
#           "temperature_factor": 0.95},
#          ],
#
#     "secondary":
#         [
#          {"longitude": 0,
#           "latitude": 40,
#           "angular_density": 3,
#           "angular_diameter": 30,
#           "temperature_factor": 1.1},
#          {"longitude": 0,
#           "latitude": 0,
#           "angular_density": 1,
#           "angular_diameter": 10,
#           "temperature_factor": 0.98}
#          ]
#      }
#
# start_time = time()
# primary = Star(mass=1.5*u.solMass,
#                surface_potential=3.15,
#                synchronicity=1.0,
#                t_eff=7000*u.K,
#                gravity_darkening=1.0,
#                discretization_factor=2,
#                spots=spots_metadata['primary'],
#                )
# secondary = Star(mass=0.9*u.solMass,
#                  surface_potential=3.15,
#                  synchronicity=1.2,
#                  t_eff=6000*u.K,
#                  gravity_darkening=0.32,
#                  discretization_factor=3,
#                  spots=spots_metadata['secondary'],
#                  )
#
# bs = BinarySystem(primary=primary,
#                   secondary=secondary,
#                   argument_of_periastron=90*u.deg,
#                   gamma=0*u.km/u.s,
#                   period=1*u.d,
#                   eccentricity=0.0,
#                   inclination=90*u.deg,
#                   primary_minimum_time=0.0*u.d,
#                   phase_shift=0.0)
#
# bs.plot(descriptor='surface',
#         phase=0,
#         # components_to_plot='primary',
#         components_to_plot='secondary',
#         edges=True,
#         # normals=True,
#         # colormap='gravity_acceleration',
#         colormap='temperature',
#         plot_axis=False,
#         )
#
#
# # bs.build_mesh()
# # bs.build_faces()
# # bs.build_temperature_distribution(components_distance=1.0)
# # bs.evaluate_normals()
#
#
# print('Elapsed time: {0:.5f} s.'.format(time() - start_time))
#
# # bs.plot('orbit', frame_of_reference='barycentric')
# # bs.plot('equipotential', plane="zx", phase=bs.orbit.periastron_phase)
#
# # bs.plot(descriptor='mesh',
#         # components_to_plot='primary',
#         # components_to_plot='secondary',
#         # plot_axis=False
#         # )
# # bs.plot(descriptor='wireframe',
#         # components_to_plot='primary',
#         # components_to_plot='secondary',
#         # plot_axis=False
#         # )
# bs.plot(descriptor='surface',
#         phase=0,
#         # components_to_plot='primary',
#         # components_to_plot='secondary',
#         # edges=True,
#         # normals=True,
#         # colormap='gravity_acceleration',
#         colormap='temperature',
#         plot_axis=False,
#         )


from engine.binary_system import BinarySystem
from engine.single_system import SingleSystem
from engine.star import Star
from engine.planet import Planet
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from engine import utils
from engine import const as c
from engine import units as U
from time import time


spots_metadata = [{"longitude": 30,
                   "latitude": 60,
                   "angular_density": 5,
                   "angular_diameter": 40,
                   "temperature_factor": 0.99},
                  # {"longitude": 70,
                  #  "latitude": 80,
                  #  "angular_density": 5,
                  #  "angular_diameter": 10,
                  #  "temperature_factor": 1.01},
                  {"longitude": 0,
                   "latitude": 45,
                   "angular_density": 5,
                   "angular_diameter": 30,
                   "temperature_factor": 0.995},
                  ]
#
# pulsations = [{'n': 1, 'l': 1, 'm': 1, 'amplitude': 50*u.K}]
#
start_time = time()
s = Star(mass=1.0*u.solMass,
         t_eff=5774*u.K,
         gravity_darkening=0.32,
         discretization_factor=5,
         spots=spots_metadata,
         )

single = SingleSystem(star=s,
                      gamma=0*u.km/u.s,
                      inclination=90*u.deg,
                      rotation_period=1.0*u.d,
                      polar_log_g=4.1*u.dex(u.cm/u.s**2))

single.gamma = 55*u.km/u.s
single.init()

single.build_mesh()
single.build_faces()

# print(s.equatorial_radius*U.DISTANCE_UNIT.to(u.solRad))
#
# s.points = single.mesh(alpha=single.discretization_factor)
# s.faces = single.surface(vertices=s.points)
# grad = single.calculate_potential_gradient()
# polar_grad = single.calculate_polar_potential_gradient()
# s.areas = s.calculate_areas()
# s.normals = s.calculate_normals(points=s.points, faces=s.faces)

# print(single.polar_radius*U.DISTANCE_UNIT.to(u.solRad))
# print(single.critical_break_up_radius()*U.DISTANCE_UNIT.to(u.solRad))
print(s.polar_gravity_acceleration)

print('Elapsed time: {0:.5f} s.'.format(time() - start_time))

# single.plot(descriptor='mesh', plot_axis=False)
# single.plot(descriptor='wireframe', plot_axis=False)

single.plot(descriptor='surface',
            # normals=True,
            edges=True,
            colormap='temperature',
            # colormap='gravity_acceleration',
            plot_axis=False,
            )
