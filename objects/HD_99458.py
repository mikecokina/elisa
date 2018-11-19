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
            {"longitude": 297,
             "latitude": 130,
             # "angular_density": 1,
             "angular_diameter": 2*88,
             "temperature_factor": 0.981},
            {"longitude": 65,
             "latitude": 18.5,
             # "angular_density": 2,
             "angular_diameter": 2*48,
             "temperature_factor": 0.944}
        ],

    "secondary":
        [
            # {"longitude": 10,
            #  "latitude": 45,
            #  # "angular_density": 3,
            #  "angular_diameter": 28,
            #  "temperature_factor": 0.7}
        ]
}
start_time = time()
primary = Star(mass=2.15*u.solMass,
               surface_potential=3.6,
               synchronicity=1.0,
               t_eff=7600*u.K,
               gravity_darkening=1.0,
               discretization_factor=3,
               spots=spots_metadata['primary'])
secondary = Star(mass=0.45*u.solMass,
                 surface_potential=5.39,
                 synchronicity=1.0,
                 t_eff=7200*u.K,
                 # t_eff=5520*u.K,
                 gravity_darkening=0.32,
                 discretization_factor=5)


bs = BinarySystem(primary=primary,
                  secondary=secondary,
                  argument_of_periastron=0*u.deg,
                  gamma=-19.17*u.km/u.s,
                  period=2.7221720400*u.d,
                  eccentricity=0.0,
                  inclination=73.22*u.deg,
                  primary_minimum_time=0.0*u.d,
                  phase_shift=0.0)

print('Elapsed time: {0:.5f} s.'.format(time() - start_time))

crit_primary_potential = bs.critical_potential('primary', 1)
print('Critical potential for primary component: {}'.format(crit_primary_potential))

crit_secondary_potential = bs.critical_potential('secondary', 1)
print('Critical potential for secondary component: {}'.format(crit_secondary_potential))

print('Primary polar radius: {}'.format(primary.polar_radius))
print('Secondary polar radius: {}'.format(secondary.polar_radius))
print('Ratio of radii R2/R1: {}'.format(secondary.polar_radius/primary.polar_radius))

# bs.plot('orbit', frame_of_reference='barycentric')
# bs.plot('equipotential', plane="zx", phase=bs.orbit.periastron_phase)

# bs.plot(descriptor='mesh', components_to_plot='both')
bs.plot(descriptor='surface',
        phase=0,
        components_to_plot='primary',
        # components_to_plot='secondary',
        # plot_axis=False,
        # edges=True,
        # normals=True,
        # colormap='gravity_acceleration',
        colormap='temperature'
        )
