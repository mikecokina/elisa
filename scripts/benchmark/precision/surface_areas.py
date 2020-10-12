from elisa.binary_system.system import BinarySystem
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.base.star import Star
from elisa import units as u
from elisa import const

import matplotlib.pyplot as plt
import numpy as np


surface_potential = 3.1
primary = Star(
    mass=2.0 * u.solMass,
    surface_potential=surface_potential,
    synchronicity=1.0,
    t_eff=7000 * u.K,
    gravity_darkening=1.0,
    discretization_factor=5,  # angular size (in degrees) of the surface elements
    albedo=0.6,
    metallicity=0.0,
)

secondary = Star(
    mass=1.2 * u.solMass,
    surface_potential=surface_potential,
    synchronicity=1.0,
    t_eff=6000 * u.K,
    gravity_darkening=1.0,
    albedo=0.6,
    metallicity=0,
)

# setattr(primary, "_mass", None)
bs = BinarySystem(
    primary=primary,
    secondary=secondary,
    argument_of_periastron=90 * u.deg,
    gamma=100 * u.km / u.s,
    period=25 * u.d,
    eccentricity=0.0,
    inclination=85 * u.deg,
    primary_minimum_time=2440000.0 * u.d,
    phase_shift=0.0,
    additional_light=0.1
)

component = 'primary'
container = OrbitalPositionContainer.from_binary_system(binary_system=bs, position=const.Position(0, 1.0, 0, 0, 0))
container.build()
comp_instance = getattr(container, component)
areas = comp_instance.areas
percentiles = np.percentile(areas, [100-68.27, 50, 68.27])
print(percentiles)
print(f'Standard deviation in triangle sizes: +{100*(percentiles[2] - percentiles[1]) / percentiles[1]}, '
      f' -{100*(percentiles[1] - percentiles[0]) / percentiles[1]} %')

plt.hist(comp_instance.areas, bins=100)
plt.show()

bs.plot.surface(
    components_to_plot='primary',
    phase=0.8,
    inclination=75,
    edges=True,
    surface_colors=('gray', 'gray'),
                )
