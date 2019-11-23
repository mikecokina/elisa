from astropy import units as u
from time import time

from elisa.conf import config
config.LOG_CONFIG = '/home/miro/ELISa/my_logging.json'
from elisa.binary_system.system import BinarySystem
from elisa.base.star import Star
from elisa.observer.observer import Observer


# config.POINTS_ON_ECC_ORBIT = 90
config.MAX_RELATIVE_D_R_POINT = 0.003

spots_primary = [
    #  Spot 1
    {"longitude": 0,
     "latitude": 45,
     "angular_radius": 30,
     "temperature_factor": 1.05,
     # "discretization_factor": 2,
    },
    #  Spot 2
    {"longitude": 30,
     "latitude": 30,
     "angular_radius": 15,
     "temperature_factor": 0.98,
    },
    #  Spot 3
    {"longitude": 40,
     "latitude": 50,
     "angular_radius": 15,
     "temperature_factor": 1.02,
    },
    #  Spot 4
    {"longitude": 0,
     "latitude": 50,
     "angular_radius": 8,
     "temperature_factor": 0.98,
    },
]

pulsations_primary = [
        {
            'l': 8,
            'm': 4,
            'amplitude': 30 * u.km/u.s,
            'frequency': 1.8 / u.d,
            'start_phase': 1.5,
            'mode_axis_theta': 30 * u.deg,
            'mode_axis_phi': 90 * u.deg,
        },
]

primary = Star(
    mass=5.15 * u.solMass,
    surface_potential=3.6,
    synchronicity=1.0,
    t_eff=10000 * u.K,
    gravity_darkening=1.0,
    # discretization_factor=5,  # angular size (in degrees) of the surface elements
    albedo=0.6,
    metallicity=0.0,
    pulsations=pulsations_primary,
    # spots=spots_primary
)

secondary = Star(
    mass=1.2 * u.solMass,
    surface_potential=4.0,
    synchronicity=1.0,
    t_eff=7000 * u.K,
    gravity_darkening=1.0,
    # discretization_factor=20,
    albedo=0.6,
    metallicity=0,
    pulsations=pulsations_primary,
)

bs = BinarySystem(
    primary=primary,
    secondary=secondary,
    argument_of_periastron=58 * u.deg,
    gamma=-30.7 * u.km / u.s,
    period=2.5 * u.d,
    eccentricity=0.2,
    inclination=85 * u.deg,
    primary_minimum_time=2440000.0 * u.d,
    phase_shift=0.0,
)

print(config.CONFIG_FILE)
# bs.plot.orbit(
#     start_phase=0.0,
#     stop_phase=1.0,
#     number_of_points=100,
#     axis_units=u.solRad,
#     frame_of_reference='barycentric'
# )

# bs.plot.equipotential(
#     plane='xy',
#     phase=-0.0,
# )

# bs.plot.mesh(
#     phase=0.2,
#     # components_to_plot='primary',
#     # plot_axis=False,
#     inclination=70,
#     # azimuth=180*u.deg,
# )

# bs.plot.wireframe(
#     phase=0.0,
#     components_to_plot='primary',
#     plot_axis=False,
#     inclination=90,
#     azimuth=180*u.deg,
# )

bs.plot.surface(
    phase=0.2,
    # plot_axis=False,
    colormap='temperature',
    # colormap='gravity_acceleration',
    components_to_plot='primary',
    # components_to_plot='secondary',
    edges=True,
    # normals=True,
    # scale='log',
    units='SI',
    axis_unit=u.solRad,
    # inclination=90,
    # azimuth=180*u.deg,
)
