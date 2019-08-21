from elisa.binary_system.system import BinarySystem
from elisa.base.star import Star
from elisa.conf import config
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from time import time
import logging
from elisa.observer.observer import Observer

logger = logging.getLogger()
# logger.setLevel(level='WARNING')
# logger.setLevel(level='DEBUG')
logger.setLevel(level='INFO')
# contact_pot = 2.8
contact_pot = 4
start_time = time()

spots_metadata = {
    "primary":
        [
            {"longitude": 0,
             "latitude": 45,
             # "angular_density": 1,
             "angular_diameter": 58,
             "temperature_factor": 1.01},
            # {"longitude": 200,
            #  "latitude": 100,
            #  # "angular_density": 2,
            #  "angular_diameter": 30,
            #  "temperature_factor": 0.97},
            # {"longitude": 60,
            #  "latitude": 90,
            #  # "angular_density": 2,
            #  "angular_diameter": 30,
            #  "temperature_factor": 0.7},
        ],

    "secondary":
        [
            {"longitude": 0,
             "latitude": 45,
             # "angular_density": 3,
             "angular_diameter": 28,
             "temperature_factor": 1.06},
            # {"longitude": 30,
            #  "latitude": 65,
            #  # "angular_density": 3,
            #  "angular_diameter": 45,
            #  "temperature_factor": 0.5},
        ]
}

pulsations_metadata = {
    'primary': [
        {
            'l': 5,
            'm': 5,
            'amplitude': 300 * u.K,
            'frequency': 3 / u.d,
            'start_phase': 0.2,
            'mode_axis_theta': 30 * u.deg,
            'mode_axis_phi': 0 * u.deg,
        },
        # {
        #     'l': 5,
        #     'm': -5,
        #     'amplitude': 300 * u.K,
        #     'frequency': 3 / u.d,
        #     'start_phase': 0.2,
        #     'mode_axis_theta': 30*u.deg,
        #     'mode_axis_phi': 0*u.deg,
        # },
    ],
    'secondary': [
        {
            'l': 4,
            'm': 4,
            'amplitude': 1000 * u.K,
            'frequency': 5 / u.d,
            'start_phase': 0
        }
    ]
}

star_time = time()
primary = Star(
    mass=1.514 * u.solMass,
    surface_potential=contact_pot,
    # synchronicity=2.0,
    synchronicity=1.0,
    # t_eff=8000 * u.K,
    t_eff=10000 * u.K,
    gravity_darkening=1.0,
    discretization_factor=5,
    albedo=0.6,
    metallicity=0.0,
    # spots=spots_metadata['primary'],
    # pulsations=pulsations_metadata['primary']
)
secondary = Star(
    mass=0.327 * u.solMass,
    # mass=1.0 * u.solMass,
    surface_potential=contact_pot,
    # synchronicity=2.0,
    synchronicity=1.0,
    t_eff=8000 * u.K,
    # t_eff=4000 * u.K,
    gravity_darkening=1.0,
    albedo=0.6,
    metallicity=0,
    # discretization_factor=5,
    # spots=spots_metadata['secondary'],
    # pulsations=pulsations_metadata['secondary']
)

bs = BinarySystem(
    primary=primary,
    secondary=secondary,
    argument_of_periastron=58 * u.deg,
    gamma=-41.7 * u.km / u.s,
    # period=0.7949859 * u.d,
    period=1.0 * u.d,
    # eccentricity=0.0,
    eccentricity=0.1,
    inclination=85 * u.deg,
    primary_minimum_time=2440862.60793 * u.d,
    phase_shift=0.0,
)

# components_min_distance = 1
print('Elapsed time during system build: {:.6f}'.format(time() - star_time))

star_time = time()
# bs.build(components_distance=1.0)

compute_curve = True
# compute_curve = False

plot_curve = True
# plot_curve = False

save_curves = True
# save_curves = False

# plot_surface = True
plot_surface = False

# animation = True
animation = False

if compute_curve:
    o = Observer(passband=['Generic.Bessell.V',
                           # 'Generic.Bessell.B',
                           # 'Generic.Bessell.U',
                           # 'Generic.Bessell.R',
                           # 'Generic.Bessell.I',
                           ],
                 system=bs)

    start_phs = -0.6
    stop_phs = 0.6
    # step = 0.1
    step = 1.0 / 100

    star_time = time()
    # config.MAX_RELATIVE_D_R_POINT = 0.005
    # config.LIMB_DARKENING_LAW = 'square_root'
    # config.LIMB_DARKENING_LAW = 'logarithmic'
    config.LIMB_DARKENING_LAW = 'linear'
    curves = o.observe(from_phase=start_phs,
                       to_phase=stop_phs,
                       phase_step=step,
                       )

    print('Elapsed time for LC gen: {:.6f}'.format(time() - star_time))

    # star_time = time()
    # # config.MAX_RELATIVE_D_R_POINT = 0.0
    # # config.POINTS_ON_ECC_ORBIT = 2000
    # config.LIMB_DARKENING_LAW = 'logarithmic'
    # curves2 = o.observe(from_phase=start_phs,
    #                     to_phase=stop_phs,
    #                     phase_step=step,
    #                    )
    #
    # print('Elapsed time for LC gen: {:.6f}'.format(time()-star_time))

    # print(utils.CUMULATIVE_TIME)

    if plot_curve:
        x = np.linspace(start_phs, stop_phs, int(round((stop_phs - start_phs) / step, 0)))
        for item in curves:
            # plt.plot(x, curves2[item]/max(curves[item]), label=item+' sqrt')
            # plt.plot(x, curves[item]/max(curves[item]), label=item)
            # plt.plot(x, curves2[item], label=item + 'sqrt')
            plt.plot(x, curves[item], label=item)
            plt.legend()
            plt.show()

if plot_surface:
    # bs.plot.orbit()
    bs.plot.surface(
        phase=0.48,
        # components_to_plot='primary',
        # components_to_plot='secondary',
        # edges=True,
        # normals=True,
        # colormap='gravity_acceleration',
        colormap='temperature',
        # plot_axis=False,
        # face_mask_primary=a,
        # face_mask_secondary=b,
        # inclination=crit_incl,
        # azimuth=azim[0],
        units='log_cgs',
        axis_unit=u.solRad,
        colorbar_orientation='horizontal',
    )

if animation:
    bs.animation.orbital_motion(
        start_phase=-0.0,
        stop_phase=0.2,
        # phase_step=0.01,
        phase_step=0.005,
        colormap='temperature',
        # savepath='/home/miro/motion.gif'
    )
