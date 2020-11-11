from astropy import units as u
import numpy as np

from elisa.binary_system.system import BinarySystem
from elisa.base.star import Star
from elisa.observer.observer import Observer

from matplotlib import pyplot as plt
from decimal import Decimal


# config.LOG_CONFIG = '/home/miro/ELISa/my_logging.json'

# config.POINTS_ON_ECC_ORBIT = 90
# config.MAX_RELATIVE_D_R_POINT = 0.003
# NUMBER_OF_PROCESSES = 4

DISCRETIZATION = [10, 7, 5, 3]
lw = [0.4, 0.7, 1.0, 1.3]
# DISCRETIZATION = [5, 3]

# pot = 115.2
pot = 530
primary = Star(
    mass=1.0 * u.solMass,
    surface_potential=pot,
    synchronicity=1.0,
    t_eff=5774 * u.K,
    gravity_darkening=0.32,
    discretization_factor=5,  # angular size (in degrees) of the surface elements
    albedo=0.6,
    metallicity=0.0,
)

secondary = Star(
    mass=1.0 * u.solMass,
    surface_potential=pot,
    # surface_potential=50.0,
    synchronicity=1.0,
    t_eff=5774 * u.K,
    gravity_darkening=0.32,
    # discretization_factor=20,
    albedo=0.6,
    metallicity=0,
)

# setattr(primary, "_mass", None)
bs = BinarySystem(
    primary=primary,
    secondary=secondary,
    argument_of_periastron=0 * u.deg,
    gamma=0 * u.km / u.s,
    period=1000 * u.d,
    eccentricity=0.0,
    inclination=90 * u.deg,
    primary_minimum_time=0.0 * u.d,
    phase_shift=0.0,
)

print((bs.primary.polar_radius * bs.semi_major_axis * u.m).to(u.solRad))
print(bs.secondary.polar_radius * bs.semi_major_axis * u.m.to(u.solRad))

lcs = {}
fig = plt.figure()
for ii, df in enumerate(DISCRETIZATION):
    bs.primary.discretization_factor = np.radians(df)
    bs.init()

    o = Observer(passband=[  # defining passbands at which calculate a light curve
        # 'Generic.Bessell.U',
        # 'Generic.Bessell.B',
        # 'Generic.Bessell.V',
        # 'Generic.Bessell.R',
        # 'Generic.Bessell.I',
        'bolometric'
    ],
    system=bs)  # specifying the binary system to use in light curve synthesis

    phases, curves = o.lc(
        from_phase=0.001,
        to_phase=0.499,
        phase_step=0.001,
        # phase_step=0.01,
        normalize=True,
    )
    lcs[df] = curves['bolometric']
    lcs[df] -= np.mean(lcs[df])

    std = lcs[df].std()
    sigma = r'$sigma$'
    lbl = f'discretization: {df}; $\sigma$ = {std:.1E}'
    plt.plot(phases, lcs[df], label=lbl, linewidth=lw[ii])
    print(f'Standard deviations for discretization factor {df}: {std:.1E}')

plt.xlabel('Phase')
plt.ylabel('Residual flux')
plt.subplots_adjust(hspace=0, right=0.98, top=0.98, left=0.155)
plt.legend()
plt.show()


# filename = '/home/miro/Documents/astrofyzika/KOLOS/2019/data/clear.txt'
# # filename = '/home/miro/Documents/astrofyzika/KOLOS/2019/data/pulsations.txt'
# with open(filename, 'wb') as f:
#     pickle.dump([phases, curves], f)

# bs.plot.surface(
#     phase=0.2,
#     plot_axis=False,
#     colormap='temperature',
#     # colormap='gravity_acceleration',
#     edges=True,
#     components_to_plot='primary',
#     )
