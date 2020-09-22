import os
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from elisa.conf import settings
from elisa.binary_system.system import BinarySystem
from elisa.base.star import Star
from elisa.observer.observer import Observer
import elisa.const as c

import pickle


settings.LOG_CONFIG = '/home/miro/ELISa/my_logging.json'
settings.NUMBER_OF_PROCESSES = os.cpu_count()


# parameters of the composite spot
longitude = 30
latitude = 110
radius = np.array([31.964, 33.741, 36.689])
width = np.array([5, 10, 20])
temp_factor = 0.90
penumbra_factor = 1.05

# defining composite spot
composite_spots = [
    [
        #  Penumbra
        {
            "longitude": longitude,
            "latitude": latitude,
            "angular_radius": r,
            "temperature_factor": temp_factor * penumbra_factor,
        },
        #  Umbra
        {
            "longitude": longitude,
            "latitude": latitude,
            "angular_radius": r - width[ii],
            "temperature_factor": temp_factor
        }] for ii, r in enumerate(radius)]


# calculating size of the equivalent spot
f = temp_factor**4
k = penumbra_factor**4
theta = np.radians(radius)
theta_u = np.radians(radius-width)
equiv_radius = np.arccos((np.cos(theta)*(1 - f*k) - np.cos(theta_u) * f*(1 - k)) / (1 - f))
equiv_radius = np.degrees(equiv_radius)
print(f'Equivalent radius: {equiv_radius}')

# equivalent spot
simple_spot = [
    {
        "longitude": longitude,
        "latitude": latitude,
        "angular_radius": np.average(equiv_radius),
        "temperature_factor": temp_factor,
    }
]

primary = Star(
    mass=1.2 * u.solMass,
    # surface_potential=40,
    surface_potential=3.0,
    synchronicity=1.0,
    t_eff=6000 * u.K,
    gravity_darkening=1.0,
    # discretization_factor=3,  # angular size (in degrees) of the surface elements
    albedo=0.6,
    metallicity=0.0,
)

secondary = Star(
    mass=0.6 * u.solMass,
    surface_potential=3.5,
    synchronicity=1.0,
    t_eff=4800 * u.K,
    gravity_darkening=1.0,
    # discretization_factor=20,
    albedo=0.6,
    metallicity=0,
)

# setattr(primary, "_mass", None)
bs = BinarySystem(
    primary=primary,
    secondary=secondary,
    argument_of_periastron=58 * u.deg,
    gamma=-30.7 * u.km / u.s,
    period=2.5 * u.d,
    eccentricity=0.0,
    inclination=85 * u.deg,
    primary_minimum_time=2440000.0 * u.d,
    phase_shift=0.0,
)

# pick only one
filters = [
    # 'Generic.Bessell.U',
    # 'Generic.Bessell.B',
    # 'Generic.Bessell.V',
    'Generic.Bessell.R',
    # 'Generic.Bessell.I',
    ]
lc_kwargs = {"from_phase": -0.6,
             "to_phase": 0.6,
             "phase_step": 0.005,
             "normalize": True}

o = Observer(passband=filters,
             system=bs)

# light curve of clear system
phases, clr_curve = o.lc(**lc_kwargs)

bs.primary.spots = simple_spot
bs.init()
o = Observer(passband=filters,
             system=bs)

# light curve with simple spot
_, simple_curve = o.lc(**lc_kwargs)

composite_curves = []
for ii, r in enumerate(radius):
    bs.primary.spots = composite_spots[ii]
    bs.init()
    o = Observer(passband=filters,
                 system=bs)
    # light curve with composite spot
    composite_curves.append(o.lc(**lc_kwargs)[1])

fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

ax1.plot(phases, clr_curve[filters[0]], label='clear system', color='black')

clrs = ['g', 'c', 'r']
for ii, r in enumerate(radius):
    ax1.plot(phases, composite_curves[ii][filters[0]], label='', color=clrs[ii])
    lbl = f'{width[ii]}' + r'$^\circ$'
    ax2.plot(phases, composite_curves[ii][filters[0]]-simple_curve[filters[0]], label=lbl, color=clrs[ii])

ax1.plot(phases, simple_curve[filters[0]], label='simple spot', color='blue')

ax1.legend()
ax2.legend(title='Penumbra width: ')
ax1.set_ylabel('Flux')
ax2.set_ylabel('Composite - simple \n model flux')
ax2.set_xlabel('Phase')
plt.subplots_adjust(hspace=0, right=0.98, top=0.98)
plt.show()
