import sys
import builtins
from importlib import import_module

builtins._ASTROPY_SETUP_ = True

# DO NOT CHANGE ANYTHING (including default units) !!!

if 'astropy.units' in sys.modules:
    u = sys.modules['astropy.units']
else:
    u = import_module('astropy.units')

# DO NOT CHANGE THIS!!!
MASS_UNIT = u.kg
TEMPERATURE_UNIT = u.K
DISTANCE_UNIT = u.m
TIME_UNIT = u.s
ARC_UNIT = u.rad
PERIOD_UNIT = u.d
VELOCITY_UNIT = DISTANCE_UNIT / TIME_UNIT
ACCELERATION_UNIT = DISTANCE_UNIT / TIME_UNIT ** 2
LOG_ACCELERATION_UNIT = u.dex(ACCELERATION_UNIT)
FREQUENCY_UNIT = u.Hz
ANGULAR_FREQUENCY_UNIT = u.rad / u.s
LUMINOSITY_UNIT = u.W
RADIANCE_UNIT = u.W / (u.m ** 2 * u.sr)

DEFAULT_UNITS = dict(
    MASS_UNIT=MASS_UNIT,
    TEMPERATURE_UNIT=TEMPERATURE_UNIT,
    DISTANCE_UNIT=DISTANCE_UNIT,
    TIME_UNIT=TIME_UNIT,
    ARC_UNIT=ARC_UNIT,
    PERIOD_UNIT=PERIOD_UNIT,
    VELOCITY_UNIT=VELOCITY_UNIT,
    ACCELERATION_UNIT=ACCELERATION_UNIT,
    LOG_ACCELERATION_UNIT=LOG_ACCELERATION_UNIT,
    FREQUENCY_UNIT=FREQUENCY_UNIT,
    ANGULAR_FREQUENCY_UNIT=ANGULAR_FREQUENCY_UNIT,
    LUMINOSITY_UNIT=LUMINOSITY_UNIT,
    RADIANCE_UNIT=RADIANCE_UNIT
)

# astropy units to avoid annoying undefined warning across the code
deg = u.deg
degree = u.degree
rad = u.rad
km = u.km
solMass = u.solMass
deg_C = u.deg_C
m = u.m
cm = u.cm
d = u.d
s = u.s
W = u.W
solRad = u.solRad
K = u.K
dimensionless_unscaled = u.dimensionless_unscaled
kg = u.kg
dex = u.dex
mag = u.mag

Unit = u.Unit
Quantity = u.quantity.Quantity
Dex = u.Dex

# _____DEFAULT PARAMETER UNITS -- DO NOT CHANGE!!______

DEFAULT_INCLINATION_UNIT = ARC_UNIT
DEFAULT_PERIOD_UNIT = PERIOD_UNIT
DEFAULT_GAMMA_UNIT = VELOCITY_UNIT

DEFAULT_SPOT_UNITS = dict(
    longitude=ARC_UNIT,
    latitude=ARC_UNIT,
    angular_radius=ARC_UNIT,
    temperature_factor=u.dimensionless_unscaled,
    discretization_factor=ARC_UNIT
)
DEFAULT_PULSATIONS_UNITS = dict(
    l=dimensionless_unscaled,
    m=dimensionless_unscaled,
    amplitude=VELOCITY_UNIT,
    frequency=FREQUENCY_UNIT,
    start_phase=ARC_UNIT,
    mode_axis_theta=ARC_UNIT,
    mode_axis_phi=ARC_UNIT,
    temperature_perturbation_phase_shift=ARC_UNIT,
    horizontal_to_radial_amplitude_ratio=dimensionless_unscaled,
    temperature_amplitude_factor=dimensionless_unscaled,
    tidally_locked=bool
)

DEFAULT_STAR_UNITS = dict(
    mass=MASS_UNIT,
    t_eff=TEMPERATURE_UNIT,
    surface_potential=dimensionless_unscaled,
    synchronicity=dimensionless_unscaled,
    metallicity=dimensionless_unscaled,
    gravity_darkening=dimensionless_unscaled,
    albedo=dimensionless_unscaled,
    discretization_factor=ARC_UNIT,
    polar_log_g=LOG_ACCELERATION_UNIT,
    equivalent_radius=DISTANCE_UNIT,
    spots=DEFAULT_SPOT_UNITS,
    pulsations=DEFAULT_PULSATIONS_UNITS
)

DEFAULT_BINARY_SYSTEM_UNITS = dict(
    system=dict(
        inclination=DEFAULT_INCLINATION_UNIT,
        period=DEFAULT_PERIOD_UNIT,
        eccentricity=dimensionless_unscaled,
        argument_of_periastron=ARC_UNIT,
        gamma=DEFAULT_GAMMA_UNIT,
        phase_shift=dimensionless_unscaled,
        additional_light=dimensionless_unscaled,
        primary_minimum_time=PERIOD_UNIT,
        semi_major_axis=DISTANCE_UNIT
    ),
    primary=dict(
        mass=DEFAULT_STAR_UNITS['mass'],
        t_eff=DEFAULT_STAR_UNITS['t_eff'],
        surface_potential=DEFAULT_STAR_UNITS['surface_potential'],
        synchronicity=DEFAULT_STAR_UNITS['synchronicity'],
        metallicity=DEFAULT_STAR_UNITS['metallicity'],
        gravity_darkening=DEFAULT_STAR_UNITS['gravity_darkening'],
        albedo=DEFAULT_STAR_UNITS['albedo'],
        discretization_factor=DEFAULT_STAR_UNITS['discretization_factor'],
        spots=DEFAULT_SPOT_UNITS,
        pulsations=DEFAULT_PULSATIONS_UNITS
    )
)
DEFAULT_BINARY_SYSTEM_UNITS['secondary'] = DEFAULT_BINARY_SYSTEM_UNITS['primary']


DEFAULT_SINGLE_SYSTEM_UNITS = dict(
    system=dict(
        inclination=DEFAULT_INCLINATION_UNIT,
        rotation_period=DEFAULT_PERIOD_UNIT,
        reference_time=PERIOD_UNIT,
        phase_shift=dimensionless_unscaled,
        additional_light=dimensionless_unscaled,
        gamma=DEFAULT_GAMMA_UNIT,
    ),
    star=dict(
        mass=DEFAULT_STAR_UNITS['mass'],
        t_eff=DEFAULT_STAR_UNITS['t_eff'],
        polar_log_g=DEFAULT_STAR_UNITS['polar_log_g'],
        metallicity=DEFAULT_STAR_UNITS['metallicity'],
        gravity_darkening=DEFAULT_STAR_UNITS['gravity_darkening'],
        discretization_factor=DEFAULT_STAR_UNITS['discretization_factor'],
        equivalent_radius=DEFAULT_STAR_UNITS['equivalent_radius'],
        spots=DEFAULT_SPOT_UNITS,
        pulsations=DEFAULT_PULSATIONS_UNITS
    )
)

# _____DEFAULT USER INPUT UNITS -- DO NOT CHANGE!!______

DEFAULT_INCLINATION_INPUT_UNIT = deg
DEFAULT_PERIOD_INPUT_UNIT = d
DEFAULT_GAMMA_INPUT_UNIT = m/s

DEFAULT_SPOT_INPUT_UNITS = dict(
    longitude=deg,
    latitude=deg,
    angular_radius=deg,
    temperature_factor=u.dimensionless_unscaled,
    discretization_factor=deg
)
DEFAULT_PULSATIONS_INPUT_UNITS = dict(
    l=dimensionless_unscaled,
    m=dimensionless_unscaled,
    amplitude=VELOCITY_UNIT,
    frequency=u.d**(-1),
    start_phase=ARC_UNIT,
    mode_axis_theta=deg,
    mode_axis_phi=deg,
    temperature_perturbation_phase_shift=ARC_UNIT,
    horizontal_to_radial_amplitude_ratio=dimensionless_unscaled,
    temperature_amplitude_factor=dimensionless_unscaled,
    tidally_locked=bool
)

DEFAULT_STAR_INPUT_UNITS = dict(
    mass=solMass,
    t_eff=K,
    surface_potential=dimensionless_unscaled,
    synchronicity=dimensionless_unscaled,
    metallicity=dimensionless_unscaled,
    gravity_darkening=dimensionless_unscaled,
    albedo=dimensionless_unscaled,
    discretization_factor=deg,
    polar_log_g=u.dex(cm/s**2),
    equivalent_radius=solRad,
    spots=DEFAULT_SPOT_INPUT_UNITS,
    pulsations=DEFAULT_PULSATIONS_INPUT_UNITS
)

DEFAULT_BINARY_SYSTEM_INPUT_UNITS = dict(
    system=dict(
        inclination=DEFAULT_INCLINATION_INPUT_UNIT,
        period=DEFAULT_PERIOD_INPUT_UNIT,
        eccentricity=dimensionless_unscaled,
        argument_of_periastron=deg,
        gamma=DEFAULT_GAMMA_INPUT_UNIT,
        phase_shift=dimensionless_unscaled,
        additional_light=dimensionless_unscaled,
        primary_minimum_time=d,
        semi_major_axis=solRad
    ),
    primary=dict(
        mass=DEFAULT_STAR_INPUT_UNITS['mass'],
        t_eff=DEFAULT_STAR_INPUT_UNITS['t_eff'],
        surface_potential=DEFAULT_STAR_INPUT_UNITS['surface_potential'],
        synchronicity=DEFAULT_STAR_INPUT_UNITS['synchronicity'],
        metallicity=DEFAULT_STAR_INPUT_UNITS['metallicity'],
        gravity_darkening=DEFAULT_STAR_INPUT_UNITS['gravity_darkening'],
        albedo=DEFAULT_STAR_INPUT_UNITS['albedo'],
        discretization_factor=DEFAULT_STAR_INPUT_UNITS['discretization_factor'],
        spots=DEFAULT_SPOT_INPUT_UNITS,
        pulsations=DEFAULT_PULSATIONS_INPUT_UNITS
    )
)
DEFAULT_BINARY_SYSTEM_INPUT_UNITS['secondary'] = DEFAULT_BINARY_SYSTEM_INPUT_UNITS['primary']

DEFAULT_SINGLE_SYSTEM_INPUT_UNITS = dict(
    system=dict(
        inclination=DEFAULT_INCLINATION_INPUT_UNIT,
        rotation_period=DEFAULT_PERIOD_INPUT_UNIT,
        reference_time=d,
        phase_shift=dimensionless_unscaled,
        additional_light=dimensionless_unscaled,
        gamma=DEFAULT_GAMMA_INPUT_UNIT,
    ),
    star=dict(
        mass=DEFAULT_STAR_INPUT_UNITS['mass'],
        t_eff=DEFAULT_STAR_INPUT_UNITS['t_eff'],
        polar_log_g=DEFAULT_STAR_INPUT_UNITS['polar_log_g'],
        metallicity=DEFAULT_STAR_INPUT_UNITS['metallicity'],
        gravity_darkening=DEFAULT_STAR_INPUT_UNITS['gravity_darkening'],
        discretization_factor=DEFAULT_STAR_INPUT_UNITS['discretization_factor'],
        equivalent_radius=DEFAULT_STAR_INPUT_UNITS['equivalent_radius'],
        spots=DEFAULT_SPOT_INPUT_UNITS,
        pulsations=DEFAULT_PULSATIONS_INPUT_UNITS
    )
)

