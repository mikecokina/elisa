import numpy as np
from astropy.time import Time

from ... import units as u
from ... atm import atm_file_prefix_to_quantity_list
from ... import settings


PARAM_PARSER = '@'
NUISANCE_PARSER = 'nuisance'

TEMPERATURES = atm_file_prefix_to_quantity_list("temperature", settings.ATM_ATLAS)
METALLICITY = atm_file_prefix_to_quantity_list("metallicity", settings.ATM_ATLAS)


COMPOSITE_FLAT_PARAMS = [
    'spot',
    'pulsation'
]

DEFAULT_NORMALIZATION_SPOT = {
    "longitude": (0, 360),
    "latitude": (0, 180),
    "angular_radius": (0, 90),
    "temperature_factor": (0.1, 3),
}

DEFAULT_NORMALIZATION_NUISANCE = {
    "ln_f": (-20, -10),
}

DEFAULT_NORMALIZATION_PULSATION = {
    "l": (0, 10),
    "m": (-10, 10),
    "amplitude": (0, 5000),
    "frequency": (0.01, 40),
    "start_phase": (0, 360),
    "mode_axis_theta": (0, 180),
    "mode_axis_phi": (0, 360)
}

DEFAULT_NORMALIZATION_STAR = {
    "mass": (0.1, 50),
    "t_eff": (np.min(TEMPERATURES), np.max(TEMPERATURES)),
    "metallicity": (np.min(METALLICITY), np.max(METALLICITY)),
    "surface_potential": (2.0, 50.0),
    "albedo": (0, 1),
    "gravity_darkening": (0, 1),
    "synchronicity": (0.01, 10),
}

DEFAULT_NORMALIZATION_SYSTEM = {
    "inclination": (0, 180),
    "eccentricity": (0, 0.9999),
    "argument_of_periastron": (0, 360),
    "gamma": (0, 1e6),
    "mass_ratio": (1e-6, 2),
    "semi_major_axis": (0.01, 100),
    "asini": (0.0001, 100),
    "period": (0.001, 100),
    "additional_light": (0, 1.0),
    "phase_shift": (-0.8, 0.8),
    "primary_minimum_time": (Time.now().jd - 365.0, Time.now().jd),
}

SPOTS_PARAMETERS = ['longitude', 'latitude', 'angular_radius', 'temperature_factor', 'angular_radius']
PULSATIONS_PARAMETERS = ['l', 'm', 'amplitude', 'frequency', 'start_phase', 'mode_axis_phi', 'mode_axis_theta']

DEFAULT_FLOAT_ANGULAR_UNIT = u.deg
DEFAULT_FLOAT_MASS_UNIT = u.solMass

DEFAULT_FLOAT_UNITS = {
    'inclination': u.DefaultSystemInputUnits.inclination,
    'eccentricity': u.DefaultBinarySystemInputUnits.system.eccentricity,
    'argument_of_periastron': u.DefaultBinarySystemInputUnits.system.argument_of_periastron,
    'gamma': u.DefaultSystemInputUnits.gamma,
    'mass': u.DefaultStarInputUnits.mass,
    't_eff': u.DefaultStarInputUnits.t_eff,
    'metallicity': u.DefaultStarInputUnits.metallicity,
    'surface_potential': u.DefaultBinarySystemInputUnits.component.surface_potential,
    'albedo': u.DefaultStarUnits.albedo,
    'gravity_darkening': u.DefaultStarInputUnits.gravity_darkening,
    'synchronicity': u.DefaultBinarySystemInputUnits.component.synchronicity,
    'mass_ratio': u.DefaultBinarySystemInputUnits.system.mass_ratio,
    'semi_major_axis': u.DefaultBinarySystemInputUnits.system.semi_major_axis,
    'asini': u.solRad,
    'period': u.DefaultBinarySystemInputUnits.system.period,
    'primary_minimum_time': u.DefaultBinarySystemInputUnits.system.primary_minimum_time,
    'additional_light': u.DefaultSystemInputUnits.additional_light,
    'phase_shift': u.DefaultSystemInputUnits.phase_shift,
    # SPOTS
    'latitude': u.DefaultSpotInputUnits.latitude,
    'longitude': u.DefaultSpotInputUnits.longitude,
    'angular_radius': u.DefaultSpotInputUnits.angular_radius,
    'temperature_factor': u.DefaultSpotInputUnits.temperature_factor,
    # PULSATIONS
    'l': u.DefaultPulsationsInputUnits.l,
    'm': u.DefaultPulsationsInputUnits.m,
    'amplitude': u.DefaultPulsationsInputUnits.amplitude,
    'frequency': u.DefaultPulsationsInputUnits.frequency,
    'start_phase': u.DefaultPulsationsInputUnits.start_phase,
    'mode_axis_theta': u.DefaultPulsationsInputUnits.mode_axis_theta,
    'mode_axis_phi': u.DefaultPulsationsInputUnits.mode_axis_phi,
    # NUISANCE
    'ln_f': None
}

PARAMS_KEY_TEX_MAP = {
    'system@argument_of_periastron': '$\\omega$',
    'system@inclination': '$i$',
    'system@eccentricity': '$e$',
    'system@gamma': '$\\gamma$',
    'system@mass_ratio': '$q$',
    'system@semi_major_axis': '$a$',
    'primary@mass': '$M_1$',
    'primary@t_eff': '$T_1^{eff}$',
    'primary@surface_potential': '$\\Omega_1$',
    'primary@gravity_darkening': '$\\beta_1$',
    'primary@albedo': '$A_1$',
    'primary@metallicity': '$M/H_1$',
    'primary@synchronicity': '$F_1$',

    'secondary@mass': '$M_2$',
    'secondary@t_eff': '$T_2^{eff}$',
    'secondary@surface_potential': '$\\Omega_2$',
    'secondary@gravity_darkening': '$\\beta_2$',
    'secondary@albedo': '$A_2$',
    'secondary@metallicity': '$M/H_2$',
    'secondary@synchronicity': '$F_2$',

    'system@asini': 'a$sin$(i)',
    'system@period': '$period$',
    'system@primary_minimum_time': '$T_0$',
    'system@additional_light': '$l_{add}$',
    'system@phase_shift': 'phase shift$',
    # SPOTS
    'longitude': '$\\varphi$',
    'latitude': '$\\vartheta$',
    'angular_radius': '$R_{spot}$',
    'temperature_factor': '$T_{spot}/T_{eff}$',
    # PULSATIONS
    'l': '$\\ell$',
    'm': '$m$',
    'amplitude': '$A$',
    'frequency': '$f$',
    'start_phase': '$\\Phi_0$',
    'mode_axis_phi': '$\\phi_{mode}$',
    'mode_axis_theta': '$\\theta_{mode}$',
    # NUISANCE
    'nuisance@ln_f': "$ln(f)$"
}
