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
GAMMA_UNIT = VELOCITY_UNIT = DISTANCE_UNIT / TIME_UNIT
ACCELERATION_UNIT = DISTANCE_UNIT / TIME_UNIT ** 2
LOG_ACCELERATION_UNIT = u.dex(ACCELERATION_UNIT)
FREQUENCY_UNIT = u.Hz
ANGULAR_FREQUENCY_UNIT = u.rad / u.s
LUMINOSITY_UNIT = u.W
RADIANCE_UNIT = u.W / (u.m ** 2 * u.sr)

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

# DEFAULT PARAMETER UNITS -- DO NOT CHANGE!!! --------------------------------------------------------------------------


class BaseUnits(object):
    @classmethod
    def __iter__(cls):
        for varname in cls.__dict__.keys():
            if str(varname).startswith("_"):
                continue
            _unit = getattr(cls, varname)
            if isinstance(_unit, Unit):
                _unit = _unit.to_string()
            yield varname, _unit

    def __getitem__(self, item):
        return getattr(self, item)

    def as_dict(self):
        _dict_repr = dict()
        for key, val in self:
            if isinstance(val, Unit):
                _val_repr = val.to_string()
            elif isinstance(val, BaseUnits):
                _val_repr = val.as_dict()
            elif isinstance(val, bool):
                _val_repr = 'boolean'
            else:
                _val_repr = str(val)

            _dict_repr[key] = _val_repr
        return _dict_repr


# DEFAULT ELISa INNER UNITS (ALL USER INPUTS ARE CONVERTED TO THIS) ----------------------------------------------------

DEFAULT_INCLINATION_UNIT = ARC_UNIT
DEFAULT_PERIOD_UNIT = PERIOD_UNIT
DEFAULT_GAMMA_UNIT = VELOCITY_UNIT


class _DefaultSpotUnits(BaseUnits):
    longitude = ARC_UNIT
    latitude = ARC_UNIT
    angular_radius = ARC_UNIT
    temperature_factor = dimensionless_unscaled
    discretization_factor = ARC_UNIT


class _DefaultPulsationsUnits(BaseUnits):
    l = dimensionless_unscaled
    m = dimensionless_unscaled
    amplitude = VELOCITY_UNIT
    frequency = FREQUENCY_UNIT
    start_phase = ARC_UNIT
    mode_axis_theta = ARC_UNIT
    mode_axis_phi = ARC_UNIT
    temperature_perturbation_phase_shift = ARC_UNIT
    horizontal_to_radial_amplitude_ratio = dimensionless_unscaled
    temperature_amplitude_factor = dimensionless_unscaled
    tidally_locked = bool


class _DefaultStarUnits(BaseUnits):
    mass = MASS_UNIT
    t_eff = TEMPERATURE_UNIT
    surface_potential = dimensionless_unscaled
    synchronicity = dimensionless_unscaled
    metallicity = dimensionless_unscaled
    gravity_darkening = dimensionless_unscaled
    albedo = dimensionless_unscaled
    discretization_factor = ARC_UNIT
    polar_log_g = LOG_ACCELERATION_UNIT
    equivalent_radius = DISTANCE_UNIT
    spots = _DefaultSpotUnits()
    pulsations = _DefaultPulsationsUnits()


class _DefaultBinarySystemUnits(BaseUnits):
    class __DefaultBinaryComponentUnits(BaseUnits):
        mass = _DefaultStarUnits.mass
        t_eff = _DefaultStarUnits.t_eff
        surface_potential = _DefaultStarUnits.surface_potential
        synchronicity = _DefaultStarUnits.synchronicity
        metallicity = _DefaultStarUnits.metallicity
        gravity_darkening = _DefaultStarUnits.gravity_darkening
        albedo = _DefaultStarUnits.albedo
        discretization_factor = _DefaultStarUnits.discretization_factor
        spots = _DefaultSpotUnits()
        pulsations = _DefaultPulsationsUnits()

    class __DefaultBinarySystemUnits(BaseUnits):
        inclination = DEFAULT_INCLINATION_UNIT
        period = DEFAULT_PERIOD_UNIT
        eccentricity = dimensionless_unscaled
        argument_of_periastron = ARC_UNIT
        gamma = DEFAULT_GAMMA_UNIT
        phase_shift = dimensionless_unscaled
        additional_light = dimensionless_unscaled
        primary_minimum_time = DEFAULT_PERIOD_UNIT
        semi_major_axis = DISTANCE_UNIT

    system = __DefaultBinarySystemUnits()
    primary = __DefaultBinaryComponentUnits()
    secondary = __DefaultBinaryComponentUnits()


class _DefaultSingleSystemUnits(BaseUnits):
    class __DefaultSingleSystemUnits(BaseUnits):
        inclination = DEFAULT_INCLINATION_UNIT
        rotation_period = DEFAULT_PERIOD_UNIT
        reference_time = DEFAULT_PERIOD_UNIT
        phase_shift = dimensionless_unscaled
        additional_light = dimensionless_unscaled
        gamma = DEFAULT_GAMMA_UNIT

    class __DefaultSingleSystemStarUnits(BaseUnits):
        mass = _DefaultStarUnits.mass
        t_eff = _DefaultStarUnits.t_eff
        polar_log_g = _DefaultStarUnits.polar_log_g
        metallicity = _DefaultStarUnits.metallicity
        gravity_darkening = _DefaultStarUnits.gravity_darkening
        discretization_factor = _DefaultStarUnits.discretization_factor
        equivalent_radius = _DefaultStarUnits.equivalent_radius
        spots = _DefaultSpotUnits()
        pulsations = _DefaultPulsationsUnits()

    system = __DefaultSingleSystemUnits()
    star = __DefaultSingleSystemStarUnits()


DefaultSpotUnits = _DefaultSpotUnits()
DefaultPulsationsUnits = _DefaultPulsationsUnits()
DefaultStarUnits = _DefaultStarUnits()
DefaultBinarySystemUnits = _DefaultBinarySystemUnits()
DefaultSingleSystemUnits = _DefaultSingleSystemUnits()

# DEFAULT ELISa OUTTER/USER INPUT UNITS (MORE CONVENIENT FOR USER BUT NOT SO MUCH FOR PROGRAMMER) ----------------------

DEFAULT_INCLINATION_INPUT_UNIT = deg
DEFAULT_PERIOD_INPUT_UNIT = d
DEFAULT_GAMMA_INPUT_UNIT = m/s


class _DefaultSpotInputUnits(BaseUnits):
    longitude = deg
    latitude = deg
    angular_radius = deg
    temperature_factor = dimensionless_unscaled
    discretization_factor = deg


class _DefaultPulsationsInputUnits(BaseUnits):
    l = dimensionless_unscaled
    m = dimensionless_unscaled
    amplitude = VELOCITY_UNIT
    frequency = u.d ** (-1)
    start_phase = ARC_UNIT
    mode_axis_theta = deg
    mode_axis_phi = deg
    temperature_perturbation_phase_shift = ARC_UNIT
    horizontal_to_radial_amplitude_ratio = dimensionless_unscaled
    temperature_amplitude_factor = dimensionless_unscaled
    tidally_locked = bool


class _DefaultStarInputUnits(BaseUnits):
    mass = solMass
    t_eff = K
    surface_potential = dimensionless_unscaled
    synchronicity = dimensionless_unscaled
    metallicity = dimensionless_unscaled
    gravity_darkening = dimensionless_unscaled
    albedo = dimensionless_unscaled
    discretization_factor = deg
    polar_log_g = u.dex(cm / s ** 2)
    equivalent_radius = solRad
    spots = _DefaultSpotInputUnits()
    pulsations = _DefaultPulsationsInputUnits()


class _DefaultBinarySystemInputUnits(BaseUnits):
    class __DefaultBinaryComponentInputUnits(BaseUnits):
        mass = _DefaultStarInputUnits.mass
        t_eff = _DefaultStarInputUnits.t_eff
        surface_potential = _DefaultStarInputUnits.surface_potential
        synchronicity = _DefaultStarInputUnits.synchronicity
        metallicity = _DefaultStarInputUnits.metallicity
        gravity_darkening = _DefaultStarInputUnits.gravity_darkening
        albedo = _DefaultStarInputUnits.albedo
        discretization_factor = _DefaultStarInputUnits.discretization_factor
        spots = _DefaultSpotInputUnits()
        pulsations = _DefaultPulsationsInputUnits()

    class __DefaultBinarySystemInputUnits(BaseUnits):
        inclination = DEFAULT_INCLINATION_INPUT_UNIT
        period = DEFAULT_PERIOD_INPUT_UNIT
        eccentricity = dimensionless_unscaled
        argument_of_periastron = deg
        gamma = DEFAULT_GAMMA_INPUT_UNIT
        phase_shift = dimensionless_unscaled
        additional_light = dimensionless_unscaled
        primary_minimum_time = d
        semi_major_axis = solRad

    system = __DefaultBinarySystemInputUnits()
    primary = __DefaultBinaryComponentInputUnits()
    secondary = __DefaultBinaryComponentInputUnits()


class _DefaultSingleSystemInputUnits(BaseUnits):
    class __DefaultSingleSystemInputUnits(BaseUnits):
        inclination = DEFAULT_INCLINATION_INPUT_UNIT
        rotation_period = DEFAULT_PERIOD_INPUT_UNIT
        reference_time = d
        phase_shift = dimensionless_unscaled
        additional_light = dimensionless_unscaled
        gamma = DEFAULT_GAMMA_INPUT_UNIT

    class __DefaultSingleSystemStarInputUnits(BaseUnits):
        mass = _DefaultStarInputUnits.mass
        t_eff = _DefaultStarInputUnits.t_eff
        polar_log_g = _DefaultStarInputUnits.polar_log_g
        metallicity = _DefaultStarInputUnits.metallicity
        gravity_darkening = _DefaultStarInputUnits.gravity_darkening
        discretization_factor = _DefaultStarInputUnits.discretization_factor
        equivalent_radius = _DefaultStarInputUnits.equivalent_radius
        spots = _DefaultSpotInputUnits()
        pulsations = _DefaultPulsationsInputUnits()

    system = __DefaultSingleSystemInputUnits()
    star = __DefaultSingleSystemStarInputUnits()


DefaultSpotInputUnits = _DefaultSpotInputUnits()
DefaultPulsationsInputUnits = _DefaultPulsationsInputUnits()
DefaultStarInputUnits = _DefaultStarInputUnits()
DefaultBinarySystemInputUnits = _DefaultBinarySystemInputUnits()
DefaultSingleSystemInputUnits = _DefaultSingleSystemInputUnits()
