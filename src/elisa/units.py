import sys
import builtins
from importlib import import_module
builtins._ASTROPY_SETUP_ = True


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
ACCELERATION_UNIT = DISTANCE_UNIT / TIME_UNIT**2
LOG_ACCELERATION_UNIT = u.dex(ACCELERATION_UNIT)
FREQUENCY_UNIT = u.Hz
ANGULAR_FREQUENCY_UNIT = u.rad / u.s
LUMINOSITY_UNIT = u.W

# astropy units to avoid annoying undefined warning accross basecode
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
