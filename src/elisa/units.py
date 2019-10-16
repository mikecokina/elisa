from astropy import units as u

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

# astropy units to avoid annoying undefined warning accross basecode
deg = u.deg
rad = u.rad
km = u.km
solMass = u.solMass
deg_C = u.deg_C
m = u.m
