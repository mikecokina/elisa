from engine.binary_system import BinarySystem
from astropy import units as u

# bs = BinarySystem(gamma=25, period=10.0, eccentricity=0.2)
bs = BinarySystem()
bs.gamma = 25 * u.km / u.s
bs.period = 10 * u.d
bs.periastron = 152 * u.deg
bs.inclination = 85 * u.deg
bs.eccentricity = 0.2

print('gamma = {}'.format(bs.gamma))
print('period = {}'.format(bs.period))
print('periastron = {}'.format(bs.periastron))
print('inclination = {}'.format(bs.inclination))
print('eccentricity = {}'.format(bs.eccentricity))

print('period from orbit = {}'.format(bs.orbit.period))
print('periastron from orbit = {}'.format(bs.orbit.periastron))
print('inclination from orbit = {}'.format(bs.orbit.inclination))
print('eccentricity from orbit = {}'.format(bs.orbit.eccentricity))

