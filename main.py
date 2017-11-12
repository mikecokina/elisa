from engine.binary_system import BinarySystem
from engine.star import Star
from astropy import units as u

# bs = BinarySystem(gamma=25, period=10.0, eccentricity=0.2)

primary = Star(mass=2.0, surface_potential=5.0)
secondary = Star(mass=1.0, surface_potential=5.0)

bs = BinarySystem(primary=primary, secondary=secondary)


# dt - dt
print(bs.primary.mass)
primary.mass = 3.0

# sd - dt
print(bs.primary.mass)
