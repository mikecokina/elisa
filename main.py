from engine.binary_system import BinarySystem
from engine.star import Star
from astropy import units as u


def main():
    primary = Star(
        mass=2.0 * u.solMass,
        surface_potential=5,
        synchronicity=1.0,
        t_eff=10000 * u.K,
        gravity_darkening=1.0,
        discretization_factor=3,
        albedo=1.0
    )

    secondary = Star(
        mass=1.0 * u.solMass,
        surface_potential=5,
        synchronicity=1.0,
        t_eff=6800 * u.K,
        gravity_darkening=1.0,
        discretization_factor=5,
        albedo=1.0
    )

    bs = BinarySystem(
        primary=primary,
        secondary=secondary,
        argument_of_periastron=90 * u.deg,
        gamma=0 * u.km / u.s,
        period=1 * u.d,
        eccentricity=0.0,
        inclination=90 * u.deg,
        primary_minimum_time=0.0 * u.d,
        phase_shift=0.0,
    )

    bs.build_mesh(components_distance=1.0)


if __name__ == "__main__":
    main()
