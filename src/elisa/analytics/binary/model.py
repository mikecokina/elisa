from elisa import units
from elisa.base.star import Star
from elisa.binary_system.system import BinarySystem


def prepare_circual_sync_binary(period, discretization, **kwargs):
    """
    Setup circular synchrnonous binary system.
    If `beta` (gravity darkening factor) or `albedo` is not supplied, then `1.0` is used as their default value.

    :param period: float;
    :param discretization; float;
    :param kwargs: Dict;
    :**kwargs options**:
        * **inclination** * -- float;
        * **p__mass** * -- float;
        * **p__t_eff** * -- float;
        * **p__surface_potential** * -- float;
        * **p__gravity_darkening** * -- float;
        * **p__albedo** * -- float;
        * **p__metallicity** * -- float;
        * **s__mass** * -- float;
        * **s__t_eff** * -- float;
        * **s__surface_potential** * -- float;
        * **s__gravity_darkening** * -- float;
        * **s__albedo** * -- float;
        * **s__metallicity** * -- float;
    :return: elisa.binary_system.system.BinarySystem;
    """

    primary = Star(
        mass=kwargs['p__mass'] * units.solMass,
        surface_potential=kwargs['p__surface_potential'],
        synchronicity=1.0,
        t_eff=kwargs['p__t_eff'] * units.K,
        gravity_darkening=kwargs.get('p__gravity_darkening', 1.0),
        discretization_factor=discretization,
        albedo=kwargs.get('p__albedo', 1.0),
        metallicity=kwargs['p__metallicity']
    )

    secondary = Star(
        mass=kwargs['s__mass'] * units.solMass,
        surface_potential=kwargs['p__surface_potential'],
        synchronicity=1.0,
        t_eff=kwargs['s__t_eff'] * units.K,
        gravity_darkening=kwargs.get('s__gravity_darkening', 1.0),
        albedo=kwargs.get('s__albedo', 1.0),
        metallicity=kwargs['s__metallicity']
    )

    bs = BinarySystem(
        primary=primary,
        secondary=secondary,
        argument_of_periastron=0.0,
        gamma=0.0,
        period=period * units.d,
        eccentricity=0.0,
        inclination=kwargs['inclination'],
        primary_minimum_time=0.0,
        phase_shift=0.0
    )
    return bs


def circular_sync_synthetic(xs, period, discretization, observer, **kwargs):
    """
    :param xs: Union[List, numpy.array];
    :param period: float;
    :param discretization: float;
    :param observer: elisa.observer.observer.Observer; instance
    :param kwargs: Dict;
    :**kwargs options**:
        * **inclination** * -- float;
        * **p__mass** * -- float;
        * **p__t_eff** * -- float;
        * **p__surface_potential** * -- float;
        * **p__gravity_darkening** * -- float;
        * **p__albedo** * -- float;
        * **p__metallicity** * -- float;
        * **s__mass** * -- float;
        * **s__t_eff** * -- float;
        * **s__surface_potential** * -- float;
        * **s__gravity_darkening** * -- float;
        * **s__albedo** * -- float;
        * **s__metallicity** * -- float;
    :return: numpy.array;
    """
    binary = prepare_circual_sync_binary(period, discretization, **kwargs)
    observer._system = binary
    lc = observer.observe.lc(phases=xs, normalize=True)
    return lc[1]
