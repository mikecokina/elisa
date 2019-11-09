import numpy as np

from elisa import units
from elisa.base.star import Star
from elisa.binary_system.system import BinarySystem
from elisa.observer.observer import Observer


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
        * **p__omega** * -- float;
        * **p__beta** * -- float;
        * **p__albedo** * -- float;
        * **p__mh** * -- float;
        * **s__mass** * -- float;
        * **s__t_eff** * -- float;
        * **s__omega** * -- float;
        * **s__beta** * -- float;
        * **s__albedo** * -- float;
        * **s__mh** * -- float;
    :return: elisa.binary_system.system.BinarySystem;
    """

    primary = Star(
        mass=kwargs['p__mass'] * units.solMass,
        surface_potential=kwargs['p__omega'],
        synchronicity=1.0,
        t_eff=kwargs['p__t_eff'] * units.K,
        gravity_darkening=kwargs.get('p__beta', 1.0),
        discretization_factor=discretization,
        albedo=kwargs.get('p__albedo', 1.0),
        metallicity=0.0,
        suppress_logger=True
    )

    secondary = Star(
        mass=kwargs['s__mass'] * units.solMass,
        surface_potential=kwargs['p__omega'],
        synchronicity=1.0,
        t_eff=kwargs['s__t_eff'] * units.K,
        gravity_darkening=kwargs.get('s__beta', 1.0),
        albedo=kwargs.get('s__albedo', 1.0),
        metallicity=0.0,
        suppress_logger=True
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
        phase_shift=0.0,
        suppress_logger=True
    )
    return bs


def circular_sync_synthetic(xs, period, passband, discretization, **kwargs):
    """
    :param xs: Union[List, numpy.array];
    :param period: float;
    :param discretization: float;
    :param passband: str;
    :param kwargs: Dict;
    :**kwargs options**:
        * **inclination** * -- float;
        * **p__mass** * -- float;
        * **p__t_eff** * -- float;
        * **p__omega** * -- float;
        * **p__beta** * -- float;
        * **p__albedo** * -- float;
        * **p__mh** * -- float;
        * **s__mass** * -- float;
        * **s__t_eff** * -- float;
        * **s__omega** * -- float;
        * **s__beta** * -- float;
        * **s__albedo** * -- float;
        * **s__mh** * -- float;
    :return: numpy.array;
    """
    binary = prepare_circual_sync_binary(period, discretization, **kwargs)
    observer = Observer(passband=passband, system=binary, suppress_logger=True)
    lc = observer.observe.lc(phases=xs, normalize=True)
    return lc[1][passband] / np.max(lc[1][passband])
