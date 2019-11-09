import numpy as np

from elisa import units
from elisa.base.star import Star
from elisa.binary_system.system import BinarySystem
from elisa.observer.observer import Observer


def prepare_binary(period, **kwargs):

    primary = Star(
        mass=kwargs['p__mass'] * units.solMass,
        surface_potential=kwargs['p__omega'],
        synchronicity=1.0,
        t_eff=kwargs['p__t_eff'] * units.K,
        gravity_darkening=1.0,
        discretization_factor=5,
        albedo=1.0,
        metallicity=0.0
    )

    secondary = Star(
        mass=kwargs['s__mass'] * units.solMass,
        surface_potential=kwargs['p__omega'],
        synchronicity=1.0,
        t_eff=kwargs['s__t_eff'] * units.K,
        gravity_darkening=1.0,
        albedo=1.0,
        metallicity=0.0
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


def synthetic(xs, period, passband, **kwargs):
    """

    :param xs: Union[List, numpy.array];
    :param period: float;
    :param passband: str;
    :param kwargs: Dict;
    :**kwargs options**:
        * **inclination** * -- float;

        * **p__mass** * -- float;
        * **p__t_eff** * -- float;
        * **p__omega** * -- float;

        * **s__mass** * -- float;
        * **s__t_eff** * -- float;
        * **s__omega** * -- float;

    :return: numpy.array;
    """

    binary = prepare_binary(period, **kwargs)
    observer = Observer(passband=passband, system=binary)
    lc = observer.observe.lc(phases=xs, normalize=True)
    return lc[1][passband] / np.max(lc[1][passband])
