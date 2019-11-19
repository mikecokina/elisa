from elisa import units
from elisa.analytics.binary import params
from elisa.base import error
from elisa.base.star import Star
from elisa.binary_system.system import BinarySystem


def _prepare_star(**kwargs):
    return Star(
        **dict(
            **dict(
                mass=kwargs['mass'] * units.solMass,
                surface_potential=kwargs['surface_potential'],
                synchronicity=kwargs.get('synchronicity', 1.0),
                t_eff=kwargs['t_eff'] * units.K,
                gravity_darkening=kwargs.get('gravity_darkening', 1.0),
                albedo=kwargs.get('albedo', 1.0),
                metallicity=kwargs.get('metallicity', 0.0)
            ),
            **{"discretization_factor": kwargs["discretization_factor"]}
            if kwargs.get("discretization_factor") else {})
    )


def _serialize_star_kwargs(component, **kwargs):
    """
    Serialize `x0` input like kawrgs to Star kwargs (truncate p__ or s__).

    :param component: str; `p` or `s`
    :param kwargs: Dict;
    :return: Dict;
    """
    return {str(k)[3:]: v for k, v in kwargs.items() if str(k).startswith(component)}


def serialize_primary_kwargs(**kwargs):
    return _serialize_star_kwargs(component='p', **kwargs)


def serialize_seondary_kwargs(**kwargs):
    return _serialize_star_kwargs(component='s', **kwargs)


def prepare_binary(period, discretization, **kwargs):
    """
    Setup circular synchrnonous binary system.
    If `beta` (gravity darkening factor) or `albedo` is not supplied, then `1.0` is used as their default value.

    :param period: float;
    :param discretization; float;
    :param kwargs: Dict;
    :**kwargs options**:
        * **argument_of_periastron** * -- float;
        * **eccentricity** * -- float;
        * **inclination** * -- float;
        * **p__mass** * -- float;
        * **p__t_eff** * -- float;
        * **p__surface_potential** * -- float;
        * **p__gravity_darkening** * -- float;
        * **p__albedo** * -- float;
        * **p__metallicity** * -- float;
        * **p__synchronicity** * -- float;
        * **s__mass** * -- float;
        * **s__t_eff** * -- float;
        * **s__surface_potential** * -- float;
        * **s__gravity_darkening** * -- float;
        * **s__albedo** * -- float;
        * **s__metallicity** * -- float;
        * **s__synchronicity** * -- float;
    :return: elisa.binary_system.system.BinarySystem;
    """

    kwargs.update({"p__discretization_factor": discretization})
    primary = _prepare_star(**serialize_primary_kwargs(**kwargs))
    secondary = _prepare_star(**serialize_seondary_kwargs(**kwargs))

    return BinarySystem(
        primary=primary,
        secondary=secondary,
        argument_of_periastron=kwargs.get('argument_of_periastron', 90.0) * units.deg,
        gamma=0.0,
        period=period * units.d,
        eccentricity=kwargs.get('eccentricity', 0.0),
        inclination=kwargs['inclination'] * units.deg,
        primary_minimum_time=0.0,
        phase_shift=0.0
    )


def synthetic_binary(xs, period, discretization, morphology, observer, **kwargs):
    """
    :param xs: Union[List, numpy.array];
    :param period: float;
    :param discretization: float;
    :param morphology: str;
    :param observer: elisa.observer.observer.Observer; instance
    :param kwargs: Dict;
    :**kwargs options**:
        * **argument_of_periastron** * -- float;
        * **eccentricity** * -- float;
        * **inclination** * -- float;
        * **p__mass** * -- float;
        * **p__t_eff** * -- float;
        * **p__surface_potential** * -- float;
        * **p__gravity_darkening** * -- float;
        * **p__albedo** * -- float;
        * **p__metallicity** * -- float;
        * **p__synchronicity** * -- float;
        * **s__mass** * -- float;
        * **s__t_eff** * -- float;
        * **s__surface_potential** * -- float;
        * **s__gravity_darkening** * -- float;
        * **s__albedo** * -- float;
        * **s__metallicity** * -- float;
        * **s__synchronicity** * -- float;
    :return: Tuple[numpy.array, str]
    """
    binary = prepare_binary(period, discretization, **kwargs)
    observer._system = binary

    if params.is_overcontact(morphology) and not params.is_overcontact(binary.morphology):
        raise error.MorphologyError(f'Expected morphology is {morphology} but obtained is {binary.morphology}')

    lc = observer.observe.lc(phases=xs, normalize=True)
    return lc[1]


def prepare_central_rv_binary(period, **kwargs):
    kwargs.update({
        "p__surface_potential": 100,
        "s__surface_potential": 100,
        "p__t_eff": 10000.0,
        "s__t_eff": 10000.0,
        "p__metallicity": 10000.0,
        "s__metallicity": 10000.0
    })
    primary = _prepare_star(**serialize_primary_kwargs(**kwargs))
    secondary = _prepare_star(**serialize_seondary_kwargs(**kwargs))

    return BinarySystem(
        primary=primary,
        secondary=secondary,
        argument_of_periastron=kwargs['argument_of_periastron'],
        gamma=kwargs['gamma'],
        period=period * units.d,
        eccentricity=kwargs['eccentricity'],
        inclination=kwargs['inclination'],
        primary_minimum_time=0.0,
        phase_shift=0.0
    )


def central_rv_synthetic(xs, period, observer, **kwargs):
    binary = prepare_central_rv_binary(period, **kwargs)
    observer._system = binary
    rv = observer.observe.rv(phases=xs, normalize=False)
    return rv[1], rv[2]
