from elisa.binary_system.curves.community import RadialVelocitySystem
from ..binary import params
from ...base import error
from ...binary_system.system import BinarySystem
from ...binary_system.utils import resolve_json_kind


def _serialize_star_kwargs(component, **kwargs):
    """
    Serialize `x0` input like kawrgs to Star kwargs (truncate p__ or s__).

    :param component: str; `p` or `s`
    :param kwargs: Dict;
    :return: Dict;
    """

    no_prefix = {str(k)[3:]: v for k, v in kwargs.items() if str(k).startswith(component)}
    return dict(
        surface_potential=no_prefix['surface_potential'],
        synchronicity=no_prefix.get('synchronicity', 1.0),
        t_eff=no_prefix['t_eff'],
        gravity_darkening=no_prefix.get('gravity_darkening', 1.0),
        albedo=no_prefix.get('albedo', 1.0),
        metallicity=no_prefix.get('metallicity', 0.0),
        **{"discretization_factor": no_prefix["discretization_factor"]}
        if no_prefix.get("discretization_factor") else {},
        **{"mass": no_prefix["mass"]} if no_prefix.get("mass") else {}
    )


def serialize_system_kwargs(**kwargs):
    return dict(
        argument_of_periastron=kwargs.get('argument_of_periastron', 90.0),
        gamma=kwargs.get('gamma', 0.0),
        period=kwargs["period"],
        eccentricity=kwargs.get('eccentricity', 0.0),
        inclination=kwargs['inclination'],
        primary_minimum_time=0.0,
        **{"semi_major_axis": kwargs["semi_major_axis"]} if kwargs.get("semi_major_axis") else {},
        **{"mass_ratio": kwargs["mass_ratio"]} if kwargs.get("mass_ratio") else {},
        **{"asini": kwargs["asini"]} if kwargs.get("asini") else {}
    )


def serialize_primary_kwargs(**kwargs):
    return _serialize_star_kwargs(component='p', **kwargs)


def serialize_seondary_kwargs(**kwargs):
    return _serialize_star_kwargs(component='s', **kwargs)


def prepare_binary(period, discretization, **kwargs):
    """
    Setup binary system.
    If `beta` (gravity darkening factor), `albedo`, `metallicity` or `synchronicity` is not supplied,
    then `1.0` is used as their default value.

    :param period: float;
    :param discretization; float;
    :param kwargs: Dict;
    :**kwargs options**:
        * **argument_of_periastron** * -- float;
        * **eccentricity** * -- float;
        * **inclination** * -- float;
        * **mass_ratio** * -- float; parameter has to be paired with `semi_major_axis`
        * **semi_major_axis** * -- float;
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

    kwargs.update({"p__discretization_factor": discretization, "period": period})
    primary_kwargs = serialize_primary_kwargs(**kwargs)
    secondary_kwargs = serialize_seondary_kwargs(**kwargs)
    system_kwargs = serialize_system_kwargs(**kwargs)
    json = {
        "primary": dict(**primary_kwargs),
        "secondary": dict(**secondary_kwargs),
        "system": dict(**system_kwargs)
    }
    return BinarySystem.from_json(json, _verify=False)


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
        * **mass_ratio** * -- float; parameter has to be paired with `semi_major_axis`
        * **semi_major_axis** * -- float;
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
        * **gamma** * -- float;
    :return: Tuple[numpy.array, str]
    """
    binary = prepare_binary(period, discretization, **kwargs)
    observer._system = binary

    if params.is_overcontact(morphology) and not params.is_overcontact(binary.morphology):
        raise error.MorphologyError(f'Expected morphology is {morphology} but obtained is {binary.morphology}')

    lc = observer.observe.lc(phases=xs, normalize=True)
    return lc[1]


def prepare_central_rv_binary(**kwargs):
    return BinarySystem.from_json(kwargs, _verify=False)


def central_rv_synthetic(xs, period, observer, **kwargs):
    """

    :param xs: Union[List, numpy.array];
    :param period: float;
    :param observer: elisa.observer.observer.Observer; instance
    :param kwargs: Dict;
    :**kwargs options**:
        * **argument_of_periastron** * -- float;
        * **eccentricity** * -- float;
        * **inclination** * -- float;
        * **p__mass** * -- float;
        * **s__mass** * -- float;
        * **gamma** * -- float;
    :return: Tuple;
    """

    kwargs.update({
        "p__surface_potential": 100,
        "s__surface_potential": 100,
        "p__t_eff": 10000.0,
        "s__t_eff": 10000.0,
        "p__metallicity": 10000.0,
        "s__metallicity": 10000.0,
        "period": period
    })

    primary_kwargs = serialize_primary_kwargs(**kwargs)
    secondary_kwargs = serialize_seondary_kwargs(**kwargs)
    system_kwargs = serialize_system_kwargs(**kwargs)
    json = {
        "primary": dict(**primary_kwargs),
        "secondary": dict(**secondary_kwargs),
        "system": dict(**system_kwargs)
    }
    kind_of = resolve_json_kind(data=json, _sin=True)
    observable = prepare_central_rv_binary(**json) if kind_of in ["std"] else RadialVelocitySystem(**json["system"])
    observer._system = observable
    rv = observer.observe.rv(phases=xs, normalize=False)
    return rv[1], rv[2]
