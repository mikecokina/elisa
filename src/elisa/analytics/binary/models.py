from elisa.binary_system import t_layer
from elisa.binary_system.curves.community import RadialVelocitySystem
from ..binary import params
from ...binary_system.system import BinarySystem
from ...binary_system.utils import resolve_json_kind


def _serialize_star_kwargs(component, **kwargs):
    """
    Serialize `x0` input like kwargs to Star kwargs (truncate primary or secondary).

    :param component: str; `primary` or `secondary`
    :param kwargs: Dict;
    :return: Dict;
    """
    # sorting pulsations and spots back to their respective hierarchies for easier access
    params_tree = params.dict_to_user_format(kwargs)[component]

    spots = [
        {
            'longitude': params_tree['spots'][key]['longitude'],
            'latitude': params_tree['spots'][key]['latitude'],
            'angular_radius': params_tree['spots'][key]['angular_radius'],
            'temperature_factor': params_tree['spots'][key]['temperature_factor']
        } for key in params_tree['spots'].keys()] if 'spots' in params_tree.keys() else None

    pulsations = [
        {
            'l': params_tree['pulsations'][key]['l'],
            'm': params_tree['pulsations'][key]['m'],
            'amplitude': params_tree['pulsations'][key]['amplitude'],
            'frequency': params_tree['pulsations'][key]['frequency'],
            'start_phase': params_tree['pulsations'][key]['start_phase'],
            'mode_axis_phi': params_tree['pulsations'][key].get('mode_axis_phi', 0),
            'mode_axis_theta': params_tree['pulsations'][key].get('mode_axis_theta', 0)
        } for key in params_tree['pulsations'].keys()] if 'pulsations' in params_tree.keys() else None

    return dict(
        surface_potential=params_tree['surface_potential'],
        synchronicity=params_tree.get('synchronicity', 1.0),
        t_eff=params_tree['t_eff'],
        gravity_darkening=params_tree.get('gravity_darkening', 1.0),
        albedo=params_tree.get('albedo', 1.0),
        metallicity=params_tree.get('metallicity', 0.0),
        **{"spots": spots} if spots is not None else {},
        **{"pulsations": pulsations} if pulsations is not None else {},
        **{"discretization_factor": params_tree["discretization_factor"]}
        if params_tree.get("discretization_factor") else {},
        **{"mass": params_tree["mass"]} if params_tree.get("mass") else {},
    )


def serialize_system_kwargs(**kwargs):
    no_prefix = {k.split(params.PARAM_PARSER, 1)[1]: v for k, v in kwargs.items() if str(k).startswith('system')}

    return dict(
        argument_of_periastron=no_prefix.get('argument_of_periastron', 90.0),
        gamma=no_prefix.get('gamma', 0.0),
        period=no_prefix["period"],
        eccentricity=no_prefix.get('eccentricity', 0.0),
        inclination=no_prefix.get('inclination'),
        primary_minimum_time=no_prefix.get('primary_minimum_time'),
        **{"semi_major_axis": no_prefix["semi_major_axis"]} if no_prefix.get("semi_major_axis") else {},
        **{"mass_ratio": no_prefix["mass_ratio"]} if no_prefix.get("mass_ratio") else {},
        **{"asini": no_prefix["asini"]} if no_prefix.get("asini") else {},
        **{"additional_light": no_prefix["additional_light"]} if no_prefix.get("additional_light") else {},
        **{"phase_shift": no_prefix["phase_shift"]} if no_prefix.get("phase_shift") else {},
    )


def serialize_primary_kwargs(**kwargs):
    return _serialize_star_kwargs(component='primary', **kwargs)


def serialize_secondary_kwargs(**kwargs):
    return _serialize_star_kwargs(component='secondary', **kwargs)


def prepare_binary(discretization=3, verify=False, **kwargs):
    """
    Setup binary system.
    If `beta` (gravity darkening factor), `albedo`, `metallicity` or `synchronicity` is not supplied,
    then `1.0` is used as their default value.

    :param discretization: float;
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
        * **additional_light** * -- float;
        * **phase_shift** * -- float;

    :return: elisa.binary_system.system.BinarySystem;
    """
    kwargs.update({"p__discretization_factor": discretization})
    primary_kwargs = serialize_primary_kwargs(**kwargs)
    secondary_kwargs = serialize_secondary_kwargs(**kwargs)
    system_kwargs = serialize_system_kwargs(**kwargs)
    json = {
        "primary": dict(**primary_kwargs),
        "secondary": dict(**secondary_kwargs),
        "system": dict(**system_kwargs)
    }
    return BinarySystem.from_json(json, _verify=verify)


def synthetic_binary(xs, discretization, morphology, observer, _raise_invalid_morphology, **kwargs):
    """
    :param _raise_invalid_morphology: bool; if morphology of system is different as expected, raise MorphologyError
    :param xs: Union[List, numpy.array];
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
        * **additional_light** * -- float;
        * **phase_shift** * -- float;
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
    :return: Tuple[numpy.array, str];
    :raise: elisa.base.errors.MorphologyError;
    """
    binary = prepare_binary(discretization, **kwargs)
    observer._system = binary

    lc = observer.observe.lc(phases=xs, normalize=True)
    return lc[1]


def prepare_central_rv_binary(**kwargs):
    return BinarySystem.from_json(kwargs, _verify=False)


def central_rv_synthetic(xs, observer, **kwargs):
    """

    :param xs: Union[List, numpy.array];
    :param observer: elisa.observer.observer.Observer; instance
    :param kwargs: Dict;
    :**kwargs options**:
        * **argument_of_periastron** * -- float;
        * **eccentricity** * -- float;
        * **inclination** * -- float;
        * **gamma** * -- float;
        * **asini** * --float;
        * **mass_ratio** * --float;
        * **primary_minimum_time** * -- float;
        * **period** * -- float;

    :return: dict;
    """

    kwargs.update({
        f"primary{params.PARAM_PARSER}surface_potential": 100,
        f"secondary{params.PARAM_PARSER}surface_potential": 100,
        f"primary{params.PARAM_PARSER}t_eff": 10000.0,
        f"secondary{params.PARAM_PARSER}t_eff": 10000.0,
        f"primary{params.PARAM_PARSER}metallicity": 0.0,
        f"secondary{params.PARAM_PARSER}metallicity": 0.0,
    })

    xs, kwargs = rvt_layer_resolver(xs, **kwargs)

    primary_kwargs = serialize_primary_kwargs(**kwargs)
    secondary_kwargs = serialize_secondary_kwargs(**kwargs)
    system_kwargs = serialize_system_kwargs(**kwargs)

    json = {
        "primary": dict(**primary_kwargs),
        "secondary": dict(**secondary_kwargs),
        "system": dict(**system_kwargs)
    }

    kind_of = resolve_json_kind(data=json, _sin=True)
    observable = prepare_central_rv_binary(**json) if kind_of in ["std"] \
        else RadialVelocitySystem(**RadialVelocitySystem.prepare_json(json["system"]))
    observer._system = observable
    _, rv = observer.observe.rv(phases=xs, normalize=False)
    return rv


def rvt_layer_resolver(xs, **kwargs):
    """
    If kwargs contain `period` and `primary_minimum_time`, then xs is expected to be JD time not phases.
    Then, xs has to be converted to phases.

    :param xs: Union[List, numpy.array];
    :param kwargs: dict;
    :return:
    """

    if params.is_time_dependent(list(kwargs.keys())):
        t0 = kwargs.pop(params.PARAM_PARSER.join(['system', 'primary_minimum_time']))
        period = kwargs[params.PARAM_PARSER.join(['system', 'period'])]
        xs = t_layer.jd_to_phase(t0, period, xs)
    return xs, kwargs


def time_layer_resolver(xs, x0):
    """
    If `x0' contain `period` and `primary_minimum_time`, then xs is expected to be JD time not phases.
    Then, xs has to be converted to phases.

    :param xs: Union[Dict, numpy.array];
    :param x0: dict;
    :return:
    """
    if params.is_time_dependent(x0.keys()):
        t0 = x0['system']['primary_minimum_time']['value']
        period = x0['system']['period']['value']
        xs = t_layer.jd_to_phase(t0, period, xs)
    return xs, x0
