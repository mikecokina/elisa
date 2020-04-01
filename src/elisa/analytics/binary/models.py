from elisa.binary_system import t_layer
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

    # sorting pulsations and spots back to their respective hierarchies for easier access
    params_tree = {'spots': {}, 'pulsations': {}}
    for param_name, param_val in no_prefix.items():
        if params.PARAM_PARSER not in param_name:
            continue

        identificators = param_name.split(params.PARAM_PARSER)

        if identificators[1] not in params_tree[identificators[0]].keys():
            params_tree[identificators[0]][identificators[1]] = {}

        params_tree[identificators[0]][identificators[1]][identificators[2]] = param_val

    spot_names = params_tree['spots'].keys()
    spots = [
        {
            'longitude': params_tree['spots'][key]['longitude'],
            'latitude': params_tree['spots'][key]['latitude'],
            'angular_radius': params_tree['spots'][key]['angular_radius'],
            'temperature_factor': params_tree['spots'][key]['temperature_factor']
        } for key in spot_names] if len(spot_names) > 0 else None

    pulsation_names = params_tree['pulsations'].keys()
    pulsations = [
        {
            'l': params_tree['pulsations'][key]['l'],
            'm': params_tree['pulsations'][key]['m'],
            'amplitude': params_tree['pulsations'][key]['amplitude'],
            'frequency': params_tree['pulsations'][key]['frequency'],
            'start_phase': params_tree['pulsations'][key]['start_phase'],
            'mode_axis_phi': params_tree['pulsations'][key].get('mode_axis_phi', 0),
            'mode_axis_theta': params_tree['pulsations'][key].get('mode_axis_theta', 0)
        } for key in pulsation_names] if len(pulsation_names) > 0 else None

    return dict(
        surface_potential=no_prefix['surface_potential'],
        synchronicity=no_prefix.get('synchronicity', 1.0),
        t_eff=no_prefix['t_eff'],
        gravity_darkening=no_prefix.get('gravity_darkening', 1.0),
        albedo=no_prefix.get('albedo', 1.0),
        metallicity=no_prefix.get('metallicity', 0.0),
        spots=spots,
        pulsations=pulsations,
        **{"discretization_factor": no_prefix["discretization_factor"]}
        if no_prefix.get("discretization_factor") else {},
        **{"mass": no_prefix["mass"]} if no_prefix.get("mass") else {},
    )


def serialize_system_kwargs(**kwargs):
    return dict(
        argument_of_periastron=kwargs.get('argument_of_periastron', 90.0),
        gamma=kwargs.get('gamma', 0.0),
        period=kwargs["period"],
        eccentricity=kwargs.get('eccentricity', 0.0),
        inclination=kwargs.get('inclination'),
        primary_minimum_time=0.0,
        **{"semi_major_axis": kwargs["semi_major_axis"]} if kwargs.get("semi_major_axis") else {},
        **{"mass_ratio": kwargs["mass_ratio"]} if kwargs.get("mass_ratio") else {},
        **{"asini": kwargs["asini"]} if kwargs.get("asini") else {},
        **{"additional_light": kwargs["additional_light"]} if kwargs.get("additional_light") else {},
        **{"phase_shift": kwargs["phase_shift"]} if kwargs.get("phase_shift") else {},
    )


def serialize_primary_kwargs(**kwargs):
    return _serialize_star_kwargs(component='p__', **kwargs)


def serialize_secondary_kwargs(**kwargs):
    return _serialize_star_kwargs(component='s__', **kwargs)


def prepare_binary(discretization=3, **kwargs):
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
    return BinarySystem.from_json(json, _verify=False)


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

    if params.is_overcontact(morphology) and not params.is_overcontact(binary.morphology) and _raise_invalid_morphology:
        raise error.MorphologyError(f'Expected morphology is {morphology} but obtained is {binary.morphology}')

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
        * **p__mass** * -- float;
        * **s__mass** * -- float;
        * **gamma** * -- float;
        * **asini** * --float;
        * **mass_ratio** * --float;
        * **primary_minimum_time** * -- float;
        * **period** * -- float;

    :return: Tuple;
    """

    kwargs.update({
        "p__surface_potential": 100,
        "s__surface_potential": 100,
        "p__t_eff": 10000.0,
        "s__t_eff": 10000.0,
        "p__metallicity": 10000.0,
        "s__metallicity": 10000.0,
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
        t0 = kwargs.pop('primary_minimum_time')
        period = kwargs['period']
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
        t0 = x0['primary_minimum_time']['value']
        period = x0['period']['value']
        xs = t_layer.jd_to_phase(t0, period, xs)
    return xs, x0
