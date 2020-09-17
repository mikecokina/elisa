from .. models import serializers
from .. tools.utils import time_layer_resolver
from ... base.error import InitialParamsError
from ... binary_system.curves.community import RadialVelocitySystem
from ... binary_system.system import BinarySystem
from ... binary_system.utils import resolve_json_kind


def central_rv_synthetic(x_data, observer, **kwargs):
    """

    :param x_data: Union[List, numpy.array];
    :param observer: elisa.observer.observer.Observer; instance
    :param kwargs: Dict;
    :**kwargs options**:
        * **system@argument_of_periastron** * -- float;
        * **system@eccentricity** * -- float;
        * **system@inclination** * -- float;
        * **primary@mass** * -- float;
        * **secondary@mass** * -- float;
        * **system@gamma** * -- float;
        * **system@asini** * --float;
        * **system@mass_ratio** * --float;
        * **system@primary_minimum_time** * -- float;
        * **system@period** * -- float;

    :return: Tuple;
    """

    kwargs.update({
        "primary@surface_potential": 100,
        "secondary@surface_potential": 100,
        "primary@t_eff": 10000.0,
        "secondary@t_eff": 10000.0,
        "primary@metallicity": 10000.0,
        "secondary@metallicity": 10000.0,
    })

    x_data, kwargs = time_layer_resolver(x_data, pop=True, **kwargs)

    system_kwargs = serializers.serialize_system_kwargs(**kwargs)
    primary_kwargs = serializers.serialize_primary_kwargs(**kwargs)
    secondary_kwargs = serializers.serialize_secondary_kwargs(**kwargs)

    json = {
        "primary": dict(**primary_kwargs),
        "secondary": dict(**secondary_kwargs),
        "system": dict(**system_kwargs)
    }

    kind_of = resolve_json_kind(data=json, _sin=True)

    if kind_of in ["std"]:
        observable = prepare_central_rv_binary(**json)
    elif kind_of in ["community"]:
        observable = RadialVelocitySystem(**RadialVelocitySystem.prepare_json(json["system"]))
    else:
        raise InitialParamsError("Initial parameters led to unknown model.")

    observer._system = observable
    _, rv = observer.observe.rv(phases=x_data, normalize=False)

    return rv


def prepare_central_rv_binary(**kwargs):
    return BinarySystem.from_json(kwargs, _verify=False)
