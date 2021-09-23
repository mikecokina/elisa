from .. models import serializers
from .. tools.utils import time_layer_resolver
from ... base.error import InitialParamsError
from ... binary_system.curves.community import RadialVelocitySystem
from ... binary_system.system import BinarySystem
from ... binary_system.utils import resolve_json_kind


def prepare_central_rv_binary(**kwargs):
    """
    Setup binary system from initial parameter object.

    :param kwargs: Dict; complete set of model parameters in flat format {'parameter@name': value, }
    :return: elisa.binary_system.system.BinarySystem
    """
    return BinarySystem.from_json(kwargs, _verify=False)


def central_rv_synthetic(phases, observer, **kwargs):
    """
    Function returns synthetic RV curve of binary system based on a set of model parameters.

    :param phases: Union[List, numpy.array]; photometric phases in which the curves will be generated
    :param observer: elisa.observer.observer.Observer; observer instance
    :param kwargs: Dict;
    :**kwargs options**:
        * **'system@argument_of_periastron'**: float;
        * **'system@eccentricity'**: float;
        * **'system@inclination'**: float;
        * **'primary@mass'**: float;
        * **'secondary@mass'**: float;
        * **'system@gamma'**: float;
        * **'system@asini'**: float;
        * **'system@mass_ratio'**: float;
        * **'system@primary_minimum_time'**: float;
        * **'system@period'**: float;

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

    x_data_resolved, kwargs = time_layer_resolver(phases, pop=True, **kwargs)

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
    observer._system_cls = type(observable)
    _, rv = observer.observe.rv(phases=x_data_resolved, normalize=False)

    return rv
