from ... binary_system.system import BinarySystem
from .. models.serializers import (
    serialize_primary_kwargs,
    serialize_secondary_kwargs,
    serialize_system_kwargs
)


def prepare_binary(discretization=5, _verify=False, **kwargs):
    """
    Setup binary system from initial parameter object.
    If `metallicity` is not supplied 0 is used to initialize the parameter. Similarly, default value of `synchronicity`
    parameter is 1.0.

    :param _verify: bool; verify input json
    :param discretization: float; primary component's surface discretization factor
    :param kwargs: Dict; complete set of model parameters in flat format {'parameter@name': value, }
    :return: elisa.binary_system.system.BinarySystem
    """
    kwargs.update({"primary@discretization_factor": discretization})
    primary_kwargs = serialize_primary_kwargs(**kwargs)
    secondary_kwargs = serialize_secondary_kwargs(**kwargs)
    system_kwargs = serialize_system_kwargs(**kwargs)
    json = {
        "primary": dict(**primary_kwargs),
        "secondary": dict(**secondary_kwargs),
        "system": dict(**system_kwargs)
    }
    return BinarySystem.from_json(json, _verify=_verify)


def synthetic_binary(phases, discretization, observer, **kwargs):
    """
    Function returns synthetic light curve of binary system based on a set of model parameters.

    :param phases: Union[List, numpy.array]; photometric phases in which the curves will be generated
    :param discretization: float; primary component's surface discretization factor
    :param observer: elisa.observer.observer.Observer; observer instance
    :param kwargs: Dict; The structure of the `kwargs` is similar to the input JSON used in BinarySystem.from_json()
                         function, with exception of using the flat format {'parameter@name': value, } instead of the
                         nested structure. The default units for parameters are the default input units
                         (see elisa.units).

    :return: Dict[str, numpy.array]; light curves in in format {'passband_name': LC in filter}
    """
    binary = prepare_binary(discretization, **kwargs)
    observer._system = binary

    lc = observer.observe.lc(phases=phases, normalize=True)
    return lc[1]
