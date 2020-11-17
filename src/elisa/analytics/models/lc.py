from ... binary_system.system import BinarySystem
from .. models.serializers import (
    serialize_primary_kwargs,
    serialize_secondary_kwargs,
    serialize_system_kwargs
)


def prepare_binary(discretization=3, _verify=False, **kwargs):
    """
    Setup binary system.
    If `beta` (gravity darkening factor), `albedo`, `metallicity` or `synchronicity` is not supplied,
    then `1.0` is used as their default value.

    :param _verify: bool; verify input json
    :param discretization: float;
    :param kwargs: Dict;
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


def synthetic_binary(x_data, discretization, observer, **kwargs):
    """
    :param x_data: Union[List, numpy.array];
    :param discretization: float;
    :param observer: elisa.observer.observer.Observer; instance
    :param kwargs: Dict;
    :return: Dict[str, numpy.array];
    """
    binary = prepare_binary(discretization, **kwargs)
    observer._system = binary

    lc = observer.observe.lc(phases=x_data, normalize=True)
    return lc[1]
