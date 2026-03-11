"""Synthetic light curve generation for binary system fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elisa.analytics.models.serializers import (
    serialize_primary_kwargs,
    serialize_secondary_kwargs,
    serialize_system_kwargs,
)
from elisa.binary_system.system import BinarySystem

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.observer.observer import Observer
    from elisa.types import Float


def prepare_binary(
    *,
    discretization: Float = 5,
    _verify: bool = False,
    **kwargs: Any,
) -> BinarySystem:
    """Set up binary system from initial parameters.

    Creates a BinarySystem instance from a complete set of model parameters
    in flat format. Default values are applied for parameters not explicitly
    supplied:

    * If ``metallicity`` is not provided, 0 is used
    * If ``synchronicity`` is not provided, 1.0 is used

    :param discretization: Primary component's surface discretization factor.
    :type discretization: Float
    :param _verify: Verify input JSON parameters for correctness.
    :type _verify: bool
    :param kwargs: Complete set of model parameters in flat format
        (format: ``{'parameter@name': value, ...}``).
    :type kwargs: dict[str, Any]
    :returns: Initialized binary system instance.
    :rtype: BinarySystem
    """
    kwargs.update({"primary@discretization_factor": discretization})
    primary_kwargs = serialize_primary_kwargs(**kwargs)
    secondary_kwargs = serialize_secondary_kwargs(**kwargs)
    system_kwargs = serialize_system_kwargs(**kwargs)

    json_config = {
        "primary": dict(**primary_kwargs),
        "secondary": dict(**secondary_kwargs),
        "system": dict(**system_kwargs),
    }

    return BinarySystem.from_json(json_config, _verify=_verify)


def synthetic_binary(
    phases: NDArray[Float],
    discretization: Float,
    observer: Observer,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Generate synthetic light curve for binary system.

    Generates a synthetic light curve of a binary system based on a set of
    model parameters. The function returns light curves in the specified
    passband(s) normalized to the baseline flux.

    The structure of ``kwargs`` follows the flat format used in the
    BinarySystem.from_json() function, employing
    ``{'parameter@name': value, ...}`` instead of nested structure.
    Default units are as defined in ``elisa.units``.

    :param phases: Orbital phases (in range 0-1) for which LC will be generated.
    :type phases: NDArray[Float]
    :param discretization: Primary component's surface discretization factor.
    :type discretization: Float
    :param observer: Observer instance with passband and system configuration.
    :type observer: Observer
    :param kwargs: Model parameters in flat format. Available parameters include
        system, primary, and secondary component parameters (see
        BinarySystem.from_json() for full parameter list).
    :type kwargs: dict[str, Any]
    :returns: Light curves in format ``{'passband_name': normalized_LC}``.
    :rtype: dict[str, NDArray[Float]]
    """
    binary = prepare_binary(discretization=discretization, **kwargs)
    observer._system = binary  # noqa: SLF001

    lc = observer.observe.lc(phases=phases, normalize=True)
    return lc[1]
