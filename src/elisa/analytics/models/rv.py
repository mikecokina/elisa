"""Radial velocity synthetic model generation for binary system fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elisa.analytics.models import serializers
from elisa.analytics.tools.utils import time_layer_resolver
from elisa.base.error import InitialParamsError
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from elisa.binary_system.utils import resolve_json_kind

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.observer.observer import Observer
    from elisa.types import Float


def prepare_central_rv_binary(**kwargs: Any) -> BinarySystem:
    """Set up binary system from initial parameters.

    Creates a BinarySystem instance from a complete set of model parameters
    in flat format. No verification of parameters is performed.

    :param kwargs: Complete set of model parameters in flat format
        (format: ``{'parameter@name': value, ...}``).
    :type kwargs: dict[str, Any]
    :returns: Initialized binary system instance.
    :rtype: BinarySystem
    """
    return BinarySystem.from_json(kwargs, _verify=False)


def central_rv_synthetic(
    phases: NDArray[Float],
    observer: Observer,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Generate synthetic radial velocity curve for binary system.

    Generates a synthetic radial velocity (RV) curve of a binary system
    based on a set of model parameters. The function supports both
    standard and community RV system types.

    :param phases: Orbital phases (in range 0-1) for which RV curve will be generated.
    :type phases: NDArray[Float]
    :param observer: Observer instance with passband and system configuration.
    :type observer: Observer
    :param kwargs: Model parameters in flat format. Available parameters include:

        * ``system@argument_of_periastron`` - Argument of periastron (degrees or radians)
        * ``system@eccentricity`` - Orbital eccentricity
        * ``system@inclination`` - Orbital inclination (degrees)
        * ``primary@mass`` - Primary component mass (solar masses)
        * ``secondary@mass`` - Secondary component mass (solar masses)
        * ``system@gamma`` - Systemic radial velocity (km/s)
        * ``system@asini`` - Semi-major axis times sine of inclination (solar radii)
        * ``system@mass_ratio`` - Mass ratio (secondary/primary)
        * ``system@primary_minimum_time`` - Time of primary eclipse
        * ``system@period`` - Orbital period (days)

    :type kwargs: dict[str, Any]
    :returns: Radial velocities for each component (primary and secondary).
    :rtype: dict[str, NDArray[Float]]
    :raises InitialParamsError: If initial parameters lead to an unknown model type.
    """
    # Set default values for parameters not provided
    kwargs.update(
        {
            "primary@surface_potential": 100,
            "secondary@surface_potential": 100,
            "primary@t_eff": 10000.0,
            "secondary@t_eff": 10000.0,
            "primary@metallicity": 10000.0,
            "secondary@metallicity": 10000.0,
        },
    )

    x_data_resolved, kwargs = time_layer_resolver(phases, pop=True, **kwargs)

    # Extract component-specific parameters
    system_kwargs = serializers.serialize_system_kwargs(**kwargs)
    primary_kwargs = serializers.serialize_primary_kwargs(**kwargs)
    secondary_kwargs = serializers.serialize_secondary_kwargs(**kwargs)

    # Build system configuration dictionary
    json_config = {
        "primary": dict(**primary_kwargs),
        "secondary": dict(**secondary_kwargs),
        "system": dict(**system_kwargs),
    }

    # Determine system type and create appropriate observable
    kind_of = resolve_json_kind(data=json_config, _sin=True)

    if kind_of == "std":
        observable = prepare_central_rv_binary(**json_config)
    elif kind_of == "community":
        observable = RadialVelocitySystem(
            **RadialVelocitySystem.prepare_json(json_config["system"]),
        )
    else:
        error_msg = "Initial parameters led to unknown model."
        raise InitialParamsError(error_msg)

    # Set observer system and retrieve RV
    observer._system = observable  # noqa: SLF001
    observer._system_cls = type(observable)  # noqa: SLF001
    _, rv = observer.observe.rv(phases=x_data_resolved, normalize=False)

    return rv
