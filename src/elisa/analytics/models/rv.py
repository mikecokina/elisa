"""Radial velocity synthetic model generation for binary system fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import const, settings
from elisa import units as u
from elisa.analytics.models import serializers
from elisa.analytics.tools.utils import time_layer_resolver
from elisa.base.error import raise_elisa_parameters_error
from elisa.base.transform import StarProperties, SystemProperties
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.system import BinarySystem
from elisa.binary_system.utils import resolve_json_kind
from elisa.conf.enums import BinaryComponent, BinarySystemKind, RVMethod

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
    standard and community orbital parameterizations for kinematic RV.
    Radiometric RV requires complete physical component parameters; it is
    available for direct synthesis, not through the public RV fitting API.

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
    :raises ElisaParametersError: If initial parameters lead to an unknown model type.
    """
    x_data_resolved, kwargs = time_layer_resolver(phases, pop=True, **kwargs)
    system_kwargs = serializers.serialize_system_kwargs(**kwargs)
    if settings.RV_METHOD == RVMethod.KINEMATIC:
        kind = resolve_json_kind(data={
            "system": system_kwargs,
            BinaryComponent.PRIMARY: {"mass": kwargs.get("primary@mass")},
            BinaryComponent.SECONDARY: {"mass": kwargs.get("secondary@mass")},
        }, _sin=True)
        if kind == BinarySystemKind.STANDARD:
            # Match BinarySystem observer phase folding for the standard model.
            x_data_resolved = np.round(x_data_resolved % 1, 9)
            primary_mass = StarProperties.mass(kwargs["primary@mass"])
            secondary_mass = StarProperties.mass(kwargs["secondary@mass"])
            period = SystemProperties.period(system_kwargs["period"])
            inclination = SystemProperties.inclination(system_kwargs["inclination"])
            period_seconds = (period * u.DefaultBinarySystemUnits.system.period).to_value(u.TIME_UNIT)
            axis = (
                const.G * (primary_mass + secondary_mass) * period_seconds**2 / (4 * const.PI**2)
            ) ** (1.0 / 3)
            system_kwargs.update(
                mass_ratio=secondary_mass / primary_mass,
                asini=axis * np.sin(inclination) * u.m,
            )
        observable = RadialVelocitySystem(**RadialVelocitySystem.prepare_json(system_kwargs))
    elif settings.RV_METHOD == RVMethod.RADIOMETRIC:
        json_config = {
            BinaryComponent.PRIMARY: serializers.serialize_primary_kwargs(**kwargs),
            BinaryComponent.SECONDARY: serializers.serialize_secondary_kwargs(**kwargs),
            "system": system_kwargs,
        }
        if resolve_json_kind(data=json_config, _sin=True) != BinarySystemKind.STANDARD:
            raise_elisa_parameters_error("Radiometric RV requires a complete physical binary model.")
        observable = prepare_central_rv_binary(**json_config)
    else:
        raise_elisa_parameters_error(f"Unsupported RV computation method {settings.RV_METHOD!r}.")

    # Set observer system and retrieve RV
    observer._system = observable  # noqa: SLF001
    observer._system_cls = type(observable)  # noqa: SLF001
    _, rv = observer.observe.rv(phases=x_data_resolved, normalize=False)

    return rv
