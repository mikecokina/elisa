"""Model parameter serializers for binary system fitting."""

from __future__ import annotations

from typing import Any

from elisa import const
from elisa.analytics.params.parameters import deflate_phenomena

# Optional system-level keys extracted when present.
_OPTIONAL_SYSTEM_KEYS: tuple[str, ...] = (
    "inclination",
    "semi_major_axis",
    "mass_ratio",
    "asini",
    "additional_light",
    "phase_shift",
)

# Optional per-component keys extracted when present.
_OPTIONAL_STAR_KEYS: tuple[str, ...] = (
    "gravity_darkening",
    "albedo",
    "metallicity",
    "mass",
    "discretization_factor",
    "atmosphere",
    "limb_darkening_coefficients",
)


def serialize_system_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Extract system-related parameters for synthetic model functions during fitting.

    The function extracts system-related parameters used in synthetic model functions
    during the fit. Those parameters are then returned as the ``system`` component of
    the input JSON used to initialize the :class:`BinarySystem` instance.

    :param kwargs: Model parameters in flat format.
    :type kwargs: dict[str, Any]
    :returns: System-related model parameters.
    :rtype: dict[str, Any]
    """
    result: dict[str, Any] = {
        "argument_of_periastron": kwargs.get("system@argument_of_periastron", const.HALF_PI),
        "gamma": kwargs.get("system@gamma", 0.0),
        "period": kwargs["system@period"],
        "eccentricity": kwargs.get("system@eccentricity", 0.0),
        "primary_minimum_time": 0.0,
    }
    # Single lookup per optional key — avoids the double-lookup pattern
    # (kwargs.get(k) check followed by kwargs[k] fetch).
    for key in _OPTIONAL_SYSTEM_KEYS:
        val = kwargs.get(f"system@{key}")
        if val is not None:
            result[key] = val
    return result


def _serialize_star_kwargs(component: str, **kwargs: Any) -> dict[str, Any]:
    """Extract component-related parameters for synthetic model functions during fitting.

    Extracts component-related parameters used in synthetic model functions during
    the fit. Those parameters are then returned as the ``primary`` or ``secondary``
    component of the input JSON used to initialize the :class:`BinarySystem` instance.

    :param component: Component identifier (``primary`` or ``secondary``).
    :type component: str
    :param kwargs: Model parameters in flat format.
    :type kwargs: dict[str, Any]
    :returns: Component-related model parameters.
    :rtype: dict[str, Any]
    """
    prefix = f"{component}@"

    # Collect raw phenomenon keys in a single pass over kwargs.
    # Keys follow the format: {component}@{phenom_type}@{label}@{property}
    # e.g. "primary@spot@spot1@longitude"
    # Simple prefix + substring checks replace the original regex lookaheads.
    spots_raw: dict[str, Any] = {}
    pulsations_raw: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key.startswith(prefix):
            if "@spot@" in key:
                spots_raw[key] = value
            elif "@pulsation@" in key:
                pulsations_raw[key] = value

    spots = [
        {k: v for k, v in item.items() if k != "label"}
        for item in deflate_phenomena(spots_raw).values()
    ]
    pulsations = [
        {k: v for k, v in item.items() if k != "label"}
        for item in deflate_phenomena(pulsations_raw).values()
    ]

    result: dict[str, Any] = {
        "surface_potential": kwargs[f"{prefix}surface_potential"],
        "synchronicity": kwargs.get(f"{prefix}synchronicity", 1.0),
        "t_eff": kwargs[f"{prefix}t_eff"],
        "spots": spots,
        "pulsations": pulsations,
    }

    # Single lookup per optional key.
    for key in _OPTIONAL_STAR_KEYS:
        val = kwargs.get(f"{prefix}{key}")
        if val is not None:
            result[key] = val

    return result


def serialize_primary_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Extract primary component parameters for synthetic model functions during fitting.

    :param kwargs: Model parameters in flat format.
    :type kwargs: dict[str, Any]
    :returns: Primary component-related model parameters.
    :rtype: dict[str, Any]
    """
    return _serialize_star_kwargs(component="primary", **kwargs)


def serialize_secondary_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Extract secondary component parameters for synthetic model functions during fitting.

    :param kwargs: Model parameters in flat format.
    :type kwargs: dict[str, Any]
    :returns: Secondary component-related model parameters.
    :rtype: dict[str, Any]
    """
    return _serialize_star_kwargs(component="secondary", **kwargs)
