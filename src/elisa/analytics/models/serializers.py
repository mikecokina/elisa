"""Model parameter serializers for binary system fitting."""

from __future__ import annotations

import re
from typing import Any

from elisa import const
from elisa.analytics.params.parameters import deflate_phenomena


def serialize_system_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Extract system-related parameters for synthetic model functions during fitting.

    The function extracts system-related parameters used in synthetic model functions
    during the fit. Those parameters are then returned as the `system` component of
    the input JSON used to initialize the BinarySystem instance.

    :param kwargs: Model parameters in flat format.
    :type kwargs: dict[str, Any]
    :returns: System-related model parameters.
    :rtype: dict[str, Any]
    """
    return dict(
        argument_of_periastron=kwargs.get("system@argument_of_periastron", const.HALF_PI),
        gamma=kwargs.get("system@gamma", 0.0),
        period=kwargs["system@period"],
        eccentricity=kwargs.get("system@eccentricity", 0.0),
        primary_minimum_time=0.0,
        **{"inclination": kwargs["system@inclination"]} if kwargs.get("system@inclination") else {},
        **{"semi_major_axis": kwargs["system@semi_major_axis"]} if kwargs.get("system@semi_major_axis") else {},
        **{"mass_ratio": kwargs["system@mass_ratio"]} if kwargs.get("system@mass_ratio") else {},
        **{"asini": kwargs["system@asini"]} if kwargs.get("system@asini") else {},
        **{"additional_light": kwargs["system@additional_light"]} if kwargs.get("system@additional_light") else {},
        **{"phase_shift": kwargs["system@phase_shift"]} if kwargs.get("system@phase_shift") else {},
    )


def _serialize_star_kwargs(component: str, **kwargs: Any) -> dict[str, Any]:
    """Extract component-related parameters for synthetic model functions during fitting.

    Extracts component-related parameters used in synthetic model functions during
    the fit. Those parameters are then returned as the `primary` or `secondary`
    component of the input JSON used to initialize the BinarySystem instance.

    :param component: Component identifier (`primary` or `secondary`).
    :type component: str
    :param kwargs: Model parameters in flat format.
    :type kwargs: dict[str, Any]
    :returns: Component-related model parameters.
    :rtype: dict[str, Any]
    """

    def _make_key(prop: str) -> str:
        """Create a prefixed parameter key."""
        return f"{component}@{prop}"

    # Extract and process phenomena parameters
    phenomena_map: dict[str, list[dict[str, Any]]] = {"spots": [], "pulsations": []}

    for phenom_name in phenomena_map:
        singular_name = phenom_name[:-1]  # e.g., 'spot' from 'spots'
        pattern = rf"^(?=.*\b{singular_name}\b)(?=.*\b{component}\b).*$"

        phenomena_params = {key: value for key, value in kwargs.items() if re.search(pattern, key)}
        phenomena_params = deflate_phenomena(phenomena_params)
        phenomena_map[phenom_name] = [
            {key: val for key, val in value.items() if key != "label"} for value in phenomena_params.values()
        ]

    spots = phenomena_map["spots"] or []
    pulsations = phenomena_map["pulsations"] or []

    return dict(
        surface_potential=kwargs[_make_key("surface_potential")],
        synchronicity=kwargs.get(_make_key("synchronicity"), 1.0),
        t_eff=kwargs[_make_key("t_eff")],
        **(
            {
                "gravity_darkening": kwargs[_make_key("gravity_darkening")],
            }
            if kwargs.get(_make_key("gravity_darkening"))
            else {}
        ),
        **(
            {
                "albedo": kwargs[_make_key("albedo")],
            }
            if kwargs.get(_make_key("albedo"))
            else {}
        ),
        **(
            {
                "metallicity": kwargs[_make_key("metallicity")],
            }
            if kwargs.get(_make_key("metallicity"))
            else {}
        ),
        **(
            {
                "mass": kwargs[_make_key("mass")],
            }
            if kwargs.get(_make_key("mass"))
            else {}
        ),
        **(
            {
                "discretization_factor": kwargs[_make_key("discretization_factor")],
            }
            if kwargs.get(_make_key("discretization_factor"))
            else {}
        ),
        **(
            {
                "atmosphere": kwargs[_make_key("atmosphere")],
            }
            if kwargs.get(_make_key("atmosphere"))
            else {}
        ),
        **(
            {
                "limb_darkening_coefficients": kwargs[
                    _make_key(
                        "limb_darkening_coefficients",
                    )
                ],
            }
            if kwargs.get(_make_key("limb_darkening_coefficients"))
            else {}
        ),
        spots=spots,
        pulsations=pulsations,
    )


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
