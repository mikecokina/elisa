from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from elisa import const
from elisa.binary_system import utils as bsutils
from elisa.pulse.container_ops import (
    generate_harmonics,
    incorporate_pulsations_to_model,
)

if TYPE_CHECKING:
    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.types import Float

ComponentSelection: TypeAlias = Literal["primary", "secondary", "all", "both"]
SurfaceComponent: TypeAlias = Literal["primary", "secondary"]


def build_harmonics(
    system: OrbitalPositionContainer,
    component: ComponentSelection,
    components_distance: Float,
) -> OrbitalPositionContainer:
    """Add precomputed spherical harmonics for each pulsation mode.

    :param system: Orbital position container instance.
    :type system: OrbitalPositionContainer
    :param component: Component selector.
    :type component: Literal["primary", "secondary", "all", "both"]
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :return: Updated orbital position container.
    :rtype: OrbitalPositionContainer
    """
    components = bsutils.component_to_list(component)

    for component_name in components:
        star = getattr(system, component_name)
        pos_correction = bsutils.correction_to_com(
            system.position.distance,
            system.mass_ratio,
            system.secondary.com,
        )[0]
        asini = system.semi_major_axis * np.sin(system.inclination)

        if star.has_pulsations():
            phase = bsutils.calculate_rotational_phase(system, component_name)
            com_x = 0.0 if component_name == "primary" else components_distance

            # LTE effect
            time_correction = (star.com[0] - pos_correction) * asini / const.C
            generate_harmonics(
                star,
                com_x=com_x,
                phase=phase,
                time=system.time + time_correction,
            )

    return system


def build_perturbations(
    system: OrbitalPositionContainer,
    component: ComponentSelection,
    components_distance: Float,
) -> OrbitalPositionContainer:
    """Incorporate pulsation perturbations into the position container.

    :param system: Orbital position container instance.
    :type system: OrbitalPositionContainer
    :param component: Component selector.
    :type component: Literal["primary", "secondary", "all", "both"]
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :return: Updated orbital position container.
    :rtype: OrbitalPositionContainer
    """
    components = bsutils.component_to_list(component)

    for component_name in components:
        star = getattr(system, component_name)
        if star.has_pulsations():
            com_x = 0.0 if component_name == "primary" else components_distance
            incorporate_pulsations_to_model(
                star,
                com_x=com_x,
                scale=system.semi_major_axis,
            )

    return system
