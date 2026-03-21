from __future__ import annotations

from typing import TYPE_CHECKING

from elisa.pulse.container_ops import (
    generate_harmonics,
    incorporate_pulsations_to_model,
)

if TYPE_CHECKING:
    from elisa.single_system.container import SinglePositionContainer


def build_harmonics(system: SinglePositionContainer) -> SinglePositionContainer:
    """Add precomputed harmonics for pulsation modes to the star in the container.

    If the star stored in :obj:`system` has pulsation modes defined, this
    function computes harmonics for those modes and assigns the result back to
    the star attribute of the supplied container.

    :param system: Container holding single-system position and star data.
    :type system: elisa.single_system.container.SinglePositionContainer
    :return: The same container with harmonics added to ``system.star`` when applicable.
    :rtype: elisa.single_system.container.SinglePositionContainer
    """
    if system.star.has_pulsations():
        system.star = generate_harmonics(system.star, com_x=0, phase=system.position.phase, time=system.time)

    return system


def build_perturbations(system: SinglePositionContainer) -> SinglePositionContainer:
    """Add surface-geometry perturbations due to pulsations to the star in the container.

    If the star stored in :obj:`system` has pulsation modes defined, this
    function computes positional perturbations for the surface mesh and assigns
    the perturbed model back to the star attribute of the supplied container.

    :param system: Container holding single-system position and star data.
    :type system: elisa.single_system.container.SinglePositionContainer
    :return: The same container with pulsation perturbations applied to ``system.star`` when applicable.
    :rtype: elisa.single_system.container.SinglePositionContainer
    """
    if system.star.has_pulsations():
        args = (system.star, 0.0, 1.0)
        system.star = incorporate_pulsations_to_model(*args)

    return system
