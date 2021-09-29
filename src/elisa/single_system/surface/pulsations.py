from ... pulse.container_ops import (
    incorporate_pulsations_to_model,
    generate_harmonics,
)


def build_harmonics(system):
    """
    Adds pre-calculated harmonics for the respective pulsation modes.

    :param system: elisa.single_system.container.SinglePositionContainer;
    """
    if system.star.has_pulsations():
        system.star = generate_harmonics(system.star, com_x=0, phase=system.position.phase, time=system.time)

    return system


def build_perturbations(system):
    """
    Function adds perturbation to the surface mesh due to pulsations.

    :param system: elisa.single_system.container.SinglePositionContainer; instance
    :return: elisa.single_system.container.SinglePositionContainer; instance
    """
    if system.star.has_pulsations():
        args = system.star, 0.0, 1.0
        system.star = incorporate_pulsations_to_model(*args)
    return system
