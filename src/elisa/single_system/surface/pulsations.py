from ... pulse.container_ops import (
    incorporate_pulsations_to_model,
    generate_harmonics,
)


def build_pulsations(system):
    """
    Adds pulsations to stellar model.

    :param system: elisa.single_system.contaier.PositionContainer; instance
    :return: elisa.single_system.contaier.PositionContainer; instance
    """
    if system.star.has_pulsations():
        system.star = generate_harmonics(system.star, com_x=0, phase=system.position.phase, time=system.time)
        args = system.star, 0.0, system.position.phase, 1.0
        system.star = incorporate_pulsations_to_model(*args)
    return system
