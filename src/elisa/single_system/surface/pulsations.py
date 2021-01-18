from ... pulse.container_ops import incorporate_pulsations_to_model, generate_harmonics


def build_pulsations(system):
    """
    Adds pulsations to stellar model.

    :param system: elisa.single_system.contaier.PositionContainer; instance
    :return: elisa.single_system.contaier.PositionContainer; instance
    """
    if system.star.has_pulsations():
        system.star = generate_harmonics(system.star, com_x=0, phase=system.position.phase, time=system.time)
        system.star = incorporate_pulsations_to_model(system.star, com_x=0.0, phase=system.position.phase)
    return system
