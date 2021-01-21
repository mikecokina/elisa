from ... pulse.container_ops import (
    incorporate_pulsations_to_model,
    generate_harmonics,
    complex_displacement
)


def build_pulsations(system, incorporate_perturbations):
    """
    Adds pulsations to stellar model.

    :param system: elisa.single_system.contaier.PositionContainer; instance
    :param incorporate_perturbations: bool; if True, only necessary pre-requisition quantities for evaluation of
                                          pulsations are calculated. The actual perturbations of surface quantities is
                                          then done by `pulse.container_ops.incorporate_pulsations_to_model`
    :return: elisa.single_system.contaier.PositionContainer; instance
    """
    if system.star.has_pulsations():
        system.star = generate_harmonics(system.star, com_x=0, phase=system.position.phase, time=system.time)
        system.star = complex_displacement(system.star, scale=1.0)
        if incorporate_perturbations:
            args = system.star, 0.0, system.position.phase, 1.0
            system.star = incorporate_pulsations_to_model(*args)
    return system
