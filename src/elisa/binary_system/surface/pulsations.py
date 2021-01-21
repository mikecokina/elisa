from .. import utils as bsutils
from ... pulse.container_ops import (
    incorporate_pulsations_to_model,
    generate_harmonics,
    complex_displacement
)


def build_pulsations(system, component, components_distance, incorporate_perturbations):
    """
    adds position perturbations to container mesh

    :param system: elisa.binary_system.contaier.OrbitalPositionContainer; instance
    :param component: Union[str, None];
    :param components_distance: float;
    :param incorporate_perturbations: bool; if True, only necessary pre-requisition quantities for evaluation of
                                          pulsations are calculated. The actual perturbations of surface quantities is
                                          then done by `pulse.container_ops.incorporate_pulsations_to_model`
    :return: elisa.binary_system.contaier.OrbitalPositionContainer; instance
    """
    components = bsutils.component_to_list(component)
    for component in components:
        star = getattr(system, component)
        if star.has_pulsations():
            phase = bsutils.calculate_rotational_phase(system, component)
            com_x = 0 if component == 'primary' else components_distance
            star = generate_harmonics(star, com_x=com_x, phase=phase, time=system.time)
            star = complex_displacement(star, scale=system.semi_major_axis)
            if incorporate_perturbations:
                incorporate_pulsations_to_model(star, com_x=com_x, phase=phase)
    return system
