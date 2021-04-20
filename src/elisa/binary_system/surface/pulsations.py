from .. import utils as bsutils
from ... pulse.container_ops import (
    incorporate_pulsations_to_model,
    generate_harmonics
)


def build_harmonics(system, component, components_distance):
    """
    Adds pre-calculated harmonics for the respective pulsation modes

    :param system: elisa.binary_system.contaier.OrbitalPositionContainer; instance
    :param component: Union[str, None];
    :param components_distance: float;
    :return: elisa.binary_system.contaier.OrbitalPositionContainer; instance
    """
    components = bsutils.component_to_list(component)
    for component in components:
        star = getattr(system, component)
        if star.has_pulsations():
            phase = bsutils.calculate_rotational_phase(system, component)
            com_x = 0 if component == 'primary' else components_distance
            generate_harmonics(star, com_x=com_x, phase=phase, time=system.time)


def build_perturbations(system, component, components_distance):
    """
    adds position perturbations to container mesh

    :param system: elisa.binary_system.contaier.OrbitalPositionContainer; instance
    :param component: Union[str, None];
    :param components_distance: float;
    :return: elisa.binary_system.contaier.OrbitalPositionContainer; instance
    """
    components = bsutils.component_to_list(component)
    for component in components:
        star = getattr(system, component)
        if star.has_pulsations():
            phase = bsutils.calculate_rotational_phase(system, component)
            com_x = 0 if component == 'primary' else components_distance
            incorporate_pulsations_to_model(star, com_x=com_x, phase=phase, scale=system.semi_major_axis)
    return system
