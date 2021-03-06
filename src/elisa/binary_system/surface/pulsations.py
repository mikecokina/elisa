from .. import utils as bsutils
from ... pulse.container_ops import incorporate_pulsations_to_model
from ... pulse.pulsations import generate_harmonics


def build_pulsations(system, component, components_distance):
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
            star = generate_harmonics(star, com_x=com_x, phase=phase, time=system.time)
            incorporate_pulsations_to_model(star, com_x=com_x, scale=system.semi_major_axis, phase=phase)
    return system
