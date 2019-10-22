from elisa import logger
from elisa.binary_system import utils as bsutils
from elisa.conf import config
from elisa.pulse import pulsations
from elisa.utils import is_empty

config.set_up_logging()
__logger__ = logger.getLogger("binary-system-temperature-module")


def build_temperature_distribution(system_container, components_distance, component="all",
                                   do_pulsations=False, phase=None):
    """
    Function calculates temperature distribution on across all faces.
    Value assigned to face is mean of values calculated in corners of given face.

    :param phase:
    :param do_pulsations:
    :param system_container: BinarySystem; instance
    :param components_distance: str
    :param component: `primary` or `secondary`
    :return:
    """
    if is_empty(component):
        __logger__.debug("no component set to build temperature distribution")
        return

    phase = 0 if phase is None else phase
    component = bsutils.component_to_list(component)

    for _component in component:
        component_instance = getattr(system_container, _component)

        __logger__.debug(f'computing effective temperature distibution '
                         f'on {_component} component name: {component_instance.name}')
        component_instance.temperatures = component_instance.calculate_effective_temperatures()

        if component_instance.has_spots():
            for spot_index, spot in component_instance.spots.items():
                __logger__.debug(f'computing temperature distribution of spot {spot_index} / {_component} component')
                spot.temperatures = spot.temperature_factor * component_instance.calculate_effective_temperatures(
                    gradient_magnitudes=spot.potential_gradient_magnitudes)

        __logger__.debug(f'renormalizing temperature map of components due to '
                         f'presence of spots in case of component {component}')
        component_instance.renormalize_temperatures()

        if component_instance.has_pulsations() and do_pulsations:
            __logger__.debug(f'adding pulsations to surface temperature distribution '
                             f'of the component instance: {_component}  / name: {component_instance.name}')

            com_x = 0 if _component == 'primary' else components_distance
            pulsations.set_misaligned_ralp(component_instance, phase, com_x=com_x)
            temp_pert, temp_pert_spot = pulsations.calc_temp_pert(component_instance, phase, system_container.period)
            component_instance.temperatures += temp_pert
            if component_instance.has_spots():
                for spot_idx, spot in component_instance.spots.items():
                    spot.temperatures += temp_pert_spot[spot_idx]

    if 'primary' in component and 'secondary' in component:
        __logger__.debug(f'calculating reflection effect with {config.REFLECTION_EFFECT_ITERATIONS} '
                         f'iterations.')
        system_container.reflection_effect(iterations=config.REFLECTION_EFFECT_ITERATIONS,
                                           components_distance=components_distance)
