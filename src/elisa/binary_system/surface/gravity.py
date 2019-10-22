from elisa import logger
from elisa.conf import config
from elisa.utils import is_empty
from elisa.binary_system import utils as bsutils
from elisa import umpy as up

config.set_up_logging()
__logger__ = logger.getLogger("binary-system-gravity-module")


def build_surface_gravity(system_container, components_distance=None, component="all"):
    """
    Function calculates gravity potential gradient magnitude (surface gravity) for each face.
    Value assigned to face is mean of values calculated in corners of given face.

    :param system_container: BinarySystem instance
    :param component: str; `primary` or `secondary`
    :param components_distance: float
    :return:
    """
    if is_empty(component):
        __logger__.debug("no component set to build surface gravity")
        return

    if is_empty(components_distance):
        raise ValueError('Component distance value was not supplied or is invalid.')

    component = bsutils.component_to_list(component)
    for _component in component:
        component_instance = getattr(system_container, _component)

        polar_gravity = system_container.calculate_polar_gravity_acceleration(_component, components_distance, logg=False)

        component_instance.polar_potential_gradient_magnitude = \
            system_container.calculate_polar_potential_gradient_magnitude(_component, components_distance)
        gravity_scalling_factor = polar_gravity / component_instance.polar_potential_gradient_magnitude

        __logger__.debug(f'computing potential gradient magnitudes distribution of {_component} component')
        component_instance.potential_gradient_magnitudes = system_container.calculate_face_magnitude_gradient(
            component=_component, components_distance=components_distance)

        component_instance.log_g = up.log10(
            gravity_scalling_factor * component_instance.potential_gradient_magnitudes)

        if component_instance.has_spots():
            for spot_index, spot in component_instance.spots.items():
                __logger__.debug(f'calculating surface SI unit gravity of {_component} component / {spot_index} spot')
                __logger__.debug(f'calculating distribution of potential gradient '
                                 f'magnitudes of spot index: {spot_index} / {_component} component')
                spot.potential_gradient_magnitudes = system_container.calculate_face_magnitude_gradient(
                    component=_component,
                    components_distance=components_distance,
                    points=spot.points, faces=spot.faces)

                spot.log_g = up.log10(gravity_scalling_factor * spot.potential_gradient_magnitudes)
