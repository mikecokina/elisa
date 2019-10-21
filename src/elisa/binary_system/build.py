import numpy as np

from copy import copy

from elisa.binary_system.surface import mesh, faces
from elisa.binary_system.surface.mesh import add_spots_to_mesh
from elisa.conf import config
from elisa.binary_system import utils as bsutils
from elisa.utils import is_empty
from elisa.pulse import pulsations


# TODO: remove
def build_mesh(self, component="all", components_distance=None, **kwargs):
    """
    Build points of surface for primary or/and secondary component. Mesh is evaluated with spots.

    :param self: BinarySystem; instance
    :param component: str or empty
    :param components_distance: float
    :return:
    """
    if is_empty(component):
        self._logger.debug("no component set to build mesh")
        return

    if components_distance is None:
        raise ValueError('Argument `component_distance` was not supplied.')
    component = bsutils.component_to_list(component)

    for _component in component:
        component_instance = getattr(self, _component)
        # in case of spoted surface, symmetry is not used
        _a, _b, _c, _d = mesh.mesh_over_contact(self, component=_component, symmetry_output=True) \
            if self.morphology == 'over-contact' \
            else mesh.mesh_detached(
            self, component=_component, components_distance=components_distance, symmetry_output=True
        )
        component_instance.points = _a
        component_instance.point_symmetry_vector = _b
        component_instance.base_symmetry_points_number = _c
        component_instance.inverse_point_symmetry_matrix = _d

    add_spots_to_mesh(self, components_distance, component="all")


def build_surface_gravity(self, component="all", components_distance=None):
    """
    Function calculates gravity potential gradient magnitude (surface gravity) for each face.
    Value assigned to face is mean of values calculated in corners of given face.

    :param self: BinarySystem instance
    :param component: str; `primary` or `secondary`
    :param components_distance: float
    :return:
    """
    if is_empty(component):
        self._logger.debug("no component set to build surface gravity")
        return

    if is_empty(components_distance):
        raise ValueError('Component distance value was not supplied or is invalid.')

    component = bsutils.component_to_list(component)
    for _component in component:
        component_instance = getattr(self, _component)

        polar_gravity = self.calculate_polar_gravity_acceleration(_component, components_distance, logg=False)

        component_instance.polar_potential_gradient_magnitude = \
            self.calculate_polar_potential_gradient_magnitude(_component, components_distance)
        gravity_scalling_factor = polar_gravity / component_instance.polar_potential_gradient_magnitude

        self._logger.debug(f'computing potential gradient magnitudes distribution of {_component} component')
        component_instance.potential_gradient_magnitudes = self.calculate_face_magnitude_gradient(
            component=_component, components_distance=components_distance)

        component_instance.log_g = np.log10(
            gravity_scalling_factor * component_instance.potential_gradient_magnitudes)

        if component_instance.has_spots():
            for spot_index, spot in component_instance.spots.items():
                self._logger.debug(f'calculating surface SI unit gravity of {_component} component / {spot_index} spot')
                self._logger.debug(f'calculating distribution of potential gradient '
                                   f'magnitudes of spot index: {spot_index} / {_component} component')
                spot.potential_gradient_magnitudes = self.calculate_face_magnitude_gradient(
                    component=_component,
                    components_distance=components_distance,
                    points=spot.points, faces=spot.faces)

                spot.log_g = np.log10(gravity_scalling_factor * spot.potential_gradient_magnitudes)


def build_faces_orientation(self, component="all", components_distance=None):
    """
    Compute face orientation (normals) for each face.
    If pulsations are present, than calculate renormalized associated
    Legendree polynomials (rALS) for each pulsation mode.

    :param self: BinarySystem instance
    :param component: str; `primary` or `secondary`
    :param components_distance: float
    orbit with misaligned pulsations, where pulsation axis drifts with star
    :return:
    """
    if is_empty(component):
        self._logger.debug("no component set to build face orientation")
        return

    component = bsutils.component_to_list(component)
    com_x = {'primary': 0.0, 'secondary': components_distance}

    for _component in component:
        component_instance = getattr(self, _component)
        component_instance.set_all_surface_centres()
        component_instance.set_all_normals(com=com_x[_component])

        # here we calculate time independent part of the pulsation modes, renormalized Legendree polynomials for each
        # pulsation mode
        if component_instance.has_pulsations():
            pulsations.set_ralp(component_instance, com_x=com_x[_component])


def build_temperature_distribution(self, component="all", components_distance=None, do_pulsations=False, phase=None):
    """
    Function calculates temperature distribution on across all faces.
    Value assigned to face is mean of values calculated in corners of given face.

    :param phase:
    :param do_pulsations:
    :param self: BinarySystem; instance
    :param components_distance: str
    :param component: `primary` or `secondary`
    :return:
    """
    if is_empty(component):
        self._logger.debug("no component set to build temperature distribution")
        return

    phase = 0 if phase is None else phase
    component = bsutils.component_to_list(component)

    for _component in component:
        component_instance = getattr(self, _component)

        self._logger.debug(f'computing effective temperature distibution '
                           f'on {_component} component name: {component_instance.name}')
        component_instance.temperatures = component_instance.calculate_effective_temperatures()

        if component_instance.has_spots():
            for spot_index, spot in component_instance.spots.items():
                self._logger.debug(f'computing temperature distribution of spot {spot_index} / {_component} component')
                spot.temperatures = spot.temperature_factor * component_instance.calculate_effective_temperatures(
                    gradient_magnitudes=spot.potential_gradient_magnitudes)

        self._logger.debug(f'renormalizing temperature map of components due to '
                           f'presence of spots in case of component {component}')
        component_instance.renormalize_temperatures()

        if component_instance.has_pulsations() and do_pulsations:
            self._logger.debug(f'adding pulsations to surface temperature distribution '
                               f'of the component instance: {_component}  / name: {component_instance.name}')

            com_x = 0 if _component == 'primary' else components_distance
            pulsations.set_misaligned_ralp(component_instance, phase, com_x=com_x)
            temp_pert, temp_pert_spot = pulsations.calc_temp_pert(component_instance, phase, self.period)
            component_instance.temperatures += temp_pert
            if component_instance.has_spots():
                for spot_idx, spot in component_instance.spots.items():
                    spot.temperatures += temp_pert_spot[spot_idx]

    if 'primary' in component and 'secondary' in component:
        self._logger.debug(f'calculating reflection effect with {config.REFLECTION_EFFECT_ITERATIONS} '
                           f'iterations.')
        self.reflection_effect(iterations=config.REFLECTION_EFFECT_ITERATIONS,
                               components_distance=components_distance)


def build_surface_map(self, colormap=None, component="all", components_distance=None, return_map=False, phase=None):
    """
    Function calculates surface maps (temperature or gravity acceleration) for star and spot faces and it can return
    them as one array if return_map=True.

    :param phase:
    :param self: BinarySystem; instance
    :param return_map: bool; if True function returns arrays with surface map including star and spot segments
    :param colormap: switch for `temperature` or `gravity` colormap to create
    :param component: `primary` or `secondary` component surface map to calculate, if not supplied
    :param components_distance: distance between components
    :return: ndarray or None
    """
    if is_empty(colormap):
        raise ValueError('Specify colormap to calculate (`temperature` or `gravity_acceleration`).')
    if is_empty(components_distance):
        raise ValueError('Component distance value was not supplied.')

    self.build_surface_areas(component)
    self.build_faces_orientation(component, components_distance)
    self.build_surface_gravity(component, components_distance)
    if colormap == 'temperature':
        self.build_temperature_distribution(component, components_distance, do_pulsations=True, phase=phase)

    component = bsutils.component_to_list(component)
    if return_map:
        return_map = {}
        for _component in component:
            component_instance = getattr(self, _component)
            if colormap == 'gravity_acceleration':
                return_map[_component] = copy(component_instance.log_g)
                # return_map[_component] = copy(component_instance.potential_gradient_magnitudes)
            elif colormap == 'temperature':
                return_map[_component] = copy(component_instance.temperatures)

            if component_instance.has_spots():
                for spot_index, spot in component_instance.spots.items():
                    if colormap == 'gravity_acceleration':
                        return_map[_component] = np.append(return_map[_component], spot._log_g)
                    elif colormap == 'temperature':
                        return_map[_component] = np.append(return_map[_component], spot.temperatures)
        return return_map
    return


def build_faces(self, component="all", components_distance=None):
    faces.build_faces(self, components_distance, component)


# def build_surface(self, component="all", components_distance=None, return_surface=False, **kwargs):
#     """
#     Function for building of general binary star component surfaces including spots. It will compute point mesh for
#     Star instance and also spots, incorporate spots and makes a triangulation.
#
#     It is possible to return computed surface (points and faces indices) if `return_surface` parametre is set to True.
#
#     :param self: elisa.binary_system.sytem.BinarySystem; instance
#     :param return_surface: bool; if True, function returns dictionary of arrays with all points and faces
#     (surface + spots) for each component
#     :param components_distance: float; distance between components
#     :param component: str; specify component, use `primary` or `secondary`
#     :return: Tuple or None
#     """
#     if not components_distance:
#         raise ValueError('components_distance value was not provided.')
#
#     ret_points, ret_faces = {}, {}
#
#     self.build_mesh(component, components_distance)
#     self.build_faces(component, components_distance)
#
#     if return_surface:
#         component = bsutils.component_to_list(component)
#         for _component in component:
#             component_instance = getattr(self, _component)
#             ret_points[_component], ret_faces[_component] = component_instance.return_whole_surface()
#         return ret_points, ret_faces
#     else:
#         return return_surface

# TODO: remove
def build_surface_with_no_spots(self, component="all", components_distance=None):
    faces.build_surface_with_no_spots(self, components_distance, component)


def build_surface_with_spots(self, component="all", components_distance=None):
    faces.build_surface_with_spots(self, components_distance, component)


def compute_all_surface_areas(self, component):
    """
    Compute surface are of all faces (spots included).

    :param self: BinaryStar instance
    :param component: str `primary` or `secondary`
    :return:
    """
    if is_empty(component):
        self._logger.debug("no component set to build surface areas")
        return

    components = bsutils.component_to_list(component)
    for component in components:
        component_instance = getattr(self, component)
        self._logger.debug(f'computing surface areas of component: '
                           f'{component_instance} / name: {component_instance.name}')
        component_instance.calculate_all_areas()
