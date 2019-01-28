import numpy as np
from copy import copy

from conf import config
from engine.binary_system import static


def build_surface_gravity(self, component: str or list=None, components_distance: float=None):
    """
    function calculates gravity potential gradient magnitude (surface gravity) for each face

    :param self:
    :param component: `primary` or `secondary`
    :param components_distance: float
    :return:
    """

    if components_distance is None:
        raise ValueError('Component distance value was not supplied.')

    component = static.component_to_list(component)
    for _component in component:
        component_instance = getattr(self, _component)

        polar_gravity = self.calculate_polar_gravity_acceleration(_component, components_distance, logg=False)

        component_instance.polar_potential_gradient_magnitude = \
            self.calculate_polar_potential_gradient_magnitude(_component, components_distance)
        gravity_scalling_factor = polar_gravity / component_instance.polar_potential_gradient_magnitude

        self._logger.debug('Computing potential gradient magnitudes distribution of {} component.'
                           ''.format(_component))
        component_instance.potential_gradient_magnitudes = self.calculate_face_magnitude_gradient(
            component=_component, components_distance=components_distance)

        component_instance._log_g = np.log10(
            gravity_scalling_factor * component_instance.potential_gradient_magnitudes)

        if component_instance.spots:
            for spot_index, spot in component_instance.spots.items():
                self._logger.debug('Calculating surface SI unit gravity of {} component / {} spot.'
                                   ''.format(_component, spot_index))
                self._logger.debug('Calculating distribution of potential gradient magnitudes of {} component / '
                                   '{} spot.'.format(_component, spot_index))
                spot.potential_gradient_magnitudes = self.calculate_face_magnitude_gradient(
                    component=_component,
                    components_distance=components_distance,
                    points=spot.points, faces=spot.faces)

                spot.log_g = np.log10(gravity_scalling_factor * spot.potential_gradient_magnitudes)


def build_faces_orientation(self, component: str or list=None, components_distance: float=None):
    component = static.component_to_list(component)
    com_x = {'primary': 0, 'secondary': components_distance}

    for _component in component:
        component_instance = getattr(self, _component)
        component_instance.set_all_surface_centres()
        component_instance.set_all_normals(com=com_x[_component])


def build_temperature_distribution(self, component=None, components_distance=None):
    """
    function calculates temperature distribution on across all faces

    :param self:
    :param components_distance:
    :param component: `primary` or `secondary`
    :return:
    """
    component = static.component_to_list(component)

    for _component in component:
        component_instance = getattr(self, _component)

        self._logger.debug('Computing effective temprature distibution on {} component.'.format(_component))
        component_instance.temperatures = component_instance.calculate_effective_temperatures()
        if component_instance.pulsations:
            self._logger.debug('Adding pulsations to surface temperature distribution '
                               'of the {} component.'.format(_component))
            component_instance.temperatures = component_instance.add_pulsations()

        if component_instance.spots:
            for spot_index, spot in component_instance.spots.items():
                self._logger.debug('Computing temperature distribution of {} component / {} spot'
                                   ''.format(_component, spot_index))
                spot.temperatures = spot.temperature_factor * component_instance.calculate_effective_temperatures(
                    gradient_magnitudes=spot.potential_gradient_magnitudes)
                if component_instance.pulsations:
                    self._logger.debug('Adding pulsations to temperature distribution of {} component / {} spot'
                                       ''.format(_component, spot_index))
                    spot.temperatures = component_instance.add_pulsations(points=spot.points, faces=spot.faces,
                                                                          temperatures=spot.temperatures)

        self._logger.debug('Renormalizing temperature map of {0} component due to presence of spots'
                           ''.format(component))
        component_instance.renormalize_temperatures()

    if 'primary' in component and 'secondary' in component:
        self.reflection_effect(iterations=config.REFLECTION_EFFECT_ITERATIONS,
                               components_distance=components_distance)


def build_surface_map(self, colormap=None, component=None, components_distance=None, return_map=False):
    """
    function calculates surface maps (temperature or gravity acceleration) for star and spot faces and it can return
    them as one array if return_map=True

    :param self:
    :param return_map: if True function returns arrays with surface map including star and spot segments
    :param colormap: switch for `temperature` or `gravity` colormap to create
    :param component: `primary` or `secondary` component surface map to calculate, if not supplied
    :param components_distance: distance between components
    :return:
    """
    if colormap is None:
        raise ValueError('Specify colormap to calculate (`temperature` or `gravity_acceleration`).')
    if components_distance is None:
        raise ValueError('Component distance value was not supplied.')

    component = static.component_to_list(component)

    for _component in component:
        component_instance = getattr(self, _component)

        # compute and assign surface areas of elements if missing
        self._logger.debug('Computing surface areas of {} elements.'.format(_component))
        component_instance.calculate_all_areas()

        self.build_surface_gravity(component=_component,
                                   components_distance=components_distance)

        # compute and assign temperature of elements
        if colormap == 'temperature':
            self._logger.debug('Computing effective temprature distibution of {} component.'.format(_component))
            # component_instance.temperatures = component_instance.calculate_effective_temperatures()
            self.build_temperature_distribution(component=_component, components_distance=components_distance)
            if component_instance.pulsations:
                self._logger.debug('Adding pulsations to surface temperature distribution '
                                   'of the {} component.'.format(_component))
                component_instance.temperatures = component_instance.add_pulsations()

    # implementation of reflection effect
    if colormap == 'temperature':
        if len(component) == 2:
            com = {'primary': 0, 'secondary': components_distance}
            for _component in component:
                component_instance = getattr(self, _component)
                component_instance.set_all_surface_centres()
                component_instance.set_all_normals(com=com[_component])

            self.reflection_effect(iterations=config.REFLECTION_EFFECT_ITERATIONS,
                                   components_distance=components_distance)
        else:
            self._logger.debug('Reflection effect can be calculated only when surface map of both components is '
                               'calculated. Skipping calculation of reflection effect.')

    if return_map:
        return_map = {}
        for _component in component:
            component_instance = getattr(self, _component)
            if colormap == 'gravity_acceleration':
                return_map[_component] = copy(component_instance._log_g)
            elif colormap == 'temperature':
                return_map[_component] = copy(component_instance.temperatures)

            if component_instance.spots:
                for spot_index, spot in component_instance.spots.items():
                    if colormap == 'gravity_acceleration':
                        return_map[_component] = np.append(return_map[_component], spot._log_g)
                    elif colormap == 'temperature':
                        return_map[_component] = np.append(return_map[_component], spot.temperatures)
        return return_map
    return


def build_mesh(self, component=None, components_distance=None, **kwargs):
    """
    build points of surface for primary or/and secondary component !!! w/o spots yet !!!

    :param self:
    :param component: str or empty
    :param components_distance: float
    :return:
    """
    if components_distance is None:
        raise ValueError('Argument `component_distance` was not supplied.')
    component = static.component_to_list(component)

    component_x_center = {'primary': 0.0, 'secondary': components_distance}
    for _component in component:
        component_instance = getattr(self, _component)
        # in case of spoted surface, symmetry is not used
        _a, _b, _c, _d = self.mesh_over_contact(component=_component, symmetry_output=True, **kwargs) \
            if self.morphology == 'over-contact' \
            else self.mesh_detached(
            component=_component, components_distance=components_distance, symmetry_output=True, **kwargs
        )
        component_instance.points = _a
        component_instance.point_symmetry_vector = _b
        component_instance.base_symmetry_points_number = _c
        component_instance.inverse_point_symmetry_matrix = _d

        component_instance = getattr(self, _component)
        self._evaluate_spots_mesh(components_distance=components_distance, component=_component)
        if self.morphology == 'over-contact':
            self._incorporate_spots_overcontact_mesh(component_instance=component_instance,
                                                     component_com=component_x_center[_component])
        else:
            self._incorporate_spots_mesh(component_instance=component_instance,
                                         component_com=component_x_center[_component])


def build_faces(self, component=None, components_distance=None):
    """
    function creates faces of the star surface for given components provided you already calculated surface points
    of the component

    :param self:
    :type components_distance: float
    :param component: `primary` or `secondary` if not supplied both components are calculated
    :return:
    """
    if not components_distance:
        raise ValueError('components_distance value was not provided.')

    component = static.component_to_list(component)
    for _component in component:
        component_instance = getattr(self, _component)
        self.build_surface_with_spots(_component, components_distance=components_distance) \
            if component_instance.spots \
            else self.build_surface_with_no_spots(_component, components_distance=components_distance)


def build_surface(self, components_distance=None, component=None, return_surface=False):
    """
    function for building of general binary star component surfaces including spots

    :param self:
    :param return_surface: bool - if true, function returns dictionary of arrays with all points and faces
                                  (surface + spots) for each component
    :param components_distance: distance between components
    :param component: specify component, use `primary` or `secondary`
    :return:
    """
    if not components_distance:
        raise ValueError('components_distance value was not provided.')

    component = static.component_to_list(component)
    ret_points, ret_faces = {}, {}

    for _component in component:
        component_instance = getattr(self, _component)

        # build mesh and incorporate spots points to given obtained object mesh
        self.build_mesh(component=_component, components_distance=components_distance)

        if not component_instance.spots:
            self.build_surface_with_no_spots(_component, components_distance=components_distance)
            if return_surface:
                ret_points[_component] = copy(component_instance.points)
                ret_faces[_component] = copy(component_instance.faces)
            continue
        else:
            self.build_surface_with_spots(_component, components_distance=components_distance)

        if return_surface:
            ret_points[_component], ret_faces[_component] = component_instance.return_whole_surface()

    return (ret_points, ret_faces) if return_surface else None


def build_surface_with_no_spots(self, component=None, components_distance=None):
    """
    function for building binary star component surfaces without spots

    :param self:
    :param components_distance: float
    :param component:
    :return:
    """
    component = static.component_to_list(component)

    for _component in component:
        component_instance = getattr(self, _component)
        # triangulating only one quarter of the star

        if self.morphology != 'over-contact':
            points_to_triangulate = component_instance.points[:component_instance.base_symmetry_points_number, :]
            triangles = self.detached_system_surface(component=_component, points=points_to_triangulate,
                                                     components_distance=components_distance)

        else:
            neck = np.max(component_instance.points[:, 0]) if component[0] == 'primary' \
                else np.min(component_instance.points[:, 0])
            points_to_triangulate = \
                np.append(component_instance.points[:component_instance.base_symmetry_points_number, :],
                          np.array([[neck, 0, 0]]), axis=0)
            triangles = self.over_contact_surface(component=_component, points=points_to_triangulate)
            # filtering out triangles containing last point in `points_to_triangulate`
            triangles = triangles[(triangles < component_instance.base_symmetry_points_number).all(1)]

        # filtering out faces on xy an xz planes
        y0_test = ~np.isclose(points_to_triangulate[triangles][:, :, 1], 0).all(1)
        z0_test = ~np.isclose(points_to_triangulate[triangles][:, :, 2], 0).all(1)
        triangles = triangles[np.logical_and(y0_test, z0_test)]

        component_instance.base_symmetry_faces_number = np.int(np.shape(triangles)[0])
        # lets exploit axial symmetry and fill the rest of the surface of the star
        all_triangles = [inv[triangles] for inv in component_instance.inverse_point_symmetry_matrix]
        component_instance.base_symmetry_faces = triangles
        component_instance.faces = np.concatenate(all_triangles, axis=0)

        base_face_symmetry_vector = np.arange(component_instance.base_symmetry_faces_number)
        component_instance.face_symmetry_vector = np.concatenate([base_face_symmetry_vector for _ in range(4)])


def build_surface_with_spots(self, component=None, components_distance=None):
    """
    function capable of triangulation of spotty stellar surfaces, it merges all surface points, triangulates them
    and then sorts the resulting surface faces under star or spot
    :param self:
    :param components_distance: float
    :param component: str `primary` or `secondary`
    :return:
    """
    component = static.component_to_list(component)
    component_com = {'primary': 0.0, 'secondary': components_distance}
    for _component in component:
        component_instance = getattr(self, _component)
        points, vertices_map = self._return_all_points(component_instance, return_vertices_map=True)

        surface_fn = self._get_surface_builder_fn()
        faces = surface_fn(component=_component, points=points, components_distance=components_distance)
        model, spot_candidates = self._initialize_model_container(vertices_map)
        model = self._split_spots_and_component_faces(
            points, faces, model, spot_candidates, vertices_map, component_instance,
            component_com=component_com[_component]
        )
        self._remove_overlaped_spots(vertices_map, component_instance)
        self._remap_surface_elements(model, component_instance, points)
