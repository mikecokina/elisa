from elisa.binary_system.surface import mesh, faces, gravity


# TODO: remove
def build_mesh(self, component="all", components_distance=None):
    mesh.build_mesh(self, components_distance, component)


# TODO: remove
def build_surface_gravity(self, components_distance=None, component="all"):
    gravity.build_surface_gravity(self, components_distance, component)


# TODO: remove
def build_faces_orientation(self, components_distance, component="all"):
    faces.build_faces_orientation(self, components_distance, component)


# def build_surface_map(self, colormap=None, component="all", components_distance=None, return_map=False, phase=None):
#     """
#     Function calculates surface maps (temperature or gravity acceleration) for star and spot faces and it can return
#     them as one array if return_map=True.
#
#     :param phase:
#     :param self: BinarySystem; instance
#     :param return_map: bool; if True function returns arrays with surface map including star and spot segments
#     :param colormap: switch for `temperature` or `gravity` colormap to create
#     :param component: `primary` or `secondary` component surface map to calculate, if not supplied
#     :param components_distance: distance between components
#     :return: ndarray or None
#     """
#     if is_empty(colormap):
#         raise ValueError('Specify colormap to calculate (`temperature` or `gravity_acceleration`).')
#     if is_empty(components_distance):
#         raise ValueError('Component distance value was not supplied.')
#
#     self.build_surface_areas(component)
#     self.build_faces_orientation(component, components_distance)
#     self.build_surface_gravity(component, components_distance)
#     if colormap == 'temperature':
#         self.build_temperature_distribution(component, components_distance, do_pulsations=True, phase=phase)
#
#     component = bsutils.component_to_list(component)
#     if return_map:
#         return_map = {}
#         for _component in component:
#             component_instance = getattr(self, _component)
#             if colormap == 'gravity_acceleration':
#                 return_map[_component] = copy(component_instance.log_g)
#                 # return_map[_component] = copy(component_instance.potential_gradient_magnitudes)
#             elif colormap == 'temperature':
#                 return_map[_component] = copy(component_instance.temperatures)
#
#             if component_instance.has_spots():
#                 for spot_index, spot in component_instance.spots.items():
#                     if colormap == 'gravity_acceleration':
#                         return_map[_component] = np.append(return_map[_component], spot._log_g)
#                     elif colormap == 'temperature':
#                         return_map[_component] = np.append(return_map[_component], spot.temperatures)
#         return return_map
#     return


# TODO: remove
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


# TODO: remove
def build_surface_with_spots(self, component="all", components_distance=None):
    faces.build_surface_with_spots(self, components_distance, component)


# TODO: remove
def compute_all_surface_areas(self, component):
    faces.compute_all_surface_areas(self, component)

