import numpy as np

from copy import deepcopy
from . surface import (
    mesh,
    faces,
    gravity,
    temperature,
    pulsations
)
from .. logger import getLogger
from .. import utils
from .. base.container import (
    StarContainer,
    PositionContainer
)

logger = getLogger("binary_system.container")


class OrbitalPositionContainer(PositionContainer):
    def __init__(self, primary: StarContainer, secondary, position, **properties):
        super().__init__(position=position)
        self._components = ['primary', 'secondary']
        self.primary = primary
        self.secondary = secondary

        # placeholder (set in loop below)
        self.period = np.nan

        for key, val in properties.items():
            setattr(self, key, val)

        # calculating a time that elapsed since t0
        self.time = self.set_time()

        # setting centre of mass
        self.set_com(self.position)

    def set_on_position_params(self, position, primary_potential=None, secondary_potential=None):
        setattr(self, "position", position)
        self.set_com(position)
        if not utils.is_empty(primary_potential):
            setattr(self.primary, "surface_potential", primary_potential)
        if not utils.is_empty(secondary_potential):
            setattr(self.secondary, "surface_potential", secondary_potential)
        return self

    def set_com(self, position):
        setattr(self.primary, 'com', np.array([0, 0, 0]))
        setattr(self.secondary, 'com', np.array([position.distance, 0, 0]))
        self.rotate_property(self.primary, 'com')
        self.rotate_property(self.secondary, 'com')

    def set_time(self):
        return 86400 * self.period * self.position.phase

    @classmethod
    def from_binary_system(cls, binary_system, position):
        binary_system.setup_components_radii(position.distance, calculate_equivalent_radius=False)
        primary = StarContainer.from_star_instance(binary_system.primary)
        secondary = StarContainer.from_star_instance(binary_system.secondary)
        primary.assign_radii(binary_system.primary)
        secondary.assign_radii(binary_system.secondary)
        return cls(primary, secondary, position, **binary_system.properties_serializer())

    def copy(self):
        return deepcopy(self)

    def has_spots(self):
        return self.primary.has_spots() or self.secondary.has_spots()

    def has_pulsations(self):
        return self.primary.has_pulsations() or self.secondary.has_pulsations()

    def build(self, components_distance=None, component="all", build_pulsations=True, **kwargs):
        """
        Main method to build binary star system from parameters given on init of BinaryStar.

        called following methods::

            - build_mesh
            - build_faces
            - build_velocities
            - build_surface_gravity
            - build_faces_orientation
            - correct_mesh
            - build_surface_areas
            - build_temperature_distribution

        :param component: str; `primary` or `secondary`
        :param components_distance: float; distance of components is SMA units
        :param build_pulsations: bool; if True, only equilibrium model is build
        :return: OrbitalPositionContainer;
        """

        components_distance = self._components_distance(components_distance)
        self.build_mesh(components_distance, component)
        self.build_from_points(components_distance, component)

        self.flatt_it()
        if build_pulsations:
            self.build_pulsations(components_distance=components_distance, component=component)
        return self

    def build_pulsations(self, components_distance=None, component="all"):
        self.build_harmonics(components_distance=components_distance, component=component)
        self.build_perturbations(components_distance=components_distance, component=component)

    def build_from_points(self, components_distance=None, component="all"):
        """
        Function is used on container to build container on which only bulid_mesh was performed. Function builds the
        rest.

        Order of methods::

            - build_faces
            - build_velocities
            - build_surface_gravity
            - build_faces_orientation
            - correct_mesh
            - build_surface_areas

        :param component: str; `primary` or `secondary`
        :param components_distance: float; distance of components is SMA units
        :return: OrbitalPositionContainer;
        """
        self.build_faces_and_kinematic_quantities(components_distance, component)
        self.build_temperature_distribution(components_distance, component)
        return self

    def build_faces_and_kinematic_quantities(self, components_distance=None, component="all"):
        """
        Function is used on container to build container on which only bulid_mesh was performed. Function builds the
        rest except for build_temperature_distribution.

        :param component: str; `primary` or `secondary`
        :param components_distance: float; distance of components is SMA units
        :return: OrbitalPositionContainer;
        """
        components_distance = self._components_distance(components_distance)
        self.build_faces(components_distance, component)
        self.build_velocities(components_distance, component)
        self.build_surface_gravity(components_distance, component)
        self.build_faces_orientation(components_distance, component)
        self.correct_mesh(components_distance, component)
        self.build_surface_areas(component)

        return self

    def build_mesh(self, components_distance=None, component="all"):
        components_distance = self._components_distance(components_distance)
        return mesh.build_mesh(self, components_distance, component)

    def correct_mesh(self, components_distance=None, component="all"):
        return mesh.correct_mesh(self, components_distance=components_distance, component=component)

    def rebuild_symmetric_detached_mesh(self, components_distance=None, component="all"):
        components_distance = self._components_distance(components_distance)
        return mesh.rebuild_symmetric_detached_mesh(self, components_distance, component)

    def build_faces(self, components_distance=None, component="all"):
        components_distance = self._components_distance(components_distance)
        return faces.build_faces(self, components_distance, component)

    def build_velocities(self, components_distance=None, component='all'):
        components_distance = self._components_distance(components_distance)
        return faces.build_velocities(self, components_distance, component)

    def build_surface_areas(self, component="all"):
        return faces.compute_all_surface_areas(self, component)

    def build_faces_orientation(self, components_distance=None, component="all"):
        components_distance = self._components_distance(components_distance)
        return faces.build_faces_orientation(self, components_distance, component)

    def build_surface_gravity(self, components_distance=None, component="all"):
        components_distance = self._components_distance(components_distance)
        return gravity.build_surface_gravity(self, components_distance, component)

    def build_temperature_distribution(self, components_distance=None, component="all", do_pulsations=False,):
        components_distance = self._components_distance(components_distance)
        return temperature.build_temperature_distribution(self, components_distance, component)

    # TODO: soon to be deprecated
    def build_temperature_perturbations(self, components_distance, component):
        return temperature.build_temperature_perturbations(self, components_distance, component)

    def build_harmonics(self, component, components_distance):
        return pulsations.build_harmonics(self, component, components_distance)

    def build_perturbations(self, component, components_distance):
        return pulsations.build_perturbations(self, component, components_distance)

    def _components_distance(self, components_distance):
        return components_distance if components_distance is not None else self.position.distance
