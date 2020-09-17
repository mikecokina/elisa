import numpy as np

from copy import deepcopy
from . surface import (
    mesh,
    faces,
    gravity,
    temperature
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

    def set_on_position_params(self, position, primary_potential=None, secondary_potential=None):
        setattr(self, "position", position)
        if not utils.is_empty(primary_potential):
            setattr(self.primary, "surface_potential", primary_potential)
        if not utils.is_empty(secondary_potential):
            setattr(self.secondary, "surface_potential", secondary_potential)
        return self

    def set_time(self):
        return 86400 * self.period * self.position.phase

    @classmethod
    def from_binary_system(cls, binary_system, position):
        primary = StarContainer.from_star_instance(binary_system.primary)
        secondary = StarContainer.from_star_instance(binary_system.secondary)
        return cls(primary, secondary, position, **binary_system.properties_serializer())

    def copy(self):
        return deepcopy(self)

    def has_spots(self):
        return self.primary.has_spots() or self.secondary.has_spots()

    def has_pulsations(self):
        return self.primary.has_pulsations() or self.secondary.has_pulsations()

    def build(self, components_distance=None, component="all", **kwargs):
        """
        Main method to build binary star system from parameters given on init of BinaryStar.

        called following methods::

            - build_mesh
            - build_faces
            - build_surface_areas
            - build_faces_orientation
            - build_surface_gravity
            - build_temperature_distribution

        :param component: str; `primary` or `secondary`
        :param components_distance: float; distance of components is SMA units
        :return: self;
        """

        components_distance = self._components_distance(components_distance)
        self.build_mesh(components_distance, component)
        self.build_from_points(components_distance, component)
        return self

    def build_mesh(self, components_distance=None, component="all"):
        components_distance = self._components_distance(components_distance)
        return mesh.build_mesh(self, components_distance, component)

    def build_faces(self, components_distance=None, component="all"):
        components_distance = self._components_distance(components_distance)
        return faces.build_faces(self, components_distance, component)

    def build_velocities(self, components_distance=None, component='all'):
        components_distance = self._components_distance(components_distance)
        return faces.build_velocities(self, components_distance, component)

    def build_pulsations_on_mesh(self, component, components_distance):
        return mesh.build_pulsations_on_mesh(self, component, components_distance)

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

    def build_temperature_perturbations(self, components_distance, component):
        return temperature.build_temperature_perturbations(self, components_distance, component)

    def build_from_points_to_temperatures(self, components_distance=None, component="all"):
        """
        Function can be used on container with built points and performs
        surface build without surface temperature distribution.

        :param component: str; `primary` or `secondary`
        :param components_distance: float; distance of components is SMA units
        :return:
        """
        components_distance = self._components_distance(components_distance)
        self.build_faces(components_distance, component)
        self.build_velocities(components_distance, component)
        self.build_pulsations_on_mesh(component, components_distance)
        self.build_surface_gravity(components_distance, component)
        self.build_faces_orientation(components_distance, component)
        self.build_surface_areas(component)

        return self

    def build_full_temperature_distribution(self, components_distance=None, component="all"):
        """
        Function can be used on container with built surface, faces, velocities, and gravity to calculate resulting
        surface temperature distribution.

        :param component: str; `primary` or `secondary`
        :param components_distance: float; distance of components is SMA units
        :return:
        """
        self.build_temperature_distribution(components_distance, component)
        self.build_temperature_perturbations(components_distance, component)

        return self

    def build_from_points(self, components_distance=None, component="all"):
        """
        Build binary system from present surface points.

        :param component: str; `primary` or `secondary`
        :param components_distance: float; distance of components is SMA units
        :return: self;
        """
        self.build_from_points_to_temperatures(components_distance, component)
        self.build_full_temperature_distribution(components_distance, component)
        return self

    def apply_eclipse_filter(self):
        """
        Just placeholder. Maybe will be used in future.

        :return: self;
        """
        raise NotImplemented("This is not implemented")

    def _components_distance(self, components_distance):
        return components_distance if components_distance is not None else self.position.distance
