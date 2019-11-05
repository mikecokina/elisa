import numpy as np

from copy import deepcopy
from elisa.binary_system import dynamic
from elisa.conf import config
from elisa import (
    logger,
    const,
    utils
)
from elisa.base.container import (
    StarContainer,
    PositionContainer
)
from elisa.binary_system.surface import (
    mesh,
    faces,
    gravity,
    temperature
)


config.set_up_logging()
__logger__ = logger.getLogger("binary-system-container-module")


class OrbitalPositionContainer(PositionContainer):
    def __init__(self, primary: StarContainer, secondary, position, **properties):
        self.primary = primary
        self.secondary = secondary
        self.position = position

        # placeholder (set in loop below)
        self.inclination = np.nan
        self._flatten = False

        for key, val in properties.items():
            setattr(self, key, val)

    def set_on_position_params(self, position, primary_potential=None, secondary_potential=None):
        setattr(self, "position", position)
        if not utils.is_empty(primary_potential):
            setattr(self.primary, "surface_potential", primary_potential)
        if not utils.is_empty(secondary_potential):
            setattr(self.secondary, "surface_potential", secondary_potential)
        return self

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

    def build(self, components_distance=None, component="all", do_pulsations=False, phase=None, **kwargs):
        """
        Main method to build binary star system from parameters given on init of BinaryStar.

        called following methods::

            - build_mesh
            - build_faces
            - build_surface_areas
            - build_faces_orientation
            - build_surface_gravity
            - build_temperature_distribution

        :param phase: float; phase to build system on
        :param do_pulsations: bool; switch to incorporate pulsations
        :param component: str; `primary` or `secondary`
        :param components_distance: float; distance of components is SMA units
        :return: self;
        """

        components_distance = self._components_distance(components_distance)
        self.build_mesh(components_distance, component)
        self.build_from_points(components_distance, component, do_pulsations=do_pulsations, phase=phase)
        return self

    def build_mesh(self, components_distance=None, component="all"):
        components_distance = self._components_distance(components_distance)
        return mesh.build_mesh(self, components_distance, component)

    def build_faces(self, components_distance=None, component="all"):
        components_distance = self._components_distance(components_distance)
        return faces.build_faces(self, components_distance, component)

    def build_surface_areas(self, component="all"):
        return faces.compute_all_surface_areas(self, component)

    def build_faces_orientation(self, components_distance=None, component="all"):
        components_distance = self._components_distance(components_distance)
        return faces.build_faces_orientation(self, components_distance, component)

    def build_surface_gravity(self, components_distance=None, component="all"):
        components_distance = self._components_distance(components_distance)
        return gravity.build_surface_gravity(self, components_distance, component)

    def build_temperature_distribution(self, components_distance=None, component="all", do_pulsations=False, phase=None):
        components_distance = self._components_distance(components_distance)
        return temperature.build_temperature_distribution(self, components_distance, component, do_pulsations, phase)

    def build_from_points(self, components_distance=None, component="all", do_pulsations=False, phase=None):
        """
        Build binary system from preset surface points

        :param component: str; `primary` or `secondary`
        :param components_distance: float; distance of components is SMA units
        :param do_pulsations: bool; switch to incorporate pulsations
        :param phase: float; phase to build system on
        :return: self;
        """
        components_distance = self._components_distance(components_distance)
        self.build_faces(components_distance, component)
        self.build_surface_areas(component)
        self.build_faces_orientation(components_distance, component)
        self.build_surface_gravity(components_distance, component)
        self.build_temperature_distribution(components_distance, component, do_pulsations=do_pulsations, phase=phase)
        return self

    def apply_eclipse_filter(self):
        """
        Just placeholder. Maybe will be used in future.

        :return: self;
        """
        raise NotImplemented("This is not implemented")

    def apply_rotation(self):
        """
        Rotate quantities defined in __PROPERTIES__ in case of components defined in __PROPERTIES__.
        Rotation is made in orbital plane and inclination direction in respective order.
        Angle are defined in self.position and self.inclination.

        :return:
        """
        __COMPONENTS__ = ["primary", "secondary"]
        __PROPERTIES__ = ["points", "normals"]

        for component in __COMPONENTS__:
            star_container = getattr(self, component)
            for prop in __PROPERTIES__:
                prop_value = getattr(star_container, prop)

                args = (self.position.azimuth - const.HALF_PI, prop_value, "z", False, False)
                prop_value = utils.around_axis_rotation(*args)

                args = (const.HALF_PI - self.inclination, prop_value, "y", False, False)
                prop_value = utils.around_axis_rotation(*args)
                setattr(star_container, prop, prop_value)
        return self

    def apply_darkside_filter(self):
        """
        Apply darkside filter on current position defined in container.
        Function iterates over components and assigns indices of visible points to EasyObject instance.

        :return: self;
        """
        __COMPONENTS__ = ["primary", "secondary"]
        __PROPERTIES__ = ["points", "normals"]

        for component in __COMPONENTS__:
            star_container = getattr(self, component)
            normals = getattr(star_container, "normals")
            valid_indices = dynamic.darkside_filter(line_of_sight=const.LINE_OF_SIGHT, normals=normals)
            setattr(star_container, "indices", valid_indices)
        return self

    def flatt_it(self):
        # naive implementation of idempotence
        if self._flatten:
            return self

        components = ["primary", "secondary"]
        for component in components:
            star_container = getattr(self, component)
            if self.primary.has_spots() or self.primary.has_pulsations():
                star_container.flatt_it()

        self._flatten = True
        return self

    def _components_distance(self, components_distance):
        return components_distance if components_distance is not None else self.position.distance
