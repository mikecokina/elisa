import numpy as np

from copy import deepcopy

from elisa.base.container import (
    StarContainer,
    PositionContainer
)
from elisa.single_system.surface import (
    mesh,
    faces,
    gravity,
    temperature
)
from elisa.logger import getLogger

logger = getLogger("single-system-container-module")


class SystemContainer(PositionContainer):
    def __init__(self, star: StarContainer, position, **properties):
        super().__init__(position=position)
        self._components = ['star']
        self.star = star

        # placeholder (set in loop below)
        self.rotation_period = np.nan

        for key, val in properties.items():
            setattr(self, key, val)

        # calculating a time that elapsed since t0
        self.time = 86400 * self.rotation_period * self.position.phase

    def set_on_position_params(self, position):
        setattr(self, "position", position)
        return self

    @classmethod
    def from_single_system(cls, single_system, position):
        star = StarContainer.from_star_instance(single_system.star)
        return cls(star, position, **single_system.properties_serializer())

    def copy(self):
        return deepcopy(self)

    def has_spots(self):
        return self.star.has_spots()

    def has_pulsations(self):
        return self.star.has_pulsations()

    def build(self, phase=None, **kwargs):
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
        :param kwargs:
        :return: self;
        """
        self.build_mesh()
        self.build_from_points(phase=phase)
        return self

    def build_mesh(self):
        return mesh.build_mesh(self)

    def build_faces(self):
        return faces.build_faces(self)

    def build_pulsations_on_mesh(self):
        return mesh.build_pulsations_on_mesh(self)

    def build_surface_areas(self):
        return faces.compute_all_surface_areas(self)

    def build_faces_orientation(self):
        return faces.build_faces_orientation(self)

    def build_surface_gravity(self):
        return gravity.build_surface_gravity(self)

    def build_temperature_distribution(self, phase=None):
        return temperature.build_temperature_distribution(self, phase)

    def build_from_points(self, phase=None):
        """
        Build single system from present surface points

        :param phase: float; phase to build system on
        :return:
        """
        self.build_faces()
        self.build_surface_areas()
        self.build_pulsations_on_mesh()
        self.build_faces_orientation()
        self.build_surface_gravity()
        self.build_temperature_distribution(phase)
        return self

    def _phase(self, phase):
        return phase if phase is not None else self.position.phase
