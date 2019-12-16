import numpy as np

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
        self.star = star
        self.position = position

        # placeholder (set in loop below)
        self.inclination = np.nan
        self._flatten = False
        self.rotation_period = np.nan

        for key, val in properties.items():
            setattr(self, key, val)

        # calculating a time that elapsed since t0
        self.time = 86400 * self.rotation_period * self.position.phase

    @classmethod
    def from_single_system(cls, single_system, position):
        star = StarContainer.from_star_instance(single_system.star)
        return cls(star, position, **single_system.properties_serializer())

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

        :param do_pulsations: bool; switch to incorporate pulsations
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
