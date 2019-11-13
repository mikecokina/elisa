import numpy as np

from elisa.base.container import (
    StarContainer,
    PositionContainer
)
from elisa.single_system.surface import (
    mesh,
    faces,
)
from elisa.logger import getLogger

logger = getLogger("single-system-container-module")


class SystemContainer(PositionContainer):
    def __init__(self, star: StarContainer, **properties):
        self.star = star

        # placeholder (set in loop below)
        self.inclination = np.nan

        for key, val in properties.items():
            setattr(self, key, val)

    def build(self, do_pulsations=False, phase=None, **kwargs):
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
        self.build_from_points(do_pulsations=do_pulsations, phase=phase)
        return self

    def build_mesh(self):
        return mesh.build_mesh(self)

    def build_faces(self):
        return faces.build_faces(self)

    def build_surface_areas(self):
        return faces.compute_all_surface_areas()

    def build_from_points(self, do_pulsations=False, phase=None):
        """
        Build single system from present surface points

        :param do_pulsations: bool; switch to incorporate pulsations
        :param phase: float; phase to build system on
        :return:
        """
        self.build_faces()




