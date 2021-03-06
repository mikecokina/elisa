import numpy as np

from copy import deepcopy
from . surface import (
    mesh,
    faces,
    gravity,
    temperature,
    pulsations
)
from .. base.container import (
    StarContainer,
    PositionContainer
)
from .. logger import getLogger

logger = getLogger("single_system.container")


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
        self.time = self.set_time()

    def set_on_position_params(self, position):
        setattr(self, "position", position)
        return self

    def set_time(self):
        return 86400 * self.rotation_period * self.position.phase

    @classmethod
    def from_single_system(cls, single_system, position):
        star = StarContainer.from_star_instance(single_system.star)
        star.assign_radii(single_system.star)
        return cls(star, position, **single_system.properties_serializer())

    def copy(self):
        return deepcopy(self)

    def has_spots(self):
        return self.star.has_spots()

    def has_pulsations(self):
        return self.star.has_pulsations()

    def build(self, build_pulsations=True, **kwargs):
        """
        Main method to build binary star system from parameters given on init of SingleStar.

        called following methods::

            - build_mesh
            - build_faces
            - build_velocities
            - build_surface_gravity
            - build_faces_orientation
            - correct_mesh
            - build_surface_areas
            - build_temperature_distribution

        :param kwargs:
        :param build_pulsations: bool; enable/disable incorporation of pulsations
        :return: self;
        """
        self.build_surface()
        self.build_from_points()

        if build_pulsations:
            self.build_pulsations()
        return self

    def build_surface(self):
        """
        Building only clear surface. (points, faces, velocities)

        :return:
        """
        self.build_mesh()
        self.build_faces()
        self.build_velocities()

    def build_from_points(self):
        """
        Build single system from present surface points

        :return:
        """
        self.build_surface_gravity()
        self.build_faces_orientation()
        self.correct_mesh()
        self.build_surface_areas()
        self.build_temperature_distribution()
        return self

    def build_mesh(self):
        return mesh.build_mesh(self)

    def correct_mesh(self):
        return mesh.correct_mesh(self)

    def build_faces(self):
        return faces.build_faces(self)

    def build_velocities(self):
        return faces.build_velocities(self)

    def build_surface_areas(self):
        return faces.compute_all_surface_areas(self)

    def build_faces_orientation(self):
        return faces.build_faces_orientation(self)

    def build_surface_gravity(self):
        return gravity.build_surface_gravity(self)

    def build_temperature_distribution(self):
        return temperature.build_temperature_distribution(self)

    # TODO: soon to be deprecated
    def build_temperature_perturbations(self):
        return temperature.build_temperature_perturbations(self)

    def build_pulsations(self):
        return pulsations.build_pulsations(self)

    def _phase(self, phase):
        return phase if phase is not None else self.position.phase
