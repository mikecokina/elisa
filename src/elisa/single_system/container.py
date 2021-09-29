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


class SinglePositionContainer(PositionContainer):
    """
    Object for handling models of a single star system at given orbital position.
    Use functions SinglePositionContainer.from_single_system or
    SingleSystem.build_container to correctly initialize this container.
    """
    def __init__(self, star: StarContainer, position, **properties):
        super().__init__(position=position)
        self._components = ['star']
        self.star = star

        # placeholder (set in loop below)
        self.rotation_period = np.nan

        for key, val in properties.items():
            setattr(self, key, val)

        # calculating a time that elapsed since t0
        self.set_time()

        # setting centre of mass
        self.set_com()

    def set_on_position_params(self, position):
        """
        Defining the orientation of the single system container with respect to the observer.

        :param position: elisa.const.Position;
        :return: SinglePositionContainer; container with set position
        """
        setattr(self, "position", position)
        return self

    def set_com(self):
        """
        Setting a centre of mass of the single star object.
        """
        setattr(self.star, 'com', np.array([0, 0, 0]))

    def set_time(self):
        """
        Calculating elapsed from `reference_time` for the instance of SinglePositionContainer.

        :return: float; time in seconds corresponding to the container
        """
        setattr(self, 'time', 86400 * self.rotation_period * self.position.phase)
        return getattr(self, 'time')

    @classmethod
    def from_single_system(cls, single_system, position):
        """
        Construct an SinglePositionContainer based on the instance of SingleSystem and position.

        :param single_system: elisa.single_system.system.SingleSystem;
        :param position: elisa.const.Position; named tuple containing information system orientation
                                               in space with respect to the observer.
        :return: SinglePositionContainer;
        """
        star = StarContainer.from_star_instance(single_system.star)
        return cls(star, position, **single_system.properties_serializer())

    def copy(self):
        """Returns a deep copy of the SinglePositionContainer."""
        return deepcopy(self)

    def has_spots(self):
        """Returns True if the star contains spots."""
        return self.star.has_spots()

    def has_pulsations(self):
        """Returns True if the star contains pulsations."""
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
        :param build_pulsations: bool;  if True, only equilibrium model is build
        :return: self;
        """
        self.build_surface()
        self.build_from_points()

        self.flat_it()
        if build_pulsations:
            self.build_pulsations()
        return self

    def build_pulsations(self):
        """
        Incorporating user-defined pulsation modes into the model.
        """
        self.build_harmonics()
        self.build_perturbations()

    def build_surface(self):
        """
        Building only clear surface. (points, faces, velocities)
        """
        self.build_mesh()
        self.build_faces()
        self.build_velocities()

    def build_from_points(self):
        """
        Build single system from present surface points

        :return: SingleSystemPosition;
        """
        self.build_surface_gravity()
        self.build_faces_orientation()
        self.correct_mesh()
        self.build_surface_areas()
        self.build_temperature_distribution()
        return self

    def build_mesh(self):
        """
        Build surface point mesh including spots.

        :return: elisa.single_system.container.SinglePositionContainer; container updated with point mesh
        """
        return mesh.build_mesh(self)

    def correct_mesh(self):
        """
        Correcting the underestimation of the surface due to the discretization.

        :return: elisa.single_system.container.SinglePositionContainer; container updated with corrected point mesh
        """
        return mesh.correct_mesh(self)

    def build_faces(self):
        """
        Function tessellates the stellar surface points into a set of triangles
        covering the star without gaps and overlaps.

        :return: elisa.single_system.container.SinglePositionContainer; container updated with faces
        """
        return faces.build_faces(self)

    def build_velocities(self):
        """
        Function calculates velocity vector for each face relative to the observer.

        :return: elisa.single_system.container.SinglePositionContainer; container updated with face velocities
        """
        return faces.build_velocities(self)

    def build_surface_areas(self):
        """
        Compute surface areas of all faces (spots included).

        :return: system; elisa.single_system.container.SinglePositionContainer; container updated with face
                                                                                (triangle) areas
        """
        return faces.compute_all_surface_areas(self)

    def build_faces_orientation(self):
        """
        Compute face orientation (normals) for each face.

        :return: elisa.single_system.container.SinglePositionContainer; container updated with correct normal vector
                                                                        orientation for each face
        """
        return faces.build_faces_orientation(self)

    def build_surface_gravity(self):
        """
        Function calculates gravity potential gradient magnitude (surface gravity) for each face.

        :return: elisa.single_system.container.SinglePositionContainer; container updated with surface gravity
                                                                        distribution
        """
        return gravity.build_surface_gravity(self)

    def build_temperature_distribution(self):
        """
        Function calculates temperature distribution on across all faces.

        :return: elisa.single_system.container.SinglePositionContainer; container updated with surface
                                                                        temperature distribution
        """
        return temperature.build_temperature_distribution(self)

    def build_harmonics(self):
        """
        Adds pre-calculated harmonics for the respective pulsation modes.

        :return: elisa.single_system.container.SinglePositionContainer; updated container
        """
        return pulsations.build_harmonics(self)

    def build_perturbations(self):
        """
        Function adds perturbation to the surface mesh due to pulsations.

        :return: elisa.single_system.container.SinglePositionContainer; container with introduced pulsations
        """
        return pulsations.build_perturbations(self)

    def _phase(self, phase):
        return phase if phase is not None else self.position.phase
