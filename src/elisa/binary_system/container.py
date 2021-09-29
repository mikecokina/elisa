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
    """
    Object for handling models of a binary system at given orbital position.
    Use functions OrbitalPositionContainer.from_binary_system or
    BinarySystem.build_container to correctly initialize this container.
    """
    def __init__(self, primary: StarContainer, secondary: StarContainer, position, **properties):
        """
        Initialization of the OrbitalPositionContainer using StarContainers

        :param primary: elisa.base.container.StarContainer;
        :param secondary: elisa.base.container.StarContainer;
        :param position: elisa.const.Position;
        :param properties: Dict; serialized binary system parameters
        """
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
        """
        Defining the orbital position and orientation of the binary system container with respect
        to the observer.

        :param position: elisa.const.Position;
        :param primary_potential: float; corrected surface potential of the primary corresponding to the `position`
        :param secondary_potential: float; corrected surface potential of the secondary corresponding to the `position`
        :return: OrbitalPositionContainer; container with set position and surface potentials
        """
        setattr(self, "position", position)
        self.set_com(position)
        if not utils.is_empty(primary_potential):
            setattr(self.primary, "surface_potential", primary_potential)
        if not utils.is_empty(secondary_potential):
            setattr(self.secondary, "surface_potential", secondary_potential)
        return self

    def set_com(self, position):
        """
        Calculate and set centre of masses of both components in frame of reference centered to the primary component.

        :param position: elisa.const.Position;
        """
        setattr(self.primary, 'com', np.array([0, 0, 0]))
        setattr(self.secondary, 'com', np.array([position.distance, 0, 0]))
        self.rotate_property(self.primary, 'com')
        self.rotate_property(self.secondary, 'com')

    def set_time(self):
        """
        Calculating elapsed from `primary_minimum_time` for the instance of OrbitalPositionContainer.

        :return: float; time in seconds corresponding to the container
        """
        setattr(self, 'time', 86400 * self.period * self.position.phase)
        return getattr(self, 'time')

    @classmethod
    def from_binary_system(cls, binary_system, position):
        """
        Construct an OrbitalPositionContainer based on the instance of BinarySystem and orbital position.

        :param binary_system: elisa.binary_system.system.BinarySystem;
        :param position: elisa.const.Position; named tuple containing information concerning orbital
                                               position of the binary components and their orientation
                                               in space with respect to the observer.
        :return: OrbitalPositionContainer;
        """
        radii = binary_system.calculate_components_radii(position.distance)
        primary = StarContainer.from_star_instance(binary_system.primary)
        secondary = StarContainer.from_star_instance(binary_system.secondary)
        primary.assign_radii(radii['primary'])
        secondary.assign_radii(radii['secondary'])
        return cls(primary, secondary, position, **binary_system.properties_serializer())

    def copy(self):
        """Returns a deep copy of the OrbitalPositionContainer."""
        return deepcopy(self)

    def has_spots(self):
        """Returns True if at least one component contains spots."""
        return self.primary.has_spots() or self.secondary.has_spots()

    def has_pulsations(self):
        """Returns True if at least one component contains pulsations."""
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
        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
        :param build_pulsations: bool; if True, only equilibrium model is build
        :return: OrbitalPositionContainer;
        """

        components_distance = self._components_distance(components_distance)
        self.build_mesh(components_distance, component)
        self.build_from_points(components_distance, component)

        # flatt it: from this point we do not require separated information about spots
        self.flat_it()
        if build_pulsations:
            self.build_pulsations(components_distance=components_distance, component=component)
        return self

    def build_pulsations(self, components_distance=None, component="all"):
        """
        Incorporating user-defined pulsation modes into the model.

        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
        :param component: str; 'primary`, `secondary` or `all`
        """
        self.build_harmonics(components_distance=components_distance, component=component)
        self.build_perturbations(components_distance=components_distance, component=component)

    def build_from_points(self, components_distance=None, component="all"):
        """
        Function is used on container to build container on which only bulid_mesh was performed. Function builds the
        rest of geometries

        Order of methods::

            - build_faces
            - build_velocities
            - build_surface_gravity
            - build_faces_orientation
            - correct_mesh
            - build_surface_areas

        and temperatures.

        Order of methods::

            - build_temperature_distribution

        :param component: str; 'primary`, `secondary` or `all`
        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
        :return: OrbitalPositionContainer; self
        """
        self.build_faces_and_kinematic_quantities(components_distance, component)
        self.build_temperature_distribution(components_distance, component)
        return self

    def build_faces_and_kinematic_quantities(self, components_distance=None, component="all"):
        """
        Function is used on container to build container on which only
        build_mesh was performed. Function builds the rest except for
        build_temperature_distribution.

        :param component: str; 'primary`, `secondary` or `all`
        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
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
        """
        Build surface points for primary or/and secondary component. In case of spots,
        the spot point mesh is incorporated into the model.

        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
        :param component: str; 'primary`, `secondary` or `all`
        :return: OrbitalPositionContainer; container updated with point mesh
        """
        components_distance = self._components_distance(components_distance)
        return mesh.build_mesh(self, components_distance, component)

    def correct_mesh(self, components_distance=None, component="all"):
        """
        Correcting the underestimation of the surface due to the discretization.

        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
        :param component: str; 'primary`, `secondary` or `all`
        :return: OrbitalPositionContainer; container updated with corrected point mesh
        """
        return mesh.correct_mesh(self, components_distance=components_distance, component=component)

    def rebuild_symmetric_detached_mesh(self, components_distance=None, component="all"):
        """
        Rebuild a mesh of a symmetrical surface using old mesh to provide azimuths for the new.
        This conserved number of points and faces and saves computational time during a surface
        recalculation in case of similar orbital positions.

        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
        :param component: str; 'primary`, `secondary` or `all`
        :return: OrbitalPositionContainer; container updated with rebuilt point mesh
        """
        components_distance = self._components_distance(components_distance)
        return mesh.rebuild_symmetric_detached_mesh(self, components_distance, component)

    def build_faces(self, components_distance=None, component="all"):
        """
        Function creates faces of the star surface for given components. Faces are
        evaluated upon points that have to be in this time already calculated.

        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
        :param component: str; 'primary`, `secondary` or `all`
        :return: OrbitalPositionContainer; container updated with faces
        """
        components_distance = self._components_distance(components_distance)
        return faces.build_faces(self, components_distance, component)

    def build_velocities(self, components_distance=None, component='all'):
        """
        Function calculates velocity vector for each face relative to the system's centre of mass.

        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
        :param component: str; 'primary`, `secondary` or `all`
        :return: OrbitalPositionContainer; container updated with face velocities
        """
        components_distance = self._components_distance(components_distance)
        return faces.build_velocities(self, components_distance, component)

    def build_surface_areas(self, component="all"):
        """
        Compute surface are of all faces (spots included).

        :param component: str; 'primary`, `secondary` or `all`
        :return: OrbitalPositionContainer; container updated with face (triangle) areas
        """
        return faces.compute_all_surface_areas(self, component)

    def build_faces_orientation(self, components_distance=None, component="all"):
        """
        Compute face orientation (normals) for each face.

        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
        :param component: str; 'primary`, `secondary` or `all`
        :return: OrbitalPositionContainer; container updated with correct normal vector orientation for each face
        """
        components_distance = self._components_distance(components_distance)
        return faces.build_faces_orientation(self, components_distance, component)

    def build_surface_gravity(self, components_distance=None, component="all"):
        """
        Function calculates gravity potential gradient magnitude (surface gravity) for each face.
        Value assigned to face is calculated as a mean of surface gravity values calculated in corners of given face.

        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
        :param component: str; 'primary`, `secondary` or `all`
        :return: OrbitalPositionContainer; container updated with surface gravity distribution
        """
        components_distance = self._components_distance(components_distance)
        return gravity.build_surface_gravity(self, components_distance, component)

    def build_temperature_distribution(self, components_distance=None, component="all", do_pulsations=False,):
        """
        Function calculates temperature distribution across all surface faces.

        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
        :param component: str; 'primary`, `secondary` or `all`
        :param do_pulsations: bool; if True, pulsations are incorporated into the system
        :return: OrbitalPositionContainer; container updated with surface temperature distribution
        """
        components_distance = self._components_distance(components_distance)
        return temperature.build_temperature_distribution(self, components_distance, component)

    def build_harmonics(self, component, components_distance):
        """
        Adds pre-calculated spherical harmonics values for each pulsation mode.

        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
        :param component: str; 'primary`, `secondary` or `all`
        :return: OrbitalPositionContainer; container updated with pre-calculated spherical harmonics
        """
        return pulsations.build_harmonics(self, component, components_distance)

    def build_perturbations(self, component, components_distance):
        """
        Incorporating perturbations of surface quantities into the PositionContainer.

        :param components_distance: Union[None, float]; distance of components is SMA units.
                                                        If None, OrbitalPositionContainer.position.distance is used
        :param component: str; 'primary`, `secondary` or `all`
        :return: OrbitalPositionContainer; container with introduced pulsations
        """
        return pulsations.build_perturbations(self, component, components_distance)

    def _components_distance(self, components_distance):
        return components_distance if components_distance is not None else self.position.distance
