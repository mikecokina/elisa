from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from elisa import umpy as up
from elisa import utils
from elisa.base.container import PositionContainer, StarContainer
from elisa.binary_system.surface import faces, gravity, mesh, pulsations, temperature
from elisa.logger import getLogger

if TYPE_CHECKING:
    from elisa.binary_system.system import BinarySystem
    from elisa.const import Position
    from elisa.types import ComponentSelection, Float

logger = getLogger("binary_system.container")


class OrbitalPositionContainer(PositionContainer):
    """Container representing a binary system at a specific orbital position.

    This object handles surface geometry, kinematic quantities, gravity,
    temperature distribution, and pulsation-related quantities for a binary
    system at a given orbital position.

    Use :meth:`from_binary_system` or ``BinarySystem.build_container`` to
    initialize this container correctly.
    """

    __slots__ = (
        "eccentricity",
        "mass",
        "mass_ratio",
        "morphology",
        "period",
        "primary",
        "secondary",
        "semi_major_axis",
        "time",
    )

    def __init__(
        self,
        primary: StarContainer,
        secondary: StarContainer,
        position: Position,
        **properties: Float,
    ) -> None:
        """Initialize the orbital position container from star containers.

        :param primary: Primary component container.
        :type primary: StarContainer
        :param secondary: Secondary component container.
        :type secondary: StarContainer
        :param position: Orbital position descriptor.
        :type position: Position
        :param properties: Serialized binary system parameters to assign to the
            container instance.
        :type properties: Float
        :return: ``None``.
        :rtype: None
        """
        super().__init__(position=position)
        self._components = ["primary", "secondary"]
        self.primary: StarContainer = primary
        self.secondary: StarContainer = secondary

        self.period = up.NaN

        for key, value in properties.items():
            setattr(self, key, value)

        self.time = self.set_time()
        self.set_com(self.position)

    def set_on_position_params(
        self,
        position: Position,
        primary_potential: Float | None = None,
        secondary_potential: Float | None = None,
    ) -> OrbitalPositionContainer:
        """Set orbital position and optional component surface potentials.

        This updates the container orientation with respect to the observer,
        updates the centers of mass, and optionally overwrites component surface
        potentials.

        :param position: Orbital position descriptor.
        :type position: Position
        :param primary_potential: Corrected surface potential of the primary
            component corresponding to ``position``.
        :type primary_potential: Float | None
        :param secondary_potential: Corrected surface potential of the
            secondary component corresponding to ``position``.
        :type secondary_potential: Float | None
        :return: Updated container instance.
        :rtype: OrbitalPositionContainer
        """
        self.position = position
        self.set_com(position)

        if not utils.is_empty(primary_potential):
            self.primary.surface_potential = primary_potential

        if not utils.is_empty(secondary_potential):
            self.secondary.surface_potential = secondary_potential

        return self

    def set_com(self, position: Position) -> None:
        """Calculate and set component centers of mass.

        The centers of mass are defined in a reference frame centered on the
        primary component, then rotated into the observer-oriented frame.

        :param position: Orbital position descriptor.
        :type position: Position
        :return: ``None``.
        :rtype: None
        """
        self.primary.com = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.secondary.com = np.array(
            [position.distance, 0.0, 0.0],
            dtype=np.float64,
        )
        self.rotate_property(self.primary, "com")
        self.rotate_property(self.secondary, "com")

    def set_time(self) -> Float:
        """Calculate elapsed time since primary minimum for this container.

        The time is derived from the orbital phase and period and returned in
        seconds.

        :return: Time corresponding to the current orbital position.
        :rtype: Float
        """
        self.time = 86400.0 * self.period * self.position.phase
        return self.time

    @classmethod
    def from_binary_system(
        cls,
        binary_system: BinarySystem,
        position: Position,
    ) -> OrbitalPositionContainer:
        """Construct a container from a binary system and orbital position.

        :param binary_system: Binary system instance.
        :type binary_system: BinarySystem
        :param position: Named tuple containing information about the orbital
            position of the binary components and their orientation in space
            with respect to the observer.
        :type position: Position
        :return: Newly constructed orbital position container.
        :rtype: OrbitalPositionContainer
        """
        radii = binary_system.calculate_components_radii(position.distance)
        primary = StarContainer.from_star_instance(binary_system.primary)
        secondary = StarContainer.from_star_instance(binary_system.secondary)
        primary.assign_radii(radii["primary"])
        secondary.assign_radii(radii["secondary"])
        return cls(
            primary=primary,
            secondary=secondary,
            position=position,
            **binary_system.properties_serializer(),
        )

    def copy(self) -> OrbitalPositionContainer:
        """Return a deep copy of the container.

        :return: Deep copy of the orbital position container.
        :rtype: OrbitalPositionContainer
        """
        return deepcopy(self)

    def has_spots(self) -> bool:
        """Return whether at least one component contains spots.

        :return: ``True`` if any component has spots, otherwise ``False``.
        :rtype: bool
        """
        return self.primary.has_spots() or self.secondary.has_spots()

    def has_pulsations(self) -> bool:
        """Return whether at least one component contains pulsations.

        :return: ``True`` if any component has pulsations, otherwise ``False``.
        :rtype: bool
        """
        return self.primary.has_pulsations() or self.secondary.has_pulsations()

    def build(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
        *,
        build_pulsations: bool = True,
        **kwargs: object,
    ) -> OrbitalPositionContainer:
        """Build the binary model for the current orbital position.

        The following methods are applied::

            - build_mesh
            - build_faces
            - build_velocities
            - build_surface_gravity
            - build_faces_orientation
            - correct_mesh
            - build_surface_areas
            - build_temperature_distribution

        Surface pulsations can be added afterward if requested.

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :param build_pulsations: If ``True``, incorporate pulsation effects
            after the equilibrium model is built.
        :type build_pulsations: bool
        :param kwargs: Unused keyword arguments preserved for compatibility.
        :type kwargs: object
        :return: Updated orbital position container.
        :rtype: OrbitalPositionContainer
        """
        del kwargs

        resolved_distance = self._components_distance(components_distance)
        self.build_mesh(
            components_distance=resolved_distance,
            component=component,
        )
        self.build_from_points(
            components_distance=resolved_distance,
            component=component,
        )

        self.flat_it()

        if build_pulsations:
            self.build_pulsations(
                components_distance=resolved_distance,
                component=component,
            )

        return self

    def build_pulsations(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
    ) -> OrbitalPositionContainer:
        """Incorporate user-defined pulsation modes into the model.

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Updated orbital position container.
        :rtype: OrbitalPositionContainer
        """
        resolved_distance = self._components_distance(components_distance)
        self.build_harmonics(
            components_distance=resolved_distance,
            component=component,
        )
        self.build_perturbations(
            components_distance=resolved_distance,
            component=component,
        )
        return self

    def build_from_points(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
    ) -> OrbitalPositionContainer:
        """Build all remaining geometry from an existing surface point mesh.

        This method assumes that :meth:`build_mesh` has already been executed.

        The following methods are applied::

            - build_faces
            - build_velocities
            - build_surface_gravity
            - build_faces_orientation
            - correct_mesh
            - build_surface_areas
            - build_temperature_distribution

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Updated orbital position container.
        :rtype: OrbitalPositionContainer
        """
        self.build_faces_and_kinematic_quantities(
            components_distance=components_distance,
            component=component,
        )
        self.build_temperature_distribution(
            components_distance=components_distance,
            component=component,
        )
        return self

    def build_faces_and_kinematic_quantities(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
    ) -> OrbitalPositionContainer:
        """Build geometry and kinematic quantities except temperature.

        This method assumes that :meth:`build_mesh` has already been executed.

        The following methods are applied::

            - build_faces
            - build_velocities
            - build_surface_gravity
            - build_faces_orientation
            - correct_mesh
            - build_surface_areas

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Updated orbital position container.
        :rtype: OrbitalPositionContainer
        """
        resolved_distance = self._components_distance(components_distance)
        self.build_faces(
            components_distance=resolved_distance,
            component=component,
        )
        self.build_velocities(
            components_distance=resolved_distance,
            component=component,
        )
        self.build_surface_gravity(
            components_distance=resolved_distance,
            component=component,
        )
        self.build_faces_orientation(
            components_distance=resolved_distance,
            component=component,
        )
        self.correct_mesh(
            components_distance=resolved_distance,
            component=component,
        )
        self.build_surface_areas(component=component)
        return self

    def build_mesh(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
    ) -> OrbitalPositionContainer:
        """Build surface point meshes for selected components.

        In systems with spots, the spot point mesh is incorporated into the
        model.

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Container updated with point meshes.
        :rtype: OrbitalPositionContainer
        """
        resolved_distance = self._components_distance(components_distance)
        return mesh.build_mesh(self, resolved_distance, component)

    def correct_mesh(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
    ) -> OrbitalPositionContainer:
        """Correct surface underestimation caused by discretization.

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Container updated with corrected point mesh.
        :rtype: OrbitalPositionContainer
        """
        return mesh.correct_mesh(
            self,
            components_distance=components_distance,
            component=component,
        )

    def rebuild_symmetric_detached_mesh(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
    ) -> OrbitalPositionContainer:
        """Rebuild a symmetric detached mesh using the existing mesh.

        The existing mesh provides azimuth sampling for the new mesh. This
        preserves the number of points and faces and reduces computational cost
        for recalculation at similar orbital positions.

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Container updated with rebuilt point mesh.
        :rtype: OrbitalPositionContainer
        """
        resolved_distance = self._components_distance(components_distance)
        return mesh.rebuild_symmetric_detached_mesh(
            self,
            resolved_distance,
            component,
        )

    def build_faces(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
    ) -> OrbitalPositionContainer:
        """Build surface faces for selected components.

        Faces are evaluated from points that must already be available.

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Container updated with faces.
        :rtype: OrbitalPositionContainer
        """
        resolved_distance = self._components_distance(components_distance)
        return faces.build_faces(self, resolved_distance, component)

    def build_velocities(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
    ) -> OrbitalPositionContainer:
        """Build face velocity vectors relative to the system center of mass.

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Container updated with face velocities.
        :rtype: OrbitalPositionContainer
        """
        resolved_distance = self._components_distance(components_distance)
        return faces.build_velocities(self, resolved_distance, component)

    def build_surface_areas(
        self,
        component: ComponentSelection = "all",
    ) -> OrbitalPositionContainer:
        """Compute areas of all faces, including spot faces.

        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Container updated with face areas.
        :rtype: OrbitalPositionContainer
        """
        return faces.compute_all_surface_areas(self, component)

    def build_faces_orientation(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
    ) -> OrbitalPositionContainer:
        """Compute correctly oriented face normals for each face.

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Container updated with oriented face normals.
        :rtype: OrbitalPositionContainer
        """
        resolved_distance = self._components_distance(components_distance)
        return faces.build_faces_orientation(self, resolved_distance, component)

    def build_surface_gravity(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
    ) -> OrbitalPositionContainer:
        """Build surface gravity distribution over all faces.

        The surface gravity assigned to a face is the mean of the gravity
        values calculated at the face corners.

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Container updated with surface gravity values.
        :rtype: OrbitalPositionContainer
        """
        resolved_distance = self._components_distance(components_distance)
        return gravity.build_surface_gravity(self, resolved_distance, component)

    def build_temperature_distribution(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
        *,
        do_pulsations: bool = False,
    ) -> OrbitalPositionContainer:
        """Build surface temperature distribution across all faces.

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :param do_pulsations: Whether pulsations should be incorporated into the
            temperature calculation.
        :type do_pulsations: bool
        :return: Container updated with surface temperature distribution.
        :rtype: OrbitalPositionContainer
        """
        del do_pulsations

        resolved_distance = self._components_distance(components_distance)
        return temperature.build_temperature_distribution(
            self,
            resolved_distance,
            component,
        )

    def build_harmonics(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
    ) -> OrbitalPositionContainer:
        """Add precomputed spherical harmonics for pulsation modes.

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Container updated with precomputed spherical harmonics.
        :rtype: OrbitalPositionContainer
        """
        resolved_distance = self._components_distance(components_distance)
        return pulsations.build_harmonics(self, component, resolved_distance)

    def build_perturbations(
        self,
        components_distance: Float | None = None,
        component: ComponentSelection = "all",
    ) -> OrbitalPositionContainer:
        """Incorporate pulsation perturbations into the container.

        :param components_distance: Distance between components in semi-major
            axis units. If ``None``, ``self.position.distance`` is used.
        :type components_distance: Float | None
        :param component: Component selector.
        :type component: Literal["primary", "secondary", "all", "both"]
        :return: Container updated with pulsation perturbations.
        :rtype: OrbitalPositionContainer
        """
        resolved_distance = self._components_distance(components_distance)
        return pulsations.build_perturbations(self, component, resolved_distance)

    def _components_distance(self, components_distance: Float | None) -> Float:
        """Resolve the component distance value to use.

        If ``components_distance`` is ``None``, the distance stored in the
        current orbital position is used.

        :param components_distance: Explicit component distance.
        :type components_distance: Float | None
        :return: Resolved component distance.
        :rtype: Float
        """
        return components_distance if components_distance is not None else self.position.distance
