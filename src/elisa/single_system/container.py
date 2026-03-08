from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import umpy as up
from elisa.base.container import PositionContainer, StarContainer
from elisa.logger import getLogger
from elisa.single_system.surface import faces, gravity, mesh, pulsations, temperature

logger = getLogger("single_system.container")


if TYPE_CHECKING:
    from elisa.const import Position
    from elisa.single_system.system import SingleSystem
    from elisa.types import Float


class SinglePositionContainer(PositionContainer):
    """Container for a single-star model at a given viewing position.

    Initialize using :meth:`from_single_system` or
    :meth:`elisa.single_system.system.SingleSystem.build_container`.

    :param star: StarContainer instance describing static star parameters.
    :type star: elisa.base.container.StarContainer
    :param position: Viewing geometry (phase, inclination, etc.).
    :type position: elisa.const.Position
    """

    def __init__(self, star: StarContainer, position: Position, **properties: Any) -> None:
        super().__init__(position=position)
        self._components = ["star"]
        self.star = star

        # placeholder (set in loop below)
        self.rotation_period = np.nan

        for key, val in properties.items():
            setattr(self, key, val)

        # calculating a time that elapsed since t0
        self.time = up.NaN  # set time to avoid warning about being set outside __init__
        self.set_time()

        # setting centre of mass
        self.set_com()

    def set_on_position_params(self, position: Position) -> SinglePositionContainer:
        """Set viewing parameters on the container.

        This assigns the supplied :class:`elisa.const.Position` namedtuple to
        the container and returns ``self`` for fluent usage.

        :param position: Viewing geometry namedtuple.
        :type position: elisa.const.Position
        :returns: Updated container with ``position`` set.
        :rtype: elisa.single_system.container.SinglePositionContainer
        """
        self.position = position
        return self

    def set_com(self) -> None:
        """Set the star centre of mass to the origin for a single-star system.

        The value is stored on the contained :class:`StarContainer` as
        ``star.com``.
        """
        self.star.com = np.array([0, 0, 0])

    def set_time(self) -> Float:
        """Compute and store elapsed time since reference in internal units.

        The result is stored on the container attribute ``time`` and also
        returned.

        :returns: Time in seconds corresponding to the container phase.
        :rtype: elisa.types.Float
        """
        self.time = 86400 * self.rotation_period * self.position.phase
        return self.time

    @classmethod
    def from_single_system(cls, single_system: SingleSystem, position: Position) -> SinglePositionContainer:
        """Create a :class:`SinglePositionContainer` from a :class:`SingleSystem`.

        :param single_system: Source single-system instance.
        :type single_system: elisa.single_system.system.SingleSystem
        :param position: Viewing geometry namedtuple.
        :type position: elisa.const.Position
        :returns: Initialized position container.
        :rtype: elisa.single_system.container.SinglePositionContainer
        """
        star = StarContainer.from_star_instance(single_system.star)
        return cls(star, position, **single_system.properties_serializer())

    def copy(self) -> SinglePositionContainer:
        """Return a deep copy of the container."""
        return deepcopy(self)

    def has_spots(self) -> bool:
        """Return ``True`` when the contained star has spots."""
        return self.star.has_spots()

    def has_pulsations(self) -> bool:
        """Return ``True`` when the contained star has pulsations."""
        return self.star.has_pulsations()

    def build(self, *, build_pulsations: bool = True, **kwargs: Any) -> SinglePositionContainer:
        """Build the per-position model for the single star.

        The method executes the standard build pipeline and optionally
        incorporates pulsations.

        :param build_pulsations: If ``True`` add pulsation perturbations (keyword-only).
        :type build_pulsations: bool
        :param kwargs: Additional keyword arguments forwarded to sub-routines.
        :type kwargs: Any
        :returns: The same container after building.
        :rtype: elisa.single_system.container.SinglePositionContainer
        """
        logger.debug("build called with kwargs: %s", kwargs)
        self.build_surface()
        self.build_from_points()

        self.flat_it()
        if build_pulsations:
            self.build_pulsations()
        return self

    def build_pulsations(self) -> None:
        """Incorporate user-defined pulsation modes into the model."""
        self.build_harmonics()
        self.build_perturbations()

    def build_surface(self) -> None:
        """Build the raw surface representation (points, faces, velocities)."""
        self.build_mesh()
        self.build_faces()
        self.build_velocities()

    def build_from_points(self) -> SinglePositionContainer:
        """Finalize derived surface quantities from already built points.

        Runs gravity, face orientation, mesh correction and temperature
        distribution calculations and returns ``self``.

        :returns: Updated container.
        :rtype: elisa.single_system.container.SinglePositionContainer
        """
        self.build_surface_gravity()
        self.build_faces_orientation()
        self.correct_mesh()
        self.build_surface_areas()
        self.build_temperature_distribution()
        return self

    def build_mesh(self) -> SinglePositionContainer:
        """Build surface point mesh including spots and return the container."""
        return mesh.build_mesh(self)

    def correct_mesh(self) -> SinglePositionContainer:
        """Apply mesh correction and return the updated container."""
        return mesh.correct_mesh(self)

    def build_faces(self) -> SinglePositionContainer:
        """Tessellate surface points into triangular faces and return container."""
        return faces.build_faces(self)

    def build_velocities(self) -> SinglePositionContainer:
        """Compute face velocity vectors and return the updated container."""
        return faces.build_velocities(self)

    def build_surface_areas(self) -> SinglePositionContainer:
        """Compute per-face surface areas and return the updated container."""
        return faces.compute_all_surface_areas(self)

    def build_faces_orientation(self) -> SinglePositionContainer:
        """Compute outward normals for faces and return the updated container."""
        return faces.build_faces_orientation(self)

    def build_surface_gravity(self) -> SinglePositionContainer:
        """Calculate surface gravity magnitudes per face and return container."""
        return gravity.build_surface_gravity(self)

    def build_temperature_distribution(self) -> SinglePositionContainer:
        """Compute temperature distribution across faces and return updated container."""
        return temperature.build_temperature_distribution(self)

    def build_harmonics(self) -> SinglePositionContainer:
        """Add precomputed harmonic components for pulsations and return container."""
        return pulsations.build_harmonics(self)

    def build_perturbations(self) -> SinglePositionContainer:
        """Apply pulsation perturbations to the mesh and return container."""
        return pulsations.build_perturbations(self)

    def _phase(self, phase: Float | None) -> Float:
        """Return supplied phase or fallback to container position phase."""
        return phase if phase is not None else self.position.phase
