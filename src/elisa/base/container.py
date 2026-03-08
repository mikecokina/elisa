from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from elisa import const, utils
from elisa import umpy as up
from elisa.base.surface.faces import mirror_face_values, symmetry_face_reduction
from elisa.base.surface.mesh import symmetry_point_reduction
from elisa.logger import getLogger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.star import Star
    from elisa.const import Position
    from elisa.types import Float

logger = getLogger("base.container")


class PropertiesContainer:
    """General container for storing model attributes."""

    def __init__(self, **kwargs) -> None:
        """Create a properties container and set provided values as attributes.

        :param kwargs: Arbitrary properties stored and exposed as attributes.
        """
        self.properties: dict = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        """Return a dictionary with stored properties."""
        return self.properties

    def __getitem__(self, item: str) -> Any:
        """Allow attribute-style access via mapping protocol."""
        return getattr(self, item)

    def __str__(self) -> str:
        return str(self.to_dict())


class StarPropertiesContainer(PropertiesContainer):
    """Container for star properties."""


class SystemPropertiesContainer(PropertiesContainer):
    """Container for system properties."""


class PositionContainer(ABC):
    """Container holding per-position model state (phase/time).

    This object groups per-component StarContainers and provides helpers
    to rotate, flatten and filter the model for the current viewing
    geometry (position and inclination).
    """

    def __init__(self, position: Position) -> None:
        self._flatten: bool = False
        self._components: list[str] = []
        self.position: Position = position
        self.inclination: Float = np.nan
        self.period: Float = np.nan
        self.gamma: Float = np.nan

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        """Build per-position data for the container."""
        raise NotImplementedError

    @abstractmethod
    def build_mesh(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_faces(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_surface_areas(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_faces_orientation(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_surface_gravity(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_temperature_distribution(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def is_flat(self) -> bool:
        return self._flatten

    def flat_it(self) -> PositionContainer:
        """Flatten per-component attributes into unified arrays.

        The method merges per-component properties (points, faces,
        temperatures, etc.) into single arrays stored on each
        component container. The operation is idempotent.
        """
        # naive implementation of idempotency
        if self._flatten:
            return self

        for component in self._components:
            star_container = getattr(self, component)
            if star_container.has_spots() or star_container.has_pulsations():
                star_container.flat_it()

        self._flatten = True
        return self

    def apply_rotation(self) -> PositionContainer:
        """Rotate per-component vector properties into the observer frame.

        Rotation is applied first around the orbital axis and then for
        inclination. The property list is defined locally.
        """
        _properties_to_rotate = ("points", "normals", "velocities", "face_centres")

        for component in self._components:
            star_container = getattr(self, component)
            for prop in _properties_to_rotate:
                self.rotate_property(star_container, prop)
        return self

    def rotate_property(self, container: StarContainer, prop: str) -> None:
        """Rotate a single property from the co-rotating frame to observer frame.

        :param container: Star container holding the property.
        :param prop: Name of the property to rotate (e.g. "points").
        """
        prop_value = getattr(container, prop)
        prop_value = utils.rotate_item(prop_value, self.position, self.inclination)
        setattr(container, prop, prop_value)

    def add_secular_velocity(self) -> PositionContainer:
        """Add systemic (gamma) velocity to component velocities.

        The gamma value is taken from the container and added to the x
        component of each surface velocity vector.
        """
        gamma = self.gamma
        for component in self._components:
            star = getattr(self, component)
            star.velocities[:, 0] += gamma
        return self

    def apply_darkside_filter(self) -> PositionContainer:
        """Apply the dark-side visibility filter for each component.

        Visible indices are computed from per-face/point cosines and
        stored on the component container as ``indices``.
        """
        for component in self._components:
            star_container = getattr(self, component)
            cosines = star_container.los_cosines
            valid_indices = self.darkside_filter(cosines=cosines)
            star_container.indices = valid_indices
        return self

    def calculate_face_angles(self, line_of_sight: np.ndarray) -> None:
        """Compute cosines between surface normals and the line-of-sight.

        The result is stored on each component container as
        ``los_cosines``.

        :param line_of_sight: Direction vector of the observer.
        """
        for component in self._components:
            star_container = getattr(self, component)
            normals = star_container.normals
            los_cosines = self.return_cosines(normals, line_of_sight=line_of_sight)
            star_container.los_cosines = los_cosines

    @staticmethod
    def return_cosines(normals: NDArray, line_of_sight: list[Float] | NDArray[Float]) -> NDArray:
        """Return cosines between normals and the provided line-of-sight.

        Uses a fast-path when the line_of_sight equals the default unit
        vector defined in :data:`const.LINE_OF_SIGHT`.
        """
        if np.array(line_of_sight == const.LINE_OF_SIGHT).all():
            return utils.calculate_cos_theta_los_x(normals=normals)
        return utils.calculate_cos_theta(normals=normals, line_of_sight_vector=line_of_sight)

    def copy(self) -> PositionContainer:
        """Return a deep copy of this PositionContainer."""
        return deepcopy(self)

    @staticmethod
    def darkside_filter(cosines: np.ndarray) -> np.ndarray:
        """Return indices of surface elements visible to the observer.

        Assumes ``cosines`` contains precomputed dot-products between
        surface normals and the observer direction.
        """
        # TODO(@author): resolve self-shadowing in contact systems (W UMa)  # noqa: FIX002, TD003
        return up.arange(np.shape(cosines)[0])[cosines > 0]


class StarContainer:
    """Container for non-static star properties and per-position data.

    This container carries properties that vary with phase/time (such as
    surface points, temperatures, and velocities) as well as static
    properties set during parent BinarySystem or SingleSystem creation.

    The container should be initialized using :meth:`from_star_instance` or
    :meth:`from_properties_container` for proper setup. Alternatively,
    experienced users may instantiate directly and set attributes manually.

    Properties gathered from the Star object include: ``mass``, ``t_eff``,
    ``synchronicity``, ``albedo``, ``discretization_factor``, ``name``,
    ``spots``, ``polar_radius``, ``equatorial_radius``, ``gravity_darkening``,
    ``surface_potential``, ``atmosphere``, ``pulsations``, ``metallicity``,
    ``polar_log_g``, ``critical_surface_potential``, and ``side_radius``.

    Optional input parameters
    -------------------------
    points : numpy.ndarray or None
        Vertex coordinates (N x 3) forming the body surface.
        Shape: (N, 3) with x, y, z cartesian coordinates.
    normals : numpy.ndarray or None
        Outward-pointing normal vectors for surface faces.
        Array containing normalised normals of corresponding faces.
    indices : numpy.ndarray or None
        Indices of visible surface faces after filtering.
    faces : numpy.ndarray or None
        Triangulation index array. Triangles stored as list of vertex
        indices. Shape: (M, 3).
    temperatures : numpy.ndarray or None
        Temperature values assigned to each face.
        Shape: (M, ) where M is the number of faces.
    log_g : numpy.ndarray or None
        Surface gravity (log g, cgs units) per face.
        Shape: (M, ).
    coverage : numpy.ndarray or None
        Surface area of each triangle visible to the observer.
    face_centres : numpy.ndarray or None
        Row-wise geometric centres of each triangular face.
        Shape: (M, 3).
    metallicity : float or None
        Metallicity value [M/H] for surface/atmosphere.
    areas : numpy.ndarray or None
        Surface area of each face. Shape: (M,).
    potential_gradient_magnitudes : numpy.ndarray or None
        Magnitude of the potential gradient at each surface element.
    ld_cfs : numpy.ndarray or None
        Limb-darkening coefficients. Can be per-passband.
    normal_radiance : numpy.ndarray or None
        Radiance normal to the surface for each element.
    los_cosines : numpy.ndarray or None
        Cosines of angle between surface normals and observer direction.
        Shape: (M,).

    Output attributes (available after building)
    -------
    points : numpy.ndarray
        Vertex array as described above.
    normals : numpy.ndarray
        Normalised outward-facing normal vectors per face.
    faces : numpy.ndarray
        Triangulation index array.
    temperatures : numpy.ndarray
        Temperature per face.
    log_g : numpy.ndarray
        Surface gravity per face.
    coverage : numpy.ndarray
        Visible coverage per face.
    indices : numpy.ndarray
        Indices of visible faces after dark-side filtering.
    face_centres : numpy.ndarray
        Geometric centre of each face.
    metallicity : float
        Metallicity value.
    areas : numpy.ndarray
        Area of each surface triangle.
    potential_gradient_magnitudes : numpy.ndarray
        Potential gradient magnitude per surface element.
    inverse_point_symmetry_matrix : numpy.ndarray
        Row-wise mapping of base symmetry quadrant points (octant for
        single star) to all other quadrants. Each row contains indices of
        points in a given quadrant with order corresponding to the order
        of points in the 1st quadrant (octant).
    base_symmetry_points_number : int
        Number of first n surface points in ``StarContainer.points`` located
        on a symmetrical part of the surface. Selects surface points from
        one quarter (or eighth) of the star in binary (or single) systems.
    base_symmetry_faces_number : int
        Number of first n triangles in ``StarContainer.faces`` located in
        the 1st quadrant (or octant). Selects temperatures and other
        per-face quantities only from triangles in the 1st quadrant (or
        octant) in binary (or single) systems.
    face_symmetry_vector : numpy.ndarray
        Array mapping each surface triangle to the symmetrical surface
        triangle located in the 1st quadrant. Contains indices between 0
        and ``base_symmetry_faces_number``.
    base_symmetry_points : numpy.ndarray
        Base symmetry points array.
    base_symmetry_faces : numpy.ndarray
        Base symmetry faces array.
    polar_potential_gradient_magnitude : float or numpy.ndarray
        Potential gradient magnitude at the stellar pole.
    ld_cfs : numpy.ndarray
        Limb-darkening coefficients.
    normal_radiance : numpy.ndarray
        Normal radiance array.
    los_cosines : numpy.ndarray
        Line-of-sight cosines array.
    """

    def __init__(
        self,
        points: NDArray | None = None,
        normals: NDArray | None = None,
        velocities: NDArray | None = None,
        accelerations: NDArray | None = None,
        indices: NDArray | None = None,
        faces: NDArray | None = None,
        temperatures: NDArray | None = None,
        log_g: NDArray | None = None,
        coverage: NDArray | None = None,
        face_centres: NDArray | None = None,
        metallicity: Float | None = None,
        areas: NDArray | None = None,
        potential_gradient_magnitudes: NDArray | None = None,
        ld_cfs: NDArray | None = None,
        normal_radiance: NDArray | None = None,
        los_cosines: NDArray | None = None,
    ) -> None:

        self.points = points
        self.normals = normals
        self.faces = faces
        self.velocities = velocities
        self.accelerations = accelerations
        self.temperatures = temperatures
        self.log_g = log_g
        self.coverage = coverage
        self.indices = indices
        self.face_centres = face_centres
        self.metallicity = metallicity
        self.areas = areas
        self.potential_gradient_magnitudes = potential_gradient_magnitudes
        self.ld_cfs = ld_cfs
        self.normal_radiance = normal_radiance
        self.los_cosines = los_cosines
        self.points_spherical = np.array([])
        self.com = np.array([])

        self.inverse_point_symmetry_matrix = np.array([])
        self.base_symmetry_points_number = 0

        self.face_symmetry_vector = np.array([])
        self.base_symmetry_faces_number = 0

        # aux variables for treating singularities at poles
        self.pole_idx = np.array([])
        self.pole_idx_neighbour = np.array([])

        # those are used only if case of spots are NOT used ------------------------------------------------------------
        self.base_symmetry_points = np.array([])
        self.base_symmetry_faces = np.array([])
        self.azimuth_args = np.array([])
        # --------------------------------------------------------------------------------------------------------------

        self.spots: dict = {}
        self.pulsations: dict = {}
        self.polar_potential_gradient_magnitude = np.nan

        # all star radii in any position (set on fly) ------------------------------------------------------------------
        # set only via `assign_radii()` method
        self.polar_radius = None
        self.forward_radius = None
        self.side_radius = None
        self.backward_radius = None
        self.equatorial_radius = None
        self.equivalent_radius = None
        # --------------------------------------------------------------------------------------------------------------

        # some defaults ------------------------------------------------------------------------------------------------
        self.t_eff = up.NaN
        self.limb_darkening_coefficients: dict[str, NDArray[Float]] | None = None
        self.mass = up.NaN
        self.discretization_factor = up.NaN
        self.gravity_darkening = up.NaN
        self.critical_surface_potential = None
        self.surface_potential = None
        self.albedo = up.NaN
        self.polar_log_g = up.NaN
        # --------------------------------------------------------------------------------------------------------------

        self._flatten = False

    @classmethod
    def from_star_instance(cls, star: Star) -> StarContainer:
        """Initialize StarContainer from Star instance.

        :param star: elisa.base.star.Star;
        :return: StarContainer;
        """
        return cls.from_properties_container(star.to_properties_container())

    @classmethod
    def from_properties_container(cls, properties_container: StarPropertiesContainer) -> StarContainer:
        """Create StarContainer from properties container.

        :param properties_container: elisa.base.container.StarPropertiesContainer;
        :return: elisa.base.container.StarContainer;
        """
        container = cls()
        container.__dict__.update(properties_container.__dict__)
        return container

    def has_spots(self) -> bool:
        """Return True if this container contains spots."""
        return len(self.spots) > 0

    def has_pulsations(self) -> bool:
        """Return True if pulsation modes are present."""
        return len(self.pulsations) > 0

    def symmetry_test(self) -> bool:
        """Return True when surface symmetry optimisations may be used."""
        return not self.has_spots() and not self.has_pulsations()

    def is_flat(self) -> bool:
        """Return True if this container is flattened."""
        return self._flatten

    def copy(self) -> StarContainer:
        """Return a deep copy of this StarContainer."""
        return deepcopy(self)

    def remove_spot(self, spot_index: int) -> None:
        """Remove n-th spot index of object.

        :param spot_index: Index of the spot to remove.
        """
        del self.spots[spot_index]

    def get_flatten_points_map(self) -> tuple[NDArray, list]:
        """Return all surface points and a vertex map describing object/spot ownership.

        :returns: Tuple of (points, vertices_map).
        """
        points = copy(self.points)
        for spot_instance in self.spots.values():
            points = up.concatenate([points, spot_instance.points])

        vertices_map = [{"type": "object", "enum": -1}] * len(self.points)
        for idx, spot_instance in enumerate(self.spots.values()):
            vertices_map = up.concatenate(
                [
                    vertices_map,
                    [{"type": "spot", "enum": idx}] * len(spot_instance.points),
                ],
            )
        return points, vertices_map

    def calculate_areas(self) -> NDArray:
        """Compute areas for each surface face (excluding spots unless flattened).

        :returns: Array of face areas.
        """
        if len(self.faces) == 0 or len(self.points) == 0:
            msg = "Faces or/and points of object {self.name} have not been set yet.\nRun build method first."
            raise ValueError(msg)
        if self.symmetry_test():
            base_areas = utils.triangle_areas(self.symmetry_faces(self.faces), self.symmetry_points())
            return self.mirror_face_values(base_areas)
        return utils.triangle_areas(self.faces, self.points)

    def calculate_all_areas(self) -> None:
        """Calculate areas for all faces and assign spot areas when present."""
        self.areas = self.calculate_areas()
        if self.has_spots() and not self.is_flat():
            for spot_instance in self.spots.values():
                spot_instance.areas = spot_instance.calculate_areas()

    def surface_serializer(self) -> tuple[NDArray, NDArray]:
        """Return all points and faces of the whole star."""
        points = copy(self.points)
        faces = copy(self.faces)
        if self.has_spots():
            for spot in self.spots.values():
                n_points = np.shape(points)[0]
                points = np.append(points, spot.points, axis=0)
                faces = np.append(faces, spot.faces + n_points, axis=0)

        return points, faces

    def reset_spots_properties(self) -> None:
        """Reset computed properties for all spots."""
        for spot_instance in self.spots.values():
            spot_instance.boundary = np.array([])
            spot_instance.boundary_center = np.array([])
            spot_instance.center = np.array([])

            spot_instance.points = np.array([])
            spot_instance.normals = np.array([])
            spot_instance.faces = np.array([])
            spot_instance.face_centres = np.array([])

            spot_instance.areas = np.array([])
            spot_instance.potential_gradient_magnitudes = np.array([])
            spot_instance.temperatures = np.array([])
            spot_instance.velocities = np.array([])

            spot_instance.log_g = np.array([])

    def get_flatten_parameter(self, prop: str) -> NDArray:
        """Return flattened array for the requested property.

        :param prop: Property name to flatten (e.g. 'points').
        :returns: Flattened numpy array for the property.
        """
        list_to_concat = [getattr(self, prop)]
        if not self.has_spots() or self._flatten:
            return list_to_concat[0]

        list_to_concat += [getattr(spot, prop) for spot in self.spots.values()]
        if prop == "faces":
            lengths = np.cumsum([np.max(item) + 1 for item in list_to_concat])
            # adjust face indices of subsequent objects by the cumulative vertex offsets
            list_to_concat[1:] = [list_to_concat[i + 1] + lengths[i] for i in range(len(self.spots))]
        return np.concatenate(list_to_concat, axis=0)

    def flat_it(self) -> StarContainer:
        """Flatten all per-spot properties into the main container arrays.

        After this call the container holds unified arrays combining star
        and spot data.
        """
        # naive implementation of idempotency
        if self._flatten:
            return self

        props_list = ["points", "normals", "faces", "temperatures", "log_g", "face_centres", "areas", "velocities"]

        for prop in props_list:
            setattr(self, prop, self.get_flatten_parameter(prop))

        self._flatten = True
        return self

    def transform_points_to_spherical_coordinates(
        self,
        kind: Literal["points", "face_centres"] = "points",
        com_x: Float = 0.0,
    ) -> NDArray:
        """Convert cartesian points to spherical coordinates (object frame).

        :param kind: Name of attribute holding cartesian points (e.g. `points` or `face_centres`).
        :param com_x: X coordinate of the object's centre of mass to subtract.
        :returns: Array of spherical coordinates.
        """
        # separating variables to convert
        centres_cartesian: NDArray = copy(getattr(self, kind))

        # transforming variables
        centres_cartesian[:, 0] -= com_x

        return utils.cartesian_to_spherical(centres_cartesian)

    def assign_radii(self, radii: dict) -> None:
        """Assign radius values from the provided mapping.

        :param radii: Mapping of radius_name → value.
        """
        for key, value in radii.items():
            setattr(self, key, value)

    def symmetry_points(self) -> NDArray:
        """Return points belonging to the base symmetry patch of the surface.

        :returns: Subset of ``points`` corresponding to the symmetric patch.
        """
        if self.has_spots():
            msg = "Surface symmetry is not applicable in this case"
            raise ValueError(msg)
        return symmetry_point_reduction(self.points, self.base_symmetry_points_number)

    def mirror_face_values(self, values: NDArray) -> NDArray:
        """Expand values defined on the symmetry patch to the full surface.

        :param values: Values on the base symmetry faces.
        :returns: Full-sized array remapped to all faces.
        """
        if not self.symmetry_test():
            msg = "Surface symmetry is not applicable in this case."
            raise ValueError(msg)
        return mirror_face_values(values, self.face_symmetry_vector)

    def symmetry_faces(self, values: NDArray) -> NDArray:
        """Reduce a full-surface distribution to the symmetric component.

        :param values: Full-surface array of values.
        :returns: Reduced array defined on the symmetry patch.
        """
        if not self.symmetry_test():
            msg = "Surface symmetry is not applicable in this case."
            raise ValueError(msg)
        return symmetry_face_reduction(values, self.base_symmetry_faces_number)
