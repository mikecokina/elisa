from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from jsonschema import ValidationError, validate

from elisa import const, settings, units
from elisa import umpy as up
from elisa.base.error import YouHaveNoIdeaError
from elisa.base.transform import SystemProperties
from elisa.binary_system import model
from elisa.binary_system.radius import calculate_side_radius
from elisa.pypex.poly2d.polygon import Polygon
from elisa.types import Float
from elisa.units import DefaultBinarySystemUnits
from elisa.utils import is_empty

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from elisa.binary_system.container import OrbitalPositionContainer

ComponentName: TypeAlias = Literal["primary", "secondary"]
ComponentSelection: TypeAlias = Literal["primary", "secondary", "all", "both"]
JSONScalar: TypeAlias = str | int | Float | bool | None
JSONValue: TypeAlias = JSONScalar | dict[str, "JSONValue"] | list["JSONValue"]
JSONDict: TypeAlias = dict[str, JSONValue]


def potential_from_radius(
    component: ComponentName,
    radius: Float,
    phi: Float,
    theta: Float,
    component_distance: Float,
    mass_ratio: Float,
    synchronicity: Float,
) -> Float:
    """Calculate the Roche potential for a component at spherical coordinates.

    The potential is evaluated for the supplied radius and angular coordinates
    using the appropriate primary or secondary potential implementation.

    :param component: Component identifier. Allowed values are ``"primary"``
        and ``"secondary"``.
    :type component: Literal["primary", "secondary"]
    :param radius: Radial distance from the component center.
    :type radius: Float
    :param phi: Azimuthal angle.
    :type phi: Float
    :param theta: Polar angle.
    :type theta: Float
    :param component_distance: Instantaneous distance between the two
        components.
    :type component_distance: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param synchronicity: Rotational synchronicity factor of the component.
    :type synchronicity: Float
    :return: Potential value evaluated at the requested coordinates.
    :rtype: Float
    """
    if component == "primary":
        precalc_fn = model.pre_calculate_for_potential_value_primary
        potential_fn = model.potential_value_primary
    else:
        precalc_fn = model.pre_calculate_for_potential_value_secondary
        potential_fn = model.potential_value_secondary

    precalc_args = (
        synchronicity,
        mass_ratio,
        component_distance,
        phi,
        theta,
    )
    args = (mass_ratio, *precalc_fn(*precalc_args))
    return potential_fn(radius, *args)


def calculate_phase(
    time: NDArray,
    period: NDArray,
    t0: Float,
    *,
    offset: Float = 0.5,
) -> NDArray[np.float64]:
    """Calculate photometric phase from observations.

    The phase is normalized to the interval shifted by ``offset``, matching the
    original implementation::

        mod((time - t0 + offset * period) / period, 1.0) - offset

    :param time: Observation times.
    :type time: NDArray
    :param period: Period values corresponding to the observations.
    :type period: NDArray
    :param t0: Reference epoch.
    :type t0: Float
    :param offset: Phase offset applied to the folded result.
    :type offset: Float
    :return: Folded photometric phase values.
    :rtype: NDArray[numpy.float64]
    """
    return np.asarray(
        up.mod((time - t0 + offset * period) / period, 1.0) - offset,
        dtype=np.float64,
    )


def faces_to_pypex_poly(t_hulls: NDArray) -> list[Polygon]:
    """Convert face hulls to :class:`~pypex.poly2d.polygon.Polygon` instances.

    Each input face is converted without validity checking, preserving the
    original behavior.

    :param t_hulls: NDArray of polygon vertex arrays.
    :type t_hulls: NDArray
    :return: Converted polygon instances.
    :rtype: list[Polygon]
    """
    return [Polygon(t_hull, _validity=False) for t_hull in t_hulls]


def pypex_poly_hull_intersection(
    pypex_faces_gen: list[Polygon],
    pypex_hull: Polygon,
) -> list[Polygon | None]:
    """Resolve intersections between polygons and a hull polygon.

    :param pypex_faces_gen: Polygons to intersect with ``pypex_hull``.
    :type pypex_faces_gen: list[Polygon]
    :param pypex_hull: Hull polygon used for clipping.
    :type pypex_hull: Polygon
    :return: Intersection results for all input polygons.
    :rtype: list[Polygon | None]
    """
    return [pypex_hull.intersection(poly) for poly in pypex_faces_gen]


def pypex_poly_surface_area(
    pypex_polys_gen: Iterable[Polygon | None],
) -> list[Float]:
    """Compute surface areas of pypex polygons.

    ``None`` entries produce zero area, preserving the original behavior.

    :param pypex_polys_gen: Polygons for which to compute surface area.
    :type pypex_polys_gen: Iterable[Polygon | None]
    :return: Surface areas of the supplied polygons.
    :rtype: list[Float]
    """
    return [poly.surface_area() if poly is not None else 0.0 for poly in pypex_polys_gen]


def hull_to_pypex_poly(hull: NDArray) -> Polygon:
    """Convert a convex hull to a pypex polygon.

    The hull can be supplied as a list-like object or a NumPy-compatible array.
    Validity checking is disabled to preserve the original behavior.

    :param hull: Convex polygon vertices.
    :type hull: NDArray
    :return: Polygon instance created from ``hull``.
    :rtype: Polygon
    """
    return Polygon(hull, _validity=False)


def component_to_list(
    component: ComponentSelection | None,
) -> list[ComponentName] | []:
    """Convert a component selector into a normalized list of components.

    If ``component`` is ``None`` or empty, an empty list is returned. Values
    ``"all"`` and ``"both"`` are expanded to both components. Values
    ``"primary"`` and ``"secondary"`` are converted to single-item lists.

    :param component: Component selection value.
    :type component: Literal["primary", "secondary", "all", "both"] | None
    :return: Normalized list of component names.
    :rtype: list[Literal["primary", "secondary"]]
    :raises ValueError: If the supplied component name is invalid.
    """
    if component in {"all", "both"}:
        return ["primary", "secondary"]

    if component in {"primary", "secondary"}:
        return [component]

    if is_empty(component):
        return []

    message = "Invalid name of the component. Use `primary`, `secondary`, `all` or `both`."
    raise ValueError(message)


def move_sys_onpos(
    init_system: OrbitalPositionContainer,
    orbital_position: const.Position,
    primary_potential: Float | None = None,
    secondary_potential: Float | None = None,
    *,
    on_copy: bool = True,
    recalculate_velocities: bool = False,
) -> OrbitalPositionContainer:
    """Prepare a position container for a given orbital position.

    Supplied ``init_system`` is not affected and remains immutable if
    ``on_copy`` is set to ``True``.

    The following methods are applied::

        system.set_on_position_params()
        system.flat_it()
        system.apply_rotation()
        system.add_secular_velocity()
        system.calculate_face_angles()
        system.apply_darkside_filter()

    If ``recalculate_velocities`` is enabled, surface element velocities are
    recomputed before flattening. This is useful while using apsidal symmetry.

    :param init_system: Initial orbital position container.
    :type init_system: OrbitalPositionContainer
    :param orbital_position: Orbital position descriptor.
    :type orbital_position: const.Position
    :param primary_potential: Explicit potential for the primary component.
    :type primary_potential: Float | None
    :param secondary_potential: Explicit potential for the secondary component.
    :type secondary_potential: Float | None
    :param on_copy: If ``True``, operate on a copied system instance.
    :type on_copy: bool
    :param recalculate_velocities: If ``True``, recalculate surface element
        velocities.
    :type recalculate_velocities: bool
    :return: Prepared system container at the requested orbital position.
    :rtype: OrbitalPositionContainer
    """
    system = init_system.copy() if on_copy else init_system
    system.set_on_position_params(
        orbital_position,
        primary_potential,
        secondary_potential,
    )

    if recalculate_velocities:
        system.build_velocities(
            components_distance=orbital_position.distance,
            component="all",
        )

    system.flat_it()
    system.apply_rotation()
    system.add_secular_velocity()
    system.calculate_face_angles(line_of_sight=const.LINE_OF_SIGHT)
    system.apply_darkside_filter()
    return system


def calculate_rotational_phase(
    system: OrbitalPositionContainer,
    component: ComponentName,
) -> Float:
    """Return the rotational phase in the co-rotating frame of reference.

    :param system: Orbital position container instance.
    :type system: OrbitalPositionContainer
    :param component: Component identifier, either ``"primary"`` or
        ``"secondary"``.
    :type component: Literal["primary", "secondary"]
    :return: Rotational phase of the selected component.
    :rtype: Float
    """
    star = getattr(system, component)
    return (star.synchronicity - 1.0) * system.position.phase


def validate_binary_json(data: JSONDict) -> bool:
    """Validate input JSON used to create a binary system instance.

    The payload is validated against both supported schemas:

    - ``binary_system_std`` for the standard physical parameterization
      ``(M1, M2)``
    - ``binary_system_community`` for the community parameterization
      ``(q, a)``

    Additional validation is performed to detect cases where the payload mixes
    parameters from both formats in a way that would otherwise pass schema
    validation.

    :param data: JSON-like mapping describing the binary system.
    :type data: JSONDict
    :return: ``True`` if validation succeeds.
    :rtype: bool
    :raises ValidationError: If the payload does not match a valid binary
        system schema or mixes parameterization styles.
    :raises YouHaveNoIdeaError: If both schemas validate simultaneously.
    """
    schema_std = settings.SCHEMA_REGISTRY.get_schema("binary_system_std")
    schema_community = settings.SCHEMA_REGISTRY.get_schema(
        "binary_system_community",
    )

    std_valid = False
    community_valid = False

    try:
        validate(instance=data, schema=schema_std)
        std_valid = True
    except ValidationError:
        pass

    try:
        validate(instance=data, schema=schema_community)
        community_valid = True
    except ValidationError:
        pass

    system_section = data.get("system", {})
    primary_section = data.get("primary", {})
    secondary_section = data.get("secondary", {})

    is_mixed = "mass_ratio" in system_section or "semi_major_axis" in system_section
    if isinstance(system_section, dict) and is_mixed and std_valid:
        message = (
            "You probably tried to input your parameters in `standard` format "
            "but your parameters include `mass_ratio` or `semi_major_axis` "
            "(use either (M1, M2) or (q, a))."
        )
        raise ValidationError(message)

    if (
        isinstance(primary_section, dict)
        and isinstance(secondary_section, dict)
        and ("mass" in primary_section or "mass" in secondary_section)
        and community_valid
    ):
        message = (
            "You probably tried to input your parameters in `community` format "
            "but your parameters include masses of the components "
            "(use either (M1, M2) or (q, a))."
        )
        raise ValidationError(message)

    if not community_valid and not std_valid:
        message = "BinarySystem cannot be created from supplied JSON schema."
        raise ValidationError(message)

    if community_valid and std_valid:
        message = (
            "Make sure that the list of fitted parameters contains only "
            "`standard` or `community` combinations of parameters "
            "(either (M1, M2) or (q, a))."
        )
        raise YouHaveNoIdeaError(message)

    return True


def resolve_json_kind(data: JSONDict, *, _sin: bool = False) -> Literal["std", "community"]:
    """Resolve whether the input JSON uses ``std`` or ``community`` format.

    ``std`` corresponds to standard physical parameters ``(M1, M2)``.
    ``community`` corresponds to astronomy community parameters ``(q, a)``.

    If ``_sin`` is ``False``, the function looks for ``semi_major_axis`` in the
    system section. Otherwise, it looks for ``asini``.

    :param data: JSON-like binary system mapping.
    :type data: JSONDict
    :param _sin: If ``False``, inspect ``semi_major_axis``. If ``True``,
        inspect ``asini`` instead.
    :type _sin: bool
    :return: Resolved JSON kind, either ``"std"`` or ``"community"``.
    :rtype: Literal["std", "community"]
    :raises LookupError: If the JSON content is insufficient or inconsistent.
    """
    lookup = "asini" if _sin else "semi_major_axis"

    primary_section = data.get("primary", {})
    secondary_section = data.get("secondary", {})
    system_section = data.get("system", {})

    m1 = primary_section.get("mass") if isinstance(primary_section, dict) else None
    m2 = secondary_section.get("mass") if isinstance(secondary_section, dict) else None
    q = system_section.get("mass_ratio") if isinstance(system_section, dict) else None
    a = system_section.get(lookup) if isinstance(system_section, dict) else None

    if m1 is not None and m2 is not None:
        return "std"

    if q is not None and a is not None:
        return "community"

    message = "It seems your JSON is invalid."
    raise LookupError(message)


def transform_json_community_to_std(data: JSONDict) -> JSONDict:
    """Transform ``community`` input JSON to ``std`` JSON.

    This function computes component masses ``M1`` and ``M2`` from the
    community parameters ``q`` and ``a`` and updates the original mapping in
    place.

    :param data: JSON-like binary system mapping in community format.
    :type data: JSONDict
    :return: Updated mapping in standard format.
    :rtype: JSONDict
    """
    system_section = data["system"]
    primary_section = data["primary"]
    secondary_section = data["secondary"]

    if not isinstance(system_section, dict):
        message = "Expected `system` section to be a dictionary."
        raise TypeError(message)

    if not isinstance(primary_section, dict):
        message = "Expected `primary` section to be a dictionary."
        raise TypeError(message)

    if not isinstance(secondary_section, dict):
        message = "Expected `secondary` section to be a dictionary."
        raise TypeError(message)

    q = system_section.pop("mass_ratio")
    a = SystemProperties.semi_major_axis(system_section.pop("semi_major_axis"))
    period_value = copy(system_section["period"])
    period = (SystemProperties.period(period_value) * DefaultBinarySystemUnits.system.period).to(units.TIME_UNIT).value

    m1 = (4.0 * const.PI**2 * a**3) / (const.G * (1.0 + q) * period**2)
    m1 = np.float64((m1 * units.kg).to(units.solMass))
    m2 = q * m1

    primary_section.update({"mass": m1})
    secondary_section.update({"mass": m2})

    return data


def correction_to_com(
    distance: Float,
    mass_ratio: Float,
    scom: NDArray,
) -> NDArray[Float]:
    """Calculate the barycentric correction from a primary-centered frame.

    The returned vector points from the primary-centered coordinate origin
    toward the barycenter.

    :param distance: Distance between the binary components.
    :type distance: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param scom: Secondary component center-of-mass vector.
    :type scom: NDArray
    :return: Correction vector to the center of mass in the primary-centered
        system.
    :rtype: NDArray[numpy.float64]
    """
    distances_to_com = distance * mass_ratio / (1.0 + mass_ratio)
    scom_array = np.asarray(scom, dtype=np.float64)
    dir_to_secondary = scom_array / np.linalg.norm(scom_array)
    return np.asarray(distances_to_com * dir_to_secondary, dtype=np.float64)


def calculate_sma_estimate(
    mass_ratio: Float,
    synchronicity: Float,
    potential: Float,
    period: Float,
    component: ComponentName,
    mid_g: Float,
) -> Float:
    """Estimate a semi-major axis value that yields the requested mean gravity.

    The estimate is useful in light-curve fitting where the semi-major axis
    must be fixed to a sensible value. The calculation chooses the semi-major
    axis so that the average surface gravity of the selected component matches
    ``mid_g``.

    The returned value is expressed in solar radii.

    :param mass_ratio: Binary mass ratio ``M2 / M1``.
    :type mass_ratio: Float
    :param synchronicity: Rotational synchronicity factor.
    :type synchronicity: Float
    :param potential: Surface potential.
    :type potential: Float
    :param period: Orbital period in days.
    :type period: Float
    :param component: Component identifier, either ``"primary"`` or
        ``"secondary"``.
    :type component: Literal["primary", "secondary"]
    :param mid_g: Desired average surface acceleration in ``m s^-2``.
    :type mid_g: Float
    :return: Estimated semi-major axis in solar radii.
    :rtype: Float
    """
    radius = calculate_side_radius(
        synchronicity,
        mass_ratio,
        1.0,
        potential,
        component,
    )

    q_funcval = 1.0 / (1.0 + mass_ratio) if component == "primary" else mass_ratio / (1.0 + mass_ratio)

    return 1.4374e-9 * mid_g * (radius * 86400.0 * period) ** 2 / (4.0 * const.PI**2 * q_funcval)
