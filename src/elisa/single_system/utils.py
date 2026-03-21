from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
from jsonschema import ValidationError, validate

from elisa import const, settings, units, utils
from elisa.base.error import YouHaveNoIdeaError
from elisa.base.transform import StarProperties, SystemProperties

if TYPE_CHECKING:
    from elisa.const import Position
    from elisa.single_system.container import SinglePositionContainer
    from elisa.single_system.system import SingleSystem
    from elisa.types import Float


def move_sys_onpos(
    system: SinglePositionContainer,
    position: Position,
    *,
    on_copy: bool = True,
) -> SinglePositionContainer:
    """Prepare and return a position container for a given orbital position.

    The supplied ``system`` is not modified when ``on_copy`` is set to
    ``True``. The function performs a series of common per-position
    transformations on the container in the following order::

        system.set_on_position_params(position)
        system.flat_it()
        system.apply_rotation()
        system.add_secular_velocity()
        system.calculate_face_angles(line_of_sight=const.LINE_OF_SIGHT)
        system.apply_darkside_filter()

    :param system: Position container to prepare.
    :type system: elisa.base.container.PositionContainer
    :param position: Orbital position namedtuple describing viewing geometry.
    :type position: elisa.const.Position
    :param on_copy: If ``True`` operate on a shallow copy and leave the
        original ``system`` unchanged (keyword-only).
    :type on_copy: bool
    :returns: Prepared position container.
    :rtype: elisa.base.container.PositionContainer
    """
    if on_copy:
        system = system.copy()

    system.set_on_position_params(position)
    system.flat_it()
    system.apply_rotation()
    system.add_secular_velocity()
    system.calculate_face_angles(line_of_sight=const.LINE_OF_SIGHT)
    system.apply_darkside_filter()
    return system


def calculate_volume(system: SinglePositionContainer | SingleSystem) -> Float:
    """Return the approximate volume of the (rotationally distorted) star.

    The function uses an ellipsoid approximation based on polar and
    equatorial radii stored on ``system.star`` and delegates the numeric
    calculation to :func:`elisa.utils.calculate_ellipsoid_volume`.

    :param system: Position container holding a ``star`` attribute.
    :type system: elisa.base.container.PositionContainer
    :returns: Volume in the model length units.
    :rtype: elisa.types.Float
    """
    args = (
        system.star.polar_radius,
        system.star.equatorial_radius,
        system.star.equatorial_radius,
    )
    return utils.calculate_ellipsoid_volume(*args)


def validate_single_json(data: dict[str, Any]) -> bool:
    """Validate JSON-like input used to construct a SingleSystem.

    The function validates ``data`` against two registry schemas
    (``single_system_std`` and ``single_system_radius``). It returns
    ``True`` when the payload matches at least one schema and raises
    :class:`elisa.base.error.YouHaveNoIdeaError` on failure or when both
    schemas match (ambiguous input).

    :param data: Parsed JSON object as dictionary.
    :type data: dict[str, Any]
    :returns: ``True`` if validation succeeds.
    :rtype: bool
    :raises elisa.base.error.YouHaveNoIdeaError: If the input does not
        match any known schema or matches both schemas.
    """
    schema_std = settings.SCHEMA_REGISTRY.get_schema("single_system_std")
    schema_radius = settings.SCHEMA_REGISTRY.get_schema("single_system_radius")

    std_valid = False
    radius_valid = False

    try:
        validate(instance=data, schema=schema_std)
        std_valid = True
    except ValidationError:
        pass

    try:
        validate(instance=data, schema=schema_radius)
        radius_valid = True
    except ValidationError:
        pass

    if not std_valid and not radius_valid:
        msg = "Make sure that list of parameters is consistent with the used schema."
        raise YouHaveNoIdeaError(msg)

    if radius_valid and std_valid:
        msg = (
            "Make sure that list of fitted parameters contain only `standard` or `radius` "
            "combination of parameter (containing either `polar_log_g` or `polar_radius`)."
        )
        raise YouHaveNoIdeaError(msg)

    return True


def resolve_json_kind(data: dict[str, Any]) -> str:
    """Determine whether supplied JSON uses the "standard" or "radius" form.

    The function inspects ``data['star']`` for presence of the keys
    ``polar_log_g`` (``std`` form) or ``equivalent_radius`` (``radius``
    form) and returns a corresponding tag.

    :param data: Parsed JSON object.
    :type data: dict[str, Any]
    :returns: ``"std"`` or ``"radius"``.
    :rtype: str
    :raises LookupError: When required keys are missing and the format cannot
        be determined.
    """
    star_section = data.get("star", {})
    polar_g = star_section.get("polar_log_g")
    polar_radius = star_section.get("equivalent_radius")

    if polar_g is not None:
        return "std"
    if polar_radius is not None:
        return "radius"

    msg = "It seems your JSON is invalid."
    raise LookupError(msg)


def transform_json_radius_to_std(data: dict[str, Any]) -> dict[str, Any]:
    """Convert ``radius``-style JSON to the ``standard`` form.

    The conversion computes ``polar_log_g`` from an input
    ``equivalent_radius`` using a simple rotationally-deformed ellipsoid
    approximation. The function updates ``data`` in-place and returns it.

    :param data: Input JSON-like dictionary in ``radius`` form.
    :type data: dict[str, Any]
    :returns: The transformed dictionary containing ``polar_log_g``.
    :rtype: dict[str, Any]
    """
    def equatorial_to_polar_radius(r_eq: Float, period_: Float, mass_: Float) -> Float:
        k = 2 * np.power(const.PI, 2) * np.power(r_eq, 3) / (const.G * mass_ * np.power(period_, 2))
        return 1 / (1 - k)

    def polar_from_equatorial_radius(r_eq: Float, period_: Float, mass_: Float) -> Float:
        rho = equatorial_to_polar_radius(r_eq, period_, mass_)
        return r_eq / np.power(rho, 2.0 / 3.0)

    mass = StarProperties.mass(data["star"]["mass"])
    # default unit of radius is the same as for the semi-major axis
    radius = SystemProperties.semi_major_axis(data["star"].pop("equivalent_radius"))

    period_val = copy(data["system"]["rotation_period"])
    period = (
        SystemProperties.period(period_val) * units.DefaultSingleSystemUnits.system.rotation_period
    ).to(units.TIME_UNIT).value

    polar_radius = polar_from_equatorial_radius(radius, period, mass)
    data["star"]["polar_log_g"] = np.log10(const.G * mass / np.power(polar_radius, 2)) + 2.0

    return data
