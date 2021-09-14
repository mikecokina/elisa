import numpy as np
from copy import copy

from jsonschema import (
    validate,
    ValidationError
)

from .. base.error import YouHaveNoIdeaError
from .. import const, utils, settings, units
from .. base.transform import StarProperties, SystemProperties


def move_sys_onpos(system, position, on_copy=True):
    """
    Prepares a postion container for given orbital position.
    Supplied `system` is not affected if `on_copy` is set to True.

    Following methods are applied::

        system.set_on_position_params()
        system.flat_it()
        system.apply_rotation()
        system.apply_darkside_filter()

    :param position: collections.namedtuple; elisa.const.Position;
    :param system: elisa.single_system.container.PositionContainer;
    :param on_copy: bool;
    :return: container; elisa.sinary_system.container.PositionContainer;
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


def calculate_volume(system):
    """
    Returns volume of rotationally squashed star based on a volume of elipsoid.

    :param system: elisa.SingleSystem;
    :return: float
    """
    args = system.star.polar_radius, system.star.equatorial_radius, system.star.equatorial_radius
    return utils.calculate_ellipsoid_volume(*args)


def validate_single_json(data):
    """
    Validate input json to create SingleSystem instance from.

    :param data: Dict; json like object
    :return: ; return True if valid schema, otherwise raise error
    :raise: ValidationError;
    """
    schema_std = settings.SCHEMA_REGISTRY.get_schema("single_system_std")
    schema_radius = settings.SCHEMA_REGISTRY.get_schema("single_system_radius")
    std_valid, radius_valid = False, False

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
        raise YouHaveNoIdeaError("Make sure that list of parameters is consistent with the used schema.")

    if radius_valid & std_valid:
        raise YouHaveNoIdeaError("Make sure that list of fitted parameters contain only `standard` or `radius` "
                                 "combination of parameter (containing either `polar_log_g` or `polar_radius`).")

    return True


def resolve_json_kind(data):
    """
    Resolve if json is in `standard` or `radius` format.

    std - size of the star defined by the polar surface gravity `polar_log_g` parameter
    community - size of the star defined by the `equivalent_radius` parameter

    :param data: Dict; json like
    :return: str; `std` or `radius`
    """
    polar_g = data['star'].get('polar_log_g')
    polar_radius = data['star'].get('equivalent_radius')

    if polar_g:
        return "std"
    elif polar_radius:
        return "radius"
    raise LookupError("It seems your JSON is invalid.")


def transform_json_radius_to_std(data):
    """
    Transform `radius` input format json to `std` json.
    Compute polar_log_g form equivalent radius.

    :param data: Dict;
    :return: Dict;
    """
    def equatorial_to_polar_radius(r_eq, period, mass):
        k = 2 * np.power(const.PI, 2) * np.power(r_eq, 3) / (const.G * mass * np.power(period, 2))
        return 1 / (1 - k)

    def polar_from_equatorial_radius(r_eq, period, mass):
        rho = equatorial_to_polar_radius(r_eq, period, mass)
        return r_eq / np.power(rho, 2.0/3.0)

    mass = (StarProperties.mass(data['star']['mass']) * units.MASS_UNIT).to(units.kg).value
    # default unit of radius is the same as for the semi-major axis
    radius = (SystemProperties.semi_major_axis(data['star'].pop('equivalent_radius'))
              * units.DISTANCE_UNIT).to(units.m).value
    period = (SystemProperties.period(copy(data["system"]["rotation_period"])) * units.PERIOD_UNIT).to(units.s).value

    polar_radius = polar_from_equatorial_radius(radius, period, mass)
    data['star']['polar_log_g'] = np.log10(const.G * mass / np.power(polar_radius, 2))

    return data
