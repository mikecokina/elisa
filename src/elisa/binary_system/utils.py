import numpy as np

from pypex.poly2d.polygon import Polygon
from jsonschema import (
    validate,
    ValidationError
)
from copy import copy

from .. import units, const
from .. import settings
from .. import umpy as up
from .. base.error import YouHaveNoIdeaError
from .. binary_system import model
from .. utils import is_empty
from .. base.transform import SystemProperties


def potential_from_radius(component, radius, phi, theta, component_distance, mass_ratio, synchronicity):
    """
    Calculate potential given spherical coordinates radius, phi, theta.

    :param component: 'primary` or `secondary`;
    :param radius: float;
    :param phi: float;
    :param theta: float;
    :param component_distance: float;
    :param mass_ratio: float;
    :param synchronicity: float;
    :return: float;
    """
    precalc_fn = model.pre_calculate_for_potential_value_primary if component == 'primary' else \
        model.pre_calculate_for_potential_value_secondary
    potential_fn = model.potential_value_primary if component == 'primary' else \
        model.potential_value_secondary

    precalc_args = (synchronicity, mass_ratio, component_distance, phi, theta)
    args = (mass_ratio, ) + precalc_fn(*precalc_args)
    return potential_fn(radius, *args)


def calculate_phase(time, period, t0, offset=0.5):
    """
    Calculates photometric phase from observations.

    :param time: array;
    :param period: array;
    :param t0: float;
    :param offset: float;
    :return: array;
    """
    return up.mod((time - t0 + offset * period) / period, 1.0) - offset


def faces_to_pypex_poly(t_hulls):
    """
    Convert all faces defined as numpy.array to pypex Polygon class instance.

    :param t_hulls: List[numpy.array];
    :return: List;
    """
    return [Polygon(t_hull, _validity=False) for t_hull in t_hulls]


def pypex_poly_hull_intersection(pypex_faces_gen, pypex_hull: Polygon):
    """
    Resolve intersection of polygons defined in `pypex_faces_gen` with polyogn `pypex_hull`.

    :param pypex_faces_gen: List[pypex.poly2d.polygon.Plygon];
    :param pypex_hull: pypex.poly2d.polygon.Plygon;
    :return: List[pypex.poly2d.polygon.Plygon];
    """
    return [pypex_hull.intersection(poly) for poly in pypex_faces_gen]


def pypex_poly_surface_area(pypex_polys_gen):
    """
    Compute surface areas of pypex.poly2d.polygon.Plygon's.

    :param pypex_polys_gen: List[pypex.poly2d.polygon.Plygon];
    :return: List[float];
    """
    return [poly.surface_area() if poly is not None else 0.0 for poly in pypex_polys_gen]


def hull_to_pypex_poly(hull):
    """
    Convert convex polygon defined by points in List or numpy.array to pypex.poly2d.polygon.Polygon.

    :param hull: Union[List, numpy.array];
    :return: pypex.poly2d.polygon.Plygon;
    """
    return Polygon(hull, _validity=False)


def component_to_list(component):
    """
    Converts component name string into list.

    :param component: str;  If None, `['primary', 'secondary']` will be returned otherwise
                            `primary` and `secondary` will be converted into lists [`primary`] and [`secondary`].
    :return: List[str]
    """
    if component in ["all", "both"]:
        component = ['primary', 'secondary']
    elif component in ['primary', 'secondary']:
        component = [component]
    elif is_empty(component):
        return []
    else:
        raise ValueError('Invalid name of the component. Use `primary`, `secondary`, `all` or `both`')
    return component


def move_sys_onpos(
        init_system,
        orbital_position,
        primary_potential: float = None,
        secondary_potential: float = None,
        on_copy: bool = True,
        recalculate_velocities: bool = False
):
    """
    Prepares a postion container for given orbital position.
    Supplied `system` is not affected (is immutable) if `on_copy` is set to True.

    Following methods are applied::

        system.set_on_position_params()
        system.flat_it()
        system.apply_rotation()
        system.add_secular_velocity()
        system.calculate_face_angles()
        system.apply_darkside_filter()

    :param init_system: elisa.binary_system.container.OrbitalPositionContainer;
    :param orbital_position: collections.namedtuple; elisa.const.Position;
    :param primary_potential: float;
    :param secondary_potential: float;
    :param on_copy: bool;
    :param recalculate_velocities: bool; if True, surface elements velocities are recalculated
                                         (usefull while using apsidal symmetry)
    :return: container; elisa.binary_system.container.OrbitalPositionContainer;
    """
    system = init_system.copy() if on_copy else init_system
    system.set_on_position_params(orbital_position, primary_potential, secondary_potential)
    if recalculate_velocities:
        system.build_velocities(components_distance=orbital_position.distance, component='all')
    system.flat_it()
    system.apply_rotation()
    system.add_secular_velocity()
    system.calculate_face_angles(line_of_sight=const.LINE_OF_SIGHT)
    system.apply_darkside_filter()
    return system


def calculate_rotational_phase(system, component):
    """
    Returns rotational phase with in co-rotating frame of reference.

    :param system: elisa.binary_system.contaier.OrbitalPositionContainer; instance
    :param component: str; `primary` or `secondary`
    :return: float;
    """
    star = getattr(system, component)
    return (star.synchronicity - 1.0) * system.position.phase


def validate_binary_json(data):
    """
    Validate input json to create binary instance from.

    :param data: Dict; json like object
    :return: bool; return True if valid schema, othervise raise error
    :raise: ValidationError;
    """
    schema_std = settings.SCHEMA_REGISTRY.get_schema("binary_system_std")
    schema_community = settings.SCHEMA_REGISTRY.get_schema("binary_system_community")
    std_valid, community_valid = False, False

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

    # previous code cannot catch error when user inputs only one argument from the other parameter input format
    if ('mass_ratio' in data['system'].keys() or 'semi_major_axis' in data['system'].keys()) and std_valid is True:
        raise ValidationError("You probably tried to input your parameters in `standard` format but your "
                              "parameters include `mass ratio` or `semi_major_axis` (use either (M1, M2) or  (q, a)).")

    if ('mass' in data['primary'].keys() or 'mass' in data['secondary'].keys()) and community_valid is True:
        raise ValidationError("You probably tried to input your parameters in `community` format but your "
                              "parameters include masses of the components (useeither (M1, M2) or  (q, a)).")

    if (not community_valid) & (not std_valid):
        raise ValidationError("BinarySystem cannot be created from supplied json schema. ")

    if community_valid & std_valid:
        raise YouHaveNoIdeaError("Make sure that list of fitted parameters contain only `standard` or `community` "
                                 "combination of parameter (either (M1, M2) or  (q, a)).")

    return True


def resolve_json_kind(data, _sin=False):
    """
    Resolve if json is `std` or `community`.

    std - standard physical parameters (M1, M2)
    community - astro community parameters (q, a)

    :param data: Dict; json like
    :param _sin: bool; if False, looking for `semi_major_axis` in given JSON, otherwise looking for `asini`
    :return: str; `std` or `community`
    """
    lookup = "asini" if _sin else "semi_major_axis"
    m1, m2 = data.get("primary", dict()).get("mass"), data.get("secondary", dict()).get("mass")
    q, a = data["system"].get("mass_ratio"), data["system"].get(lookup)

    if m1 and m2:
        return "std"
    if q and a:
        return "community"
    raise LookupError("It seems your JSON is invalid.")


def transform_json_community_to_std(data):
    """
    Transform `community` input json to `std` json.
    Compute `M1` and `M2` from `q` and `a`.

    :param data: Dict;
    :return: Dict;
    """
    q = data["system"].pop("mass_ratio")
    a = SystemProperties.semi_major_axis(data["system"].pop("semi_major_axis"))
    period = (SystemProperties.period(copy(data["system"]["period"])) * units.PERIOD_UNIT).to(units.s).value
    m1 = ((4.0 * const.PI ** 2 * a ** 3) / (const.G * (1.0 + q) * period ** 2))
    m1 = np.float64((m1 * units.kg).to(units.solMass))
    m2 = q * m1

    data["primary"].update({"mass": m1})
    data["secondary"].update({"mass": m2})

    return data


def correction_to_com(distance, mass_ratio, scom):
    """
    Calculates the correction for com from primary-centered coordinate system to barycentric.

    :param distance: float;
    :param mass_ratio: float
    :param scom: float; secondary component component of mass
    :return: float; correction to com in primary-centered system
    """
    distances_to_com = distance * mass_ratio / (1 + mass_ratio)
    dir_to_secondary = scom / np.linalg.norm(scom)
    return distances_to_com * dir_to_secondary
