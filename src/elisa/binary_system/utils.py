import numpy as np
from pypex.poly2d.polygon import Polygon

from elisa.binary_system import model
from elisa.utils import is_empty
from elisa import (
    umpy as up,
    utils
)


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


def get_visible_projection(obj):
    """
    Returns yz projection of nearside points.

    :param obj: instance;
    :return: numpy.array
    """
    return utils.plane_projection(
        obj.points[
            np.unique(obj.faces[obj.indices])
        ], "yz"
    )


def renormalize_async_result(result):
    """
    Renormalize multiprocessing output to native form.
    Multiprocessing will return several dicts with same passband (due to supplied batches), but continuous
    computaion require dict in form like::

        [{'passband': [all fluxes]}]

    instead::

        [[{'passband': [fluxes in batch]}], [{'passband': [fluxes in batch]}], ...]

    :param result: List;
    :return: Dict[str; numpy.array]
    """
    # todo: come with something more sophisticated
    placeholder = {key: np.array([]) for key in result[-1]}
    for record in result:
        for passband in placeholder:
            placeholder[passband] = record[passband] if is_empty(placeholder[passband]) else np.hstack(
                (placeholder[passband], record[passband]))
    return placeholder
