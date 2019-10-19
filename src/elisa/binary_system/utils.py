from pypex.poly2d.polygon import Polygon
from elisa import umpy as up
from elisa.binary_system import model
from elisa.utils import is_empty


def get_flaten_properties(component):
    """
    Return flatten ndarrays of points, faces, etc. from object instance and spot instances for given object.
    :param component: Star instance
    :return: Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]

    ::

        Tuple(points, normals, faces, temperatures, log_g, rals, face_centres)
    """
    points = component.points
    normals = component.normals
    faces = component.faces
    temperatures = component.temperatures
    log_g = component.log_g
    rals = {mode_idx: mode.rals[0] for mode_idx, mode in component.pulsations.items()}
    centres = component.face_centres

    if isinstance(component.spots, (dict,)):
        for idx, spot in component.spots.items():
            faces = up.concatenate((faces, spot.faces + len(points)), axis=0)
            points = up.concatenate((points, spot.points), axis=0)
            normals = up.concatenate((normals, spot.normals), axis=0)
            temperatures = up.concatenate((temperatures, spot.temperatures), axis=0)
            log_g = up.concatenate((log_g, spot.log_g), axis=0)
            for mode_idx, mode in component.pulsations.items():
                rals[mode_idx] = up.concatenate((rals[mode_idx], mode.rals[1][idx]), axis=0)
            centres = up.concatenate((centres, spot.face_centres), axis=0)

    return points, normals, faces, temperatures, log_g, rals, centres


def potential_from_radius(component, radius, phi, theta, component_distance, mass_ratio, synchronicity):
    """
    Calculate potential given spherical coordinates radius, phi, theta.
    :param component: 'primary` or `secondary`
    :param radius: float
    :param phi: float
    :param theta: float
    :param component_distance: float
    :param mass_ratio: float
    :param synchronicity: float
    :return: float
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
    :param time: array
    :param period: array
    :param t0: float
    :param offset: float
    :return: array
    """
    return up.mod((time - t0 + offset * period) / period, 1.0) - offset


def faces_to_pypex_poly(t_hulls):
    """
    Convert all faces defined as numpy.array to pypex Polygon class instance.
    :param t_hulls: List[numpy.array]
    :return: List
    """
    return [Polygon(t_hull, _validity=False) for t_hull in t_hulls]


def pypex_poly_hull_intersection(pypex_faces_gen, pypex_hull: Polygon):
    """
    Resolve intersection of polygons defined in `pypex_faces_gen` with polyogn `pypex_hull`.
    :param pypex_faces_gen: List[pypex.poly2d.polygon.Plygon]
    :param pypex_hull: pypex.poly2d.polygon.Plygon
    :return: List[pypex.poly2d.polygon.Plygon]
    """
    return [pypex_hull.intersection(poly) for poly in pypex_faces_gen]


def pypex_poly_surface_area(pypex_polys_gen):
    """
    Compute surface areas of pypex.poly2d.polygon.Plygon's.
    :param pypex_polys_gen: List[pypex.poly2d.polygon.Plygon]
    :return: List[float]
    """
    return [poly.surface_area() if poly is not None else 0.0 for poly in pypex_polys_gen]


def hull_to_pypex_poly(hull):
    """
    Convert convex polygon defined by points in List or numpy.array to pypex.poly2d.polygon.Plygon.
    :param hull: List or numpy.array
    :return: pypex.poly2d.polygon.Plygon
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
