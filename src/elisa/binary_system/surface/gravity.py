import numpy as np

from elisa import logger, const
from elisa.conf import config
from elisa.utils import is_empty
from elisa.binary_system import utils as bsutils
from elisa import umpy as up

config.set_up_logging()
__logger__ = logger.getLogger("binary-system-gravity-module")


def eval_args_for_magnitude_gradient(star_container):
    if star_container.spots:
        points, faces = star_container.points, star_container.faces
    else:
        points = star_container.points[:star_container.base_symmetry_points_number]
        faces = star_container.faces[:star_container.base_symmetry_faces_number]
    return points, faces


def calculate_potential_gradient(components_distance, component, points, synchronicity, mass_ratio):
    """
    Return outter gradients in each point of star surface or defined points.
    If points are not supplied, component instance points are used.

    :param component: str; define target component to compute critical potential; `primary` or `secondary`
    :param components_distance: float, in SMA distance
    :param points: List or numpy.array
    :param synchronicity:
    :param mass_ratio:
    :return: numpy.array
    """
    r3 = up.power(np.linalg.norm(points, axis=1), 3)
    r_hat3 = up.power(np.linalg.norm(points - np.array([components_distance, 0., 0]), axis=1), 3)

    if component == 'primary':
        f2 = up.power(synchronicity, 2)
        domega_dx = - points[:, 0] / r3 + mass_ratio * (
                components_distance - points[:, 0]) / r_hat3 + f2 * (
                            mass_ratio + 1) * points[:, 0] - mass_ratio / up.power(components_distance, 2)
    elif component == 'secondary':
        f2 = up.power(synchronicity, 2)
        domega_dx = - points[:, 0] / r3 + mass_ratio * (
                components_distance - points[:, 0]) / r_hat3 - f2 * (
                            mass_ratio + 1) * (
                            components_distance - points[:, 0]) * points[:, 0] + 1 / up.power(
            components_distance, 2)
    else:
        raise ValueError(f'Invalid value `{component}` of argument `component`.\n Use `primary` or `secondary`.')

    domega_dy = - points[:, 1] * (1 / r3 + mass_ratio / r_hat3 - f2 * (mass_ratio + 1))
    domega_dz = - points[:, 2] * (1 / r3 + mass_ratio / r_hat3)
    return -np.column_stack((domega_dx, domega_dy, domega_dz))


def calculate_face_magnitude_gradient(components_distance, component, points, faces, synchronicity, mass_ratio,
                                      face_symmetry_vector=None):
    """
    Return array of face magnitude gradients calculated as a mean of magnitude gradients on vertices.
    If neither points nor faces are supplied, method runs over component instance points and faces.

    :param component: str; define target component to compute critical potential; `primary` or `secondary`
    :param components_distance: float; distance of componetns in SMA units
    :param points: points in which to calculate magnitude of gradient, if False/None take star points
    :param faces: faces corresponding to given points
    :param synchronicity:
    :param mass_ratio:
    :param face_symmetry_vector: Union[numpy.array, None]
    :return: numpy.array
    """

    gradients = calculate_potential_gradient(components_distance, component, points, synchronicity, mass_ratio)
    domega_dx, domega_dy, domega_dz = gradients[:, 0], gradients[:, 1], gradients[:, 2]
    points_gradients = up.power(up.power(domega_dx, 2) + up.power(domega_dy, 2) + up.power(domega_dz, 2), 0.5)

    return np.mean(points_gradients[faces], axis=1) if is_empty(face_symmetry_vector) \
        else np.mean(points_gradients[faces], axis=1)[face_symmetry_vector]


def calculate_polar_potential_gradient_magnitude(components_distance, mass_ratio, polar_radius, component):
    """
    Calculate magnitude of polar potential gradient.

    :param components_distance: float; in SMA distance
    :param polar_radius: float;
    :param mass_ratio: float;
    :param component: str;
    :return: float
    """
    points = [0., 0., polar_radius] if component == 'primary' else [components_distance, 0., polar_radius]
    points = np.array(points)
    r3 = up.power(np.linalg.norm(points), 3)
    r_hat3 = up.power(np.linalg.norm(points - np.array([components_distance, 0., 0.])), 3)

    if component == 'primary':
        domega_dx = mass_ratio * components_distance / r_hat3 - mass_ratio / up.power(components_distance, 2)
    elif component == 'secondary':
        domega_dx = + 0 - points[0] / r3 + mass_ratio * (components_distance - points[0]) / r_hat3 \
                    + 1. / up.power(components_distance, 2)
    else:
        raise ValueError(f'Invalid value `{component}` of argument `component`. \nUse `primary` or `secondary`.')
    domega_dz = - points[2] * (1. / r3 + mass_ratio / r_hat3)
    return up.power(up.power(domega_dx, 2) + up.power(domega_dz, 2), 0.5)


def calculate_polar_gravity_acceleration(star_container, components_distance, mass_ratio, component,
                                         semi_major_axis, logg=False):
    """
    Calculates polar gravity acceleration for component of binary system.
    Calculated from gradient of Roche potential::

        d_Omega/dr using transformation g = d_Psi/dr = (GM_component/semi_major_axis**2) * d_Omega/dr
        ( * 1/q in case of secondary component )

    :param star_container:
    :param components_distance: float; (in SMA units)
    :param mass_ratio: float;
    :param component: str;
    :param semi_major_axis: float;
    :param logg: bool; if True log g is returned, otherwise values are not in log10
    :return: numpy.array; surface gravity or log10 of surface gravity
    """
    pgm = calculate_polar_potential_gradient_magnitude(components_distance, mass_ratio,
                                                       star_container.polar_radius, component)
    gradient = const.G * star_container.mass * pgm / up.power(semi_major_axis, 2)
    gradient = gradient / mass_ratio if component == 'secondary' else gradient
    return up.log10(gradient) if logg else gradient


def build_surface_gravity(system_container, components_distance=None, component="all"):
    """
    Function calculates gravity potential gradient magnitude (surface gravity) for each face.
    Value assigned to face is mean of values calculated in corners of given face.

    :param system_container: BinarySystem instance
    :param component: str; `primary` or `secondary`
    :param components_distance: float
    :return:
    """
    if is_empty(component):
        __logger__.debug("no component set to build surface gravity")
        return

    if is_empty(components_distance):
        raise ValueError('Component distance value was not supplied or is invalid.')

    components = bsutils.component_to_list(component)
    mass_ratio = system_container.mass_ratio
    semi_major_axis = system_container.semi_major_axis

    for component in components:
        star_container = getattr(system_container, component)
        synchronicity = star_container.synchronicity
        polar_gravity = calculate_polar_gravity_acceleration(star_container, components_distance, mass_ratio,
                                                             component, semi_major_axis, logg=False)

        pgm = calculate_polar_potential_gradient_magnitude(components_distance, mass_ratio,
                                                           star_container.polar_radius, component)
        setattr(star_container, "polar_potential_gradient_magnitude", pgm)
        gravity_scalling_factor = polar_gravity / pgm

        __logger__.debug(f'computing potential gradient magnitudes distribution of {component} component')

        pgms_args = eval_args_for_magnitude_gradient(star_container) + (synchronicity, mass_ratio)
        pgms_kwargs = dict(
            **{"face_symmetry_vector": star_container.face_symmetry_vector} if not star_container.has_spots() else {})
        pgms = calculate_face_magnitude_gradient(components_distance, component, *pgms_args, **pgms_kwargs)
        setattr(star_container, "potential_gradient_magnitudes", pgms)

        logg = up.log10(gravity_scalling_factor * star_container.potential_gradient_magnitudes)
        setattr(star_container, "log_g", logg)

        if star_container.has_spots():
            for spot_index, spot in star_container.spots.items():
                __logger__.debug(f'calculating surface SI unit gravity of {component} component / {spot_index} spot')
                __logger__.debug(f'calculating distribution of potential gradient '
                                 f'magnitudes of spot index: {spot_index} / {component} component')

                spot_pgms = calculate_face_magnitude_gradient(components_distance, component, spot.points, spot.faces,
                                                              synchronicity, mass_ratio, face_symmetry_vector=None)
                setattr(spot, "potential_gradient_magnitudes", spot_pgms)
                spot_logg = up.log10(gravity_scalling_factor * spot.potential_gradient_magnitudes)
                setattr(spot, "log_g", spot_logg)
