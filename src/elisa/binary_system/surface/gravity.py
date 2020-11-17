import numpy as np

from .. import utils as bsutils
from ... utils import is_empty
from ... logger import getLogger
from ... base.surface import gravity as bgravity
from ... pulse import pulsations
from ... import (
    umpy as up,
    const
)

logger = getLogger("binary_system.surface.gravity")


def calculate_potential_gradient(components_distance, component, points, synchronicity, mass_ratio):
    """
    Return outter gradients in each point of star surface or defined points.
    If points are not supplied, component instance points are used.

    :param component: str; define target component to compute critical potential; `primary` or `secondary`
    :param components_distance: float, in SMA distance
    :param points: numpy.array;
    :param synchronicity: float;
    :param mass_ratio: float;
    :return: numpy.array;
    """
    r3 = up.power(np.linalg.norm(points, axis=1), 3)
    r_hat3 = up.power(np.linalg.norm(points - np.array([components_distance, 0., 0]), axis=1), 3)

    f2 = up.power(synchronicity, 2)
    if component == 'primary':
        domega_dx = - points[:, 0] / r3 + \
                    mass_ratio * (components_distance - points[:, 0]) / r_hat3 + \
                    f2 * (mass_ratio + 1) * points[:, 0] - \
                    mass_ratio / up.power(components_distance, 2)
    elif component == 'secondary':
        domega_dx = - points[:, 0] / r3 + \
                    mass_ratio * (components_distance - points[:, 0]) / r_hat3 - \
                    f2 * (mass_ratio + 1) * (components_distance - points[:, 0]) + \
                    1 / up.power(components_distance, 2)
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
    :param points: numpy.array; points in which to calculate magnitude of gradient, if False/None take star points
    :param faces: numpy.array; faces corresponding to given points
    :param synchronicity: float;
    :param mass_ratio: float;
    :param face_symmetry_vector: Union[numpy.array, None];
    :return: numpy.array;
    """
    gradients = calculate_potential_gradient(components_distance, component, points, synchronicity, mass_ratio)
    domega_dx, domega_dy, domega_dz = gradients[:, 0], gradients[:, 1], gradients[:, 2]
    points_gradients = up.power(up.power(domega_dx, 2) + up.power(domega_dy, 2) + up.power(domega_dz, 2), 0.5)

    return np.mean(points_gradients[faces], axis=1) if is_empty(face_symmetry_vector) \
        else np.mean(points_gradients[faces], axis=1)[face_symmetry_vector]


def calculate_polar_potential_gradient_magnitude(components_distance, mass_ratio, polar_radius, component,
                                                 synchronicity):
    """
    Calculate magnitude of polar potential gradient.

    :param components_distance: float; in SMA distance
    :param polar_radius: float;
    :param mass_ratio: float;
    :param component: str;
    :param synchronicity: float;
    :return: float;
    """
    points = [0., 0., polar_radius] if component == 'primary' else [components_distance, 0., polar_radius]
    points = np.array(points)
    r3 = up.power(np.linalg.norm(points), 3)
    r_hat3 = up.power(np.linalg.norm(points - np.array([components_distance, 0., 0.])), 3)

    if component == 'primary':
        domega_dx = mass_ratio * components_distance / r_hat3 - mass_ratio / up.power(components_distance, 2)
    elif component == 'secondary':
        domega_dx = - points[0] / r3 + mass_ratio * (components_distance - points[0]) / r_hat3 \
                    - np.power(synchronicity, 2) * (mass_ratio + 1) * (1 - points[0])\
                    + 1. / up.power(components_distance, 2)
    else:
        raise ValueError(f'Invalid value `{component}` of argument `component`. \nUse `primary` or `secondary`.')
    domega_dz = - points[2] * (1. / r3 + mass_ratio / r_hat3)
    return up.power(up.power(domega_dx, 2) + up.power(domega_dz, 2), 0.5)


def calculate_polar_gravity_acceleration(star, components_distance, mass_ratio, component,
                                         semi_major_axis, synchronicity, logg=False):
    """
    Calculates polar gravity acceleration for component of binary system.
    Calculated from gradient of Roche potential::

        d_Omega/dr using transformation g = d_Psi/dr = (GM_component/semi_major_axis**2) * d_Omega/dr
        ( * 1/q in case of secondary component )

    :param star: elisa.base.container.StarContainer;
    :param components_distance: float; (in SMA units)
    :param mass_ratio: float;
    :param component: str;
    :param semi_major_axis: float;
    :param logg: bool; if True log g is returned, otherwise values are not in log10
    :param synchronicity: float;
    :return: numpy.array; surface gravity or log10 of surface gravity
    """
    pgm = calculate_polar_potential_gradient_magnitude(components_distance, mass_ratio,
                                                       star.polar_radius, component, synchronicity)
    gradient = const.G * star.mass * pgm / up.power(semi_major_axis, 2)
    gradient = gradient / mass_ratio if component == 'secondary' else gradient
    return up.log10(gradient) if logg else gradient


def build_surface_gravity(system, components_distance, component="all"):
    """
    Function calculates gravity potential gradient magnitude (surface gravity) for each face.
    Value assigned to face is mean of values calculated in corners of given face.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param component: str; `primary` or `secondary`
    :param components_distance: float;
    :return: system: elisa.binary_system.container.OrbitalPositionContainer;
    """
    if is_empty(component):
        logger.debug("no component set to build surface gravity")
        return

    if is_empty(components_distance):
        raise ValueError('Component distance value was not supplied or is invalid.')

    components = bsutils.component_to_list(component)
    mass_ratio = system.mass_ratio

    for component in components:
        star = getattr(system, component)
        synchronicity = star.synchronicity

        pgm = calculate_polar_potential_gradient_magnitude(components_distance, mass_ratio,
                                                           star.polar_radius, component, star.synchronicity)
        setattr(star, "polar_potential_gradient_magnitude", pgm)

        logger.debug(f'computing potential gradient magnitudes distribution of {component} component')

        points, faces = bgravity.eval_args_for_magnitude_gradient(star)

        scaling_factor = const.G * system.primary.mass / system.semi_major_axis**2
        p_grad = calculate_potential_gradient(components_distance, component, points=points,
                                              synchronicity=synchronicity, mass_ratio=mass_ratio)
        g_acc_vector = scaling_factor * p_grad

        g_acc_vector_spot = dict()
        if star.has_spots():
            for spot_index, spot in star.spots.items():
                logger.debug(f'calculating surface SI unit gravity of {component} component / {spot_index} spot')
                logger.debug(f'calculating distribution of potential gradient '
                             f'magnitudes of spot index: {spot_index} / {component} component')

                p_grad = calculate_potential_gradient(components_distance, component, points=spot.points,
                                                      synchronicity=synchronicity, mass_ratio=mass_ratio)
                g_acc_vector_spot.update({spot_index: scaling_factor * p_grad})

        if star.has_pulsations():
            g_acc_vector, g_acc_vector_spot = \
                pulsations.incorporate_gravity_perturbation(star, g_acc_vector, g_acc_vector_spot,
                                                            phase=system.position.phase)

        gravity = np.mean(np.linalg.norm(g_acc_vector, axis=1)[faces], axis=1) if star.symmetry_test else \
            np.mean(np.linalg.norm(g_acc_vector, axis=1), axis=1)
        setattr(star, 'potential_gradient_magnitudes', gravity[star.face_symmetry_vector]) \
            if star.symmetry_test() else setattr(star, 'potential_gradient_magnitudes', gravity)
        setattr(star, 'log_g', np.log10(star.potential_gradient_magnitudes))

        if star.has_spots():
            for spot_index, spot in star.spots.items():
                setattr(spot, 'potential_gradient_magnitudes',
                        np.mean(np.linalg.norm(g_acc_vector_spot[spot_index], axis=1)[spot.faces], axis=1))
                setattr(spot, 'log_g', np.log10(spot.potential_gradient_magnitudes))

    return system
