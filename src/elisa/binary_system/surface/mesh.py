import numpy as np

from .. import utils as butils
from .. import (
    utils as bsutils,
    model
)
from ... base.error import MaxIterationError, SpotError
from ... base.spot import incorporate_spots_mesh
from ... import settings
from ... opt.fsolver import fsolver, fsolve
from ... utils import is_empty
from ... logger import getLogger
from ... pulse import pulsations
from ... import (
    umpy as up,
    utils,
    opt,
    const
)

logger = getLogger("binary_system.surface.mesh")


def build_mesh(system, components_distance, component="all"):
    """
    Build points of surface for primary or/and secondary component. Mesh is evaluated with spots.
    Points are assigned to system.

    :param system: elisa.binary_system.contaier.OrbitalPositionContainer; instance
    :param component: Union[str, None];
    :param components_distance: float;
    :return: system; elisa.binary_system.contaier.OrbitalPositionContainer; instance
    """
    components = bsutils.component_to_list(component)

    for component in components:
        star = getattr(system, component)
        # in case of spoted surface, symmetry is not used
        if getattr(system, 'morphology') == 'over-contact':
            a, b, c, d = mesh_over_contact(system, component, symmetry_output=True)
        else:
            a, b, c, d = mesh_detached(system, components_distance, component, symmetry_output=True)

        star.points = a
        star.point_symmetry_vector = b
        star.base_symmetry_points_number = c
        star.inverse_point_symmetry_matrix = d

    add_spots_to_mesh(system, components_distance, component="all")

    return system


def build_pulsations_on_mesh(system, component, components_distance):
    """
    adds position perturbations to container mesh

    :param system: elisa.binary_system.contaier.OrbitalPositionContainer; instance
    :param component: Union[str, None];
    :param components_distance: float;
    :return: elisa.binary_system.contaier.OrbitalPositionContainer; instance
    """
    components = bsutils.component_to_list(component)
    for component in components:
        star = getattr(system, component)
        if star.has_pulsations():
            phase = butils.calculate_rotational_phase(system, component)
            com_x = 0 if component == 'primary' else components_distance
            star = pulsations.generate_harmonics(star, com_x=com_x, phase=phase, time=system.time)
            pulsations.incorporate_pulsations_to_mesh(star, com_x=com_x)
    return system


def pre_calc_azimuths_for_detached_points(discretization):
    """
    Returns azimuths for the whole quarter surface in specific order::

        (near point, equator, far point and the rest)

    separator gives you information about position of these sections.

    :param discretization: float; discretization factor
    :return: Tuple; (phi: numpy.array, theta: numpy.array, separator: numpy.array)
    """
    separator = []

    # azimuths for points on equator
    num = int(const.PI // discretization)
    phi = np.linspace(0., const.PI, num=num + 1)
    theta = np.array([const.HALF_PI for _ in phi])
    separator.append(np.shape(theta)[0])

    # azimuths for points on meridian
    num = int(const.HALF_PI // discretization)
    phi_meridian = np.array([const.PI for _ in range(num - 1)] + [0 for _ in range(num)])
    theta_meridian = up.concatenate((np.linspace(const.HALF_PI - discretization, discretization, num=num - 1),
                                     np.linspace(0., const.HALF_PI, num=num, endpoint=False)))

    phi = up.concatenate((phi, phi_meridian))
    theta = up.concatenate((theta, theta_meridian))
    separator.append(np.shape(theta)[0])

    # azimuths for rest of the quarter
    num = int(const.HALF_PI // discretization)
    thetas = np.linspace(discretization, const.HALF_PI, num=num - 1, endpoint=False)
    phi_q, theta_q = [], []
    for tht in thetas:
        alpha_corrected = discretization / up.sin(tht)
        num = int(const.PI // alpha_corrected)
        alpha_corrected = const.PI / (num + 1)
        phi_q_add = [alpha_corrected * ii for ii in range(1, num + 1)]
        phi_q += phi_q_add
        theta_q += [tht for _ in phi_q_add]

    phi = up.concatenate((phi, phi_q))
    theta = up.concatenate((theta, theta_q))

    return phi, theta, separator


def pre_calc_azimuths_for_overcontact_farside_points(discretization):
    """
    Calculates azimuths (directions) to the surface points of over-contact component on its far-side (convex part).

    :param discretization: float; discretization factor
    :return: Tuple; (phi: numpy.array, theta: numpy.array, separtor: numpy.array)
    """
    separator = []

    # calculating points on farside equator
    num = int(const.HALF_PI // discretization)
    phi = np.linspace(const.HALF_PI, const.PI, num=num + 1)
    theta = np.array([const.HALF_PI for _ in phi])
    separator.append(np.shape(theta)[0])

    # calculating points on phi = pi meridian
    phi_meridian1 = np.array([const.PI for _ in range(num)])
    theta_meridian1 = np.linspace(0., const.HALF_PI - discretization, num=num)
    phi = up.concatenate((phi, phi_meridian1))
    theta = up.concatenate((theta, theta_meridian1))
    separator.append(np.shape(theta)[0])

    # calculating points on phi = pi/2 meridian, perpendicular to component`s distance vector
    num -= 1
    phi_meridian2 = np.array([const.HALF_PI for _ in range(num)])
    theta_meridian2 = np.linspace(discretization, const.HALF_PI, num=num, endpoint=False)
    phi = up.concatenate((phi, phi_meridian2))
    theta = up.concatenate((theta, theta_meridian2))
    separator.append(np.shape(theta)[0])

    # calculating the rest of the surface on farside
    thetas = np.linspace(discretization, const.HALF_PI, num=num, endpoint=False)
    phi_q1, theta_q1 = [], []
    for tht in thetas:
        alpha_corrected = discretization / up.sin(tht)
        num = int(const.HALF_PI // alpha_corrected)
        alpha_corrected = const.HALF_PI / (num + 1)
        phi_q_add = [const.HALF_PI + alpha_corrected * ii for ii in range(1, num + 1)]
        phi_q1 += phi_q_add
        theta_q1 += [tht for _ in phi_q_add]
    phi = up.concatenate((phi, phi_q1))
    theta = up.concatenate((theta, theta_q1))
    separator.append(np.shape(theta)[0])

    return phi, theta, separator


def pre_calc_azimuths_for_overcontact_neck_points(
        discretization, neck_position, neck_polynomial, polar_radius, component):
    """
    Calculates azimuths (directions) to the surface points of over-contact component on neck.

    :param discretization: float; doscretiozation factor
    :param neck_position: float; x position of neck of over-contact binary
    :param neck_polynomial: scipy.Polynome; polynome that define neck profile in plane `xz`
    :param polar_radius: float;
    :param component: str; `primary` or `secondary`
    :return: Tuple; (phi: numpy.array, z: numpy.array, separator: numpy.array)
    """
    # generating the neck

    # lets define cylindrical coordinate system r_n, phi_n, z_n for our neck where z_n = x, phi_n = 0 heads along
    # z axis
    delta_z = discretization * polar_radius

    # test radii on neck_position
    r_neck, separator = [], []

    if component == 'primary':
        num = 15 * int(neck_position // (polar_radius * discretization))
        # position of z_n adapted to the slope of the neck, gives triangles with more similar areas
        x_curve = np.linspace(0., neck_position, num=num, endpoint=True)
        z_curve = np.polyval(neck_polynomial, x_curve)

        # radius on the neck
        mid_r = np.min(z_curve)

        curve = np.column_stack((x_curve, z_curve))
        neck_lengths = up.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
        neck_length = np.sum(neck_lengths)
        segment = neck_length / (int(neck_length // delta_z))

        k = 1
        z_ns, line_sum = [], 0.0
        for ii in range(num - 2):
            line_sum += neck_lengths[ii]
            if line_sum > k * segment:
                z_ns.append(x_curve[ii + 1])
                r_neck.append(z_curve[ii])
                k += 1
        z_ns.append(neck_position)
        r_neck.append(mid_r)
        z_ns = np.array(z_ns)
    else:
        num = 15 * int((1 - neck_position) // (polar_radius * discretization))
        # position of z_n adapted to the slope of the neck, gives triangles with more similar areas
        x_curve = np.linspace(neck_position, 1, num=num, endpoint=True)
        z_curve = np.polyval(neck_polynomial, x_curve)

        # radius on the neck
        mid_r = np.min(z_curve)

        curve = np.column_stack((x_curve, z_curve))
        neck_lengths = up.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
        neck_length = np.sum(neck_lengths)
        segment = neck_length / (int(neck_length // delta_z))

        k = 1
        z_ns, line_sum = [1 - neck_position], 0.0
        r_neck.append(mid_r)
        for ii in range(num - 2):
            line_sum += neck_lengths[ii]
            if line_sum > k * segment:
                z_ns.append(1 - x_curve[ii + 1])
                r_neck.append(z_curve[ii])
                k += 1

        z_ns = np.array(z_ns)

    # equator azimuths
    phi = np.array([const.HALF_PI for _ in z_ns])
    z = z_ns
    separator.append(np.shape(z)[0])
    # meridian azimuths
    phi = up.concatenate((phi, np.array([0 for _ in z_ns])))
    z = up.concatenate((z, z_ns))
    separator.append(np.shape(z)[0])

    phi_n, z_n = [], []
    for ii, zz in enumerate(z_ns):
        num = int(0.93 * (const.HALF_PI * r_neck[ii] // delta_z))
        num = 1 if num == 0 else num
        start_val = const.HALF_PI / num
        phis = np.linspace(start_val, const.HALF_PI, num=num - 1, endpoint=False)
        z_n += [zz for _ in phis]
        phi_n += [phi for phi in phis]
    phi = up.concatenate((phi, np.array(phi_n)))
    z = up.concatenate((z, np.array(z_n)))
    separator.append(np.shape(z)[0])

    return phi, z, separator


def get_surface_points(*args):
    """
    Function solves radius for given azimuths that are passed in `args`.
    It use `scipy.optimize.fsolve` method. Function to solve is specified as last parameter in `args` Tuple.
    :param args: Tuple;

    ::

        Tuple[
                phi: numpy.array,
                theta: numpy.array,
                x0: float,
                components_distance: float,
                precalc_fn: callable,
                potential_fn: callable,
                fprime: callable,
                surface_potential: float,
                mass_ratio: float
                synchronicity: float
            ]

    :return: numpy.array
    """
    phi, theta, x0, components_distance, precalc_fn, potential_fn, fprime, potential, q, synchronicity = args
    max_iter = settings.MAX_SOLVER_ITERS
    precalc_vals = precalc_fn(*(synchronicity, q, components_distance, phi, theta), return_as_tuple=True)
    x0 = x0 * np.ones(phi.shape)
    radius_kwargs = dict(fprime=fprime, maxiter=max_iter, args=((q, ) + precalc_vals, potential), rtol=1e-10)
    radius = opt.newton.newton(potential_fn, x0, **radius_kwargs)
    if (radius < 0.0).any():
        raise ValueError('Solver found at least one point in the opposite direction. Check you points. ')
    return utils.spherical_to_cartesian(np.column_stack((radius, phi, theta)))


def get_surface_points_cylindrical(*args):
    """
    Function solves radius for given azimuths that are passed in `args`.

    :param args: Tuple;

    ::

         Tuple[
                phi: numpy.array,
                z: numpy.array,
                components_distance: float,
                x0: float,
                precalc_fn: callable,
                potential_fn: callable,
                fprime: callable (fprime),
                surface_potential: float,
                mass_ratio: float,
                synchronicity: float
              ]

    :return: numpy.array;
    """
    phi, z, components_distance, x0, precalc_fn, potential_fn, fprime, potential, q, synchronicity = args
    max_iter = settings.MAX_SOLVER_ITERS
    precalc_vals = precalc_fn(*(synchronicity, q, phi, z, components_distance), return_as_tuple=True)
    x0 = x0 * np.ones(phi.shape)
    radius_kwargs = dict(fprime=fprime, maxiter=max_iter, rtol=1e-10, args=((q,) + precalc_vals, potential))
    radius = opt.newton.newton(potential_fn, x0, **radius_kwargs)
    return utils.cylindrical_to_cartesian(np.column_stack((up.abs(radius), phi, z)))


def mesh_detached(system, components_distance, component, symmetry_output=False):
    """
    Creates surface mesh of given binary star component in case of detached or semi-detached system.

    :param system: elisa.binary_system.contaienr.OrbitalPositionContainer;
    :param symmetry_output: bool; if True, besides surface points are returned also `symmetry_vector`,
                                  `base_symmetry_points_number`, `inverse_symmetry_matrix`
    :param component: str; `primary` or `secondary`
    :param components_distance: numpy.float
    :return: Union[Tuple, numpy.array]; (if `symmetry_output` is False)

    Array of surface points if symmetry_output = False::

         numpy.array([[x1 y1 z1],
                      [x2 y2 z2],
                       ...
                      [xN yN zN]])

    othervise::

        (
         numpy.array([[x1 y1 z1],
                      [x2 y2 z2],
                        ...
                      [xN yN zN]]) - array of surface points,
         numpy.array([indices_of_symmetrical_points]) - array which remapped surface points to symmetrical one
                                                          quarter of surface,
         numpy.float - number of points included in symmetrical one quarter of surface,
         numpy.array([quadrant[indexes_of_remapped_points_in_quadrant]) - matrix of four sub matrices that
                                                                            mapped basic symmetry quadrant to all
                                                                            others quadrants
        )
    """
    star = getattr(system, component)
    discretization_factor = star.discretization_factor
    synchronicity = star.synchronicity
    mass_ratio = system.mass_ratio
    potential = star.surface_potential

    potential_fn = getattr(model, f"potential_{component}_fn")
    precalc_fn = getattr(model, f"pre_calculate_for_potential_value_{component}")
    fprime = getattr(model, f"radial_{component}_potential_derivative")

    # pre calculating azimuths for surface points on quarter of the star surface
    phi, theta, separator = pre_calc_azimuths_for_detached_points(discretization_factor)

    # calculating mesh in cartesian coordinates for quarter of the star
    # args = phi, theta, components_distance, precalc_fn, potential_fn
    args = phi, theta, star.side_radius, components_distance, precalc_fn, \
        potential_fn, fprime, potential, mass_ratio, synchronicity

    logger.debug(f'calculating surface points of {component} component in mesh_detached '
                 f'function using single process method')
    points_q = get_surface_points(*args)

    equator = points_q[:separator[0], :]
    # assigning equator points and nearside and farside points A and B
    x_a, x_eq, x_b = equator[0, 0], equator[1: -1, 0], equator[-1, 0]
    y_a, y_eq, y_b = equator[0, 1], equator[1: -1, 1], equator[-1, 1]
    z_a, z_eq, z_b = equator[0, 2], equator[1: -1, 2], equator[-1, 2]

    # calculating points on phi = 0 meridian
    meridian = points_q[separator[0]: separator[1], :]
    x_meridian, y_meridian, z_meridian = meridian[:, 0], meridian[:, 1], meridian[:, 2]

    # the rest of the surface
    quarter = points_q[separator[1]:, :]
    x_q, y_q, z_q = quarter[:, 0], quarter[:, 1], quarter[:, 2]

    # stiching together 4 quarters of stellar surface in order:
    # north hemisphere: left_quadrant (from companion point of view):
    #                   nearside_point, farside_point, equator, quarter, meridian
    #                   right_quadrant:
    #                   quadrant, equator
    # south hemisphere: right_quadrant:
    #                   quadrant, meridian
    #                   left_quadrant:
    #                   quadrant
    x = np.array([x_a, x_b])
    y = np.array([y_a, y_b])
    z = np.array([z_a, z_b])
    x = up.concatenate((x, x_eq, x_q, x_meridian, x_q, x_eq, x_q, x_meridian, x_q))
    y = up.concatenate((y, y_eq, y_q, y_meridian, -y_q, -y_eq, -y_q, -y_meridian, y_q))
    z = up.concatenate((z, z_eq, z_q, z_meridian, z_q, z_eq, -z_q, -z_meridian, -z_q))

    x = -x + components_distance if component == 'secondary' else x
    points = np.column_stack((x, y, z))
    if symmetry_output:
        equator_length = np.shape(x_eq)[0]
        meridian_length = np.shape(x_meridian)[0]
        quarter_length = np.shape(x_q)[0]
        quadrant_start = 2 + equator_length
        base_symmetry_points_number = 2 + equator_length + quarter_length + meridian_length
        symmetry_vector = up.concatenate((up.arange(base_symmetry_points_number),  # 1st quadrant
                                          up.arange(quadrant_start, quadrant_start + quarter_length),
                                          up.arange(2, quadrant_start),  # 2nd quadrant
                                          up.arange(quadrant_start, base_symmetry_points_number),  # 3rd quadrant
                                          up.arange(quadrant_start, quadrant_start + quarter_length)
                                          ))

        points_length = np.shape(x)[0]
        inverse_symmetry_matrix = \
            np.array([up.arange(base_symmetry_points_number),  # 1st quadrant
                      up.concatenate(([0, 1],
                                      up.arange(base_symmetry_points_number + quarter_length,
                                                base_symmetry_points_number + quarter_length + equator_length),
                                      up.arange(base_symmetry_points_number,
                                                base_symmetry_points_number + quarter_length),
                                      up.arange(base_symmetry_points_number - meridian_length,
                                                base_symmetry_points_number))),  # 2nd quadrant
                      up.concatenate(([0, 1],
                                      up.arange(base_symmetry_points_number + quarter_length,
                                                base_symmetry_points_number + quarter_length + equator_length),
                                      up.arange(base_symmetry_points_number + quarter_length + equator_length,
                                                base_symmetry_points_number + 2 * quarter_length + equator_length +
                                                meridian_length))),  # 3rd quadrant
                      up.concatenate((up.arange(2 + equator_length),
                                      up.arange(points_length - quarter_length, points_length),
                                      up.arange(base_symmetry_points_number + 2 * quarter_length + equator_length,
                                                base_symmetry_points_number + 2 * quarter_length + equator_length +
                                                meridian_length)))  # 4th quadrant
                      ])

        return points, symmetry_vector, base_symmetry_points_number, inverse_symmetry_matrix
    else:
        return points


def mesh_over_contact(system, component="all", symmetry_output=False):
    """
    Creates surface mesh of given binary star component in case of over-contact system.

    :param system: elisa.binary_system.contaienr.OrbitalPositionContainer;
    :param symmetry_output: bool; if true, besides surface points are returned also `symmetry_vector`,
                                 `base_symmetry_points_number`, `inverse_symmetry_matrix`
    :param component: str; `primary` or `secondary`
    :return: Union[Tuple, numpy.array]; (if symmetry_output is False)

    Array of surface points if symmetry_output = False::

        numpy.array([[x1 y1 z1],
                     [x2 y2 z2],
                      ...
                     [xN yN zN]])

    otherwise::

             numpy.array([[x1 y1 z1],
                          [x2 y2 z2],
                           ...
                          [xN yN zN]]) - array of surface points,

             numpy.array([indices_of_symmetrical_points]) - array which remapped surface points to symmetrical one
             quarter of surface,
             numpy.float - number of points included in symmetrical one quarter of surface,
             numpy.array([quadrant[indexes_of_remapped_points_in_quadrant]) - matrix of four sub matrices that
             mapped basic symmetry quadrant to all others quadrants
    """
    star = getattr(system, component)
    discretization_factor = star.discretization_factor
    synchronicity = star.synchronicity
    q = system.mass_ratio
    potential = star.surface_potential
    r_polar = star.polar_radius

    # calculating distance between components
    components_distance = 1.0  # system.orbit.orbital_motion(phase=0)[0][0]

    fn = getattr(model, f"potential_{component}_fn")
    fn_cylindrical = getattr(model, f"potential_{component}_cylindrical_fn")
    precalc = getattr(model, f"pre_calculate_for_potential_value_{component}")
    precal_cylindrical = getattr(model, f"pre_calculate_for_potential_value_{component}_cylindrical")
    fprime = getattr(model, f"radial_{component}_potential_derivative")
    cylindrical_fprime = getattr(model, f"radial_{component}_potential_derivative_cylindrical")

    # precalculating azimuths for farside points
    phi_farside, theta_farside, separator_farside = \
        pre_calc_azimuths_for_overcontact_farside_points(discretization_factor)

    # generating the azimuths for neck
    neck_position, neck_polynomial = calculate_neck_position(system, return_polynomial=True)
    phi_neck, z_neck, separator_neck = \
        pre_calc_azimuths_for_overcontact_neck_points(discretization_factor, neck_position, neck_polynomial,
                                                      polar_radius=star.polar_radius,
                                                      component=component)

    # solving points on farside
    # here implement multiprocessing
    args = phi_farside, theta_farside, r_polar, components_distance, precalc, fn, fprime, potential, q, synchronicity
    logger.debug(f'calculating farside points of {component} component in mesh_overcontact '
                 f'function using single process method')
    points_farside = get_surface_points(*args)

    # assigning equator points and point A (the point on the tip of the farside equator)
    equator_farside = points_farside[:separator_farside[0], :]
    x_eq1, x_a = equator_farside[: -1, 0], equator_farside[-1, 0]
    y_eq1, y_a = equator_farside[: -1, 1], equator_farside[-1, 1]
    z_eq1, z_a = equator_farside[: -1, 2], equator_farside[-1, 2]

    # assigning points on phi = pi
    meridian_farside1 = points_farside[separator_farside[0]: separator_farside[1], :]
    x_meridian1, y_meridian1, z_meridian1 = meridian_farside1[:, 0], meridian_farside1[:, 1], meridian_farside1[:, 2]

    # assigning points on phi = pi/2 meridian, perpendicular to component`s distance vector
    meridian_farside2 = points_farside[separator_farside[1]: separator_farside[2], :]
    x_meridian2, y_meridian2, z_meridian2 = meridian_farside2[:, 0], meridian_farside2[:, 1], meridian_farside2[:, 2]

    # assigning the rest of the surface on farside
    quarter = points_farside[separator_farside[2]:, :]
    x_q1, y_q1, z_q1 = quarter[:, 0], quarter[:, 1], quarter[:, 2]

    # solving points on neck
    args = phi_neck, z_neck, components_distance, star.polar_radius, \
        precal_cylindrical, fn_cylindrical, cylindrical_fprime, \
        star.surface_potential, system.mass_ratio, synchronicity
    logger.debug(f'calculating neck points of {component} component in mesh_overcontact '
                 f'function using single process method')
    points_neck = get_surface_points_cylindrical(*args)

    # assigning equator points on neck
    r_eqn = points_neck[:separator_neck[0], :]
    z_eqn, y_eqn, x_eqn = r_eqn[:, 0], r_eqn[:, 1], r_eqn[:, 2]

    # assigning points on phi = 0 meridian, perpendicular to component`s distance vector
    r_meridian_n = points_neck[separator_neck[0]: separator_neck[1], :]
    z_meridian_n, y_meridian_n, x_meridian_n = r_meridian_n[:, 0], r_meridian_n[:, 1], r_meridian_n[:, 2]

    # assigning the rest of the surface on neck
    r_n = points_neck[separator_neck[1]:, :]
    z_n, y_n, x_n = r_n[:, 0], r_n[:, 1], r_n[:, 2]

    # building point blocks similar to those in detached system (equator pts, meridian pts and quarter pts)
    x_eq = up.concatenate((x_eqn, x_eq1), axis=0)
    y_eq = up.concatenate((y_eqn, y_eq1), axis=0)
    z_eq = up.concatenate((z_eqn, z_eq1), axis=0)
    x_q = up.concatenate((x_n, x_meridian2, x_q1), axis=0)
    y_q = up.concatenate((y_n, y_meridian2, y_q1), axis=0)
    z_q = up.concatenate((z_n, z_meridian2, z_q1), axis=0)
    x_meridian = up.concatenate((x_meridian_n, x_meridian1), axis=0)
    y_meridian = up.concatenate((y_meridian_n, y_meridian1), axis=0)
    z_meridian = up.concatenate((z_meridian_n, z_meridian1), axis=0)

    x = up.concatenate((np.array([x_a]), x_eq, x_q, x_meridian, x_q, x_eq, x_q, x_meridian, x_q))
    y = up.concatenate((np.array([y_a]), y_eq, y_q, y_meridian, -y_q, -y_eq, -y_q, -y_meridian, y_q))
    z = up.concatenate((np.array([z_a]), z_eq, z_q, z_meridian, z_q, z_eq, -z_q, -z_meridian, -z_q))

    x = -x + components_distance if component == 'secondary' else x
    points = np.column_stack((x, y, z))
    if symmetry_output:
        equator_length = np.shape(x_eq)[0]
        meridian_length = np.shape(x_meridian)[0]
        quarter_length = np.shape(x_q)[0]
        quadrant_start = 1 + equator_length
        base_symmetry_points_number = 1 + equator_length + quarter_length + meridian_length
        symmetry_vector = up.concatenate((up.arange(base_symmetry_points_number),  # 1st quadrant
                                          up.arange(quadrant_start, quadrant_start + quarter_length),
                                          up.arange(1, quadrant_start),  # 2nd quadrant
                                          up.arange(quadrant_start, base_symmetry_points_number),  # 3rd quadrant
                                          up.arange(quadrant_start, quadrant_start + quarter_length)
                                          ))

        points_length = np.shape(x)[0]
        inverse_symmetry_matrix = \
            np.array([up.arange(base_symmetry_points_number),  # 1st quadrant
                      up.concatenate(([0],
                                      up.arange(base_symmetry_points_number + quarter_length,
                                                base_symmetry_points_number + quarter_length + equator_length),
                                      up.arange(base_symmetry_points_number,
                                                base_symmetry_points_number + quarter_length),
                                      up.arange(base_symmetry_points_number - meridian_length,
                                                base_symmetry_points_number))),  # 2nd quadrant
                      up.concatenate(([0],
                                      up.arange(base_symmetry_points_number + quarter_length,
                                                base_symmetry_points_number + quarter_length + equator_length),
                                      up.arange(base_symmetry_points_number + quarter_length + equator_length,
                                                base_symmetry_points_number + 2 * quarter_length + equator_length +
                                                meridian_length))),  # 3rd quadrant
                      up.concatenate((up.arange(1 + equator_length),
                                      up.arange(points_length - quarter_length, points_length),
                                      up.arange(base_symmetry_points_number + 2 * quarter_length + equator_length,
                                                base_symmetry_points_number + 2 * quarter_length + equator_length +
                                                meridian_length)))  # 4th quadrant
                      ])

        return points, symmetry_vector, base_symmetry_points_number, inverse_symmetry_matrix
    else:
        return points


def mesh_spots(system, components_distance, component="all"):
    """
    Compute points of each spots and assigns values to spot container instance.
    If any of any spot point cannot be obtained, entire spot will be omitted.

    :param system: elisa.binary_system.contaienr.OrbitalPositionContainer;
    :param component: str;
    :param components_distance: float;
    :return: bool;
    """

    def solver_condition(x, *_args):
        if isinstance(x, np.ndarray):
            x = x[0]
        point = utils.spherical_to_cartesian([x, _args[1], _args[2]])
        point[0] = point[0] if component == "primary" else components_distance - point[0]
        # ignore also spots where one of points is situated just on the neck
        if getattr(system, "morphology") == "over-contact":
            if (component == "primary" and point[0] >= neck_position) or \
                    (component == "secondary" and point[0] <= neck_position):
                return False
        return True

    components = bsutils.component_to_list(component)
    fns = {
        "primary": (model.potential_primary_fn, model.pre_calculate_for_potential_value_primary,
                    model.radial_primary_potential_derivative),
        "secondary": (model.potential_secondary_fn, model.pre_calculate_for_potential_value_secondary,
                      model.radial_secondary_potential_derivative)
    }
    fns = {component: fns[component] for component in components}

    # in case of wuma system, get separation and make additional test of location of each point (if primary
    # spot doesn't intersect with secondary, if does, then such spot will be skipped completly)
    neck_position = calculate_neck_position(system) if system.morphology == "over-contact" else 1e10

    for component, functions in fns.items():
        logger.debug(f"evaluating spots for {component} component")
        potential_fn, precalc_fn, fprime = functions
        component_instance = getattr(system, component)

        if not component_instance.spots:
            logger.debug(f"no spots to evaluate for {component} component - continue")
            continue

        # iterate over spots
        for spot_index, spot_instance in list(component_instance.spots.items()):
            # lon -> phi, lat -> theta
            lon, lat = spot_instance.longitude, spot_instance.latitude

            alpha = spot_instance.discretization_factor \
                if spot_instance.discretization_factor < spot_instance.angular_radius else spot_instance.angular_radius
            spot_radius = spot_instance.angular_radius
            synchronicity = component_instance.synchronicity
            mass_ratio = system.mass_ratio
            potential = component_instance.surface_potential

            # initial radial vector
            radial_vector = np.array([1.0, lon, lat])  # unit radial vector to the center of current spot
            center_vector = utils.spherical_to_cartesian([1.0, lon, lat])
            args1, use = (synchronicity, mass_ratio, components_distance, radial_vector[1], radial_vector[2]), False
            args2 = ((system.mass_ratio,) + precalc_fn(*args1), potential)
            kwargs = {'original_kwargs': args1}
            solution, use = fsolver(potential_fn, solver_condition, *args2, **kwargs)

            if not use:
                # in case of spots, each point should be usefull, otherwise remove spot from
                # component spot list and skip current spot computation
                logger.warning(f"center of spot {spot_instance.kwargs_serializer()} "
                               f"doesn't satisfy reasonable conditions and entire spot will be omitted")

                component_instance.remove_spot(spot_index=spot_index)
                continue

            spot_center_r = solution
            spot_center = utils.spherical_to_cartesian([spot_center_r, lon, lat])

            # compute euclidean distance of two points on spot (x0)
            # we have to obtain distance between center and 1st point in 1st inner ring of spot
            args1, use = (synchronicity, mass_ratio, components_distance, lon, lat + alpha), False
            args2 = ((system.mass_ratio,) + precalc_fn(*args1), potential)
            kwargs = {'original_kwargs': args1}
            solution, use = fsolver(potential_fn, solver_condition, *args2, **kwargs)

            if not use:
                # in case of spots, each point should be usefull, otherwise remove spot from
                # component spot list and skip current spot computation
                logger.warning(f"first inner ring of spot {spot_instance.kwargs_serializer()} "
                               f"doesn't satisfy reasonable conditions and entire spot will be omitted")

                component_instance.remove_spot(spot_index=spot_index)
                continue

            x0 = up.sqrt(spot_center_r ** 2 + solution ** 2 - (2.0 * spot_center_r * solution * up.cos(alpha)))

            # number of points in latitudal direction
            # + 1 to obtain same discretization as object itself
            num_radial = int(np.round(spot_radius / alpha)) + 1
            logger.debug(f'number of rings in spot {spot_instance.kwargs_serializer()} is {num_radial}')
            thetas = np.linspace(lat, lat + spot_radius, num=num_radial, endpoint=True)

            num_azimuthal = [1 if i == 0 else int(i * 2.0 * const.PI * x0 // x0) for i in range(0, len(thetas))]
            deltas = [np.linspace(0., const.FULL_ARC, num=num, endpoint=False) for num in num_azimuthal]

            spot_phi, spot_theta = [], []
            for theta_index, theta in enumerate(thetas):
                # first point of n-th ring of spot (counting start from center)
                default_spherical_vector = [1.0, lon % const.FULL_ARC, theta]

                for delta_index, delta in enumerate(deltas[theta_index]):
                    # rotating default spherical vector around spot center vector and thus generating concentric
                    # circle of points around centre of spot
                    delta_vector = utils.arbitrary_rotation(theta=delta, omega=center_vector,
                                                            vector=utils.spherical_to_cartesian(
                                                                default_spherical_vector),
                                                            degrees=False,
                                                            omega_normalized=True)

                    spherical_delta_vector = utils.cartesian_to_spherical(delta_vector)

                    spot_phi.append(spherical_delta_vector[1])
                    spot_theta.append(spherical_delta_vector[2])

            spot_phi, spot_theta = np.array(spot_phi), np.array(spot_theta)
            args = spot_phi, spot_theta, spot_center_r, components_distance, precalc_fn, \
                potential_fn, fprime, potential, mass_ratio, synchronicity
            try:
                spot_points = get_surface_points(*args)
            except (MaxIterationError, ValueError) as e:
                raise SpotError(f"Solver could not find at least some surface points of spot "
                                f"{spot_instance.kwargs_serializer()}. Probable reason is that your spot is"
                                f"intersecting neck which is currently not supported.")

            if getattr(system, "morphology") == "over-contact":
                if spot_points.ndim == 2:
                    validity_test = (spot_points[:, 0] <= neck_position).all() if component == 'primary' else \
                        (spot_points[:, 0] <= (1 - neck_position)).all()
                else:
                    validity_test = False

                if not validity_test:
                    raise SpotError(f"Your spot {spot_instance.kwargs_serializer()} "
                                    f"is intersecting neck which is currently not supported.")

            boundary_points = spot_points[-len(deltas[-1]):]

            if component == "primary":
                spot_instance.points = np.array(spot_points)
                spot_instance.boundary = np.array(boundary_points)
                spot_instance.center = np.array(spot_center)
            else:
                spot_instance.points = np.array([np.array([components_distance - point[0], -point[1], point[2]])
                                                 for point in spot_points])

                spot_instance.boundary = np.array([np.array([components_distance - point[0], -point[1], point[2]])
                                                   for point in boundary_points])

                spot_instance.center = np.array([components_distance - spot_center[0], -spot_center[1], spot_center[2]])


def calculate_neck_position(system, return_polynomial=False):
    """
    Function calculates x-coordinate of the `neck` (the narrowest place) of an over-contact system.

    :return: Union[Tuple (if return_polynomial is True), float];

    If return_polynomial is set to True::

        (neck position: float, polynomial degree: int)

    otherwise::

        float
    """
    neck_position = None
    components_distance = 1.0
    components = ['primary', 'secondary']
    points_primary, points_secondary = [], []

    # generating only part of the surface that I'm interested in (neck in xy plane for x between 0 and 1)
    angles = np.linspace(0., const.HALF_PI, 100, endpoint=True)
    for component in components:
        for angle in angles:
            component_instance = getattr(system, component)
            synchronicity = component_instance.synchronicity
            q = system.mass_ratio
            potential = component_instance.surface_potential

            fn = getattr(model, f"potential_{component}_fn")
            precalc_fn = getattr(model, f"pre_calculate_for_potential_value_{component}")

            args, use = (components_distance, angle, const.HALF_PI), False
            scipy_solver_init_value = np.array([components_distance / 10000.0])
            args = ((system.mass_ratio,) + precalc_fn(*((synchronicity, q) + args)), potential)
            solution, _, ier, _ = fsolve(fn, scipy_solver_init_value, full_output=True, args=args, xtol=1e-12)

            # check for regular solution
            if ier == 1 and not up.isnan(solution[0]):
                solution = solution[0]
                if 30 >= solution >= 0:
                    use = True
            else:
                continue

            if use:
                if component == 'primary':
                    points_primary.append([solution * up.cos(angle), solution * up.sin(angle)])
                elif component == 'secondary':
                    points_secondary.append([- (solution * up.cos(angle) - components_distance),
                                             solution * up.sin(angle)])

    neck_points = np.array(points_secondary + points_primary)
    # fitting of the neck with polynomial in order to find minimum
    polynomial_fit = np.polyfit(neck_points[:, 0], neck_points[:, 1], deg=15)
    polynomial_fit_differentiation = np.polyder(polynomial_fit)
    roots = np.roots(polynomial_fit_differentiation)
    roots = [np.real(xx) for xx in roots if np.imag(xx) == 0]
    # choosing root that is closest to the middle of the system, should work...
    # idea is to rule out roots near 0 or 1
    comparision_value = 1
    for root in roots:
        new_value = abs(0.5 - root)
        if new_value < comparision_value:
            comparision_value = new_value
            neck_position = root
    if return_polynomial:
        return neck_position, polynomial_fit
    else:
        return neck_position


def add_spots_to_mesh(system, components_distance, component="all"):
    """
    Function implements surface points into clean mesh and removes stellar
    points and other spot points under the given spot if such overlapped spots exists.

    :param system: elisa.binary_system.contaienr.OrbitalPositionContainer;
    :param components_distance: float;
    :param component: Union[str, None];
    """

    components = bsutils.component_to_list(component)

    if is_empty(components):
        # skip building if not required
        return

    component_com = {'primary': 0.0, 'secondary': components_distance}
    for component in components:
        star = getattr(system, component)
        mesh_spots(system, components_distance=components_distance, component=component)
        incorporate_spots_mesh(star, component_com=component_com[component])
