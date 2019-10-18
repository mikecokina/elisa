import numpy as np

from elisa import umpy as up, utils, opt, logger, const
from elisa.conf import config

config.set_up_logging()
__logger__ = logger.getLogger("binary-system-mesh-module")


def pre_calc_azimuths_for_detached_points(deiscretization_factor):
    """
    Returns azimuths for the whole quarter surface in specific order::

        (near point, equator, far point and the rest)

    separator gives you information about position of these sections.

    :param deiscretization_factor: float; discretization factor
    :return: Tuple; (phi: numpy.array, theta: numpy.array, separtor: numpy.array)
    """
    separator = []

    # azimuths for points on equator
    num = int(const.PI // deiscretization_factor)
    phi = np.linspace(0., const.PI, num=num + 1)
    theta = np.array([const.HALF_PI for _ in phi])
    separator.append(np.shape(theta)[0])

    # azimuths for points on meridian
    num = int(const.HALF_PI // deiscretization_factor)
    phi_meridian = np.array([const.PI for _ in range(num - 1)] + [0 for _ in range(num)])
    theta_meridian = up.concatenate((np.linspace(const.HALF_PI - deiscretization_factor,
                                                 deiscretization_factor, num=num - 1),
                                     np.linspace(0., const.HALF_PI, num=num, endpoint=False)))

    phi = up.concatenate((phi, phi_meridian))
    theta = up.concatenate((theta, theta_meridian))
    separator.append(np.shape(theta)[0])

    # azimuths for rest of the quarter
    num = int(const.HALF_PI // deiscretization_factor)
    thetas = np.linspace(deiscretization_factor, const.HALF_PI, num=num - 1, endpoint=False)
    phi_q, theta_q = [], []
    for tht in thetas:
        alpha_corrected = deiscretization_factor / up.sin(tht)
        num = int(const.PI // alpha_corrected)
        alpha_corrected = const.PI / (num + 1)
        phi_q_add = [alpha_corrected * ii for ii in range(1, num + 1)]
        phi_q += phi_q_add
        theta_q += [tht for _ in phi_q_add]

    phi = up.concatenate((phi, phi_q))
    theta = up.concatenate((theta, theta_q))

    return phi, theta, separator


def pre_calc_azimuths_for_overcontact_farside_points(alpha):
    """
    Calculates azimuths (directions) to the surface points of over-contact component on its far-side.

    :param alpha: float; discretization factor
    :return: Tuple; (phi: numpy.array, theta: numpy.array, separtor: numpy.array)
    """
    separator = []

    # calculating points on farside equator
    num = int(const.HALF_PI // alpha)
    phi = np.linspace(const.HALF_PI, const.PI, num=num + 1)
    theta = np.array([const.HALF_PI for _ in phi])
    separator.append(np.shape(theta)[0])

    # calculating points on phi = pi meridian
    phi_meridian1 = np.array([const.PI for _ in range(num)])
    theta_meridian1 = np.linspace(0., const.HALF_PI - alpha, num=num)
    phi = up.concatenate((phi, phi_meridian1))
    theta = up.concatenate((theta, theta_meridian1))
    separator.append(np.shape(theta)[0])

    # calculating points on phi = pi/2 meridian, perpendicular to component`s distance vector
    num -= 1
    phi_meridian2 = np.array([const.HALF_PI for _ in range(num)])
    theta_meridian2 = np.linspace(alpha, const.HALF_PI, num=num, endpoint=False)
    phi = up.concatenate((phi, phi_meridian2))
    theta = up.concatenate((theta, theta_meridian2))
    separator.append(np.shape(theta)[0])

    # calculating the rest of the surface on farside
    thetas = np.linspace(alpha, const.HALF_PI, num=num, endpoint=False)
    phi_q1, theta_q1 = [], []
    for tht in thetas:
        alpha_corrected = alpha / up.sin(tht)
        num = int(const.HALF_PI // alpha_corrected)
        alpha_corrected = const.HALF_PI / (num + 1)
        phi_q_add = [const.HALF_PI + alpha_corrected * ii for ii in range(1, num + 1)]
        phi_q1 += phi_q_add
        theta_q1 += [tht for _ in phi_q_add]
    phi = up.concatenate((phi, phi_q1))
    theta = up.concatenate((theta, theta_q1))
    separator.append(np.shape(theta)[0])

    return phi, theta, separator


def pre_calc_azimuths_for_overcontact_neck_points(alpha, neck_position, neck_polynomial, polar_radius, component):
    """
    Calculates azimuths (directions) to the surface points of over-contact component on neck.

    :param alpha: float; doscretiozation factor
    :param neck_position: float; x position of neck of over-contact binary
    :param neck_polynomial: scipy.Polynome; polynome that define neck profile in plane `xz`
    :param polar_radius: float
    :param component: str; `primary` or `secondary`
    :return: Tuple; (phi: numpy.array, z: numpy.array, separator: numpy.array)
    """
    # generating the neck

    # lets define cylindrical coordinate system r_n, phi_n, z_n for our neck where z_n = x, phi_n = 0 heads along
    # z axis
    delta_z = alpha * polar_radius

    # test radii on neck_position
    r_neck = []
    separator = []

    if component == 'primary':
        num = 15 * int(neck_position // (polar_radius * alpha))
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
        num = 15 * int((1 - neck_position) // (polar_radius * alpha))
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
    Function solves radius for given azimuths that are passed in *args.
    It use `scipy.optimize.fsolve` method. Function to solve is specified as last parameter in *args Tuple.

    :param args: Tuple;

    ::

        Tuple[
                phi: numpy.array,
                theta: numpy.array,
                x0: float,
                components_distance: float,
                precalc: method,
                fn: method,
                derivative_fn: method]

    :return: numpy.array
    """
    phi, theta, x0, components_distance, precalc_fn, potential_fn, potential_derivative_fn, surface_potential = args
    precalc_vals = precalc_fn(*(components_distance, phi, theta), return_as_tuple=True)
    x0 = x0 * np.ones(phi.shape)
    radius = opt.newton.newton(potential_fn, x0, fprime=potential_derivative_fn,
                               maxiter=config.MAX_SOLVER_ITERS, args=(precalc_vals, surface_potential), rtol=1e-10)
    return utils.spherical_to_cartesian(np.column_stack((radius, phi, theta)))


def get_surface_points_cylindrical(*args):
    """
    Function solves radius for given azimuths that are passed in *args.

    :param args: Tuple;

    ::

         Tuple[
                phi: numpy.array,
                z: numpy.array,
                components_distance: float,
                x0: float,
                precalc: method,
                fn: method
                fprime: method
              ]

    :return: numpy.array
    """
    phi, z, components_distance, x0, precalc_fn, potential_fn, potential_derivative_fn, surface_potential = args
    precalc_vals = precalc_fn(*(phi, z, components_distance), return_as_tuple=True)
    x0 = x0 * np.ones(phi.shape)

    radius = opt.newton.newton(potential_fn, x0, fprime=potential_derivative_fn, args=(precalc_vals, surface_potential),
                               maxiter=config.MAX_SOLVER_ITERS, rtol=1e-10)
    return utils.cylindrical_to_cartesian(np.column_stack((up.abs(radius), phi, z)))


def mesh_detached(system, components_distance, component, symmetry_output=False, **kwargs):
    """
    Creates surface mesh of given binary star component in case of detached or semi-detached system.

    :param system: 
    :param symmetry_output: bool; if True, besides surface points are returned also `symmetry_vector`,
                                  `base_symmetry_points_number`, `inverse_symmetry_matrix`
    :param component: str; `primary` or `secondary`
    :param components_distance: numpy.float
    :return: Tuple or numpy.array (if `symmetry_output` is False)

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
    star_container = getattr(system, component)
    discretization_factor = star_container.discretization_factor

    if component == 'primary':
        potential_fn = system.potential_primary_fn
        precalc_fn = system.pre_calculate_for_potential_value_primary
        potential_derivative_fn = system.radial_primary_potential_derivative
    elif component == 'secondary':
        potential_fn = system.potential_secondary_fn
        precalc_fn = system.pre_calculate_for_potential_value_secondary
        potential_derivative_fn = system.radial_secondary_potential_derivative
    else:
        raise ValueError('Invalid value of `component` argument: `{}`. Expecting '
                         '`primary` or `secondary`.'.format(component))

    # pre calculating azimuths for surface points on quarter of the star surface
    phi, theta, separator = pre_calc_azimuths_for_detached_points(discretization_factor)

    if config.NUMBER_OF_THREADS == 1:
        # calculating mesh in cartesian coordinates for quarter of the star
        # args = phi, theta, components_distance, precalc_fn, potential_fn
        args = phi, theta, star_container.side_radius, \
               components_distance, precalc_fn, potential_fn, potential_derivative_fn, \
               star_container.surface_potential
        __logger__.debug(f'calculating surface points of {component} component in mesh_detached '
                         f'function using single process method')
        points_q = get_surface_points(*args)
    else:
        # todo: consider to remove following multiproc line if "parallel" solver implemented
        # calculating mesh in cartesian coordinates for quarter of the star
        args = phi, theta, components_distance, precalc_fn, potential_fn

        __logger__.debug(f'calculating surface points of {component} component in mesh_detached '
                         f'function using multi process method')
        points_q = system.get_surface_points_multiproc(*args)

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


def mesh_over_contact(system, component="all", symmetry_output=False, **kwargs):
    """
    Creates surface mesh of given binary star component in case of over-contact system.

    :param system:
    :param symmetry_output: bool; if true, besides surface points are returned also `symmetry_vector`,
    `base_symmetry_points_number`, `inverse_symmetry_matrix`
    :param component: str; `primary` or `secondary`
    :return: Tuple or numpy.array (if symmetry_output is False)

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
    star_container = getattr(system, component)
    discretization_factor = star_container.discretization_factor

    # calculating distance between components
    components_distance = system.orbit.orbital_motion(phase=0)[0][0]

    if component == 'primary':
        fn = system.potential_primary_fn
        fn_cylindrical = system.potential_primary_cylindrical_fn
        precalc = system.pre_calculate_for_potential_value_primary
        precal_cylindrical = system.pre_calculate_for_potential_value_primary_cylindrical
        potential_derivative_fn = system.radial_primary_potential_derivative
        cylindrical_potential_derivative_fn = system.radial_primary_potential_derivative_cylindrical
    elif component == 'secondary':
        fn = system.potential_secondary_fn
        fn_cylindrical = system.potential_secondary_cylindrical_fn
        precalc = system.pre_calculate_for_potential_value_secondary
        precal_cylindrical = system.pre_calculate_for_potential_value_secondary_cylindrical
        potential_derivative_fn = system.radial_secondary_potential_derivative
        cylindrical_potential_derivative_fn = system.radial_secondary_potential_derivative_cylindrical
    else:
        raise ValueError(f'Invalid value of `component` argument: `{component}`.\n'
                         f'Expecting `primary` or `secondary`.')

    # precalculating azimuths for farside points
    phi_farside, theta_farside, separator_farside = \
        pre_calc_azimuths_for_overcontact_farside_points(discretization_factor)

    # generating the azimuths for neck
    neck_position, neck_polynomial = system.calculate_neck_position(return_polynomial=True)
    phi_neck, z_neck, separator_neck = \
        pre_calc_azimuths_for_overcontact_neck_points(discretization_factor, neck_position, neck_polynomial,
                                                      polar_radius=star_container.polar_radius,
                                                      component=component)

    # solving points on farside
    # here implement multiprocessing
    if config.NUMBER_OF_THREADS == 1:
        args = phi_farside, theta_farside, star_container.polar_radius, \
               components_distance, precalc, fn, potential_derivative_fn, star_container.surface_potential
        __logger__.debug(f'calculating farside points of {component} component in mesh_overcontact '
                         f'function using single process method')
        points_farside = get_surface_points(*args)
    else:
        args = phi_farside, theta_farside, components_distance, precalc, fn
        __logger__.debug(f'calculating farside points of {component} component in mesh_overcontact '
                         f'function using multi process method')
        points_farside = system.get_surface_points_multiproc(*args)

    # assigning equator points and point A (the point on the tip of the farside equator)
    equator_farside = points_farside[:separator_farside[0], :]
    x_eq1, x_a = equator_farside[: -1, 0], equator_farside[-1, 0]
    y_eq1, y_a = equator_farside[: -1, 1], equator_farside[-1, 1]
    z_eq1, z_a = equator_farside[: -1, 2], equator_farside[-1, 2]

    # assigning points on phi = pi
    meridian_farside1 = points_farside[separator_farside[0]: separator_farside[1], :]
    x_meridian1, y_meridian1, z_meridian1 = \
        meridian_farside1[:, 0], meridian_farside1[:, 1], meridian_farside1[:, 2]

    # assigning points on phi = pi/2 meridian, perpendicular to component`s distance vector
    meridian_farside2 = points_farside[separator_farside[1]: separator_farside[2], :]
    x_meridian2, y_meridian2, z_meridian2 = \
        meridian_farside2[:, 0], meridian_farside2[:, 1], meridian_farside2[:, 2]

    # assigning the rest of the surface on farside
    quarter = points_farside[separator_farside[2]:, :]
    x_q1, y_q1, z_q1 = quarter[:, 0], quarter[:, 1], quarter[:, 2]

    # solving points on neck
    if config.NUMBER_OF_THREADS:
        args = phi_neck, z_neck, components_distance, star_container.polar_radius, \
               precal_cylindrical, fn_cylindrical, cylindrical_potential_derivative_fn, \
               star_container.surface_potential
        __logger__.debug(f'calculating neck points of {component} component in mesh_overcontact '
                         f'function using single process method')
        points_neck = get_surface_points_cylindrical(*args)
    else:
        args = phi_neck, z_neck, components_distance, precal_cylindrical, fn_cylindrical
        __logger__.debug(f'calculating neck points of {component} component in mesh_overcontact '
                         f'function using multi process method')
        points_neck = system.get_surface_points_multiproc_cylindrical(*args)

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

    x = np.array([x_a])
    y = np.array([y_a])
    z = np.array([z_a])
    x = up.concatenate((x, x_eq, x_q, x_meridian, x_q, x_eq, x_q, x_meridian, x_q))
    y = up.concatenate((y, y_eq, y_q, y_meridian, -y_q, -y_eq, -y_q, -y_meridian, y_q))
    z = up.concatenate((z, z_eq, z_q, z_meridian, z_q, z_eq, -z_q, -z_meridian, -z_q))

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
