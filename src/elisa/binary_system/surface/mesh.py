import numpy as np

from .. import (
    utils as bsutils,
    model,
)
from ... base.error import MaxIterationError, SpotError
from ... base.spot import incorporate_spots_mesh
from ... base.surface.mesh import correct_component_mesh
from ... import settings
from ... opt.fsolver import fsolver
from ... utils import is_empty
from ... logger import getLogger
from ... import (
    umpy as up,
    utils,
    opt,
    const
)

logger = getLogger("binary_system.surface.mesh")

CORRECTION_FACTORS = {
    'detached': np.load(settings.PATH_TO_DETACHED_CORRECTIONS, allow_pickle=False),
    'over-contact': np.load(settings.PATH_TO_OVER_CONTACT_CORRECTIONS, allow_pickle=False)
}
CORRECTION_FACTORS['semi-detached'] = CORRECTION_FACTORS['detached']
CORRECTION_FACTORS['double-contact'] = CORRECTION_FACTORS['detached']


def build_mesh(system, components_distance, component="all"):
    """
    Build surface points for primary or/and secondary component. In case of spots,
    the spot point mesh is incorporated into the model. Points are assigned to system.

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
            a, c, d = mesh_over_contact(system, component, symmetry_output=True)
        else:
            a, c, d = mesh_detached(system, components_distance, component, symmetry_output=True)

        star.points = a
        star.base_symmetry_points_number = c
        star.inverse_point_symmetry_matrix = d

    add_spots_to_mesh(system, components_distance, component="all")

    return system


def rebuild_symmetric_detached_mesh(system, components_distance, component):
    """
    Rebuild a mesh of a symmetrical surface using old mesh to provide azimuths for the new. This conserved number
    of points and faces.

    :param system: elisa.binary_system.contaier.OrbitalPositionContainer; instance
    :param component: Union[str, None];
    :param components_distance: float;
    :return: system; elisa.binary_system.contaier.OrbitalPositionContainer; instance
    """
    components = bsutils.component_to_list(component)

    for component in components:
        star = getattr(system, component)
        setattr(star, "points", rebuild_mesh_detached(system, components_distance, component))

    return system


def pre_calc_azimuths_for_detached_points(discretization, star):
    """
    Router for discretization method of detached surfaces.

    :param discretization: float; discretization factor
    :param star:
    :return: Tuple; (phi: numpy.array, theta: numpy.array, separator: numpy.array)
    """
    if settings.MESH_GENERATOR in ['auto', 'improved_trapezoidal']:
        rel_radii = (star.forward_radius - star.polar_radius) / star.polar_radius
        if rel_radii > settings.DEFORMATION_TOL or settings.MESH_GENERATOR == 'improved_trapezoidal':
            args = discretization, star.forward_radius, star.polar_radius, star.side_radius, star.backward_radius
            return improved_trapezoidal_mesh(*args)

    return trapezoidal_mesh(discretization)


def trapezoidal_mesh(discretization):
    """
    Caculates azimuths for every surface point using trapezoidal discretization. Good for nearly spherical stars.

    :param discretization: float;
    :return: Tuple;
    """
    # vertical alpha needs to be corrected to maintain similar sizes of triangle sizes
    # vertical_alpha = const.POINT_ROW_SEPARATION_FACTOR * discretization
    vertical_alpha = discretization

    separator = []

    # azimuths for points on equator
    num = int(const.PI // (const.SEAM_CONST * discretization))
    phi = np.linspace(0., const.PI, num=num + 1)
    theta = np.full(phi.shape, const.HALF_PI)
    separator.append(np.shape(theta)[0])

    # azimuths for points on meridian
    v_num = int(const.HALF_PI // (const.SEAM_CONST * vertical_alpha))
    # v_num = int(const.HALF_PI // discretization)
    phi_meridian = np.concatenate((const.PI * np.ones(v_num - 1), np.zeros(v_num)))
    theta_meridian = up.concatenate((np.linspace(const.HALF_PI, 0, num=v_num + 1)[1:-1],
                                     np.linspace(0., const.HALF_PI, num=v_num, endpoint=False)))

    phi = up.concatenate((phi, phi_meridian))
    theta = up.concatenate((theta, theta_meridian))
    separator.append(np.shape(theta)[0])

    v_num = int(const.HALF_PI // vertical_alpha)
    # azimuths for rest of the quarter
    thetas = np.linspace(discretization, const.HALF_PI, num=v_num - 1, endpoint=False)
    for tht in thetas:
        alpha_corrected = discretization / up.sin(tht)
        num = int(const.PI // alpha_corrected)
        alpha_corrected = const.PI / (num + 1)
        phi_q_add = alpha_corrected * np.arange(1, num+1)
        phi = up.concatenate((phi, phi_q_add))
        theta = up.concatenate((theta, tht*np.ones(phi_q_add.shape[0])))

    return phi, theta, separator


def improved_trapezoidal_mesh(discretization, forward_radius, polar_radius, side_radius, backward_radius):
    """
    Calculates azimuths for every surface point using trapezoidal discretization. Function conserves areas of the
    triangles better than trapezoidal method.

    :param discretization: numpy.float;
    :param forward_radius: numpy.float;
    :param polar_radius: numpy.float;
    :param side_radius: numpy.float;
    :param backward_radius: numpy.float;
    :return: Tuple; (phi: numpy.array, theta: numpy.array, separtor: List)
    """
    # vertical alpha needs to be corrected to maintain similar sizes of triangle sizes
    # vertical_alpha = const.POINT_ROW_SEPARATION_FACTOR * discretization
    vertical_alpha = discretization

    separator = []

    # azimuths for points on equator
    num = int(const.PI // (const.SEAM_CONST * discretization))
    phi = np.linspace(0., const.PI, num=num + 1)
    theta = np.full(phi.shape, const.HALF_PI)
    separator.append(np.shape(theta)[0])
    # obliqueness correction
    inner_mask = (phi < const.HALF_PI) & (phi > 0)
    outer_mask = (phi > const.HALF_PI) & (phi < const.PI)
    inner_phis = phi[inner_mask]
    outer_phis = phi[outer_mask]
    tan_phs1 = up.tan(inner_phis)
    tan_phs2 = up.tan(outer_phis)
    inner_corr = up.arctan((side_radius - forward_radius) * tan_phs1 /
                           (side_radius + forward_radius * tan_phs1 ** 2))
    outer_corr = up.arctan((side_radius - backward_radius) * tan_phs2 /
                           (side_radius + backward_radius * tan_phs2 ** 2))
    phi[inner_mask] += inner_corr
    phi[outer_mask] += outer_corr

    # azimuths for points on meridian
    v_num = int(const.HALF_PI // (1.07*vertical_alpha))
    # v_num = int(const.HALF_PI // discretization)
    phi_meridian = np.concatenate((const.PI * np.ones(v_num - 1), np.zeros(v_num)))
    theta_meridian = up.concatenate((np.linspace(const.HALF_PI, 0, num=v_num + 1)[1:-1],
                                     np.linspace(0., const.HALF_PI, num=v_num, endpoint=False)))

    # azimuths for rest of the quarter
    v_num = int(const.HALF_PI // vertical_alpha)
    thetas_lin = np.linspace(discretization, const.HALF_PI, num=v_num - 1, endpoint=False)

    # correcting theta for obliqueness
    est_eqt_r = (side_radius + forward_radius + backward_radius) / 3.0
    tan_tht = np.tan(theta_meridian)
    theta_meridian += up.arctan((est_eqt_r - polar_radius) * tan_tht /
                                (polar_radius + est_eqt_r * tan_tht ** 2))

    phi = up.concatenate((phi, phi_meridian))
    theta = up.concatenate((theta, theta_meridian))
    separator.append(np.shape(theta)[0])

    # correcting theta for obliqueness
    tan_tht = np.tan(thetas_lin)
    thetas = thetas_lin + up.arctan((est_eqt_r - polar_radius) * tan_tht /
                                    (polar_radius + est_eqt_r * tan_tht ** 2))

    for ii, tht in enumerate(thetas):
        alpha_corrected = discretization / up.sin(tht)
        num = int(const.PI // alpha_corrected)
        alpha_corrected = const.PI / (num + 1)
        phi_q_add = alpha_corrected * np.arange(1, num+1)

        # correction for obliqness
        inner_mask = phi_q_add < const.HALF_PI
        outer_mask = phi_q_add > const.HALF_PI
        inner_phis = phi_q_add[inner_mask]
        outer_phis = phi_q_add[outer_mask]

        tan_phs1 = up.tan(inner_phis)
        tan_phs2 = up.tan(outer_phis)
        scaling_factor = np.sin(tht)
        inner_corr = np.arctan(scaling_factor * (side_radius - forward_radius) * tan_phs1 /
                               (side_radius + forward_radius * tan_phs1 ** 2))
        outer_corr = np.arctan(scaling_factor * (side_radius - backward_radius) * tan_phs2 /
                               (side_radius + backward_radius * tan_phs2 ** 2))
        phi_q_add[inner_mask] += inner_corr
        phi_q_add[outer_mask] += outer_corr

        phi = up.concatenate((phi, phi_q_add))
        thetas_add = tht * np.ones(phi_q_add.shape[0])
        theta = up.concatenate((theta, thetas_add))
    return phi, theta, separator


def pre_calc_azimuths_for_overcontact_points(discretization, star, component, neck_position, neck_polynomial):
    """
    Router for farside over-contact discretization methods.

    :param discretization: numpy.float;
    :param star: elisa.base.container.StarContainer;
    :param component: str;;
    :param neck_position: numpy.float;
    :param neck_polynomial: numpy.polynomial.Polynomial;
    :return: Tuple;
    """
    if settings.MESH_GENERATOR in ['auto', 'improved_trapezoidal']:
        rel_radii = (star.backward_radius - star.polar_radius) / star.polar_radius
        if rel_radii > settings.DEFORMATION_TOL or settings.MESH_GENERATOR == 'improved_trapezoidal':
            far_azim = improved_trapezoidal_overcontact_farside_points(discretization, star.polar_radius,
                                                                       star.side_radius, star.backward_radius)
            near_azim = improved_trapezoidal_overcontact_neck_points(discretization, neck_position, neck_polynomial,
                                                                     star.polar_radius, star.side_radius, component)
            return far_azim, near_azim

    far_azim = trapezoidal_overcontact_farside_points(discretization)
    near_azim = trapezoidal_overcontact_neck_points(discretization, neck_position, neck_polynomial, star.polar_radius,
                                                    component)
    return far_azim, near_azim


def trapezoidal_overcontact_farside_points(discretization):
    """
    Calculates azimuths (directions) to the surface points of over-contact component on its far-side (convex part).

    :param discretization: float; discretization factor
    :return: Tuple; (phi: numpy.array, theta: numpy.array, separtor: list)
    """
    # vertical alpha needs to be corrected to maintain similar sizes of triangle sizes
    # vertical_alpha = const.POINT_ROW_SEPARATION_FACTOR * discretization
    vertical_alpha = discretization
    separator = []

    # calculating points on farside equator
    num = int(const.HALF_PI // (const.SEAM_CONST * discretization))
    phi = np.linspace(const.HALF_PI, const.PI, num=num + 2)
    theta = np.full(phi.shape, const.HALF_PI)
    separator.append(np.shape(theta)[0])

    # calculating points on phi = pi meridian
    v_num = int(const.HALF_PI / (const.SEAM_CONST * vertical_alpha))
    phi_meridian1 = np.full(v_num - 1, const.PI)
    theta_meridian1 = np.linspace(0., const.HALF_PI, num=v_num - 1, endpoint=False)
    phi = up.concatenate((phi, phi_meridian1))
    theta = up.concatenate((theta, theta_meridian1))
    separator.append(np.shape(theta)[0])

    # calculating points on phi = pi/2 meridian, perpendicular to component`s distance vector
    v_num -= 1
    phi_meridian2 = np.full(v_num - 1, const.HALF_PI)
    theta_meridian2 = np.linspace(0, const.HALF_PI, num=v_num, endpoint=False)[1:]
    phi = up.concatenate((phi, phi_meridian2))
    theta = up.concatenate((theta, theta_meridian2))
    separator.append(np.shape(theta)[0])

    v_num = int(const.HALF_PI / vertical_alpha)
    theta_meridian = np.linspace(0., const.HALF_PI, num=v_num - 1, endpoint=False)
    # calculating the rest of the surface on farside
    for tht in theta_meridian[1:]:
        alpha_corrected = discretization / up.sin(tht)
        num = int(const.HALF_PI // alpha_corrected)
        alpha_corrected = const.HALF_PI / (num + 1)
        phi_q_add = const.HALF_PI + alpha_corrected * np.arange(1, num + 1)
        phi = np.concatenate((phi, phi_q_add))
        theta = np.concatenate((theta, np.full(phi_q_add.shape, tht)))
    separator.append(np.shape(theta)[0])

    return phi, theta, separator


def improved_trapezoidal_overcontact_farside_points(discretization, polar_radius, side_radius, backward_radius):
    """
    Calculates azimuths (directions) to the surface points of over-contact component on its far-side (convex part)
    using improved trapezoidal mesh approach.

    :param discretization: float; discretization factor
    :param polar_radius: numpy.float;
    :param side_radius: numpy.float;
    :param backward_radius: numpy.float;
    :return: Tuple; (phi: numpy.array, theta: numpy.array, separtor: List)
    """
    # vertical alpha needs to be corrected to maintain similar sizes of triangle sizes
    # vertical_alpha = const.POINT_ROW_SEPARATION_FACTOR * discretization
    vertical_alpha = discretization
    separator = []

    # calculating points on farside equator
    num = int(const.HALF_PI / (const.SEAM_CONST * discretization))
    phi = np.linspace(const.HALF_PI, const.PI, num=num + 2)
    theta = np.full(phi.shape, const.HALF_PI)
    separator.append(np.shape(theta)[0])
    # obliqueness correction
    tan_phs = up.tan(phi)
    corr = up.arctan((side_radius - backward_radius) * tan_phs /
                     (side_radius + backward_radius * tan_phs ** 2))
    phi += corr

    # calculating points on phi = pi meridian
    v_num = int(const.HALF_PI / (const.SEAM_CONST * vertical_alpha))
    phi_meridian1 = np.full(v_num - 1, const.PI)
    theta_meridian1 = np.linspace(0., const.HALF_PI, num=v_num - 1, endpoint=False)
    # obliqueness correction
    est_eqt_r = (side_radius + 2*backward_radius) / 3.0
    tan_tht = np.tan(theta_meridian1)
    theta_meridian1 += up.arctan((est_eqt_r - polar_radius) * tan_tht /
                                 (polar_radius + est_eqt_r * tan_tht ** 2))

    phi = up.concatenate((phi, phi_meridian1))
    theta = up.concatenate((theta, theta_meridian1))
    separator.append(np.shape(theta)[0])

    # calculating points on phi = pi/2 meridian, perpendicular to component`s distance vector
    v_num -= 1
    phi_meridian2 = np.full(v_num - 1, const.HALF_PI)
    theta_meridian2 = theta_meridian1[1:]
    phi = up.concatenate((phi, phi_meridian2))
    theta = up.concatenate((theta, theta_meridian2))
    separator.append(np.shape(theta)[0])

    v_num = int(const.HALF_PI / vertical_alpha)
    theta_meridian = np.linspace(0., const.HALF_PI, num=v_num - 1, endpoint=False)
    tan_tht = np.tan(theta_meridian)
    theta_meridian += up.arctan((est_eqt_r - polar_radius) * tan_tht /
                                (polar_radius + est_eqt_r * tan_tht ** 2))
    # calculating the rest of the surface on farside
    for tht in theta_meridian[1:]:
        alpha_corrected = discretization / up.sin(tht)
        num = int(const.HALF_PI // alpha_corrected)
        alpha_corrected = const.HALF_PI / (num + 1)
        phi_q_add = const.HALF_PI + alpha_corrected * np.arange(1, num + 1)
        # obliqueness correction
        scaling_factor = np.sin(tht)
        tan_phs = up.tan(phi_q_add)
        corr = np.arctan(scaling_factor * (side_radius - backward_radius) * tan_phs /
                         (side_radius + backward_radius * tan_phs ** 2))
        phi_q_add += corr

        phi = np.concatenate((phi, phi_q_add))
        theta = np.concatenate((theta, np.full(phi_q_add.shape, tht)))
    separator.append(np.shape(theta)[0])

    return phi, theta, separator


def _generate_neck_zs(delta_z, component, neck_position, neck_polynomial):
    """
    Common generating function for azimuths on the neck of over-contact systems.

    :param delta_z: numpy.float;
    :param component: str;
    :param neck_position: numpy.float;
    :param neck_polynomial: numpy.polynomial.Polynomial
    :return: Tuple
    """
    # lets define cylindrical coordinate system r_n, phi_n, z_n for our neck where z_n = x, phi_n = 0 heads along
    # z axis

    # alpha along cylindrical axis z needs to be corrected to maintain similar sizes of triangle sizes
    # delta_z = const.POINT_ROW_SEPARATION_FACTOR * delta_z
    delta_z = delta_z
    delta_z_polar = const.SEAM_CONST * delta_z
    # test radii on neck_position
    separator = []

    if component == 'primary':
        num = 100 * int(neck_position // delta_z)
        # position of z_n adapted to the slope of the neck, gives triangles with more similar areas
        x_curve = np.linspace(0., neck_position, num=num, endpoint=True)
        z_curve = np.polyval(neck_polynomial, x_curve)

        curve = np.column_stack((x_curve, z_curve))
        lengths = up.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
        neck_lengths = np.cumsum(lengths)
        num_z = int(neck_lengths[-1] // delta_z)
        num_z_polar = int(neck_lengths[-1] // delta_z_polar)
        segments = np.linspace(0, neck_lengths[-1], num=num_z)[1:]
        segments_polar = np.linspace(0, neck_lengths[-1], num=num_z_polar)[1:]

        z_ns = np.interp(segments, neck_lengths, x_curve[1:])
        z_ns_polar = np.interp(segments_polar, neck_lengths, x_curve[1:])
        r_neck = np.polyval(neck_polynomial, z_ns)
    else:
        num = 100 * int((1 - neck_position) // delta_z)
        # position of z_n adapted to the slope of the neck, gives triangles with more similar areas
        x_curve = np.linspace(neck_position, 1, num=num, endpoint=True)
        z_curve = np.polyval(neck_polynomial, x_curve)

        curve = np.column_stack((x_curve, z_curve))
        lengths = up.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
        neck_lengths = np.cumsum(lengths)
        num_z = int(neck_lengths[-1] // delta_z)
        num_z_polar = int(neck_lengths[-1] // delta_z_polar)
        segments = np.linspace(0, neck_lengths[-1], num=num_z)[:-1]
        segments_polar = np.linspace(0, neck_lengths[-1], num=num_z_polar)[:-1]

        z_ns = np.interp(segments, neck_lengths, x_curve[:-1])
        z_ns_polar = np.interp(segments_polar, neck_lengths, x_curve[:-1])
        r_neck = np.polyval(neck_polynomial, z_ns)
        z_ns = 1 - z_ns
        z_ns_polar = 1 - z_ns_polar

    # equator azimuths
    phi = np.full(z_ns_polar.shape, const.HALF_PI)
    z = z_ns_polar
    separator.append(np.shape(z)[0])
    # meridian azimuths
    phi = up.concatenate((phi, np.zeros(z_ns_polar.shape)))
    z = up.concatenate((z, z_ns_polar))
    separator.append(np.shape(z)[0])

    return phi, z, z_ns, r_neck, separator


def trapezoidal_overcontact_neck_points(
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
    delta_z = discretization * polar_radius

    phi, z, z_ns, r_neck, separator = _generate_neck_zs(delta_z, component, neck_position, neck_polynomial)

    for ii, zz in enumerate(z_ns):
        num = int(const.HALF_PI * r_neck[ii] // delta_z)
        num = num + 1 if num < 5 else num
        phis = np.linspace(0, const.HALF_PI, num=int(num), endpoint=False)[1:]
        z_n = np.full(phis.shape, zz)

        phi = np.concatenate((phi, phis))
        z = up.concatenate((z, z_n))

    separator.append(np.shape(z)[0])

    return phi, z, separator


def improved_trapezoidal_overcontact_neck_points(
        discretization, neck_position, neck_polynomial, polar_radius, side_radius, component):
    """
    Calculates azimuths (directions) to the surface points of over-contact component on neck.

    :param discretization: float; doscretization factor
    :param neck_position: float; x position of neck of over-contact binary
    :param neck_polynomial: numpy.polynomial.Polynomial; polynome that define neck profile in plane `xz`
    :param polar_radius: float;
    :param side_radius: float;
    :param component: str; `primary` or `secondary`
    :return: Tuple; (phi: numpy.array, z: numpy.array, separator: numpy.array)
    """
    # generating the neck
    delta_z = discretization * polar_radius

    phi, z, z_ns, r_neck, separator = _generate_neck_zs(delta_z, component, neck_position, neck_polynomial)

    eq_coeff = side_radius / polar_radius
    for ii, zz in enumerate(z_ns):
        num = const.HALF_PI * r_neck[ii] // delta_z
        num = num + 1 if num < 4 else num
        phis = np.linspace(0, const.HALF_PI, num=int(num), endpoint=False)[1:]
        # obliqueness correction
        tan_phis = np.tan(phis)
        phis += up.arctan((eq_coeff - 1) * tan_phis /
                          (1 + eq_coeff * tan_phis ** 2))

        z_n = np.full(phis.shape, zz)

        phi = np.concatenate((phi, phis))
        z = up.concatenate((z, z_n))

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
    x0 = np.full(phi.shape, x0)
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
         numpy.float - number of points included in symmetrical one quarter of surface,
         numpy.array([quadrant[indexes_of_remapped_points_in_quadrant]) - matrix of four sub matrices that
                                                                          map basic symmetry quadrant points to all
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
    phi, theta, separator = pre_calc_azimuths_for_detached_points(discretization_factor, star)
    setattr(star, "azimuth_args", (phi, theta, separator))
    # calculating mesh in cartesian coordinates for quarter of the star, the forward point nearest to the L1 is ommitted
    # it was found that in rare instances, newton performs badly near L1
    args = phi[1:], theta[1:], star.side_radius, components_distance, precalc_fn, \
        potential_fn, fprime, potential, mass_ratio, synchronicity

    logger.debug(f'calculating surface points of {component} component in mesh_detached '
                 f'function using single process method')
    points_q = get_surface_points(*args)
    # inserting forward surface point from pre-calculated forward radius
    points_q = np.insert(points_q, 0, [star.forward_radius, 0.0, 0.0], axis=0)
    points = stitch_quarters_in_detached(points_q, separator, component, components_distance)

    if symmetry_output:
        equator_length = separator[0] - 2
        meridian_length = separator[1] - separator[0]
        quarter_length = np.shape(points_q)[0] - separator[1]
        base_symmetry_points_number = 2 + equator_length + quarter_length + meridian_length

        points_length = np.shape(points)[0]
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

        return points, base_symmetry_points_number, inverse_symmetry_matrix
    else:
        return points


def rebuild_mesh_detached(system, components_distance, component):
    """
    Rebuilds symmetrical surface mesh of given binary star component in case of detached or semi-detached system.

    :param system: elisa.binary_system.contaienr.OrbitalPositionContainer;
    :param component: str; `primary` or `secondary`
    :param components_distance: numpy.float
    :return: Union[Tuple, numpy.array]; (if `symmetry_output` is False)

    Array of surface points if symmetry_output = False::

         numpy.array([[x1 y1 z1],
                      [x2 y2 z2],
                       ...
                      [xN yN zN]])

    """
    star = getattr(system, component)
    synchronicity = star.synchronicity
    mass_ratio = system.mass_ratio
    potential = star.surface_potential

    if is_empty(star.points):
        raise RuntimeError('This function can be used only on container with already built mesh')
    if star.base_symmetry_points_number == 0:
        raise RuntimeError('This function can be used only on symmetrical meshes')

    potential_fn = getattr(model, f"potential_{component}_fn")
    precalc_fn = getattr(model, f"pre_calculate_for_potential_value_{component}")
    fprime = getattr(model, f"radial_{component}_potential_derivative")

    phi, theta, separator = star.azimuth_args

    args = phi, theta, star.side_radius, components_distance, precalc_fn, \
           potential_fn, fprime, potential, mass_ratio, synchronicity

    logger.debug(f're calculating surface points of {component} component in rebuild_mesh_detached ')
    points_q = np.round(get_surface_points(*args), 15)
    return stitch_quarters_in_detached(points_q, separator, component, components_distance)


def stitch_quarters_in_detached(points_q, separator, component, components_distance):
    """
    Stitching together surface from symmetrical quarter

    :param points_q: numpy.array; point on the symmetrical quarter including equator and meridian
    :param separator: numpy.array; indices that separates equatorial meridian and inside points
    :param component: str;
    :param components_distance: numpy.float;
    :return: numpy.array; stitched surface
    """
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
    return np.column_stack((x, y, z))


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

             numpy.float - number of points included in symmetrical one quarter of surface,
             numpy.array([quadrant[indexes_of_remapped_points_in_quadrant]) - matrix of four sub matrices that
                                                                              map base symmetry quadrant to all others
                                                                              quadrants
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

    # precalculating azimuths for farside points and nearside points
    neck_position, neck_polynomial = calculate_neck_position(system, return_polynomial=True)
    (phi_farside, theta_farside, separator_farside), (phi_neck, z_neck, separator_neck) = \
        pre_calc_azimuths_for_overcontact_points(discretization_factor, star, component, neck_position, neck_polynomial)

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
    args = phi_neck, z_neck, components_distance, 0.25 * star.polar_radius, \
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
        base_symmetry_points_number = 1 + equator_length + quarter_length + meridian_length

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

        return points, base_symmetry_points_number, inverse_symmetry_matrix
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
                if not settings.SUPPRESS_WARNINGS:
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
                if not settings.SUPPRESS_WARNINGS:
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
    n_points = int(100 * np.radians(5) / system.primary.discretization_factor)
    degree = 15
    components_distance = 1.0

    # solving for neck points
    star = system.primary
    precal_cylindrical = getattr(model, "pre_calculate_for_potential_value_primary_cylindrical")
    fn_cylindrical = getattr(model, "potential_primary_cylindrical_fn")
    cylindrical_fprime = getattr(model, "radial_primary_potential_derivative_cylindrical")

    phi = np.zeros(n_points)
    z = np.linspace(0, 1, num=n_points)
    args = \
        phi, z, components_distance, 0.5 * star.polar_radius, precal_cylindrical, fn_cylindrical, cylindrical_fprime, \
        star.surface_potential, system.mass_ratio, 1.0

    points_neck = get_surface_points_cylindrical(*args)
    x = np.abs(points_neck[:, 2])
    r_c = np.abs(points_neck[:, 0])

    # fiting polynomial to the neck points and searching for neck
    polynomial_fit = np.polyfit(x, r_c, deg=degree)
    polynomial_fit_differentiation = np.polyder(polynomial_fit)
    roots = np.roots(polynomial_fit_differentiation)

    # discarding imaginary solutions
    roots = np.real(roots[np.imag(roots) == 0])

    # choosing root that is closest to the middle of the system, should work...
    # idea is to rule out roots near 0 or 1
    dist_to_cntr = np.abs(roots - 0.5)
    neck_position = roots[np.argmin(dist_to_cntr)]

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


def correct_mesh(system, components_distance=None, component='all'):
    """
    Correcting the underestimation of the surface due to the discretization.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param components_distance: float; distance of components in SMA units
    :param component: str;
    :return: elisa.binary_system.container.OrbitalPositionContainer;
    """
    components = bsutils.component_to_list(component)

    com = {'primary': 0, 'secondary': components_distance}
    for component in components:
        star = getattr(system, component)
        correct_component_mesh(star, com=com[component], correction_factors=CORRECTION_FACTORS[system.morphology])

    return system
