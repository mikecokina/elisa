from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import elisa.umpy as up
from elisa import const, opt, settings, utils
from elisa.base.error import MaxIterationError
from elisa.base.spot import incorporate_spots_mesh
from elisa.base.surface.mesh import correct_component_mesh
from elisa.base.types import FLOAT, INT
from elisa.logger import getLogger
from elisa.single_system import model
from elisa.single_system.radius import calculate_radius

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from elisa.single_system.container import SinglePositionContainer
    from elisa.types import Float, Int


logger = getLogger("single_system.surface.mesh")

CORRECTION_FACTORS = np.load(settings.PATH_TO_SINGLE_CORRECTIONS, allow_pickle=False)


def build_mesh(system: SinglePositionContainer) -> SinglePositionContainer:
    """Build the stellar surface point mesh including spots.

    :param system: Single star position container instance.
    :type system: SinglePositionContainer
    :returns: Updated single star position container instance.
    :rtype: SinglePositionContainer
    """
    points, base_symmetry_points_number, inverse_symmetry_matrix = mesh(
        system_container=system,
        symmetry_output=True,
    )

    system.star.points = points
    system.star.base_symmetry_points_number = base_symmetry_points_number
    system.star.inverse_point_symmetry_matrix = inverse_symmetry_matrix

    add_spots_to_mesh(system)
    return system


def mesh(
    system_container: SinglePositionContainer,
    *,
    symmetry_output: bool = False,
) -> NDArray[Float] | tuple[NDArray[Float], Int, NDArray[np.int_]]:
    """Create a surface mesh for a single star system.

    If ``symmetry_output`` is ``False``, the function returns an array of
    Cartesian surface points with shape ``(N, 3)``.

    If ``symmetry_output`` is ``True``, the function returns:

    - the array of Cartesian surface points with shape ``(N, 3)``,
    - the number of points included in the base symmetry region,
    - the inverse symmetry matrix mapping the base symmetry octant to all
      remaining octants.

    :param system_container: Single star system container.
    :type system_container: SinglePositionContainer
    :param symmetry_output: Whether to also return symmetry metadata.
    :type symmetry_output: bool
    :returns: Surface mesh points, optionally with symmetry metadata.
    :rtype: numpy.typing.NDArray[numpy.float64] |
            tuple[numpy.typing.NDArray[numpy.float64], Int, numpy.typing.NDArray[numpy.int_]]
    :raises ValueError: If the discretization factor is greater than
        ``const.HALF_PI``.
    """
    star_container = system_container.star
    discretization_factor = star_container.discretization_factor
    if discretization_factor > const.HALF_PI:
        message = "Invalid value of alpha parameter. Use value less than 90."
        raise ValueError(message)

    potential_fn = model.potential_fn
    precalc_fn = model.pre_calculate_for_potential_value
    potential_derivative_fn = model.radial_potential_derivative

    characteristic_distance = discretization_factor * star_container.polar_radius

    # calculating equatorial part
    equator_points = calculate_equator_points(
        characteristic_distance,
        star_container.equatorial_radius,
    )
    x_eq = equator_points[:, 0]
    y_eq = equator_points[:, 1]
    z_eq = equator_points[:, 2]

    # axial symmetry, therefore calculating latitudes
    thetas = pre_calc_latitudes(
        discretization_factor,
        star_container.polar_radius,
        star_container.equatorial_radius,
    )
    thetas_meridian = pre_calc_latitudes(
        const.SEAM_CONST * discretization_factor,
        star_container.polar_radius,
        star_container.equatorial_radius,
    )

    x0 = 0.5 * (star_container.equatorial_radius + star_container.polar_radius)
    args = (
        thetas,
        x0,
        precalc_fn,
        potential_fn,
        potential_derivative_fn,
        star_container.surface_potential,
        star_container.mass,
        system_container.angular_velocity,
    )
    args_meridian = (
        thetas_meridian,
        x0,
        precalc_fn,
        potential_fn,
        potential_derivative_fn,
        star_container.surface_potential,
        star_container.mass,
        system_container.angular_velocity,
    )

    radius = get_surface_points_radii(*args)
    radius_meridian = get_surface_points_radii(*args_meridian)

    # converting this eighth of surface to cartesian coordinates
    quarter_points = calculate_points_on_quarter_surface(
        radius,
        thetas,
        characteristic_distance,
    )
    x_q = quarter_points[:, 0]
    y_q = quarter_points[:, 1]
    z_q = quarter_points[:, 2]

    meridian_points = calculate_points_on_meridian(radius_meridian, thetas_meridian)
    x_mer = meridian_points[:, 0]
    y_mer = meridian_points[:, 1]
    z_mer = meridian_points[:, 2]

    x = np.concatenate(
        (
            np.array([0.0]),
            x_mer,
            x_eq,
            x_q,
            -y_mer,
            -y_eq,
            -y_q,
            -x_mer,
            -x_eq,
            -x_q,
            y_mer,
            y_eq,
            y_q,
            np.array([0.0]),
            x_mer,
            x_q,
            -y_mer,
            -y_q,
            -x_mer,
            -x_q,
            y_mer,
            y_q,
        ),
    )
    y = np.concatenate(
        (
            np.array([0.0]),
            y_mer,
            y_eq,
            y_q,
            x_mer,
            x_eq,
            x_q,
            -y_mer,
            -y_eq,
            -y_q,
            -x_mer,
            -x_eq,
            -x_q,
            np.array([0.0]),
            y_mer,
            y_q,
            x_mer,
            x_q,
            -y_mer,
            -y_q,
            -x_mer,
            -x_q,
        ),
    )
    z = np.concatenate(
        (
            np.array([star_container.polar_radius]),
            z_mer,
            z_eq,
            z_q,
            z_mer,
            z_eq,
            z_q,
            z_mer,
            z_eq,
            z_q,
            z_mer,
            z_eq,
            z_q,
            np.array([-star_container.polar_radius]),
            -z_mer,
            -z_q,
            -z_mer,
            -z_q,
            -z_mer,
            -z_q,
            -z_mer,
            -z_q,
        ),
    )

    points = np.column_stack((x, y, z))

    if symmetry_output:
        quarter_equator_length = len(x_eq)
        meridian_length = len(x_mer)
        quarter_length = len(x_q)
        base_symmetry_points_number = 1 + meridian_length + quarter_equator_length + quarter_length + meridian_length

        south_pole_index = 4 * (base_symmetry_points_number - meridian_length) - 3
        reduced_bspn = base_symmetry_points_number - meridian_length  # auxiliary variable1
        reduced_bspn2 = base_symmetry_points_number - quarter_equator_length
        inverse_symmetry_matrix = np.array(
            [
                np.arange(base_symmetry_points_number + 1),  # 1st quadrant (north hem)
                # 2nd quadrant (north hem)
                np.concatenate(
                    (
                        [0],
                        np.arange(
                            reduced_bspn,
                            2 * base_symmetry_points_number - meridian_length,
                        ),
                    ),
                ),
                # 3rd quadrant (north hem)
                np.concatenate(
                    (
                        [0],
                        np.arange(
                            2 * reduced_bspn - 1,
                            3 * reduced_bspn + meridian_length - 1,
                        ),
                    ),
                ),
                # 4th quadrant (north hem)
                np.concatenate(
                    (
                        [0],
                        np.arange(3 * reduced_bspn - 2, 4 * reduced_bspn - 3),
                        np.arange(1, meridian_length + 2),
                    ),
                ),
                # 1st quadrant (south hemisphere)
                np.concatenate(
                    (
                        np.arange(
                            south_pole_index,
                            meridian_length + 1 + south_pole_index,
                        ),
                        np.arange(
                            1 + meridian_length,
                            1 + meridian_length + quarter_equator_length,
                        ),
                        np.arange(
                            meridian_length + 1 + south_pole_index,
                            base_symmetry_points_number - quarter_equator_length + south_pole_index,
                        ),
                        [base_symmetry_points_number],
                    ),
                ),
                # 2nd quadrant (south hem)
                np.concatenate(
                    (
                        [south_pole_index],
                        np.arange(
                            reduced_bspn2 - meridian_length + south_pole_index,
                            reduced_bspn2 + south_pole_index,
                        ),
                        np.arange(
                            base_symmetry_points_number,
                            base_symmetry_points_number + quarter_equator_length,
                        ),
                        np.arange(
                            reduced_bspn2 + south_pole_index,
                            2 * reduced_bspn2 - meridian_length - 1 + south_pole_index,
                        ),
                        [2 * base_symmetry_points_number - meridian_length - 1],
                    ),
                ),
                # 3rd quadrant (south hem)
                np.concatenate(
                    (
                        [south_pole_index],
                        np.arange(
                            2 * reduced_bspn2 - 2 * meridian_length - 1 + south_pole_index,
                            2 * reduced_bspn2 - meridian_length - 1 + south_pole_index,
                        ),
                        np.arange(
                            2 * base_symmetry_points_number - meridian_length - 1,
                            2 * base_symmetry_points_number - meridian_length + quarter_equator_length - 1,
                        ),
                        np.arange(
                            2 * reduced_bspn2 - meridian_length - 1 + south_pole_index,
                            3 * reduced_bspn2 - 2 * meridian_length - 2 + south_pole_index,
                        ),
                        [3 * reduced_bspn + meridian_length - 2],
                    ),
                ),
                # 4th quadrant (south hem)
                np.concatenate(
                    (
                        [south_pole_index],
                        np.arange(
                            3 * reduced_bspn2 - 3 * meridian_length - 2 + south_pole_index,
                            3 * reduced_bspn2 - 2 * meridian_length - 2 + south_pole_index,
                        ),
                        np.arange(
                            3 * reduced_bspn + meridian_length - 2,
                            3 * reduced_bspn + meridian_length - 2 + quarter_equator_length,
                        ),
                        np.arange(
                            3 * reduced_bspn2 - 2 * meridian_length - 2 + south_pole_index,
                            len(x),
                        ),
                        np.arange(
                            1 + south_pole_index,
                            meridian_length + south_pole_index + 1,
                        ),
                        [1 + meridian_length],
                    ),
                ),
            ],
        )

        return points, INT(base_symmetry_points_number + 1), inverse_symmetry_matrix

    return points


def pre_calc_latitudes(
    alpha: Float,
    polar_radius: Float,
    equatorial_radius: Float,
) -> NDArray[Float]:
    """Pre-calculate stellar surface latitudes excluding the pole and equator.

    :param alpha: Angular distance of neighbouring points.
    :type alpha: Float
    :param polar_radius: Polar radius of the star.
    :type polar_radius: Float
    :param equatorial_radius: Equatorial radius of the star.
    :type equatorial_radius: Float
    :returns: Latitudes for mesh construction.
    :rtype: numpy.typing.NDArray[numpy.float64]
    """
    # alpha_corr = const.POINT_ROW_SEPARATION_FACTOR * alpha
    alpha_corr = alpha
    num = int(const.HALF_PI // alpha_corr)
    thetas = np.linspace(0.0, const.HALF_PI, num=num, endpoint=True)[1:-1]

    # solving non-uniform sampling along theta coordinates for squashed stars
    auto_test = (
        settings.MESH_GENERATOR == "auto"
        and (equatorial_radius - polar_radius) / polar_radius > settings.DEFORMATION_TOL
    )
    if auto_test or settings.MESH_GENERATOR == "improved_trapezoidal":
        thetas += up.arctan(
            (equatorial_radius - polar_radius)
            * up.tan(thetas)
            / (polar_radius + equatorial_radius * up.tan(thetas) ** 2),
        )
    return thetas


def get_surface_points_radii(
    theta: NDArray[Float],
    x0: Float,
    precalc_fn: Callable[..., tuple[object, ...]],
    potential_fn: Callable[..., NDArray[Float] | Float],
    potential_derivative_fn: Callable[..., NDArray[Float] | Float],
    surface_potential: Float,
    mass: Float,
    angular_velocity: Float,
) -> NDArray[Float]:
    """Solve radii for the given latitudes.

    The function being solved is specified by the supplied potential callback
    and its derivative callback.

    :param theta: Latitudes for which the radius should be computed.
    :type theta: numpy.typing.NDArray[numpy.float64]
    :param x0: Initial guess for the Newton solver.
    :type x0: Float
    :param precalc_fn: Function used to pre-calculate values for the potential.
    :type precalc_fn: collections.abc.Callable[..., tuple[object, ...]]
    :param potential_fn: Potential function passed to the Newton solver.
    :type potential_fn: collections.abc.Callable[..., numpy.typing.NDArray[numpy.float64] | Float]
    :param potential_derivative_fn: Derivative of the potential function.
    :type potential_derivative_fn: collections.abc.Callable[..., numpy.typing.NDArray[numpy.float64] | Float]
    :param surface_potential: Surface potential value.
    :type surface_potential: Float
    :param mass: Stellar mass.
    :type mass: Float
    :param angular_velocity: Angular velocity of the system.
    :type angular_velocity: Float
    :returns: Solved radii for all supplied latitudes.
    :rtype: numpy.typing.NDArray[numpy.float64]
    """
    precalc_vals = precalc_fn(
        *(mass, angular_velocity, theta),
        return_as_tuple=True,
    )
    initial_guess = x0 * np.ones(theta.shape)
    return opt.newton.newton(
        potential_fn,
        initial_guess,
        fprime=potential_derivative_fn,
        maxiter=settings.MAX_SOLVER_ITERS,
        args=(precalc_vals, surface_potential),
        rtol=1e-10,
    )


def calculate_points_on_quarter_surface(
    radius: NDArray[Float],
    thetas: NDArray[Float],
    characteristic_distance: Float,
) -> NDArray[Float]:
    """Return Cartesian coordinates for points on the quarter of the surface.

    :param radius: Radii corresponding to ``thetas``.
    :type radius: numpy.typing.NDArray[numpy.float64]
    :param thetas: Latitude values.
    :type thetas: numpy.typing.NDArray[numpy.float64]
    :param characteristic_distance: Mean distance between points.
    :type characteristic_distance: Float
    :returns: Array of shape ``(N, 3)`` with ``x, y, z`` coordinates.
    :rtype: numpy.typing.NDArray[numpy.float64]
    """
    r_q: list[NDArray[np.float64]] = []
    phi_q: list[NDArray[np.float64]] = []
    theta_q: list[NDArray[np.float64]] = []

    for ii, theta in enumerate(thetas):
        num = int(const.HALF_PI * radius[ii] * np.sin(theta) / characteristic_distance)
        alpha = const.HALF_PI / num
        num -= 1 if num > 0 else num
        r_q.append(radius[ii] * np.ones(num))
        theta_q.append(theta * np.ones(num))
        phi_q.append(np.linspace(alpha, const.HALF_PI - alpha, num=num, endpoint=True))

    r_q_array = np.concatenate(r_q)
    theta_q_array = np.concatenate(theta_q)
    phi_q_array = np.concatenate(phi_q)
    return utils.spherical_to_cartesian(
        np.column_stack((r_q_array, phi_q_array, theta_q_array)),
    )


def calculate_points_on_meridian(
    radius: NDArray[Float],
    thetas: NDArray[Float],
) -> NDArray[Float]:
    """Return Cartesian coordinates for points on the surface meridian.

    :param radius: Radii corresponding to ``thetas``.
    :type radius: numpy.typing.NDArray[numpy.float64]
    :param thetas: Latitude values.
    :type thetas: numpy.typing.NDArray[numpy.float64]
    :returns: Array of shape ``(N, 3)`` with ``x, y, z`` coordinates.
    :rtype: numpy.typing.NDArray[numpy.float64]
    """
    phi = 0.0 * np.ones(radius.shape)
    return utils.spherical_to_cartesian(np.column_stack((radius, phi, thetas)))


def calculate_equator_points(
    characteristic_distance: Float,
    equatorial_radius: Float,
) -> NDArray[Float]:
    """Return Cartesian coordinates for points on the equator.

    :param characteristic_distance: Characteristic spacing used to determine
        the number of equatorial points on the quarter arc.
    :type characteristic_distance: Float
    :param equatorial_radius: Equatorial radius of the star.
    :type equatorial_radius: Float
    :returns: Array of shape ``(N, 3)`` with ``x, y, z`` coordinates.
    :rtype: numpy.typing.NDArray[numpy.float64]
    """
    num = int(
        const.HALF_PI * equatorial_radius / (const.SEAM_CONST * characteristic_distance),
    )
    radii = equatorial_radius * np.ones(num)
    thetas = const.HALF_PI * np.ones(num)
    phis = np.linspace(0.0, const.HALF_PI, num=num, endpoint=False)
    return utils.spherical_to_cartesian(np.column_stack((radii, phis, thetas)))


def mesh_spots(system_container: SinglePositionContainer) -> None:  # noqa: PLR0915
    """Compute points for each spot and assign them to the spot container.

    :param system_container: Single star system container.
    :type system_container: SinglePositionContainer
    :returns: ``None``
    :rtype: None
    """
    logger.info("evaluating spots")
    star_container = system_container.star
    if not star_container.has_spots():
        logger.info("no spots to evaluate")
        return

    potential_fn = model.potential_fn
    precalc_fn = model.pre_calculate_for_potential_value
    potential_derivative_fn = model.radial_potential_derivative

    # iterate over spots
    for spot_index, spot_instance in list(star_container.spots.items()):
        # lon -> phi, lat -> theta
        lon = spot_instance.longitude
        lat = spot_instance.latitude

        alpha = spot_instance.discretization_factor
        spot_radius = spot_instance.angular_radius

        # initial containers for current spot
        boundary_points: list[NDArray[np.float64]] = []
        spot_points: NDArray[Float]

        # initial radial vector
        radial_vector = np.array(
            [1.0, lon, lat],
        )  # unit radial vector to the center of current spot
        center_vector = utils.spherical_to_cartesian(np.asarray([1.0, lon, lat], dtype=FLOAT))

        args = (radial_vector[2],)

        solution = calculate_radius(
            star_container.mass,
            system_container.angular_velocity,
            star_container.surface_potential,
            *args,
        )

        if solution > star_container.equatorial_radius or solution < star_container.polar_radius:
            # in case of spots, each point should be usefull, otherwise remove spot from
            # component spot list and skip current spot computation
            logger.info(
                "center of spot %s doesn't satisfy reasonable conditions and entire spot will be omitted",
                spot_instance.kwargs_serializer(),
            )

            star_container.remove_spot(spot_index=spot_index)
            continue

        spot_center_r = solution
        spot_center = utils.spherical_to_cartesian(np.asarray([spot_center_r, lon, lat], dtype=FLOAT))

        # compute Euclidean distance of two points on spot (x0)
        # we have to obtain distance between center and 1st point in 1st ring of spot
        args = (lat + alpha,)
        solution = calculate_radius(
            star_container.mass,
            system_container.angular_velocity,
            star_container.surface_potential,
            *args,
        )

        if solution > star_container.equatorial_radius or solution < star_container.polar_radius:
            # in case of spots, each point should be usefull, otherwise remove spot from
            # component spot list and skip current spot computation
            logger.info(
                "first ring of spot %s doesn't satisfy reasonable conditions and entire spot will be omitted",
                spot_instance.kwargs_serializer(),
            )
            star_container.remove_spot(spot_index=spot_index)
            continue

        x0 = np.sqrt(
            spot_center_r**2 + solution**2 - (2.0 * spot_center_r * solution * np.cos(alpha)),
        )

        # number of points in latitudal direction
        num_radial = int(np.round(spot_radius / alpha)) + 1
        thetas = np.linspace(lat, lat + spot_radius, num=num_radial, endpoint=True)

        num_azimuthal = [1 if i == 0 else int(i * 2.0 * np.pi * x0 // x0) for i in range(len(thetas))]
        deltas = [np.linspace(0.0, const.FULL_ARC, num=num, endpoint=False) for num in num_azimuthal]

        spot_phi: list[Float] = []
        spot_theta: list[Float] = []
        for theta_index, theta in enumerate(thetas):
            # first point of n-th ring of spot (counting start from center)
            default_spherical_vector = np.asarray([1.0, lon % const.FULL_ARC, theta], dtype=FLOAT)

            for _delta_index, delta in enumerate(deltas[theta_index]):
                # rotating default spherical vector around spot center vector and thus generating concentric
                # circle of points around centre of spot
                delta_vector = utils.arbitrary_rotation(
                    theta=delta,
                    omega=center_vector,
                    vector=utils.spherical_to_cartesian(default_spherical_vector),
                    degrees=False,
                    omega_normalized=True,
                )

                spherical_delta_vector = utils.cartesian_to_spherical(delta_vector)

                spot_phi.append(spherical_delta_vector[1])
                spot_theta.append(spherical_delta_vector[2])

        spot_phi_array = np.asarray(spot_phi, dtype=FLOAT)
        spot_theta_array = np.asarray(spot_theta, dtype=FLOAT)
        args = (
            spot_theta_array,
            spot_center_r,
            precalc_fn,
            potential_fn,
            potential_derivative_fn,
            star_container.surface_potential,
            star_container.mass,
            system_container.angular_velocity,
        )
        try:
            spot_points_radii = get_surface_points_radii(*args)
        except MaxIterationError:
            if not settings.SUPPRESS_WARNINGS:
                logger.warning(
                    "at least 1 point of spot %s doesn't satisfy reasonable conditions and entire spot will be omitted",
                    spot_instance.kwargs_serializer(),
                )
            star_container.remove_spot(spot_index=spot_index)
            continue

        spherical_points_: NDArray[Float] = np.column_stack(
            (spot_points_radii, spot_phi_array, spot_theta_array),
        ).astype(FLOAT)
        spot_points = utils.spherical_to_cartesian(spherical_points_).astype(FLOAT)

        spot_instance.points = np.array(spot_points)
        spot_instance.boundary = np.array(boundary_points)
        spot_instance.boundary_center = spot_points[0]
        spot_instance.center = np.array(spot_center)


def add_spots_to_mesh(system: SinglePositionContainer) -> None:
    """Incorporate spot points into the clean stellar mesh.

    The function also removes stellar points and spot points hidden under a
    given spot if overlapping spots are present.

    :param system: Single star system container.
    :type system: SinglePositionContainer
    :returns: ``None``
    :rtype: None
    """
    mesh_spots(system)
    incorporate_spots_mesh(system.star, component_com=0.0)


def correct_mesh(system: SinglePositionContainer) -> SinglePositionContainer:
    """Correct the surface underestimation caused by discretization.

    :param system: Single star system container.
    :type system: SinglePositionContainer
    :returns: Corrected system container.
    :rtype: SinglePositionContainer
    """
    star = system.star
    correct_component_mesh(star, com=0.0, correction_factors=CORRECTION_FACTORS)

    return system
