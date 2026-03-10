"""Horizon estimation utilities for binary star systems."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import BinarySystem, opt, settings, u, utils
from elisa import umpy as up
from elisa.base.types import FLOAT
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.model import (
    potential_primary_fn,
    pre_calculate_for_potential_value_primary,
    radial_primary_potential_derivative,
)
from elisa.binary_system.surface.coverage import get_eclipse_boundary_path
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.const import HALF_PI, Position
from elisa.pypex import Polygon

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer

LINE_OF_SIGHT: NDArray = np.array([1, 0, 0])
BINARY_DEFINITION: dict = {
    "system": {
        "argument_of_periastron": 90.0,
        "gamma": 0.0,
        "period": 5.0,
        "eccentricity": 0.0,
        "inclination": 95.0,
        "primary_minimum_time": 0.0,
        "phase_shift": 0.0,
    },
    "primary": {
        "mass": 3.0,
        "surface_potential": 4.2,
        "synchronicity": 1.0,
        "t_eff": 6000.0,
        "gravity_darkening": 0.09,
        "albedo": 0.5,
        "metallicity": 0.0,
        "discretization_factor": 5,
    },
    "secondary": {
        "mass": 0.5,
        "surface_potential": 5.0,
        "synchronicity": 1.0,
        "t_eff": 5000.0,
        "gravity_darkening": 0.09,
        "albedo": 0.5,
        "metallicity": 0.0,
    },
}


def _horizon_base_component(
    binary: BinarySystem,
    position: Position,
    *,
    analytic: bool = True,
) -> OrbitalPositionContainer:
    """Build base component container for horizon calculation.

    Constructs a position container for the binary system and builds
    mesh, faces, and surface properties. For non-analytic mode, applies
    rotations and filters.

    :param binary: Binary system instance
    :type binary: BinarySystem
    :param position: Orbital position.
    :type position: Position
    :param analytic: If True, use analytic computation; otherwise discrete
    :type analytic: bool
    :return: Position container with primary component data
    :rtype: OrbitalPositionContainer
    """
    if analytic:
        binary.primary.discretization_factor = 1.0 * u.deg
        binary.secondary.discretization_factor = 10.0 * u.deg
        binary.init()

    position_container = OrbitalPositionContainer.from_binary_system(binary, position)
    position_container.build_mesh(components_distance=1.0, component="primary")
    position_container.build_faces(components_distance=1.0, component="primary")
    position_container.build_faces_orientation(
        components_distance=1.0,
        component="primary",
    )
    position_container.build_surface_areas(component="primary")

    if not analytic:
        # rotate
        for prop in ["points", "normals"]:
            prop_value = getattr(position_container.primary, prop)

            prop_value = utils.around_axis_rotation(
                position_container.position.azimuth - HALF_PI,
                prop_value,
                "z",
            )
            prop_value = utils.around_axis_rotation(
                HALF_PI - position_container.inclination,
                prop_value,
                "y",
            )
            setattr(position_container.primary, prop, prop_value)

        # compute los cosines
        normals = position_container.primary.normals
        los_cosines = normals[:, 0]
        position_container.primary.los_cosines = los_cosines

        # apply darkside filter (horizon)
        cosines = position_container.primary.los_cosines
        valid_indices = position_container.darkside_filter(cosines=cosines)
        position_container.primary.indices = valid_indices

    return position_container


def estimate_analytic_horizon(
    binary: BinarySystem | None = None,
    *,
    phase: float = 0.0,
    threshold: float = -1e-6,
    polar: bool = False,
    cosine_precision: bool = False,
    _3d: bool = False,
) -> NDArray | tuple[NDArray, float | int | None]:
    """Estimate analytic horizon of primary component.

    This uses very dense discretization of surface to approach real horizon
    without solving complicated equations.

    :param binary: Binary system instance; if None, default system is used
    :type binary: BinarySystem | None
    :param phase: Orbital phase at which to estimate horizon (default: 0.0)
    :type phase: float
    :param threshold: Threshold for boundary detection (default: -1e-6)
    :type threshold: float
    :param polar: If True, convert result to polar coordinates (default: False)
    :type polar: bool
    :param cosine_precision: If True, also return cosine precision value
    :type cosine_precision: bool
    :param _3d: If True, return 3D points instead of 2D projection (default: False)
    :type _3d: bool
    :return: Horizon points or (horizon, precision) tuple if cosine_precision=True
    :rtype: NDArray | tuple[NDArray, float | int | None]
    """

    def _cosine_precision(cosine: NDArray) -> float:
        """Calculate cosine precision threshold."""
        precisions = reversed(1.0 / np.power(10, list(range(1, 11))))
        for _precision in precisions:
            if np.sum(cosine > _precision) == 0:
                return float(_precision)
        return -1.0

    if binary is None:
        binary = BinarySystem.from_json(BINARY_DEFINITION)
    orbital_motion_data = binary.orbit.orbital_motion(phase=phase)[0]
    position = Position(
        idx=0,
        distance=orbital_motion_data[0],
        azimuth=orbital_motion_data[1],
        true_anomaly=orbital_motion_data[2],
        phase=orbital_motion_data[3],
    )
    container = _horizon_base_component(binary, position, analytic=True)
    star: StarContainer = container.primary
    normals = calculate_potential_gradient(
        position.distance,
        "primary",
        star.points,
        star.synchronicity,
        binary.mass_ratio,
    )
    properties: list[NDArray] = []
    for props in [star.points, normals]:
        prop_value = utils.around_axis_rotation(
            position.azimuth - HALF_PI,
            props,
            "z",
        )
        prop_value = utils.around_axis_rotation(
            HALF_PI - container.inclination,
            prop_value,
            "y",
        )
        properties.append(prop_value)

    del container

    points, normals = properties
    cosines = np.inner(normals, LINE_OF_SIGHT)
    valid_indices = OrbitalPositionContainer.darkside_filter(cosines)

    visible_projection = utils.plane_projection(points[valid_indices], "yz")
    bb_path = get_eclipse_boundary_path(visible_projection)
    horizon_indices = up.invert(
        bb_path.contains_points(visible_projection, radius=threshold),
    )

    precision: float | int | None = None
    if cosine_precision:
        precision = _cosine_precision(cosines[valid_indices][horizon_indices])

    if _3d:
        return points[valid_indices][horizon_indices]

    horizon = visible_projection[horizon_indices]

    if polar:
        horizon = utils.cartesian_to_polar(horizon)
        horizon_argsort = np.argsort(horizon.T[1])
        horizon = horizon[horizon_argsort]

    return (horizon, precision) if cosine_precision else horizon


def get_analytics_horizon(
    binary: BinarySystem | None = None,
    *,
    phase: float = 0.0,
    tol: float = 1e-4,
    polar: bool = False,
    phi_density: int = 100,
    theta_density: int = 1000,
) -> NDArray:
    """Get analytically computed horizon of primary component.

    Computes the horizon using numerical root finding to solve the potential
    equation. Rotations and coordinate transformations are applied to account
    for orbital phase and inclination.

    :param binary: Binary system instance; if None, default system is used
    :type binary: BinarySystem | None
    :param phase: Orbital phase at which to compute horizon (default: 0.0)
    :type phase: float
    :param tol: Tolerance for horizon point selection (default: 1e-4)
    :type tol: float
    :param polar: If True, convert result to polar coordinates (default: False)
    :type polar: bool
    :param phi_density: Number of points in azimuthal direction (default: 100)
    :type phi_density: int
    :param theta_density: Number of points in polar angle direction (default: 1000)
    :type theta_density: int
    :return: Horizon points in 2D (y, z) or polar (r, theta) coordinates
    :rtype: NDArray
    :raises ValueError: If no horizon points found within tolerance
    """
    if binary is None:
        binary = BinarySystem.from_json(BINARY_DEFINITION)

    star = binary.primary
    orbital_motion_data = binary.orbit.orbital_motion(phase=phase)[0]
    position = Position(
        idx=0,
        distance=orbital_motion_data[0],
        azimuth=orbital_motion_data[1],
        true_anomaly=orbital_motion_data[2],
        phase=orbital_motion_data[3],
    )

    # rotate line of sight to simulate phase and inclination
    zv: NDArray = np.array([0.0, 0.0, 1.0])

    xv: NDArray = utils.around_axis_rotation(
        HALF_PI - binary.inclination,
        LINE_OF_SIGHT,
        "y",
    )
    xv = utils.around_axis_rotation(
        position.azimuth - HALF_PI,
        xv,
        "z",
    )

    zv = utils.around_axis_rotation(
        HALF_PI - binary.inclination,
        zv,
        "y",
    )
    zv = utils.around_axis_rotation(
        position.azimuth - HALF_PI,
        zv,
        "z",
    )

    # perpendicular vector to find theta-like rotation
    yv: NDArray = np.cross(zv, xv)

    potential_fn = potential_primary_fn
    precalc_fn = pre_calculate_for_potential_value_primary
    fprime = radial_primary_potential_derivative

    # prepare-phi like vector
    phi_range: NDArray = np.linspace(np.radians(0), np.radians(360), phi_density)
    theta_range: NDArray = np.linspace(
        np.radians(-5),
        np.radians(5),
        theta_density,
    )

    # prepare theta-like vectors via rotation around phi and then around yv in -/+ range
    vectors: list[NDArray] = []
    for d_phi in phi_range:
        # first rotate zv about phi around vector `xv`
        vector: NDArray = utils.arbitrary_rotation(d_phi, xv, vector=zv)
        _yv: NDArray = utils.arbitrary_rotation(d_phi, xv, vector=yv)
        for d_theta in theta_range:
            vectors.extend(
                [utils.arbitrary_rotation(d_theta, _yv, vector=vector)],
            )

    vectors_array: NDArray = np.array(utils.cartesian_to_spherical(np.array(vectors, dtype=FLOAT)))

    phi: NDArray = vectors_array[:, 1]
    theta: NDArray = vectors_array[:, 2]
    args: tuple = (star.synchronicity, binary.mass_ratio, position.distance, phi, theta)
    precalc_vals: tuple = precalc_fn(*args, return_as_tuple=True)
    x0: NDArray = star.side_radius * np.ones(phi.shape)
    radius_kwargs: dict = {
        "fprime": fprime,
        "maxiter": settings.MAX_SOLVER_ITERS,
        "rtol": 1e-10,
        "args": (
            (binary.mass_ratio, *precalc_vals),
            star.surface_potential,
        ),
    }

    radius: NDArray = opt.newton.newton(potential_fn, x0, **radius_kwargs)
    points: NDArray = utils.spherical_to_cartesian(
        np.array([radius, phi, theta]).T,
    )
    normals = calculate_potential_gradient(
        1.0,
        "primary",
        points,
        star.synchronicity,
        binary.mass_ratio,
    )

    cosines: NDArray = np.inner(normals, xv)
    cosines = cosines.reshape(phi_density, theta_density)
    cosines_gtz: list[NDArray] = [up.arange(np.shape(row)[0])[row > 0] for row in cosines]
    # take only smallest values (but greater than zero) in each theta line
    cosines_argmin: list[int] = [np.argmin(row[gtz]) for row, gtz in zip(cosines, cosines_gtz, strict=True)]

    # find cosines in tolerance (tol)
    valid_argmin: list[bool] = [
        tol > row[gtz][argmin] for row, gtz, argmin in zip(cosines, cosines_gtz, cosines_argmin, strict=True)
    ]

    points = points.reshape(phi_density, theta_density, 3)
    horizon_points: list[NDArray] = [
        row[gtz][argmin]
        for row, gtz, argmin, valid in zip(
            points,
            cosines_gtz,
            cosines_argmin,
            valid_argmin,
            strict=True,
        )
        if valid
    ]

    if utils.is_empty(horizon_points):
        error_msg: str = f"No horizon points found in given tolerance {tol}. Decrease tolerance."
        raise ValueError(error_msg)

    horizon_points_array: NDArray = np.array(horizon_points)
    horizon_points_array = utils.around_axis_rotation(
        position.azimuth - HALF_PI,
        horizon_points_array,
        "z",
    )
    horizon_points_array = utils.around_axis_rotation(
        HALF_PI - binary.inclination,
        horizon_points_array,
        "y",
    )
    horizon_points_array = horizon_points_array.T[1:3].T

    if polar:
        horizon_points_array = utils.cartesian_to_polar(horizon_points_array)
        horizon_argsort = np.argsort(horizon_points_array.T[1])
        horizon_points_array = horizon_points_array[horizon_argsort]

    return horizon_points_array


def _cover_horizon_edges(horizon: NDArray) -> NDArray:
    """Cover horizon edges with interpolated points.

    Fills the edges of the discrete horizon polygon with interpolated
    points at small intervals (0.01 of edge length).

    :param horizon: Discrete horizon vertex points
    :type horizon: NDArray
    :return: Horizon with interpolated edge points
    :rtype: NDArray
    """
    horizon_polygon = Polygon(horizon)
    horizon_interpolated: list[NDArray] = []
    for edge in horizon_polygon.edges(as_line=True):
        parametrized = edge.parametrized()
        horizon_interpolated.extend(
            [parametrized(t) for t in np.arange(0, 1, 0.01)],
        )
    return np.array(horizon_interpolated)


def get_discrete_horizon(
    binary: BinarySystem | None = None,
    *,
    phase: float = 0.0,
    threshold: float = -1e-6,
    polar: bool = False,
) -> tuple[NDArray, NDArray]:
    """Get discrete horizon of primary component.

    Computes the horizon using surface discretization. Returns both the
    interpolated horizon (with edge points filled) and the original vertex
    horizon for comparison or further analysis.

    :param binary: Binary system instance; if None, default system is used
    :type binary: BinarySystem | None
    :param phase: Orbital phase at which to compute horizon (default: 0.0)
    :type phase: float
    :param threshold: Threshold for boundary detection (default: -1e-6)
    :type threshold: float
    :param polar: If True, convert result to polar coordinates (default: False)
    :type polar: bool
    :return: Tuple of (interpolated_horizon, original_horizon)
    :rtype: tuple[NDArray, NDArray]
    """
    if binary is None:
        binary = BinarySystem.from_json(BINARY_DEFINITION)

    orbital_motion_data = binary.orbit.orbital_motion(phase=phase)[0]
    position = Position(
        idx=0,
        distance=orbital_motion_data[0],
        azimuth=orbital_motion_data[1],
        true_anomaly=orbital_motion_data[2],
        phase=orbital_motion_data[3],
    )
    position_container = _horizon_base_component(binary, position, analytic=False)
    position_container.correct_mesh(
        component="primary",
        components_distance=position_container.position.distance,
    )
    star = position_container.primary
    visible_projection = utils.get_visible_projection(star)

    bb_path = get_eclipse_boundary_path(visible_projection)
    horizon_indices = up.invert(
        bb_path.contains_points(visible_projection, radius=threshold),
    )
    horizon = visible_projection[horizon_indices]
    origin_horizon = horizon
    horizon = _cover_horizon_edges(horizon)

    if polar:
        horizon = utils.cartesian_to_polar(horizon)
        horizon_argsort = np.argsort(horizon.T[1])

        origin_horizon = utils.cartesian_to_polar(origin_horizon)
        origin_horizon_argsort = np.argsort(origin_horizon.T[1])
        return horizon[horizon_argsort], origin_horizon[origin_horizon_argsort]

    return horizon, origin_horizon


if __name__ == "__main__":
    _phase = 0.25

    discrete_horizon, origin_discrete_horizon = get_discrete_horizon(
        phase=_phase,
        polar=False,
    )

    # show full path of discrete horizon
    # phi_argsort = np.argsort(discrete_horizon.T[1] % FULL_ARC)
    # rs, phis = discrete_horizon[phi_argsort].T[0], discrete_horizon[phi_argsort].T[1] % FULL_ARC
    # rs, phis = rs[:-1], phis[:-1]
    # plt.plot(phis % FULL_ARC, rs * 10, c="r")

    # analytic_horizon = get_analytics_horizon(phase=_phase, tol=1e-2, polar=False, phi_density=50, theta_density=1000)
    #
    # plt.scatter(analytic_horizon.T[0], analytic_horizon.T[1], c="r")
    # plt.show()

    # # show vertex path of discrete horizon
    # phi_argsort = np.argsort(origin_discrete_horizon.T[1] % FULL_ARC)
    # rs, phis = origin_discrete_horizon[phi_argsort].T[0], origin_discrete_horizon[phi_argsort].T[1] % FULL_ARC
    # rs, phis = rs[:-1], phis[:-1]
    #
    # plt.scatter(phis % FULL_ARC, rs * 10, c="r")
    #
    # # analytic horizon
    # analytic_horizon = get_analytics_horizon(phase=_phase, tol=1e-2, polar=True, phi_density=100, theta_density=1000)
    # phi_argsort = np.argsort(analytic_horizon.T[1] % FULL_ARC)
    # rs, phis = analytic_horizon[phi_argsort].T[0], analytic_horizon[phi_argsort].T[1] % FULL_ARC
    # rs, phis = rs[:-1], phis[:-1]
    #
    # plt.plot(phis % FULL_ARC, rs * 10, c="b")
    # plt.xlabel(r"$\theta$")
    # plt.ylabel(r"$\varrho$")
    # plt.legend()
    # plt.show()


