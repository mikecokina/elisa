import numpy as np

from elisa import u, settings, opt, BinarySystem
from elisa import umpy as up
from elisa.binary_system.model import (
    pre_calculate_for_potential_value_primary,
    radial_primary_potential_derivative,
    potential_primary_fn)
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.const import Position, HALF_PI, FULL_ARC
from elisa import utils
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.surface.coverage import get_eclipse_boundary_path
from pypex import Polygon


LINE_OF_SIGHT = np.array([1, 0, 0])
BINARY_DEFINITION = {
    "system": {
        "argument_of_periastron": 90.0,
        "gamma": 0.0,
        "period": 5.0,
        "eccentricity": 0.0,
        "inclination": 95.0,
        "primary_minimum_time": 0.0,
        "phase_shift": 0.0
    },
    "primary": {
        "mass": 3.0,
        "surface_potential": 4.2,
        "synchronicity": 1.0,
        "t_eff": 6000.0,
        "gravity_darkening": 0.09,
        "albedo": 0.5,
        "metallicity": 0.0,
        "discretization_factor": 5
    },
    "secondary": {
        "mass": 0.5,
        "surface_potential": 5.0,
        "synchronicity": 1.0,
        "t_eff": 5000.0,
        "gravity_darkening": 0.09,
        "albedo": 0.5,
        "metallicity": 0.0
    }
}


def _horizon_base_component(binary, position, analytic=True):
    if analytic:
        binary.primary.discretization_factor = 1.0 * u.deg
        binary.secondary.discretization_factor = 10.0 * u.deg
        binary.init()

    position_container = OrbitalPositionContainer.from_binary_system(binary, position)
    position_container.build_mesh(components_distance=1.0, component="primary")
    position_container.build_faces(components_distance=1.0, component="primary")
    position_container.build_faces_orientation(components_distance=1.0, component="primary")
    position_container.build_surface_areas(component="primary")

    if not analytic:
        # rotate
        for prop in ["points", "normals"]:
            prop_value = getattr(position_container.primary, prop)

            args = (position_container.position.azimuth - HALF_PI, prop_value, "z", False, False)
            prop_value = utils.around_axis_rotation(*args)

            args = (HALF_PI - position_container.inclination, prop_value, "y", False, False)
            prop_value = utils.around_axis_rotation(*args)
            setattr(position_container.primary, prop, prop_value)

        # compute los cosines
        normals = getattr(position_container.primary, "normals")
        los_cosines = normals[:, 0]
        setattr(position_container.primary, "los_cosines", los_cosines)

        # apply darkside filter (horizon)
        cosines = getattr(position_container.primary, "los_cosines")
        valid_indices = position_container.darkside_filter(cosines=cosines)
        setattr(position_container.primary, "indices", valid_indices)

    return position_container


def estimate_analytic_horizon(binary=None, phase=0.0, threshold=-1e-6, polar=False, cosine_precision=False, _3d=False):
    """
    Estimate analytic horizon of primary component.
    This use very dense discretization of surface to approach real horizon
    without solving complicated equations.
    """

    def _cosine_precision(cosine):
        precisions = reversed(1.0 / np.power(10, [x for x in range(1, 11)]))
        for _precision in precisions:
            if np.sum(cosine > _precision) == 0:
                return _precision
        return -1

    if binary is None:
        binary = BinarySystem.from_json(BINARY_DEFINITION)
    position = Position(*((0,) + tuple(binary.orbit.orbital_motion(phase=phase)[0])))
    container = _horizon_base_component(binary, position, analytic=True)
    star = container.primary
    normals = calculate_potential_gradient(position.distance, "primary", star.points,
                                           star.synchronicity, binary.mass_ratio)
    properties = []
    for props in [star.points, normals]:
        args = (position.azimuth - HALF_PI, props, "z", False, False)
        prop_value = utils.around_axis_rotation(*args)
        args = (HALF_PI - container.inclination, prop_value, "y", False, False)
        prop_value = utils.around_axis_rotation(*args)
        properties.append(prop_value)

    del container

    points, normals = properties
    cosines = np.inner(normals, LINE_OF_SIGHT)
    valid_indices = OrbitalPositionContainer.darkside_filter(cosines)

    visible_projection = utils.plane_projection(points[valid_indices], "yz")
    bb_path = get_eclipse_boundary_path(visible_projection)
    horizon_indices = up.invert(bb_path.contains_points(visible_projection, radius=threshold))

    precision = None
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


def get_analytics_horizon(binary=None, phase=0.0, tol=1e-4, polar=False, phi_density=100, theta_density=1000):
    if binary is None:
        binary = BinarySystem.from_json(BINARY_DEFINITION)

    star = binary.primary
    position = Position(*((0,) + tuple(binary.orbit.orbital_motion(phase=phase)[0])))

    # rotate line of sight to simulate phase and inclination
    zv = np.array([0.0, 0.0, 1.0])

    xv = utils.around_axis_rotation(HALF_PI - binary.inclination, LINE_OF_SIGHT, axis="y", inverse=True)
    xv = utils.around_axis_rotation(position.azimuth - HALF_PI, xv, axis="z", inverse=True)

    zv = utils.around_axis_rotation(HALF_PI - binary.inclination, zv, axis="y", inverse=True)
    zv = utils.around_axis_rotation(position.azimuth - HALF_PI, zv, axis="z", inverse=True)

    # perpendicular vector to find theta-like rotation
    yv = np.cross(zv, xv)

    potential_fn = potential_primary_fn
    precalc_fn = pre_calculate_for_potential_value_primary
    fprime = radial_primary_potential_derivative

    # prepare-phi like vector
    phi_range = np.linspace(np.radians(0), np.radians(360), phi_density)
    theta_range = np.linspace(np.radians(-5), np.radians(5), theta_density)

    # prepare theta-like vectors via rotation around phi and then aroun yv in -/+ range
    vectors = []
    for d_phi in phi_range:
        # girst rotate zv about phi around vector `xv`
        inner_vectors = []
        vector = utils.arbitrary_rotation(d_phi, xv, vector=zv)
        _yv = utils.arbitrary_rotation(d_phi, xv, vector=yv)
        for d_theta in theta_range:
            vectors.append(utils.arbitrary_rotation(d_theta, _yv, vector=vector))

    vectors = np.array(utils.cartesian_to_spherical(vectors))

    phi, theta = vectors[:, 1], vectors[:, 2]
    args = (star.synchronicity, binary.mass_ratio, position.distance, phi, theta)
    precalc_vals = precalc_fn(*args, return_as_tuple=True)
    x0 = star.side_radius * np.ones(phi.shape)
    radius_kwargs = dict(fprime=fprime, maxiter=settings.MAX_SOLVER_ITERS, rtol=1e-10,
                         args=((binary.mass_ratio,) + precalc_vals, star.surface_potential))

    radius = opt.newton.newton(potential_fn, x0, **radius_kwargs)
    points = utils.spherical_to_cartesian(np.array([radius, phi, theta]).T)
    normals = calculate_potential_gradient(1.0, "primary", points, star.synchronicity, binary.mass_ratio)

    cosines = np.inner(normals, xv)
    cosines = cosines.reshape(phi_density, theta_density)
    cosines_gtz = [up.arange(np.shape(row)[0])[row > 0] for row in cosines]
    # take only smallest values (but greater than zero) in each theta line
    cosines_argmin = [np.argmin(row[gtz]) for row, gtz in zip(cosines, cosines_gtz)]

    # find cosines in tolerance (tol)
    valid_argmin = [tol > row[gtz][argmin]
                    for row, gtz, argmin in zip(cosines, cosines_gtz, cosines_argmin)]

    points = points.reshape(phi_density, theta_density, 3)
    horizon_points = [row[gtz][argmin] for row, gtz, argmin, valid in
                      zip(points, cosines_gtz, cosines_argmin, valid_argmin) if valid]

    if utils.is_empty(horizon_points):
        raise ValueError(f"No horizon points found in given tolerance {tol}. Decrease tolerance.")

    horizon_points = utils.around_axis_rotation(position.azimuth - HALF_PI, horizon_points, axis="z", inverse=False)
    horizon_points = utils.around_axis_rotation(HALF_PI - binary.inclination, horizon_points, axis="y", inverse=False)
    horizon_points = horizon_points.T[1:3].T

    if polar:
        horizon_points = utils.cartesian_to_polar(horizon_points)
        horizon_argsort = np.argsort(horizon_points.T[1])
        horizon_points = horizon_points[horizon_argsort]

    return horizon_points


def _cover_horizon_edges(horizon):
    horizon_polygon = Polygon(horizon)
    horizon = list()
    for edge in horizon_polygon.edges(as_line=True):
        parametrized = edge.parametrized()
        for t in np.arange(0, 1, 0.01):
            horizon.append(parametrized(t))
    return np.array(horizon)


def get_discrete_horizon(binary=None, phase=0.0, threshold=-1e-6, polar=False):
    """
    Get discrete horizon of primary component.
    """
    if binary is None:
        binary = BinarySystem.from_json(BINARY_DEFINITION)

    position = Position(*((0,) + tuple(binary.orbit.orbital_motion(phase=phase)[0])))
    position_container = _horizon_base_component(binary, position, analytic=False)
    position_container.correct_mesh(component="primary", components_distance=position_container.position.distance)
    star = position_container.primary
    visible_projection = utils.get_visible_projection(star)

    bb_path = get_eclipse_boundary_path(visible_projection)
    horizon_indices = up.invert(bb_path.contains_points(visible_projection, radius=threshold))
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
    from matplotlib import pyplot as plt
    _phase = 0.25

    discrete_horizon, origin_discrete_horizon = get_discrete_horizon(phase=_phase, polar=False)

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
