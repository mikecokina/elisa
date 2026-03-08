from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from scipy.spatial import Delaunay

from elisa import const
from elisa import umpy as up
from elisa import units as u
from elisa.base import spot
from elisa.base.surface.faces import (
    calculate_normals,
    initialize_model_container,
    mirror_triangulation,
    set_all_surface_centres,
    split_spots_and_component_faces,
)
from elisa.base.types import INT
from elisa.binary_system import utils as bsutils
from elisa.binary_system.orbit import orbit
from elisa.logger import getLogger
from elisa.types import ComponentName
from elisa.utils import is_empty

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.types import Float, Int

logger = getLogger("binary_system.surface.faces")

ComponentSelection: TypeAlias = Literal["primary", "secondary", "all", "both"]
SurfaceComponent: TypeAlias = ComponentName


POTENTIAL_ERROR_BOUNDARY = 0.01


def visibility_test(
    centres: NDArray[np.float64],
    xlim: Float,
    component: SurfaceComponent,
) -> NDArray[np.bool_]:
    """Test whether faces are visible from the other star.

    :param centres: Face centres.
    :type centres: NDArray[np.float64]
    :param xlim: Visibility threshold on the x-axis for the given component.
    :type xlim: Float
    :param component: Component selector.
    :type component: Literal["primary", "secondary"]
    :return: Visibility mask.
    :rtype: NDArray[np.bool_]
    """
    return centres[:, 0] >= xlim if component == "primary" else centres[:, 0] <= xlim


def get_visibility_tests(
    centres: NDArray[np.float64],
    xlim: Float,
    component: SurfaceComponent,
    morphology: str,
    *,
    q_test: bool,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_] | None]:
    """Calculate visibility tests for illumination from the companion.

    Used in reflection-effect calculations.

    :param centres: Face centres.
    :type centres: NDArray[np.float64]
    :param q_test: Whether quarter-star symmetry filtering should be used.
    :type q_test: bool
    :param xlim: Visibility threshold on the x-axis for the given component.
    :type xlim: Float
    :param component: Component selector.
    :type component: Literal["primary", "secondary"]
    :param morphology: System morphology.
    :type morphology: str
    :return: Visibility masks for the full geometry and the symmetry-reduced
        geometry.
    :rtype: tuple[NDArray[np.bool_], NDArray[np.bool_] | None]
    """
    if q_test:
        y_test = centres[:, 1] > 0
        z_test = centres[:, 2] > 0

        # this branch is activated in case of clean surface where symmetries can be used
        # excluding quadrants that can be mirrored using symmetries
        quadrant_exclusion = (
            up.logical_or(y_test, z_test) if morphology == "over-contfact" else np.array([True] * len(centres))
        )

        single_quadrant = up.logical_and(y_test, z_test)

        # excluding faces on far sides of components
        test1 = visibility_test(centres, xlim, component)

        # this variable contains faces that can seen from base symmetry part of the other star
        vis_test = up.logical_and(test1, quadrant_exclusion)
        vis_test_symmetry = up.logical_and(test1, single_quadrant)
    else:
        vis_test = visibility_test(centres, xlim, component)
        vis_test_symmetry = None

    return vis_test, vis_test_symmetry


def faces_visibility_x_limits(
    primary_polar_radius: Float,
    secondary_polar_radius: Float,
    components_distance: Float,
) -> dict[str, Float]:
    """Return x limits of surface elements visible from the companion.

    :param primary_polar_radius: Polar radius of the primary component.
    :type primary_polar_radius: Float
    :param secondary_polar_radius: Polar radius of the secondary component.
    :type secondary_polar_radius: Float
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :return: Mapping with x visibility limits for primary and secondary.
    :rtype: dict[str, Float]
    """
    # this section calculates the visibility of each surface face
    # don't forget to treat system visibility of faces on the same star in over-contact system

    # if stars are too close and with too different radii, you can see more (less) than a half of the stellar
    # surface, calculating excess angle

    primary_polar_r = primary_polar_radius
    secondary_polar_r = secondary_polar_radius
    sin_theta = up.abs(primary_polar_r - secondary_polar_r) / components_distance
    x_corr_primary = primary_polar_r * sin_theta
    x_corr_secondary = secondary_polar_r * sin_theta

    # visibility of faces is given by their x position
    xlim: dict[str, Float] = {}
    if primary_polar_r > secondary_polar_r:
        xlim["primary"], xlim["secondary"] = (
            x_corr_primary,
            components_distance + x_corr_secondary,
        )
    else:
        xlim["primary"], xlim["secondary"] = (
            -x_corr_primary,
            components_distance - x_corr_secondary,
        )

    return xlim


def get_surface_builder_fn(morphology: str) -> Callable:
    """Return the triangulation function appropriate for the morphology.

    :param morphology: System morphology.
    :type morphology: str
    :return: Surface builder function.
    :rtype: callable
    """
    return over_contact_system_surface if morphology == "over-contact" else detached_system_surface


def build_faces(
    system: OrbitalPositionContainer,
    components_distance: Float,
    component: ComponentSelection = "all",
) -> OrbitalPositionContainer:
    """Build surface faces for selected components.

    Faces are evaluated from points that must already be available.

    :param system: Orbital position container instance.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param component: Component selector.
    :type component: Literal["primary", "secondary", "all", "both"]
    :return: Updated orbital position container.
    :rtype: OrbitalPositionContainer
    """
    if is_empty(component):
        logger.debug("no component set to build faces")
        return system

    if is_empty(components_distance):
        msg = "Value of `components_distance` was not provided."
        raise ValueError(msg)

    components = bsutils.component_to_list(component)
    for component_name in components:
        star = getattr(system, component_name)
        if star.has_spots():
            build_surface_with_spots(system, components_distance, component_name)
        else:
            build_surface_with_no_spots(system, components_distance, component_name)

    return system


def build_surface_with_no_spots(
    system: OrbitalPositionContainer,
    components_distance: Float,
    component: ComponentSelection = "all",
) -> OrbitalPositionContainer:
    """Build stellar surfaces for components without spots.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param component: Component selector.
    :type component: Literal["primary", "secondary", "all", "both"]
    :return: Updated orbital position container.
    :rtype: OrbitalPositionContainer
    """
    components = bsutils.component_to_list(component)

    for component_name in components:
        star = getattr(system, component_name)

        # triangulating only one quarter of the star
        triangulated_pts = star.symmetry_points()

        if system.morphology != "over-contact":
            triangles = detached_system_surface(
                system,
                components_distance,
                triangulated_pts,
                component_name,
            )
        else:
            neck = np.max(triangulated_pts[:, 0]) if component_name == "primary" else np.min(triangulated_pts[:, 0])
            triangulated_pts = np.append(
                triangulated_pts,
                np.array([[neck, 0, 0]]),
                axis=0,
            )
            triangles = over_contact_system_surface(
                system,
                triangulated_pts,
                component_name,
            )

            # filtering out triangles containing last point in `points_to_triangulate`
            triangles = triangles[np.array(triangles < star.base_symmetry_points_number).all(axis=1)]

        # filtering out faces on xy a xz planes
        y0_test = np.bitwise_not(
            np.isclose(triangulated_pts[triangles][:, :, 1], 0).all(axis=1),
        )
        z0_test = np.bitwise_not(
            np.isclose(triangulated_pts[triangles][:, :, 2], 0).all(axis=1),
        )
        triangles = triangles[up.logical_and(y0_test, z0_test)]

        star.base_symmetry_faces_number = INT(np.shape(triangles)[0])

        # let's exploit axial symmetry and fill the rest with the surface of the star
        star.base_symmetry_faces = triangles
        star.faces = mirror_triangulation(
            triangles,
            star.inverse_point_symmetry_matrix,
        )

        base_face_symmetry_vector = up.arange(star.base_symmetry_faces_number)
        star.face_symmetry_vector = up.concatenate(
            [base_face_symmetry_vector for _ in range(4)],
        )

    return system


def build_surface_with_spots(
    system: OrbitalPositionContainer,
    components_distance: Float,
    component: ComponentSelection = "all",
) -> OrbitalPositionContainer:
    """Triangulate stellar surfaces that contain spots.

    Surface and spot points are merged, triangulated together, and the
    resulting faces are then split back into stellar and spot subsets.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param component: Component selector.
    :type component: Literal["primary", "secondary", "all", "both"]
    :return: Updated orbital position container.
    :rtype: OrbitalPositionContainer
    """
    components = bsutils.component_to_list(component)
    component_com = {"primary": 0.0, "secondary": components_distance}

    for component_name in components:
        star_container = getattr(system, component_name)
        points, vertices_map = star_container.get_flatten_points_map()

        surface_fn = get_surface_builder_fn(system.morphology)
        surface_fn_kwargs = {
            "component": component_name,
            "points": points,
            "components_distance": components_distance,
        }
        faces = surface_fn(system, **surface_fn_kwargs)

        model, spot_candidates = initialize_model_container(vertices_map)
        model = split_spots_and_component_faces(
            star_container,
            points,
            faces,
            model,
            spot_candidates,
            vertices_map,
            component_com[component_name],
        )
        spot.remove_overlaped_spots_by_vertex_map(star_container, vertices_map)
        spot.remap_surface_elements(star_container, model, points)

    return system


def detached_system_surface(
    system: OrbitalPositionContainer,
    components_distance: Float,
    points: NDArray[np.float64] | None = None,
    component: SurfaceComponent = "primary",
) -> NDArray[Int]:
    """Calculate surface faces for detached or semi-contact systems.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param points: Surface points to triangulate. If ``None``, component points
        from the container are used.
    :type points: NDArray[np.float64] | None
    :param component: Component selector.
    :type component: Literal["primary", "secondary"]
    :return: Array of triangle vertex indices with shape ``(N, 3)``.
    :rtype: NDArray[Int]
    """
    component_instance = getattr(system, component)
    if points is None:
        points = component_instance.points

    if not np.any(points):
        msg = (
            f"{component} component, with class instance name "
            f"{component_instance.name} do not contain any valid surface "
            "point to triangulate"
        )
        raise ValueError(msg)

    # there is a problem with triangulation of near over-contact system, delaunay is not good with pointy surfaces
    critical_pot = (
        system.primary.critical_surface_potential
        if component == "primary"
        else system.secondary.critical_surface_potential
    )
    potential = system.primary.surface_potential if component == "primary" else system.secondary.surface_potential

    if potential - critical_pot > POTENTIAL_ERROR_BOUNDARY:
        logger.debug(
            "triangulating surface of %s component using standard method",
            component,
        )
        triangulation = Delaunay(points)
        triangles_indices = triangulation.convex_hull
    else:
        logger.debug(
            "surface of %s component is near or at critical potential; "
            "therefore custom triangulation method for (near)critical "
            "potential surfaces will be used",
            component,
        )

        # calculating closest point to the barycentre
        r_near = np.max(points[:, 0]) if component == "primary" else np.min(points[:, 0])

        # projection of component's far side surface into ``sphere`` with radius r1
        points_to_transform = copy(points)
        if component == "secondary":
            points_to_transform[:, 0] -= components_distance

        projected_points = r_near * points_to_transform / np.linalg.norm(points_to_transform, axis=1)[:, None]

        if component == "secondary":
            projected_points[:, 0] += components_distance

        triangulation = Delaunay(projected_points)
        triangles_indices = triangulation.convex_hull

    return triangles_indices


# noinspection PyUnusedLocal
def over_contact_system_surface(
    system: OrbitalPositionContainer,
    points: NDArray[np.float64] | None = None,
    component: SurfaceComponent = "primary",
    **kwargs: Any,
) -> NDArray[Int]:
    """Calculate surface faces for an over-contact system.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param points: Surface points to triangulate.
    :type points: NDArray[np.float64] | None
    :param component: Component selector.
    :type component: Literal["primary", "secondary"]
    :param kwargs: Unused keyword arguments preserved for compatibility.
    :type kwargs: object
    :return: Array of triangle vertex indices with shape ``(N, 3)``.
    :rtype: NDArray[Int]
    """
    # do not remove kwargs, keep compatible interface w/ detached where components distance has to be provided
    # in this case, components distance is sinked in kwargs and not used
    del kwargs

    component_instance = getattr(system, component)

    if points is None:
        points = component_instance.points

    if up.isnan(points).any():
        msg = (
            f"{component} component, with class instance name "
            f"{component_instance.name} contain any valid point to triangulate"
        )
        raise ValueError(msg)

    # calculating position of the neck
    neck_x = np.max(points[:, 0]) if component == "primary" else np.min(points[:, 0])

    projected_points = points.copy()
    projected_points[:, 0] -= 1 if component == "secondary" else 0
    projected_points = neck_x * projected_points / np.linalg.norm(projected_points, axis=1)[:, None]

    triangulation = Delaunay(projected_points)
    triangles_indices = triangulation.convex_hull

    # removal of faces on top of the neck
    neck_test = ~(up.equal(points[triangles_indices][:, :, 0], neck_x).all(axis=-1))
    return triangles_indices[neck_test]



def compute_all_surface_areas(
    system: OrbitalPositionContainer,
    component: ComponentSelection,
) -> OrbitalPositionContainer | None:
    """Compute areas of all faces, including spot faces.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param component: Component selector.
    :type component: Literal["primary", "secondary", "all", "both"]
    :return: Updated orbital position container, or ``None`` when no component
        is selected.
    :rtype: OrbitalPositionContainer | None
    """
    if is_empty(component):
        logger.debug("no component set to build surface areas")
        return None

    components = bsutils.component_to_list(component)
    for component_name in components:
        star = getattr(system, component_name)
        logger.debug(
            "computing surface areas of component: %s / name: %s",
            star,
            star.name,
        )
        star.calculate_all_areas()

    return system


def build_faces_orientation(
    system: OrbitalPositionContainer,
    components_distance: Float,
    component: ComponentSelection = "all",
) -> OrbitalPositionContainer:
    """Compute face normals for each face.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param component: Component selector.
    :type component: Literal["primary", "secondary", "all", "both"]
    :return: Updated orbital position container.
    :rtype: OrbitalPositionContainer
    """
    if is_empty(component):
        logger.debug("no component set to build face orientation")
        return system

    components = bsutils.component_to_list(component)
    com_x = {"primary": 0.0, "secondary": components_distance}

    for component_name in components:
        star = getattr(system, component_name)
        set_all_surface_centres(star)
        set_all_normals(star, com=com_x[component_name])

    return system


def set_all_normals(
    star_container: StarContainer,
    com: Float,
) -> StarContainer:
    """Calculate normals for all faces and assign them to the container.

    This includes spot normals when present.

    :param star_container: Container on which normals should be assigned.
    :type star_container: StarContainer
    :param com: x coordinate of the component centre of mass.
    :type com: Float
    :return: Updated star container.
    :rtype: StarContainer
    """
    points = star_container.points
    faces = star_container.faces
    cntrs = star_container.face_centres

    if star_container.symmetry_test():
        normals1 = calculate_normals(
            star_container.symmetry_points(),
            star_container.symmetry_faces(faces),
            star_container.symmetry_faces(cntrs),
            com,
        )
        normals2 = normals1 * np.array([1.0, -1.0, 1.0])
        normals3 = normals1 * np.array([1.0, -1.0, -1.0])
        normals4 = normals1 * np.array([1.0, 1.0, -1.0])
        star_container.normals = np.concatenate(
            (normals1, normals2, normals3, normals4),
            axis=0,
        )
    else:
        star_container.normals = calculate_normals(points, faces, cntrs, com)

    if star_container.has_spots() and not star_container.is_flat():
        for spot_index in star_container.spots:
            star_container.spots[spot_index].normals = calculate_normals(
                star_container.spots[spot_index].points,
                star_container.spots[spot_index].faces,
                star_container.spots[spot_index].face_centres,
                com,
            )

    return star_container


def build_velocities(
    system: OrbitalPositionContainer,
    components_distance: Float,
    component: ComponentSelection = "all",
) -> OrbitalPositionContainer:
    """Calculate velocity vector for each face relative to the system COM.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param component: Component selector.
    :type component: Literal["primary", "secondary", "all", "both"]
    :return: Updated orbital position container.
    :rtype: OrbitalPositionContainer
    """
    if is_empty(component):
        logger.debug("no component set to build face orientation")
        return system

    components = bsutils.component_to_list(component)
    com_x = {
        "primary": np.array([0.0, 0.0, 0.0]),
        "secondary": np.array([components_distance, 0.0, 0.0]),
    }

    velocities = orbit.create_orb_vel_vectors(system, components_distance)

    orb_period = (system.period * u.DefaultBinarySystemUnits.system.period).to(u.TIME_UNIT).value
    omega_orb = np.array([0, 0, const.FULL_ARC / orb_period])

    for component_name in components:
        star = getattr(system, component_name)
        points = (star.points - com_x[component_name][None, :]) * system.semi_major_axis
        omega = star.synchronicity * omega_orb

        # orbital velocity + rotational velocity
        p_velocities = velocities[component_name] + np.cross(
            omega[None, :],
            points,
            axisa=1,
        )
        star.velocities = np.mean(p_velocities[star.faces], axis=1)

        if star.has_spots():
            for spot_inst in star.spots.values():
                points = (
                    spot_inst.points - com_x[component_name][None, :]
                ) * system.semi_major_axis
                p_velocities = velocities[component_name] + np.cross(
                    omega[None, :],
                    points,
                    axisa=1,
                )
                spot_inst.velocities = np.mean(p_velocities[spot_inst.faces], axis=1)

    return system
