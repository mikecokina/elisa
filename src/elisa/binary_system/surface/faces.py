import numpy as np

from copy import copy
from scipy.spatial.qhull import Delaunay
from .. import utils as bsutils
from .. orbit import orbit
from ... base import spot
from ... utils import is_empty
from ... logger import getLogger
from ... import (
    const,
    umpy as up,
    units as u
)
from ... base.surface.faces import (
    initialize_model_container,
    split_spots_and_component_faces,
    set_all_surface_centres,
    calculate_normals,
    mirror_triangulation
)

logger = getLogger("binary_system.surface.faces")


def visibility_test(centres, xlim, component):
    """
    Tests if given faces are visible from the other star.

    :param component: str;
    :param centres: numpy.array;
    :param xlim: visibility threshold in x axis for given component
    :return: numpy.array[bool];
    """
    return centres[:, 0] >= xlim if component == 'primary' else centres[:, 0] <= xlim


def get_visibility_tests(centres, q_test, xlim, component, morphology):
    """
    Method calculates tests for visibilities of faces from other component.
    Used in reflection effect.

    :param centres: np.array; of face centres
    :param q_test: use_quarter_star_test
    :param xlim: visibility threshold in x axis for given component
    :param component: `primary` or `secondary`
    :param morphology: str;
    :return: visual tests for normal and symmetrical star
    """
    if q_test:
        y_test, z_test = centres[:, 1] > 0, centres[:, 2] > 0
        # this branch is activated in case of clean surface where symmetries can be used
        # excluding quadrants that can be mirrored using symmetries
        quadrant_exclusion = up.logical_or(y_test, z_test) if morphology == 'over-contfact' \
            else np.array([True] * len(centres))

        single_quadrant = up.logical_and(y_test, z_test)
        # excluding faces on far sides of components
        test1 = visibility_test(centres, xlim, component)
        # this variable contains faces that can seen from base symmetry part of the other star
        vis_test = up.logical_and(test1, quadrant_exclusion)
        vis_test_symmetry = up.logical_and(test1, single_quadrant)

    else:
        vis_test = centres[:, 0] >= xlim if component == 'primary' else centres[:, 0] <= xlim
        vis_test_symmetry = None

    return vis_test, vis_test_symmetry


def faces_visibility_x_limits(primary_polar_radius, secondary_polar_radius, components_distance):
    """
    Returns x coordinates of `primary` and `secondary` surface elements which can be visible from the other star.

    :param primary_polar_radius: float;
    :param secondary_polar_radius: float;
    :param components_distance: float; in SMA
    :return: Tuple; x_min for primary, x_max for secondary
    """
    # this section calculates the visibility of each surface face
    # don't forget to treat system visibility of faces on the same star in over-contact system

    # if stars are too close and with too different radii, you can see more (less) than a half of the stellar
    # surface, calculating excess angle

    primary_polar_r, secondary_polar_r = primary_polar_radius, secondary_polar_radius
    sin_theta = up.abs(primary_polar_r - secondary_polar_r) / components_distance
    x_corr_primary, x_corr_secondary = primary_polar_r * sin_theta, secondary_polar_r * sin_theta

    # visibility of faces is given by their x position
    xlim = {}
    (xlim['primary'], xlim['secondary']) = (x_corr_primary, components_distance + x_corr_secondary) \
        if primary_polar_r > secondary_polar_r else (-x_corr_primary, components_distance - x_corr_secondary)
    return xlim


def get_surface_builder_fn(morphology):
    """
    Returns suitable triangulation function depending on morphology.

    :return: callable; method that performs generation surface faces
    """
    return over_contact_system_surface if morphology == "over-contact" else detached_system_surface


def build_faces(system, components_distance, component="all"):
    """
    Function creates faces of the star surface for given components. Faces are evaluated upon points that
    have to be in this time already calculated.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;; instance
    :type components_distance: float;
    :param component: `primary` or `secondary` if not supplied both component are calculated
    :return: system; elisa.binary_system.contaier.OrbitalPositionContainer; instance
    """
    if is_empty(component):
        logger.debug("no component set to build faces")
        return system

    if is_empty(components_distance):
        raise ValueError('Value of `components_distance` was not provided.')

    components = bsutils.component_to_list(component)
    for component in components:
        star = getattr(system, component)
        if star.has_spots():
            build_surface_with_spots(system, components_distance, component)
        else:
            build_surface_with_no_spots(system, components_distance, component)
    return system


def build_surface_with_no_spots(system, components_distance, component="all"):
    """
    Function for building binary star component surfaces without spots.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param components_distance: float;
    :param component: str; `primary` or `secondary` if not supplied both component are calculated
    :return: system; elisa.binary_system.contaier.OrbitalPositionContainer; instance
    """
    components = bsutils.component_to_list(component)

    for component in components:
        star = getattr(system, component)
        # triangulating only one quarter of the star

        triangulated_pts = star.symmetry_points()
        if system.morphology != 'over-contact':
            triangles = detached_system_surface(system, components_distance, triangulated_pts, component)
        else:
            neck = np.max(triangulated_pts[:, 0]) if component == 'primary' else np.min(triangulated_pts[:, 0])
            triangulated_pts = \
                np.append(triangulated_pts, np.array([[neck, 0, 0]]), axis=0)
            triangles = over_contact_system_surface(system, triangulated_pts, component)
            # filtering out triangles containing last point in `points_to_triangulate`
            triangles = triangles[np.array(triangles < star.base_symmetry_points_number).all(1)]

        # filtering out faces on xy an xz planes
        y0_test = np.bitwise_not(np.isclose(triangulated_pts[triangles][:, :, 1], 0).all(1))
        z0_test = np.bitwise_not(np.isclose(triangulated_pts[triangles][:, :, 2], 0).all(1))
        triangles = triangles[up.logical_and(y0_test, z0_test)]

        setattr(star, "base_symmetry_faces_number", np.int(np.shape(triangles)[0]))
        # lets exploit axial symmetry and fill the rest of the surface of the star
        star.base_symmetry_faces = triangles
        star.faces = mirror_triangulation(triangles, star.inverse_point_symmetry_matrix)

        base_face_symmetry_vector = up.arange(star.base_symmetry_faces_number)
        star.face_symmetry_vector = up.concatenate([base_face_symmetry_vector for _ in range(4)])
    return system


def build_surface_with_spots(system, components_distance, component="all"):
    """
    Function capable of triangulation of spotty stellar surfaces.
    It merges all surface points, triangulates them and then sorts the resulting surface faces under star or spot.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param components_distance: float;
    :param component: str `primary` or `secondary`
    :return: system; elisa.binary_system.contaier.OrbitalPositionContainer; instance
    """
    components = bsutils.component_to_list(component)
    component_com = {'primary': 0.0, 'secondary': components_distance}
    for component in components:
        star_container = getattr(system, component)
        points, vertices_map = star_container.get_flatten_points_map()

        surface_fn = get_surface_builder_fn(system.morphology)
        surface_fn_kwargs = dict(component=component, points=points, components_distance=components_distance)
        faces = surface_fn(system, **surface_fn_kwargs)
        model, spot_candidates = initialize_model_container(vertices_map)
        model = split_spots_and_component_faces(star_container, points, faces, model,
                                                spot_candidates, vertices_map, component_com[component])
        spot.remove_overlaped_spots_by_vertex_map(star_container, vertices_map)
        spot.remap_surface_elements(star_container, model, points)
    return system


def detached_system_surface(system, components_distance, points=None, component="all"):
    """
    Calculates surface faces from the given component's points in case of detached or semi-contact system.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param components_distance: float;
    :param points: numpy.array;
    :param component: str;
    :return: numpy.array; N x 3 array of vertices indices
    """
    component_instance = getattr(system, component)
    if points is None:
        points = component_instance.points

    if not np.any(points):
        raise ValueError(f"{component} component, with class instance name {component_instance.name} do not "
                         "contain any valid surface point to triangulate")
    # there is a problem with triangulation of near over-contact system, delaunay is not good with pointy surfaces
    critical_pot = system.primary.critical_surface_potential if component == 'primary' \
        else system.secondary.critical_surface_potential
    potential = system.primary.surface_potential if component == 'primary' \
        else system.secondary.surface_potential
    if potential - critical_pot > 0.01:
        logger.debug(f'triangulating surface of {component} component using standard method')
        triangulation = Delaunay(points)
        triangles_indices = triangulation.convex_hull
    else:
        logger.debug(f'surface of {component} component is near or at critical potential; therefore custom '
                     f'triangulation method for (near)critical potential surfaces will be used')
        # calculating closest point to the barycentre
        r_near = np.max(points[:, 0]) if component == 'primary' else np.min(points[:, 0])
        # projection of component's far side surface into ``sphere`` with radius r1

        points_to_transform = copy(points)
        if component == 'secondary':
            points_to_transform[:, 0] -= components_distance
        projected_points = \
            r_near * points_to_transform / np.linalg.norm(points_to_transform, axis=1)[:, None]
        if component == 'secondary':
            projected_points[:, 0] += components_distance

        triangulation = Delaunay(projected_points)
        triangles_indices = triangulation.convex_hull

    return triangles_indices


def over_contact_system_surface(system, points=None, component="all", **kwargs):
    # do not remove kwargs, keep compatible interface w/ detached where components distance has to be provided
    # in this case, components distance is sinked in kwargs and not used
    """
    Calculates surface faces from the given component's points in case of over-contact system.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param points: numpy.array; - points to triangulate
    :param component: str; `primary` or `secondary`
    :return: numpy.array; N x 3 array of vertice indices
    """
    del kwargs

    component_instance = getattr(system, component)
    if up.isnan(points).any():
        raise ValueError(f"{component} component, with class instance name {component_instance.name} "
                         f"contain any valid point to triangulate")
    # calculating position of the neck
    neck_x = np.max(points[:, 0]) if component == 'primary' else np.min(points[:, 0])

    projected_points = points.copy()
    projected_points[:, 0] -= 1 if component == 'secondary' else 0
    projected_points = neck_x * projected_points / np.linalg.norm(projected_points, axis=1)[:, None]

    triangulation = Delaunay(projected_points)
    triangles_indices = triangulation.convex_hull

    # removal of faces on top of the neck
    neck_test = ~(up.equal(points[triangles_indices][:, :, 0], neck_x).all(-1))
    new_triangles_indices = triangles_indices[neck_test]

    return new_triangles_indices


def compute_all_surface_areas(system, component):
    """
    Compute surface are of all faces (spots included).

    :param system: elisa.binary_system.container.OrbitalPositionContainer; instance
    :param component: str; `primary` or `secondary`
    :return: system; elisa.binary_system.contaier.OrbitalPositionContainer; instance
    """
    if is_empty(component):
        logger.debug("no component set to build surface areas")
        return

    components = bsutils.component_to_list(component)
    for component in components:
        star = getattr(system, component)
        logger.debug(f'computing surface areas of component: '
                     f'{star} / name: {star.name}')
        star.calculate_all_areas()
    return system


def build_faces_orientation(system, components_distance, component="all"):
    """
    Compute face orientation (normals) for each face.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param component: str; `primary` or `secondary`
    :param components_distance: float; orbit with misaligned pulsations, where pulsation axis drifts with star
    :return: system; elisa.binary_system.contaier.OrbitalPositionContainer; instance
    """
    if is_empty(component):
        logger.debug("no component set to build face orientation")
        return system

    component = bsutils.component_to_list(component)
    com_x = {'primary': 0.0, 'secondary': components_distance}

    for _component in component:
        star = getattr(system, _component)
        set_all_surface_centres(star)
        set_all_normals(star, com=com_x[_component])
    return system


def set_all_normals(star_container, com):
    """
    Function calculates normals for each face of given body (including spots) and assign it to object.

    :param star_container: instance of container to set normals on;
    :param com: numpy.array;
    :param star_container: instance of container to set normals on;
    """
    points, faces, cntrs = star_container.points, star_container.faces, star_container.face_centres
    if star_container.symmetry_test():
        normals1 = calculate_normals(star_container.symmetry_points(),
                                     star_container.symmetry_faces(faces),
                                     star_container.symmetry_faces(cntrs), com)
        normals2 = normals1 * np.array([1.0, -1.0,  1.0])
        normals3 = normals1 * np.array([1.0, -1.0, -1.0])
        normals4 = normals1 * np.array([1.0,  1.0, -1.0])
        star_container.normals = np.concatenate((normals1, normals2, normals3, normals4), axis=0)
    else:
        star_container.normals = calculate_normals(points, faces, cntrs, com)

    if star_container.has_spots() and not star_container.is_flat():
        for spot_index in star_container.spots:
            star_container.spots[spot_index].normals = calculate_normals(star_container.spots[spot_index].points,
                                                                         star_container.spots[spot_index].faces,
                                                                         star_container.spots[spot_index].face_centres,
                                                                         com)
    return star_container


def build_velocities(system, components_distance, component='all'):
    """
    Function calculates velocity vector for each face relative to the system's centre of mass.

    :param system: elisa.binary_system.container.SystemContainer;
    :param components_distance: float;
    :param component: str;
    :return: elisa.binary_system.container.SystemContainer;
    """
    if is_empty(component):
        logger.debug("no component set to build face orientation")
        return system

    component = bsutils.component_to_list(component)
    com_x = {'primary': np.array([0.0, 0.0, 0.0]), 'secondary': np.array([components_distance, 0.0, 0.0])}

    velocities = orbit.create_orb_vel_vectors(system, components_distance)

    orb_period = (system.period * u.PERIOD_UNIT).to(u.s).value
    omega_orb = np.array([0, 0, const.FULL_ARC / orb_period])

    for _component in component:
        star = getattr(system, _component)
        points = (star.points - com_x[_component][None, :]) * system.semi_major_axis
        omega = star.synchronicity * omega_orb

        # orbital velocity + rotational velocity
        p_velocities = velocities[_component] + np.cross(omega[None, :], points, axisa=1)
        star.velocities = np.mean(p_velocities[star.faces], axis=1)

        if star.has_spots():
            for spot_inst in star.spots.values():
                points = (spot_inst.points - com_x[_component][None, :]) * system.semi_major_axis
                p_velocities = velocities[_component] + np.cross(omega[None, :], points, axisa=1)
                spot_inst.velocities = np.mean(p_velocities[spot_inst.faces], axis=1)

    return system
