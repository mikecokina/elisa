import numpy as np

from . import utils as putils
from . surface import kinematics
from . import pulsations
from elisa import utils, const
from .. base.surface.faces import (
    set_all_surface_centres,
    calculate_normals
)


def generate_harmonics(star_container, com_x, phase, time):
    """
    Generating spherical harmonics Y_l^m in shapes(2, n_points) and (2, n_faces)
    and its derivatives to be subsequently used for calculation of perturbed properties.

    :param star_container: elisa.base.container.StarContainer;
    :param com_x: float; centre of mass for the component
    :param phase: float; rotational/orbital phase
    :param time: float; time of the observation
    :return: elisa.base.container.StarContainer; Star container with updated harmonics
    """
    if not star_container.is_flat():
        raise ValueError('Pulsations can be calculated only on flattened container.')
    _kwargs = dict(kind='points', com_x=com_x)
    star_container.points_spherical = star_container.transform_points_to_spherical_coordinates(**_kwargs)

    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star_container, phase)
    star_container.pulsations[0].tilt_phi, star_container.pulsations[0].tilt_theta = tilt_phi, tilt_theta

    tilted_points = putils.tilt_mode_coordinates(
        star_container.points_spherical, star_container.pulsations[0].tilt_phi, star_container.pulsations[0].tilt_theta
    )

    # assigning tilted points in spherical coordinates only to the first mode (the rest will share the same points)
    star_container.pulsations[0].points = tilted_points

    exponential, norm_constant = dict(), dict()
    for mode_index, mode in star_container.pulsations.items():
        # beware of this in case you want use the container at different phase
        exponential[mode_index] = putils.generate_time_exponential(mode, time)

        # generating harmonics Y_m^l and Y_m+1^l for star and spot points
        harmonics = np.zeros((2, tilted_points.shape[0]), dtype=np.complex)
        harmonics[0] = pulsations.spherical_harmonics(mode, tilted_points, exponential[mode_index])
        if mode.m != mode.l:
            _args, _kwargs = (mode, tilted_points, exponential[mode_index]), dict(order=mode.m + 1, degree=mode.l)
            harmonics[1] = pulsations.spherical_harmonics(*_args, **_kwargs)

        # generating derivatives of spherical harmonics by phi an theta
        derivatives = np.empty((2, tilted_points.shape[0]), dtype=np.complex)
        derivatives[0] = pulsations.diff_spherical_harmonics_by_phi(mode, harmonics)
        _args = (mode, harmonics, tilted_points[:, 1], tilted_points[:, 2])
        derivatives[1] = pulsations.diff_spherical_harmonics_by_theta(*_args)

        # renormalizing horizontal amplitude to 1
        if mode.l > 0:
            norm_constant[mode_index] = pulsations.horizontal_displacement_normalization(derivatives, harmonics)
            derivatives *= norm_constant[mode_index]

        # assignment of harmonics to mode instance variables
        mode.point_harmonics = harmonics[0]
        mode.point_harmonics_derivatives = derivatives

    return star_container


def incorporate_pulsations_to_model(star_container, com_x, scale=1.0):
    """
    Function adds perturbation to the surface mesh due to pulsations.

    :param star_container: base.container.StarContainer;
    :param com_x: float;
    :param scale: numpy.float; scale of the perturbations
    :return: base.container.StarContainer;
    """
    # calculating kinematics quantities
    complex_displacement(star_container, scale=scale)

    # treating polar regions
    putils.pole_neighbours(star_container)

    position_perturbation(star_container, com_x=com_x, update_container=True, return_perturbation=False)
    velocity_perturbation(star_container, scale=scale, update_container=True, return_perturbation=False)
    gravity_acc_perturbation(star_container, scale=scale, update_container=True, return_perturbation=False)
    temp_perturbation(star_container, scale=scale, update_container=True, return_perturbation=False)

    # recalculating normals and areas
    set_all_surface_centres(star_container)
    args = (star_container.points, star_container.faces, star_container.face_centres, com_x)
    star_container.normals = calculate_normals(*args)
    star_container.calculate_all_areas()
    return star_container


def complex_displacement(star, scale):
    """
    Assigning complex displacement for surface points. Complex displacement
    is then used to calculate the kinematic quantities (r, v, a).

    :param star: base.container.StarContainer;
    :param scale: float;
    :return: base.container.StarContainer;
    """
    if not star.is_flat():
        raise ValueError('Pulsations can be calculated only on flattened container.')

    for mode_index, mode in star.pulsations.items():
        mode.complex_displacement = kinematics.calculate_displacement_coordinates(
            mode, star.pulsations[0].points, mode.point_harmonics, mode.point_harmonics_derivatives,
            star.points_spherical[:, 0], scale=scale
        )

    return star


def position_perturbation(
        star,
        com_x=0,
        update_container=True,
        return_perturbation=False,
        spherical_perturbation=False
):
    """
    Calculates the deformation of the surface mesh due to the pulsations.

    :param com_x: float; x coordinate of the compotnents centre of mass in corotating frame
    :param star: base.container.StarContainer;
    :param update_container: bool; if True, perturbation is incorporated into star.points
    :param return_perturbation: bool; if True, calculated displacement (in cartesian coordinates) is returned
    :param spherical_perturbation: bool; if True, perturbations are returned in spherical coordinates
    :return: Union[numpy.array, None];
    """
    displacement = None

    tilt_displacement_sph = np.sum([
        kinematics.calculate_mode_angular_displacement(mode.complex_displacement) for mode in star.pulsations.values()
    ], axis=0)

    points_spherical = putils.derotate_surface_points(
        star.pulsations[0].points + tilt_displacement_sph,
        star.pulsations[0].tilt_phi, star.pulsations[0].tilt_theta
    )

    points = utils.spherical_to_cartesian(points_spherical)
    if return_perturbation:
        if spherical_perturbation:
            displacement = points_spherical - getattr(star, 'points_spherical')
            displacement[displacement[:, 1] > const.PI, 1] -= const.FULL_ARC
        else:
            points - utils.spherical_to_cartesian(star.points_spherical)
    if update_container:
        com = np.array([com_x, 0, 0])
        setattr(star, 'points', points + com[None, :])

    return displacement if return_perturbation else None


def velocity_perturbation(
        star,
        scale,
        update_container=False,
        return_perturbation=False,
        spherical_perturbation=False,
        point_perturbations=False
):
    """
    Calculates velocity perturbation on a surface of a pulsating star.

    :param star: base.container.StarContainer;
    :param scale: float; scaling factor of the system (a in case of BinarySystem)
    :param update_container: bool; if true, the perturbations are added into surface element velocities
    :param return_perturbation: bool; if True, velocity perturbation itself is returned
    :param spherical_perturbation: bool; if True, velocity perturbation in spherical coordinates (d_r, d_phi, d_theta)
                                         is returned.
    :param point_perturbations: bool; if True perturbations on surface pints are returned instead face averaged values
    :return: Union[None, numpy.array];
    """
    # calculating perturbed velocity in spherical coordinates
    tilt_velocity_sph = np.sum([
        kinematics.calculate_mode_derivatives(
            displacement=mode.complex_displacement, angular_frequency=mode.angular_frequency
        ) for mode in star.pulsations.values()
    ], axis=0)

    velocity_pert_sph = putils.derotate_surface_displacements(
        tilt_velocity_sph, star.pulsations[0].points, star.points_spherical,
        star.pulsations[0].tilt_phi, star.pulsations[0].tilt_theta
    )
    velocity_pert_sph[star.pole_idx] = velocity_pert_sph[star.pole_idx_neighbour]
    points_cartesian = utils.spherical_to_cartesian(star.points_spherical)
    args = (velocity_pert_sph, points_cartesian, 0.0)
    velocity_pert = putils.transform_spherical_displacement_to_cartesian(*args) * scale

    velocity_pert_face = velocity_pert[star.faces].mean(axis=1)

    if update_container:
        star.velocities += velocity_pert_face

    if return_perturbation:
        if spherical_perturbation:
            velocity_pert_sph[:, 0] *= scale
            return velocity_pert_sph[star.faces].mean(axis=1) if not point_perturbations else velocity_pert_sph
        else:
            return velocity_pert if point_perturbations else velocity_pert_face
    else:
        return None


def gravity_acc_perturbation(
        star,
        scale,
        update_container=False,
        return_perturbation=False,
        spherical_perturbation=False,
        point_perturbations=False
):
    """
    Calculates acceleration perturbation on a surface of a pulsating star.

    :param star: base.container.StarContainer;
    :param scale: float; scaling factor of the system (a in case of BinarySystem)
    :param update_container: bool; if true, the perturbations are added into surface element velocities
    :param return_perturbation: bool; if True, velocity perturbation itself is returned
    :param spherical_perturbation: bool; if True, velocity perturbation in spherical coordinates (d_r, d_phi, d_theta)
                                         is returned.
    :param point_perturbations: bool; if True perturbations on surface pints are returned instead face averaged values
    :return:
    """
    # calculating perturbed acceleration in tilted spherical coordinates
    tilt_acc_sph = np.sum([
        kinematics.calculate_mode_second_derivatives(
            displacement=mode.complex_displacement, angular_frequency=mode.angular_frequency
        ) for mode in star.pulsations.values()
    ], axis=0)

    acc_pert_sph = putils.derotate_surface_displacements(
        tilt_acc_sph, star.pulsations[0].points, star.points_spherical,
        star.pulsations[0].tilt_phi, star.pulsations[0].tilt_theta
    )
    acc_pert_sph[star.pole_idx] = acc_pert_sph[star.pole_idx_neighbour]
    points_cartesian = utils.spherical_to_cartesian(star.points_spherical)
    acc_pert = putils.transform_spherical_displacement_to_cartesian(acc_pert_sph, points_cartesian, 0.0) * scale

    # treating singularities at poles
    acc_pert[star.pole_idx] = acc_pert[star.pole_idx_neighbour]

    acc_pert_face = acc_pert[star.faces].mean(axis=1)

    if update_container:
        g_eq = - np.power(10, star.log_g)[:, None] * star.normals
        total_acc = np.linalg.norm(g_eq + acc_pert_face, axis=1)
        star.log_g = np.log10(total_acc)

    if return_perturbation:
        if spherical_perturbation:
            acc_pert_sph[:, 0] *= scale
            return acc_pert_sph[star.faces].mean(axis=1) if not point_perturbations else acc_pert_sph
        else:
            return acc_pert if point_perturbations else acc_pert_face
    else:
        return None


def temp_perturbation(star, scale=1.0, update_container=False, return_perturbation=False, point_perturbations=False):
    """
    Calculates temperature perturbation on a surface of a pulsating star.

    :param star: base.container.StarContainer;
    :param scale: numpy.float; scale of the system
    :param update_container: bool; if true, the perturbations are added into surface element temperatures
    :param return_perturbation:
    :param point_perturbations: bool; if True perturbations on surface pints are returned instead face averaged values
    :return:
    """
    temp_pert = np.sum([kinematics.calculate_temperature_pert_factor(mode, scale)
                        for mode in star.pulsations.values()], axis=0)

    temp_pert_face = temp_pert[star.faces].mean(axis=1) * star.temperatures
    if update_container:
        star.temperatures += temp_pert_face

    if return_perturbation:
        return temp_pert*star.t_eff if point_perturbations else temp_pert_face
