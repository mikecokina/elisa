import numpy as np

from . import utils as putils
from .surface import kinematics
from . import pulsations


def generate_harmonics(star_container, com_x, phase, time):
    """
    Generating spherical harmonics Y_l^m in shapes(2, n_points) and (2, n_faces) and its derivatives to be subsequently
    used for calculation of perturbed properties.

    :param star_container: elisa.base.container.StarContainer;
    :param com_x: float; centre of mass for the component
    :param phase: float; rotational/orbital phase
    :param time: float; time of the observation
    :return: elisa.base.container.StarContainer; Star container with updated harmonics
    """
    if not star_container.is_flat():
        raise ValueError('Pulsations can be calculated only on flattened container.')
    star_container.points_spherical = \
        star_container.transform_points_to_spherical_coordinates(kind='points', com_x=com_x)

    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star_container, phase)
    tilted_points = putils.tilt_mode_coordinates(
        star_container.points_spherical, tilt_phi, tilt_theta
    )

    # assigning tilted points in spherical coordinates only to the first mode (the rest will share the same points)
    star_container.pulsations[0].points = tilted_points

    exponential = dict()
    norm_constant = dict()
    for mode_index, mode in star_container.pulsations.items():
        # TODO: beware of this in case you want use the container at different phase
        exponential[mode_index] = putils.generate_time_exponential(mode, time)

        # generating harmonics Y_m^l and Y_m+1^l for star and spot points
        harmonics = np.zeros((2, tilted_points.shape[0]), dtype=np.complex)
        harmonics[0] = pulsations.spherical_harmonics(mode, tilted_points, exponential[mode_index])
        if mode.m != mode.l:
            harmonics[1] = pulsations.spherical_harmonics(mode, tilted_points, exponential[mode_index],
                                                          order=mode.m + 1, degree=mode.l)

        # generating derivatives of spherical harmonics by phi an theta
        derivatives = np.empty((2, tilted_points.shape[0]), dtype=np.complex)
        derivatives[0] = pulsations.diff_spherical_harmonics_by_phi(mode, harmonics)
        derivatives[1] = pulsations.diff_spherical_harmonics_by_theta(mode, harmonics, tilted_points[:, 1],
                                                                      tilted_points[:, 2])

        # renormalizing horizontal amplitude to 1
        norm_constant[mode_index] = pulsations.horizontal_displacement_normalization(derivatives, harmonics)
        derivatives *= norm_constant[mode_index]

        # assignment of harmonics to mode instance variables
        mode.point_harmonics = harmonics[0]
        mode.face_harmonics = np.mean(mode.point_harmonics[star_container.faces], axis=1)

        mode.point_harmonics_derivatives = derivatives
        mode.face_harmonics_derivatives = np.mean(derivatives[:, star_container.faces], axis=1)

    return star_container


def incorporate_pulsations_to_model(star_container, com_x, phase, scale=1.0):
    """
    Function adds perturbation to the surface mesh due to pulsations.

    :param phase: numpy.float; (0, 1)
    :param star_container: base.container.StarContainer;
    :param com_x: float;
    :param scale: numpy.float; scale of the perturbations
    :return: base.container.StarContainer;
    """
    # calculating kinematics quantities
    complex_displacement(star_container, scale=scale)

    position_perturbation(star_container, com_x, phase, update_container=True, return_perturbation=False)
    velocity_perturbation(star_container, phase, update_container=True, return_perturbation=False)
    return star_container


def complex_displacement(star, scale):
    """
    Assigning complex displacement for surface points. Complex displacement is then used to calculate the kinematic
    quantities (r,v,a).

    :param star: base.container.StarContainer;
    :param scale: float;
    :return: base.container.StarContainer;
    """
    if not star.is_flat():
        raise ValueError('Pulsations can be calculated only on flattened container.')

    for mode_index, mode in star.pulsations.items():
        mode.complex_displacement = kinematics.calculate_displacement_coordinates(
            mode, star.pulsations[0].points, mode.point_harmonics, mode.point_harmonics_derivatives,
            scale=scale
        )

    return star


def position_perturbation(star, com_x, phase, update_container=True, return_perturbation=False):
    """
    Calculates the deformation of the surface mesh due to the pulsations.

    :param star: base.container.StarContainer;
    :param com_x: float; x-component of system's centre of mass
    :param phase: float;
    :param update_container: bool; if True, perturbation is incorporated into star.points
    :param return_perturbation: bool; if True, calculated displacement (in cartesian coordinates) is returned
    :return: Union[numpy.array, None];
    """
    # initializing cumulative variables for displacement
    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star, phase)
    displacement = None

    tilt_displacement_sph = np.sum([
        kinematics.calculate_mode_angular_displacement(mode.complex_displacement) for mode in star.pulsations.values()
    ], axis=0)

    points = putils.derotate_surface_points(
        star.pulsations[0].points + tilt_displacement_sph,
        tilt_phi, tilt_theta, com_x
    )
    if return_perturbation:
        displacement = points - getattr(star, 'points')
    if update_container:
        setattr(star, 'points', points)

    return displacement if return_perturbation else None


def velocity_perturbation(star, phase, update_container=False, return_perturbation=False):
    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star, phase)

    # calculating perturbed velocity in spherical coordinates
    tilt_velocity_sph = np.sum([
        kinematics.calculate_mode_angular_derivatives(
            displacement=mode.complex_displacement, angular_frequency=mode.angular_frequency
        ) for mode in star.pulsations.values()
    ], axis=0)

    velocity_pert = putils.derotate_surface_displacements(
        tilt_velocity_sph, star.pulsations[0].points, star.points_spherical, tilt_phi, tilt_theta
    )
    velocity_pert = velocity_pert[star.faces].mean(axis=1)

    if update_container:
        star.velocities += velocity_pert

    return velocity_pert if return_perturbation else None
