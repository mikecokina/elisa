import numpy as np

from . import utils as putils
from . surface import kinematics


def incorporate_pulsations_to_model(star_container, com_x, scale=1.0):
    """
    Function adds perturbation to the surface mesh due to pulsations.

    :param star_container: base.container.StarContainer;
    :param com_x: float;
    :param scale: numpy.float; scale of the perturbations
    :return: base.container.StarContainer;
    """
    tilted_points, tilted_points_spot = star_container.pulsations[0].points, star_container.pulsations[0].spot_points

    # initializing cumulative variables for displacement, velocity,
    displacement = np.zeros(tilted_points.shape, dtype=np.float64)
    displacement_spots = {spot_idx: np.zeros(spot.shape, dtype=np.float64)
                          for spot_idx, spot in tilted_points_spot.items()}
    velocity = np.zeros(tilted_points.shape, dtype=np.float64)
    velocity_spots = {spot_idx: np.zeros(spot.shape, dtype=np.float64)
                          for spot_idx, spot in tilted_points_spot.items()}

    # calculating kinematics quantities
    for mode_index, mode in star_container.pulsations.items():
        angular_displacement = kinematics.calculate_displacement_coordinates(
            mode, tilted_points, mode.point_harmonics, mode.point_harmonics_derivatives, scale=scale
        )
        displacement += np.real(angular_displacement)

        for spot_idx, spoints in tilted_points_spot.items():
            spot_angular_displacement = kinematics.calculate_displacement_coordinates(
                mode, spoints, mode.spot_point_harmonics[spot_idx], mode.spot_point_harmonics_derivatives[spot_idx],
                scale=scale
            )
            displacement_spots[spot_idx] += np.real(spot_angular_displacement)

    # displacement
    star_container.points = putils.derotate_surface_points(
        tilted_points + displacement,
        star_container.pulsations[0].mode_axis_phi,
        star_container.pulsations[0].mode_axis_theta,
        com_x
    )

    for spot_idx, spot in star_container.spots.items():
        # displacement
        setattr(spot, 'points',
                putils.derotate_surface_points(tilted_points_spot[spot_idx] + displacement_spots[spot_idx],
                                               star_container.pulsations[0].mode_axis_phi,
                                               star_container.pulsations[0].mode_axis_theta,
                                               com_x))

    return star_container
