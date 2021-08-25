import numpy as np

from elisa import utils


def correct_component_mesh(star, com, correction_factors):
    """
    Correcting the underestimation of the surface due to the discretization.

    :param correction_factors: numpy.array; (2*N) [discretization factor, correction factor],
    :param star: elisa.base.container.StarContainer;
    :return: elisa.base.container.StarContainer;
    """
    comv = np.array([com, 0, 0])
    args = (star.discretization_factor, correction_factors)
    centered_points = utils.discretization_correction_factor(*args) * (star.points - comv[None, :])
    star.points = centered_points + comv[None, :]

    if star.has_spots():
        for spot in star.spots.values():
            args = (spot.discretization_factor, correction_factors)
            centered_points = utils.discretization_correction_factor(*args) * (spot.points - comv[None, :])
            spot.points = centered_points + comv[None, :]

    return star


def symmetry_point_reduction(array, base_symmetry_points_number):
    """
    Returns portion of the points (or other surface point related distribution) contained in the 1st quadrant (octant).
    This is used during utilization of surface symmetries to reduce number of surface points that code works with.

    :param array: numpy.array; surface points or point related surface distribution
    :param base_symmetry_points_number: int; number of first n surface points stored in StarContainer.points that are
                                             located on a symmetrical part of the surface. E.g.:

                                             ::

                                                StarContainer.points[:StarContainer.base_symmetry_points_number]

                                              will select surface points from one quarter (eighth) of the star in case
                                              of binary (single) star system
    :return: numpy.array; reduced form of an input array from the symmetrical part of the surface
    """
    return array[:base_symmetry_points_number]
