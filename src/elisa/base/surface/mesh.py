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
